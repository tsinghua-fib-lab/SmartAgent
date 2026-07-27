import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from typing import Dict, Any
import yaml
from copy import deepcopy
from similarities import BertSimilarity
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--stage_0', type=bool, default=False) # end-to-end training
parser.add_argument('--stage_1', type=bool, default=False) # two-stage training #1
parser.add_argument('--stage_2', type=bool, default=False) # two-stage training #2
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--comment', type=str)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# calculate underlying cosine similarity
def calculate_BertSimilarity(pred=str, label=str):
    sim_model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-chinese")
    Cosine_score = sim_model.similarity(pred, label)
    Cosine_score = round(Cosine_score.item(), 4)
    return Cosine_score

action_type_to_id = {
    'click': 4,
    'type': 3,
    'scroll up': 1,
    'scroll down': 0,
    'scroll left': 8,
    'scroll right': 9,
    'exit': 6,
    'complete': 10,
    'recommendation': 2 
}

# convert action to prediction format (and return the groundtruth bbox)
def action2step(action, image_size, action_type_to_id, return_bbox=False):
    action_type = action["action_type"]
    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]   
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
                action["bbox"]["y"] + action["bbox"]["height"]]
        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type == 'click': 
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_to_id[action_type], click_point)

    elif action_type == 'type':
        typed_text = action["type_text"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"type_text\": \"{}\"}}".format(action_type_to_id[action_type], click_point, typed_text)

    elif action_type == 'recommendation':
        user_preference = action["user_preference"]
        action_step = "{{\"action_type\": {}, \"preference\": \"{}\"}}".format(action_type_to_id[action_type], user_preference)

    elif action_type in ['scroll up', 'scroll down', 'scroll left', 'scroll right', 'exit', 'complete']:
        action_step = "{{\"action_type\": {}}}".format(action_type_to_id[action_type])

    if return_bbox:
        return action_step, bbox
    else:
        return action_step

# Perceiver model
model_path_root = '<pre-trained model directory>'
model_path = os.path.join(model_path_root, args.model_path)
qwen_path = 'Qwen/Qwen-VL-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval() # load with model checkpoint
# model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()  # load with lora checkpoint
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

image_root_dir = '<image directory>'
annot_root_dir = '<annotation directory>'
annot_root_dir = os.path.join(annot_root_dir, args.test_data) 
test_data = json.load(open(annot_root_dir, 'r'))

results = []
thought = {}


# Reasoner model
Qwen_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-VL-Chat', device_map="cuda", trust_remote_code=True, bf16=True).eval()
Qwen_model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)


# Prompts
prompt_origin = "Please generate the next action according to the ui screenshot, user's instruction: {} and previous actions: {}. You have to generate as the format for each action: for action 4: {{\"action_type\": , \"click_point\": }}, for action 3: {{\"action_type\": , \"click_point\": ,\"type_text\": }}, for the others: {{\"action_type\": }}"
think_prompt = "Given the action goal ({}), the action ({}) just clicked a button/content on the screenshot. Tell me the text of the clicked button/content in ONE simple Chinese sentence. Within the action, action_type indicates what type of actions (4 for click, 3 for type), click_point means the point clicked on the screenshot, and type_text means text typed on the screenshot. Output Example1: 该动作点击了名为'单人餐'的按钮. Output Example2: 该动作键入了一段文本：'北京菜'. Output Example3: 该动作点击了一处文本内容：'14-17点'"
summary_prompt = "Complete the user's ask ({}), using keywords in the step-wise actions ({}). Output Format: 用户似乎在寻找XXX/用户可能是需要XXX/看起来用户想要找XXX. Example1: 我需要从杭州前往济南航班，请给我一些推荐 -> 用户似乎在寻找从杭州前往济南[经济舱]航班，出发时间为[后天]. Example2: 你能告诉我一些关于昆明的酒店的信息吗 -> 用户似乎在寻找昆明的[四星]酒店，价格在[￥400-600]之间并提供[新装修]. Example3: 你能推荐一些北京菜吗-> 用户似乎打算尝试一下[单人餐]的北京菜，时间为[10-14点]. Example4: 帮我找一些止痛药 -> 用户想要[专送]的[成人]止痛药. Example5: 你能推荐一些在映影片吗 -> 用户可能在寻找[可改签票]的在映影片，希望在最近的[Dolby Cinema厅]影院观看."

for episode in tqdm(test_data):
    goal = episode["goal"]
    ep_id = episode["episode_id"]
    underlying = episode["underlying"]
    previous_actions = []
    results_actions = []
    episode_thought = ""

    for j, step in enumerate(episode["actions"]):    
        
        if args.stage_1 and step["action_type"] == "recommendation":
            continue

        # get screenshot
        filename = step["action_uid"] + '.PNG'
        filechannel = os.path.join(step["action_uid"].split('-')[-1].split('_')[0], episode["episode_id"])
        if 'MUT' in filename:
            if 'FOOD' in filename: filechannel = os.path.join('FLIGHT+FOOD', episode["episode_id"])
            if 'HOTEL' in filename: filechannel = os.path.join('FLIGHT+HOTEL', episode["episode_id"])
        img_path = os.path.join(os.path.join(image_root_dir, filechannel), filename)
        image = Image.open(img_path)

        # get previous_step
        previous_step = ""
        for i, action in enumerate(previous_actions[-8:]):
            previous_step += 'Step' + str(i) + ': ' + action + ' '

        action_step = action2step(step, image.size, action_type_to_id)
        previous_actions.append(action_step)
        
        # get input prompt
        prompt = prompt_origin.format(goal, previous_step)

        # get groundtruth
        action_step_ref, bbox_ref = action2step(step, image.size, action_type_to_id, return_bbox=True)
        try:
            action_step_ref = ast.literal_eval(action_step_ref)
        except:
            logging.info("groundtruth format wrong")

        # get response
        query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}])
        with torch.no_grad():
            response, history = model.chat(tokenizer, query=query, history=None)

        step_result = {
            "step_id": step["action_uid"], "action_type": action_step_ref["action_type"], 
            "response": response, 
            "Op_match": False, "Ele_match": False, "Op_F1": 0, "Exp_match": 0, "Imp_match": False
            }

        think_result = {
            "step_id": step["action_uid"], "action_type": action_step_ref["action_type"], 
            "think": "" 
        }
            

        try:
            action_pred = ast.literal_eval(response)

            # get step thought
            if args.stage_1 and step["action_type"] in ["click", "type"]:
                query_think = tokenizer.from_list_format([{'image': img_path}, {'text': think_prompt.format(goal, action_pred)}])
                with torch.no_grad():
                    think, history = Qwen_model.chat(tokenizer, query=query_think, history=None)
                think_result["think"] = think
                episode_thought += f'Step{j}: ' + think + '; '
                logging.info(f"{think_result}")

            # get episode summarization
            if args.stage_1 and step["action_type"] in ["complete"]:
                query_summary = tokenizer.from_list_format([{'text': summary_prompt.format(goal, episode_thought)}])
                with torch.no_grad():
                    summary, history = Qwen_model.chat(tokenizer, query=query_summary, history=None)
                thought[ep_id] = summary

                # check Exp_match
                Exp_match = calculate_BertSimilarity(summary, underlying)
                step_result["Exp_match"] = Exp_match

                logging.info(f"SUMMARY: {summary}, Exp_match: {Exp_match}")

            ## check Op_match
            if action_pred["action_type"] == action_step_ref["action_type"]:
                step_result["Op_match"] = True

            ## check Ele_match
            if action_pred["action_type"] == action_type_to_id["click"] or action_pred["action_type"] == action_type_to_id["type"]:
                click_point = action_pred["click_point"]
                if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                    step_result["Ele_match"] = True
            
            ## check Op_F1
            pred_str = str(action_pred["action_type"])
            if action_pred["action_type"] == action_type_to_id["type"]: 
                pred_str += ' '
                pred_str += action_pred["type_text"]

            ref_str = str(action_step_ref["action_type"])
            if action_step_ref["action_type"] == action_type_to_id["type"]:
                ref_str += ' '
                ref_str += action_step_ref["type_text"]

            op_f1 = calculate_f1(pred_str, ref_str)
            step_result["Op_F1"] = op_f1

            ## check Imp_match
            if action_pred["action_type"] == action_type_to_id["recommendation"] and action_pred["preference"] == action_step_ref["preference"]:
                step_result["Imp_match"] = True

            logging.info(f"{step_result}")

        except: 
            logging.info("format wrong")

        results_actions.append(step_result)
    
    results.append(results_actions)

# save results
results_to_save = {}
results_to_save["model_path"] = model_path
results_to_save["test_data"] = annot_root_dir
results_to_save["results"] = results
results_to_save["thought"] = thought

save_url = os.path.join('<save directory>', args.comment + model_path.split('/')[-1])
if not os.path.exists(save_url): 
    os.makedirs(save_url)
json.dump(results_to_save, open(os.path.join(save_url, f"use_{model_path.split('/')[-1]}.json"), "w"))


# update goals for second-stage training
if args.stage_1: 
    annotation_root_dir = '<annotation directory>'
    phase_1 = os.path.join(annotation_root_dir, '<annotation file>')
    phase_1_data = json.load(open(phase_1, 'r', encoding='utf-8'))

    for episode in tqdm(phase_1_data):
        if episode["episode_id"] in thought:
            episode["pred_underlying"] = thought[episode["episode_id"]]
        else:
            episode["pred_underlying"] = episode["goal"]
            
    json.dump(phase_1_data, open(os.path.join(annotation_root_dir, '<new annotation file>'), "w"), ensure_ascii=False)


# calculate overall metrics
num_episode = 0
num_step = 0
num_step_success = 0
num_episode_success = 0

num_op_match = 0

num_ele_match = 0
num_ele_step = 0

op_f1 = {val: [] for key, val in action_type_to_id.items() if key in ['type']}

macro_ele_acc = {}
macro_action_f1 = {}
macro_step_acc = {}
macro_explicit_preference_acc = []
macro_implicit_preference_acc = {}

for i, episode in enumerate(results): 
    macro_ele_acc[i] = []
    macro_action_f1[i] = []
    macro_implicit_preference_acc[i] = []
    macro_step_acc[i] = []

    num_episode += 1
    episode_success = True

    for step_result in episode: 
        num_step += 1 
        if step_result["Op_match"]:
            num_op_match += 1

        # macro_ele_acc
        if step_result["action_type"] in [action_type_to_id["click"], action_type_to_id["type"]]:
            num_ele_step += 1
            if step_result["Ele_match"]:
                num_ele_match += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

        # macro_action_f1
        if step_result["action_type"] in [action_type_to_id["type"]]:
            op_f1[step_result["action_type"]].append(step_result["Op_F1"])
        macro_action_f1[i].append(step_result["Op_F1"])

        # macro_explicit_preference_acc
        if step_result["action_type"] == action_type_to_id["complete"]:
            macro_explicit_preference_acc.append(step_result["Exp_match"])

        # macro_implicit_preference_acc
        if step_result["action_type"] == action_type_to_id["recommendation"]:
            macro_implicit_preference_acc[i].append(step_result["Imp_match"])

        # macro_step_acc
        if step_result["Op_match"]:
            if step_result["action_type"] == action_type_to_id["click"] and step_result["Ele_match"]: 
                num_step_success += 1; macro_step_acc[i].append(1)
            elif step_result["action_type"] == action_type_to_id["type"] and step_result["Ele_match"] and step_result["Op_F1"] == 1.0: 
                num_step_success += 1; macro_step_acc[i].append(1)
            elif step_result["action_type"] == action_type_to_id["recommendation"] and step_result["Imp_match"]: 
                num_step_success += 1; macro_step_acc[i].append(1)
            elif step_result["action_type"] in [1, 0, 8, 9, 6, 10]: 
                num_step_success += 1; macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False
        else:
            macro_step_acc[i].append(0)
            episode_success = False

    if episode_success:
        num_episode_success += 1

logging.info("pretrain ckpt used: " + model_path)
logging.info("--"*30)

logging.info("Step Success: " + str(num_step_success / num_step))
logging.info("Episode Success (GP): " + str(num_episode_success / num_episode))

logging.info("Element Acc: " + str(num_ele_match / num_ele_step))
logging.info("Operation F1: " + str(np.mean([np.mean(x) for x in op_f1.values()])))
logging.info("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

logging.info("--"*30)
logging.info("Macro Ele Acc: " + str(np.mean([np.mean(x) for x in macro_ele_acc.values()])))
logging.info("Macro Op F1: " + str(np.mean([np.mean(x) for x in macro_action_f1.values()])))
logging.info("Macro Step SR: " + str(np.mean([np.mean(x) for x in macro_step_acc.values()])))
logging.info("Macro_Exp_pref_acc: " + str(np.mean(macro_explicit_preference_acc)))
logging.info("Macro_Imp_pref_acc: " + str(np.mean([np.mean(x) for x in macro_implicit_preference_acc.values()])))


def which_channel(ep_id=str):
    if 'FLIGHT' in ep_id and 'FOOD' in ep_id: return 'FLIGHT_FOOD'
    if 'FLIGHT' in ep_id and 'HOTEL' in ep_id: return 'FLIGHT_HOTEL'
    if 'FOOD' in ep_id: return 'FOOD'
    if 'HOTEL' in ep_id: return 'HOTEL'
    if 'FLIGHT' in ep_id: return 'FLIGHT'
    if 'MOVIE' in ep_id: return 'MOVIE'


# calculate channel metrics
num_op_match = 0

num_ele_match = 0
num_ele_step = 0

op_f1 = {val: [] for key, val in action_type_to_id.items() if key in ['type']}

base_metrics = {
    'num_episode': 0,
    'num_step': 0,
    'num_step_success': 0,
    'num_episode_success': 0,
    'macro_ele_acc': {},
    'macro_action_f1': {}, 
    'macro_step_acc': {}, 
    'macro_explicit_preference_acc': [], 
    'macro_implicit_preference_acc': {}
}
categories = ["FOOD", "HOTEL", "FLIGHT", "MOVIE", "FLIGHT_FOOD", "FLIGHT_HOTEL"]
sub_metrics = {category: deepcopy(base_metrics) for category in categories}

for i, episode in enumerate(results): 
    channel = which_channel(episode[0]["step_id"])
    
    sub_metrics[channel]["macro_ele_acc"][i] = []
    sub_metrics[channel]["macro_action_f1"][i] = []
    sub_metrics[channel]["macro_step_acc"][i] = []
    sub_metrics[channel]["macro_implicit_preference_acc"][i] = []

    sub_metrics[channel]["num_episode"] += 1
    episode_success = True

    for step_result in episode: 
        sub_metrics[channel]["num_step"] += 1 
        if step_result["Op_match"]:
            num_op_match += 1

        # macro_ele_acc
        if step_result["action_type"] in [action_type_to_id["click"], action_type_to_id["type"]]:
            num_ele_step += 1
            if step_result["Ele_match"]:
                num_ele_match += 1
                sub_metrics[channel]["macro_ele_acc"][i].append(1)
            else:
                sub_metrics[channel]["macro_ele_acc"][i].append(0)

        # macro_action_f1
        if step_result["action_type"] in [action_type_to_id["type"]]: 
            op_f1[step_result["action_type"]].append(step_result["Op_F1"])
        sub_metrics[channel]["macro_action_f1"][i].append(step_result["Op_F1"])

        # macro_explicit_preference_acc
        if step_result["action_type"] == action_type_to_id["complete"]:
            sub_metrics[channel]["macro_explicit_preference_acc"].append(step_result["Exp_match"])

        # macro_implicit_preference_acc
        if step_result["action_type"] == action_type_to_id["recommendation"]: 
            sub_metrics[channel]["macro_implicit_preference_acc"][i].append(step_result["Imp_match"])

        # macro_step_acc
        if step_result["Op_match"]:
            if step_result["action_type"] == action_type_to_id["click"] and step_result["Ele_match"]: 
                sub_metrics[channel]["num_step_success"] += 1; sub_metrics[channel]["macro_step_acc"][i].append(1)
            elif step_result["action_type"] == action_type_to_id["type"] and step_result["Ele_match"] and step_result["Op_F1"] == 1.0: 
                sub_metrics[channel]["num_step_success"] += 1; sub_metrics[channel]["macro_step_acc"][i].append(1)
            elif step_result["action_type"] == action_type_to_id["recommendation"] and step_result["Imp_match"]: 
                sub_metrics[channel]["num_step_success"] += 1; sub_metrics[channel]["macro_step_acc"][i].append(1)
            elif step_result["action_type"] in [1, 0, 8, 9, 6, 10]: 
                sub_metrics[channel]["num_step_success"] += 1; sub_metrics[channel]["macro_step_acc"][i].append(1)
            else:
                sub_metrics[channel]["macro_step_acc"][i].append(0)
                episode_success = False
        else:
            sub_metrics[channel]["macro_step_acc"][i].append(0)
            episode_success = False

    if episode_success:
        sub_metrics[channel]["num_episode_success"] += 1

logging.info("--"*15 + " each channel " + "--"*15)
for key,channel_i in sub_metrics.items():
    logging.info(f"channel: {key}:")
    logging.info("Step Success: " + str(channel_i["num_step_success"] / channel_i["num_step"]))
    logging.info("Episode Success (GP): " + str(channel_i["num_episode_success"] / channel_i["num_episode"]))

    logging.info("--"*20)
    logging.info("Macro Ele Acc: " + str(np.mean([np.mean(x) for x in channel_i["macro_ele_acc"].values()])))
    logging.info("Macro Op F1: " + str(np.mean([np.mean(x) for x in channel_i["macro_action_f1"].values()])))
    logging.info("Macro Step SR: " + str(np.mean([np.mean(x) for x in channel_i["macro_step_acc"].values()])))
    logging.info("Macro_Exp_pref_acc: " + str(np.mean(channel_i["macro_explicit_preference_acc"])))
    logging.info("Macro_Imp_pref_acc: " + str(np.mean([np.mean(x) for x in channel_i["macro_implicit_preference_acc"].values()])))
    logging.info("--"*40)
