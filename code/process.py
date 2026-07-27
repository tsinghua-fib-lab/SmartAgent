import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import json
import os
from tqdm import tqdm
import random
import argparse
import numpy as np
from typing import Dict, Any
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--stage_0', type=bool, default=False) # end-to-end training
parser.add_argument('--stage_1', type=bool, default=False) # two-stage training #1
parser.add_argument('--stage_2', type=bool, default=False) # two-stage training #2
parser.add_argument('--train_json_file', type=str)
args = parser.parse_args()

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

# convert action to prediction format
def action2step(action, image_size, action_type_to_id):
    action_type = action["action_type"]
    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

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

    return action_step, action_type_to_id[action_type]


image_root_dir = '<image directory>'
annotation_root_dir = '<annotation directory>'
annotation_data = os.path.join(annotation_root_dir, args.train_json_file)
train_data = json.load(open(annotation_data, 'r', encoding='utf-8'))
train_step = []

prompt_origin = "Please generate the next action according to the ui screenshot, user's instruction: {} and previous actions: {}. You have to generate as the format for each action: for action 4: {{\"action_type\": , \"click_point\": }}, for action 3: {{\"action_type\": , \"click_point\": ,\"type_text\": }}, for the others: {{\"action_type\": }}"
step_i = 0

for episode in tqdm(train_data):
    goal = episode["goal"]
    annot_id = episode["episode_id"]
    underlying = episode["underlying"]
    previous_actions = []
    previous_items_preferred = []

    if args.stage_2:
        new_goal = episode["pred_underlying"]

    for step in episode["actions"]:
        
        if args.stage_1 and step["action_type"] == "recommendation":
            continue            

        # Few actions can not find its corresponding bbox, jump these actions
        if "bbox" not in step:
            print("action not found")
            continue

        filename = step["action_uid"] + '.PNG'
        filechannel = os.path.join(step["action_uid"].split('-')[-1].split('_')[0], episode["episode_id"])
        if 'MUT' in filename:
            if 'FOOD' in filename: filechannel = os.path.join('FLIGHT+FOOD', episode["episode_id"])
            if 'HOTEL' in filename: filechannel = os.path.join('FLIGHT+HOTEL', episode["episode_id"])

        img_path = os.path.join(os.path.join(image_root_dir, filechannel), filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)

        previous_step = ""
        for i, action in enumerate(previous_actions[-8:]):
            previous_step += 'Step' + str(i) + ':' + action + ' '

        previous_items = ""
        for i, item in enumerate(previous_items_preferred[-8:]):
            previous_items += 'Item' + str(i) + ':' + item + ' '

        action_step, action_id = action2step(step, image.size, action_type_to_id)

        if action_id == 2:
            previous_items_preferred.append(action_step)
        else:
            previous_actions.append(action_step)

        prompt = prompt_origin.format(goal, previous_step) 
        if args.stage_2:
            prompt = prompt_origin.format(new_goal, previous_step) 

        conversations = []
        conv_user = {"from": "user", "value": "Picture: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        conv_ai = {"from": "assistant", "value": str(action_step)}
        conversations.append(conv_user)
        conversations.append(conv_ai)

        train_step.append({"id": "SmartSpot_1_phase_{}".format(step_i), "conversations": conversations})
        step_i += 1

random.shuffle(train_step)
print("Num of total step: " + str(len(train_step)))
json.dump(train_step, open(os.path.join(annotation_root_dir, 'train_data_sft.json'), "w"))
