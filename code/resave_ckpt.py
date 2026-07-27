from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    '<lora checkpoint directory>', 
    device_map="auto",
    trust_remote_code=True
).eval()
print("load success")
merged_model = model.merge_and_unload()
merged_model.save_pretrained('<base model checkpoint directory>', max_shard_size="2048MB", safe_serialization=True)
print("save success")
