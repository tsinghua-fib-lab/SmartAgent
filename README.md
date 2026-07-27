# SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World
This is the anonymous repository for submission "SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World".

# Requirements
```
pip install -r requirements.txt
```
# File Description
`process.py` - data preprocess \
`finetune.py` - main model file \
`test` - model evaluation \
`resave_ckpt.py` - save lora checkpoints as standard base model checkpoints \
`run.sh` - running bash script \
`requirements.txt` - python environments required \
`ds_config_zero2` - deepspeed setup for GPU acceleration 

# Datasets
We release the SmartSpot benchmark, including [images](https://drive.google.com/file/d/1LCXAxkhVRnHLh1e5TK3L_nzDym3qZa2b/view?usp=drive_link) and [anotations](https://drive.google.com/file/d/16Q7pa5ifgkmiLJ6x6qN0CuF1aLyNP72l/view?usp=drive_link). We also provide the [anotations in English](https://drive.google.com/file/d/1tmVyjdjKPFJt3bPMqpcvCL4iwmZ5G3J4/view?usp=drive_link).

