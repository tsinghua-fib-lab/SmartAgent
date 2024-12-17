<p align="center" width="100%">
  <img src='assets/SmartAgent_banner2_1.png' width="100%">
</p>


# SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World
The official repository of our paper "[SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World](https://arxiv.org/pdf/2412.07472)". We will release code and data upon paper notification.

# Blogs
Find SmartAgent in [Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning) ![GitHub stars](https://img.shields.io/github/stars/atfortes/Awesome-LLM-Reasoning?color=yellow) with more brilliant works on LLMs/MLLMs reasoning.

# Chain-of-User-Thought (COUT) Reasoning Paradigm
We formulate COUT to achieve embodied personalized agent training in terms of three stages of thought. In Thought #1, according to a user's instruction, an agent performs GUI actions to search for an item pool. In Thought #2 with seeing the pool, the agent reasons underlying requirements behind the original instruction, as implied by the previous actions. In Thought #3, based on the underlying thought, the agent recommends items within the pool to complete the user's instruction. 

<p align="center" width="100%">
  <img src='assets/COUT.png' width="100%">
</p>
By leveraging user-oriented thoughts, this COUT could enable full-stage embodied personalized capabilities across various information systems. 

# SmartSpot Benchmark
The first environment supports full-stage embodied personalized evaluation.

<p align="center" width="100%">
  <img src='assets/Datasets_statistics.png' width="100%">
</p>


# SmartAgent Model
The capabilities of SmartAgent from basic embodied operations to personalized reasoning.

<p align="center" width="100%">
  <img src='assets/SmartAgent_capabilities.png' width="100%">
</p>


Two-stage training paradigm of SmartAgent.

<p align="center" width="100%">
  <img src='assets/Two_stage_model.png' width="100%">
</p>

# Citation
Please consider citing our paper and staring this repo if you find SmartAgent helpful in your work, thanks!

```bib
@article{zhang2024smartagent,
      title={SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World}, 
      author={Zhang, Jiaqi and Gao, Chen and Zhang, Liyuan and Li, Yong and Yin, Hongzhi},
      journal={arXiv preprint arXiv:2412.07472},
      year={2024}
}
```
