# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-31

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器人学 (Robotics) (3)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning](https://arxiv.org/abs/2510.25320)
*Jiaqi Wu, Qinlao Zhao, Zefeng Chen, Kai Qin, Yifei Zhao, Xueqian Wang, Yuhang Yao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous agents powered by large language models (LLMs) have shown impressive capabilities in tool manipulation for complex task-solving. However, existing paradigms such as ReAct rely on sequential reasoning and execution, failing to exploit the inherent parallelism among independent sub-tasks. This sequential bottleneck leads to inefficient tool utilization and suboptimal performance in multi-step reasoning scenarios. We introduce Graph-based Agent Planning (GAP), a novel framework that explicitly models inter-task dependencies through graph-based planning to enable adaptive parallel and serial tool execution. Our approach trains agent foundation models to decompose complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies. This dependency-aware orchestration achieves substantial improvements in both execution efficiency and task accuracy. To train GAP, we construct a high-quality dataset of graph-based planning traces derived from the Multi-Hop Question Answering (MHQA) benchmark. We employ a two-stage training strategy: supervised fine-tuning (SFT) on the curated dataset, followed by reinforcement learning (RL) with a correctness-based reward function on strategically sampled queries where tool-based reasoning provides maximum value. Experimental results on MHQA datasets demonstrate that GAP significantly outperforms traditional ReAct baselines, particularly on multi-step retrieval tasks, while achieving dramatic improvements in tool invocation efficiency through intelligent parallelization. The project page is available at: https://github.com/WJQ7777/Graph-Agent-Planning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.25320) | **Categories:** cs.AI, cs.CL

---

### [2] [Navigation in a Three-Dimensional Urban Flow using Deep Reinforcement Learning](https://arxiv.org/abs/2510.25679)
*Federica Tonti, Ricardo Vinuesa*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unmanned Aerial Vehicles (UAVs) are increasingly populating urban areas for delivery and surveillance purposes. In this work, we develop an optimal navigation strategy based on Deep Reinforcement Learning. The environment is represented by a three-dimensional high-fidelity simulation of an urban flow, characterized by turbulence and recirculation zones. The algorithm presented here is a flow-aware Proximal Policy Optimization (PPO) combined with a Gated Transformer eXtra Large (GTrXL) architecture, giving the agent richer information about the turbulent flow field in which it navigates. The results are compared with a PPO+GTrXL without the secondary prediction tasks, a PPO combined with Long Short Term Memory (LSTM) cells and a traditional navigation algorithm. The obtained results show a significant increase in the success rate (SR) and a lower crash rate (CR) compared to a PPO+LSTM, PPO+GTrXL and the classical Zermelo's navigation algorithm, paving the way to a completely reimagined UAV landscape in complex urban environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.25679) | **Categories:** cs.AI, physics.flu-dyn

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DrivingScene: A Multi-Task Online Feed-Forward 3D Gaussian Splatting Method for Dynamic Driving Scenes](https://arxiv.org/abs/2510.24734)
*Qirui Hou, Wenzhang Sun, Chang Zeng, Chunfeng Wang, Hao Li, Jianxun Cui*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Real-time, high-fidelity reconstruction of dynamic driving scenes is challenged by complex dynamics and sparse views, with prior methods struggling to balance quality and efficiency. We propose DrivingScene, an online, feed-forward framework that reconstructs 4D dynamic scenes from only two consecutive surround-view images. Our key innovation is a lightweight residual flow network that predicts the non-rigid motion of dynamic objects per camera on top of a learned static scene prior, explicitly modeling dynamics via scene flow. We also introduce a coarse-to-fine training paradigm that circumvents the instabilities common to end-to-end approaches. Experiments on nuScenes dataset show our image-only method simultaneously generates high-quality depth, scene flow, and 3D Gaussian point clouds online, significantly outperforming state-of-the-art methods in both dynamic reconstruction and novel view synthesis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24734) | **Categories:** cs.CV, cs.LG, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [SoraNav: Adaptive UAV Task-Centric Navigation via Zeroshot VLM Reasoning](https://arxiv.org/abs/2510.25191)
*Hongyu Song, Rishabh Dev Yadav, Cheng Guo, Wei Pan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Interpreting visual observations and natural language instructions for complex task execution remains a key challenge in robotics and AI. Despite recent advances, language-driven navigation is still difficult, particularly for UAVs in small-scale 3D environments. Existing Vision-Language Navigation (VLN) approaches are mostly designed for ground robots and struggle to generalize to aerial tasks that require full 3D spatial reasoning. The emergence of large Vision-Language Models (VLMs), such as GPT and Claude, enables zero-shot semantic reasoning from visual and textual inputs. However, these models lack spatial grounding and are not directly applicable to navigation. To address these limitations, SoraNav is introduced, an adaptive UAV navigation framework that integrates zero-shot VLM reasoning with geometry-aware decision-making. Geometric priors are incorporated into image annotations to constrain the VLM action space and improve decision quality. A hybrid switching strategy leverages navigation history to alternate between VLM reasoning and geometry-based exploration, mitigating dead-ends and redundant revisits. A PX4-based hardware-software platform, comprising both a digital twin and a physical micro-UAV, enables reproducible evaluation. Experimental results show that in 2.5D scenarios, our method improves Success Rate (SR) by 25.7% and Success weighted by Path Length (SPL) by 17%. In 3D scenarios, it improves SR by 29.5% and SPL by 18.5% relative to the baseline.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.25191) | **Categories:** cs.RO

---

### [2] [SCOUT: A Lightweight Framework for Scenario Coverage Assessment in Autonomous Driving](https://arxiv.org/abs/2510.24949)
*Anil Yildiz, Sarah M. Thornton, Carl Hildebrandt, Sreeja Roy-Singh, Mykel J. Kochenderfer*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Assessing scenario coverage is crucial for evaluating the robustness of autonomous agents, yet existing methods rely on expensive human annotations or computationally intensive Large Vision-Language Models (LVLMs). These approaches are impractical for large-scale deployment due to cost and efficiency constraints. To address these shortcomings, we propose SCOUT (Scenario Coverage Oversight and Understanding Tool), a lightweight surrogate model designed to predict scenario coverage labels directly from an agent's latent sensor representations. SCOUT is trained through a distillation process, learning to approximate LVLM-generated coverage labels while eliminating the need for continuous LVLM inference or human annotation. By leveraging precomputed perception features, SCOUT avoids redundant computations and enables fast, scalable scenario coverage estimation. We evaluate our method across a large dataset of real-life autonomous navigation scenarios, demonstrating that it maintains high accuracy while significantly reducing computational cost. Our results show that SCOUT provides an effective and practical alternative for large-scale coverage analysis. While its performance depends on the quality of LVLM-generated training labels, SCOUT represents a major step toward efficient scenario coverage oversight in autonomous systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24949) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [3] [Integrating Legal and Logical Specifications in Perception, Prediction, and Planning for Automated Driving: A Survey of Methods](https://arxiv.org/abs/2510.25386)
*Kumar Manas, Mert Keser, Alois Knoll*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This survey provides an analysis of current methodologies integrating legal and logical specifications into the perception, prediction, and planning modules of automated driving systems. We systematically explore techniques ranging from logic-based frameworks to computational legal reasoning approaches, emphasizing their capability to ensure regulatory compliance and interpretability in dynamic and uncertain driving environments. A central finding is that significant challenges arise at the intersection of perceptual reliability, legal compliance, and decision-making justifiability. To systematically analyze these challenges, we introduce a taxonomy categorizing existing approaches by their theoretical foundations, architectural implementations, and validation strategies. We particularly focus on methods that address perceptual uncertainty and incorporate explicit legal norms, facilitating decisions that are both technically robust and legally defensible. The review covers neural-symbolic integration methods for perception, logic-driven rule representation, and norm-aware prediction strategies, all contributing toward transparent and accountable autonomous vehicle operation. We highlight critical open questions and practical trade-offs that must be addressed, offering multidisciplinary insights from engineering, logic, and law to guide future developments in legally compliant autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.25386) | **Categories:** cs.RO, cs.AI, cs.SY, eess.SY

---


## eess.SY [eess.SY]
### [1] [Incorporating Social Awareness into Control of Unknown Multi-Agent Systems: A Real-Time Spatiotemporal Tubes Approach](https://arxiv.org/abs/2510.25597)
*Siddhartha Upadhyay, Ratnangshu Das, Pushpak Jagtap*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a decentralized control framework that incorporates social awareness into multi-agent systems with unknown dynamics to achieve prescribed-time reach-avoid-stay tasks in dynamic environments. Each agent is assigned a social awareness index that quantifies its level of cooperation or self-interest, allowing heterogeneous social behaviors within the system. Building on the spatiotemporal tube (STT) framework, we propose a real-time STT framework that synthesizes tubes online for each agent while capturing its social interactions with others. A closed-form, approximation-free control law is derived to ensure that each agent remains within its evolving STT, thereby avoiding dynamic obstacles while also preventing inter-agent collisions in a socially aware manner, and reaching the target within a prescribed time. The proposed approach provides formal guarantees on safety and timing, and is computationally lightweight, model-free, and robust to unknown disturbances. The effectiveness and scalability of the framework are validated through simulation and hardware experiments on a 2D omnidirectional

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.25597) | **Categories:** eess.SY, cs.RO, cs.SY

---
