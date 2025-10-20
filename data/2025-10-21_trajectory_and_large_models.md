# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-21

## 目录

- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (3)](#cs-ro)

## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [TGT: Text-Grounded Trajectories for Locally Controlled Video Generation](https://arxiv.org/abs/2510.15104)
*Guofeng Zhang, Angtian Wang, Jacob Zhiyuan Fang, Liming Jiang, Haotian Yang, Bo Liu, Yiding Yang, Guang Chen, Longyin Wen, Alan Yuille, Chongyang Ma*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Text-to-video generation has advanced rapidly in visual fidelity, whereas standard methods still have limited ability to control the subject composition of generated scenes. Prior work shows that adding localized text control signals, such as bounding boxes or segmentation masks, can help. However, these methods struggle in complex scenarios and degrade in multi-object settings, offering limited precision and lacking a clear correspondence between individual trajectories and visual entities as the number of controllable objects increases. We introduce Text-Grounded Trajectories (TGT), a framework that conditions video generation on trajectories paired with localized text descriptions. We propose Location-Aware Cross-Attention (LACA) to integrate these signals and adopt a dual-CFG scheme to separately modulate local and global text guidance. In addition, we develop a data processing pipeline that produces trajectories with localized descriptions of tracked entities, and we annotate two million high quality video clips to train TGT. Together, these components enable TGT to use point trajectories as intuitive motion handles, pairing each trajectory with text to control both appearance and motion. Extensive experiments show that TGT achieves higher visual quality, more accurate text alignment, and improved motion controllability compared with prior approaches. Website: https://textgroundedtraj.github.io.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15104) | **Categories:** cs.CV

---

### [2] [UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](https://arxiv.org/abs/2510.15018)
*Mingxuan Liu, Honglin He, Elisa Ricci, Wayne Wu, Bolei Zhou*

Main category: cs.CV

TL;DR: 本文提出了UrbanVerse，一个将众包城市旅游视频转换为具有物理感知和交互式仿真场景的数据驱动的real-to-sim系统。


<details>
  <summary>Details</summary>
Motivation: 现有的手工制作或程序生成的仿真场景缺乏可扩展性或无法捕捉真实世界的复杂性，因此需要能够生成多样且高保真城市环境的系统来训练城市具身AI智能体。

Method: 该方法包括：(i) UrbanVerse-100K，一个包含10万+带语义和物理属性的城市3D资产仓库；(ii) UrbanVerse-Gen，一个自动流程，可以从视频中提取场景布局并使用检索到的资产实例化度量尺度的3D仿真。

Result: 实验表明，UrbanVerse场景保留了真实世界的语义和布局，实现了与手动制作场景相当的人工评估真实感。在城市导航中，在UrbanVerse中训练的策略表现出缩放幂律和强大的泛化能力，与先前的方法相比，在仿真中成功率提高了+6.3%，在零样本sim-to-real迁移中提高了+30.1%，仅通过两次干预就完成了300米的真实世界任务。

Conclusion: UrbanVerse提供了一种有效的方法来生成高质量、真实的城市仿真环境，可用于训练和评估城市具身AI智能体。

Abstract: 城市具身AI智能体，例如送货机器人和四足机器人，正越来越多地出现在我们的城市中，在混乱的街道上导航，以提供最后一公里的连接。训练这些智能体需要多样化、高保真的城市环境来扩展，但现有的人工制作或程序生成的仿真场景要么缺乏可扩展性，要么无法捕捉真实世界的复杂性。我们介绍了UrbanVerse，这是一个数据驱动的real-to-sim系统，可以将众包的城市旅游视频转换为具有物理感知和交互式的仿真场景。UrbanVerse包括：（i）UrbanVerse-100K，一个包含10万+带语义和物理属性的城市3D资产仓库；（ii）UrbanVerse-Gen，一个自动流程，可以从视频中提取场景布局并使用检索到的资产实例化度量尺度的3D仿真。UrbanVerse在IsaacSim中运行，提供来自24个国家/地区的160个高质量构建场景，以及10个艺术家设计的测试场景的精选基准。实验表明，UrbanVerse场景保留了真实世界的语义和布局，实现了与手动制作场景相当的人工评估真实感。在城市导航中，在UrbanVerse中训练的策略表现出缩放幂律和强大的泛化能力，与先前的方法相比，在仿真中成功率提高了+6.3%，在零样本sim-to-real迁移中提高了+30.1%，仅通过两次干预就完成了300米的真实世界任务。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15018) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [3] [DriveGen3D: Boosting Feed-Forward Driving Scene Generation with Efficient Video Diffusion](https://arxiv.org/abs/2510.15264)
*Weijie Wang, Jiagang Zhu, Zeyu Zhang, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Chaojun Ni, Haoxiao Wang, Guan Huang, Xinze Chen, Yukun Zhou, Wenkang Qin, Duochao Shi, Haoyun Li, Guanghong Jia, Jiwen Lu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present DriveGen3D, a novel framework for generating high-quality and highly controllable dynamic 3D driving scenes that addresses critical limitations in existing methodologies. Current approaches to driving scene synthesis either suffer from prohibitive computational demands for extended temporal generation, focus exclusively on prolonged video synthesis without 3D representation, or restrict themselves to static single-scene reconstruction. Our work bridges this methodological gap by integrating accelerated long-term video generation with large-scale dynamic scene reconstruction through multimodal conditional control. DriveGen3D introduces a unified pipeline consisting of two specialized components: FastDrive-DiT, an efficient video diffusion transformer for high-resolution, temporally coherent video synthesis under text and Bird's-Eye-View (BEV) layout guidance; and FastRecon3D, a feed-forward reconstruction module that rapidly builds 3D Gaussian representations across time, ensuring spatial-temporal consistency. Together, these components enable real-time generation of extended driving videos (up to $424\times800$ at 12 FPS) and corresponding dynamic 3D scenes, achieving SSIM of 0.811 and PSNR of 22.84 on novel view synthesis, all while maintaining parameter efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15264) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Internalizing World Models via Self-Play Finetuning for Agentic RL](https://arxiv.org/abs/2510.15047)
*Shiqi Chen, Tongyao Zhu, Zian Wang, Jinghan Zhang, Kangrui Wang, Siyang Gao, Teng Xiao, Yee Whye Teh, Junxian He, Manling Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground their internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; we observe Pass@k--the probability that at least one of (k) sampled trajectories succeeds--drops markedly across training steps, indicating brittle exploration and limited generalization. Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision-making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling. Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised finetuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance. Experiments across diverse environments like Sokoban, FrozenLake, and Sudoku show that our approach significantly improves performance. For example, SPA boosts the Sokoban success rate from 25.6% to 59.8% and raises the FrozenLake score from 22.1% to 70.9% for the Qwen2.5-1.5B-Instruct model.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15047) | **Categories:** cs.LG, cs.CL

---

### [2] [Spatiotemporal Transformers for Predicting Avian Disease Risk from Migration Trajectories](https://arxiv.org/abs/2510.15254)
*Dingya Feng, Dingyuan Xue*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate forecasting of avian disease outbreaks is critical for wildlife conservation and public health. This study presents a Transformer-based framework for predicting the disease risk at the terminal locations of migratory bird trajectories. We integrate multi-source datasets, including GPS tracking data from Movebank, outbreak records from the World Organisation for Animal Health (WOAH), and geospatial context from GADM and Natural Earth. The raw coordinates are processed using H3 hierarchical geospatial encoding to capture spatial patterns. The model learns spatiotemporal dependencies from bird movement sequences to estimate endpoint disease risk. Evaluation on a held-out test set demonstrates strong predictive performance, achieving an accuracy of 0.9821, area under the ROC curve (AUC) of 0.9803, average precision (AP) of 0.9299, and an F1-score of 0.8836 at the optimal threshold. These results highlight the potential of Transformer architectures to support early-warning systems for avian disease surveillance, enabling timely intervention and prevention strategies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15254) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [VDRive: Leveraging Reinforced VLA and Diffusion Policy for End-to-end Autonomous Driving](https://arxiv.org/abs/2510.15446)
*Ziang Guo, Zufeng Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In autonomous driving, dynamic environment and corner cases pose significant challenges to the robustness of ego vehicle's state understanding and decision making. We introduce VDRive, a novel pipeline for end-to-end autonomous driving that explicitly models state-action mapping to address these challenges, enabling interpretable and robust decision making. By leveraging the advancement of the state understanding of the Vision Language Action Model (VLA) with generative diffusion policy-based action head, our VDRive guides the driving contextually and geometrically. Contextually, VLA predicts future observations through token generation pre-training, where the observations are represented as discrete codes by a Conditional Vector Quantized Variational Autoencoder (CVQ-VAE). Geometrically, we perform reinforcement learning fine-tuning of the VLA to predict future trajectories and actions based on current driving conditions. VLA supplies the current state tokens and predicted state tokens for the action policy head to generate hierarchical actions and trajectories. During policy training, a learned critic evaluates the actions generated by the policy and provides gradient-based feedback, forming an actor-critic framework that enables a reinforcement-based policy learning pipeline. Experiments show that our VDRive achieves state-of-the-art performance in the Bench2Drive closed-loop benchmark and nuScenes open-loop planning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15446) | **Categories:** cs.RO

---

### [2] [Perfect Prediction or Plenty of Proposals? What Matters Most in Planning for Autonomous Driving](https://arxiv.org/abs/2510.15505)
*Aron Distelzweig, Faris Janjoš, Oliver Scheel, Sirish Reddy Varra, Raghu Rajan, Joschka Boedecker*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditionally, prediction and planning in autonomous driving (AD) have been treated as separate, sequential modules. Recently, there has been a growing shift towards tighter integration of these components, known as Integrated Prediction and Planning (IPP), with the aim of enabling more informed and adaptive decision-making. However, it remains unclear to what extent this integration actually improves planning performance. In this work, we investigate the role of prediction in IPP approaches, drawing on the widely adopted Val14 benchmark, which encompasses more common driving scenarios with relatively low interaction complexity, and the interPlan benchmark, which includes highly interactive and out-of-distribution driving situations. Our analysis reveals that even access to perfect future predictions does not lead to better planning outcomes, indicating that current IPP methods often fail to fully exploit future behavior information. Instead, we focus on high-quality proposal generation, while using predictions primarily for collision checks. We find that many imitation learning-based planners struggle to generate realistic and plausible proposals, performing worse than PDM - a simple lane-following approach. Motivated by this observation, we build on PDM with an enhanced proposal generation method, shifting the emphasis towards producing diverse but realistic and high-quality proposals. This proposal-centric approach significantly outperforms existing methods, especially in out-of-distribution and highly interactive settings, where it sets new state-of-the-art results.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15505) | **Categories:** cs.RO

---

### [3] [Few-Shot Demonstration-Driven Task Coordination and Trajectory Execution for Multi-Robot Systems](https://arxiv.org/abs/2510.15686)
*Taehyeon Kim, Vishnunandan L. N. Venkatesh, Byung-Cheol Min*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this paper, we propose a novel few-shot learning framework for multi-robot systems that integrate both spatial and temporal elements: Few-Shot Demonstration-Driven Task Coordination and Trajectory Execution (DDACE). Our approach leverages temporal graph networks for learning task-agnostic temporal sequencing and Gaussian Processes for spatial trajectory modeling, ensuring modularity and generalization across various tasks. By decoupling temporal and spatial aspects, DDACE requires only a small number of demonstrations, significantly reducing data requirements compared to traditional learning from demonstration approaches. To validate our proposed framework, we conducted extensive experiments in task environments designed to assess various aspects of multi-robot coordination-such as multi-sequence execution, multi-action dynamics, complex trajectory generation, and heterogeneous configurations. The experimental results demonstrate that our approach successfully achieves task execution under few-shot learning conditions and generalizes effectively across dynamic and diverse settings. This work underscores the potential of modular architectures in enhancing the practicality and scalability of multi-robot systems in real-world applications. Additional materials are available at https://sites.google.com/view/ddace.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15686) | **Categories:** cs.RO

---
