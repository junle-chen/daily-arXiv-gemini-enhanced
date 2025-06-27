# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-28

## 目录

- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (7)](#cs-ro)

## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction](https://arxiv.org/abs/2506.21121)
*Muleilan Pei, Shaoshuai Shi, Lu Zhang, Peiliang Li, Shaojie Shen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction for surrounding agents is a challenging task in autonomous driving due to its inherent uncertainty and underlying multimodality. Unlike prevailing data-driven methods that primarily rely on supervised learning, in this paper, we introduce a novel Graph-oriented Inverse Reinforcement Learning (GoIRL) framework, which is an IRL-based predictor equipped with vectorized context representations. We develop a feature adaptor to effectively aggregate lane-graph features into grid space, enabling seamless integration with the maximum entropy IRL paradigm to infer the reward distribution and obtain the policy that can be sampled to induce multiple plausible plans. Furthermore, conditioned on the sampled plans, we implement a hierarchical parameterized trajectory generator with a refinement module to enhance prediction accuracy and a probability fusion strategy to boost prediction confidence. Extensive experimental results showcase our approach not only achieves state-of-the-art performance on the large-scale Argoverse & nuScenes motion forecasting benchmarks but also exhibits superior generalization abilities compared to existing supervised models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21121) | **Categories:** cs.CV, cs.RO

---

### [2] [Out-of-Distribution Semantic Occupancy Prediction](https://arxiv.org/abs/2506.21185)
*Yuheng Zhang, Mengfei Duan, Kunyu Peng, Yuhang Wang, Ruiping Liu, Fei Teng, Kai Luo, Zhiyong Li, Kailun Yang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: 3D Semantic Occupancy Prediction is crucial for autonomous driving, providing a dense, semantically rich environmental representation. However, existing methods focus on in-distribution scenes, making them susceptible to Out-of-Distribution (OoD) objects and long-tail distributions, which increases the risk of undetected anomalies and misinterpretations, posing safety hazards. To address these challenges, we introduce Out-of-Distribution Semantic Occupancy Prediction, targeting OoD detection in 3D voxel space. To fill the gaps in the dataset, we propose a Synthetic Anomaly Integration Pipeline that injects synthetic anomalies while preserving realistic spatial and occlusion patterns, enabling the creation of two datasets: VAA-KITTI and VAA-KITTI-360. We introduce OccOoD, a novel framework integrating OoD detection into 3D semantic occupancy prediction, with Voxel-BEV Progressive Fusion (VBPF) leveraging an RWKV-based branch to enhance OoD detection via geometry-semantic fusion. Experimental results demonstrate that OccOoD achieves state-of-the-art OoD detection with an AuROC of 67.34% and an AuPRCr of 29.21% within a 1.2m region, while maintaining competitive occupancy prediction performance. The established datasets and source code will be made publicly available at https://github.com/7uHeng/OccOoD.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21185) | **Categories:** cs.CV, cs.RO, eess.IV

---

### [3] [Real-Time ESFP: Estimating, Smoothing, Filtering, and Pose-Mapping](https://arxiv.org/abs/2506.21234)
*Qifei Cui, Yuang Zhou, Ruichen Deng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents ESFP, an end-to-end pipeline that converts monocular RGB video into executable joint trajectories for a low-cost 4-DoF desktop arm. ESFP comprises four sequential modules. (1) Estimating: ROMP lifts each frame to a 24-joint 3-D skeleton. (2) Smoothing: the proposed HPSTM-a sequence-to-sequence Transformer with self-attention-combines long-range temporal context with a differentiable forward-kinematics decoder, enforcing constant bone lengths and anatomical plausibility while jointly predicting joint means and full covariances. (3) Filtering: root-normalized trajectories are variance-weighted according to HPSTM's uncertainty estimates, suppressing residual noise. (4) Pose-Mapping: a geometric retargeting layer transforms shoulder-elbow-wrist triples into the uArm's polar workspace, preserving wrist orientation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21234) | **Categories:** cs.CV, cs.RO

---

### [4] [Whole-Body Conditioned Egocentric Video Prediction](https://arxiv.org/abs/2506.21552)
*Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, Jitendra Malik*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We train models to Predict Ego-centric Video from human Actions (PEVA), given the past video and an action represented by the relative 3D body pose. By conditioning on kinematic pose trajectories, structured by the joint hierarchy of the body, our model learns to simulate how physical human actions shape the environment from a first-person point of view. We train an auto-regressive conditional diffusion transformer on Nymeria, a large-scale dataset of real-world egocentric video and body pose capture. We further design a hierarchical evaluation protocol with increasingly challenging tasks, enabling a comprehensive analysis of the model's embodied prediction and control abilities. Our work represents an initial attempt to tackle the challenges of modeling complex real-world environments and embodied agent behaviors with video prediction from the perspective of a human.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21552) | **Categories:** cs.CV, cs.AI, cs.LG, cs.MM, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Omniwise: Predicting GPU Kernels Performance with LLMs](https://arxiv.org/abs/2506.20886)
*Zixian Wang, Cole Ramos, Muhammad A. Awad, Keith Lowery*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In recent years, the rapid advancement of deep neural networks (DNNs) has revolutionized artificial intelligence, enabling models with unprecedented capabilities in understanding, generating, and processing complex data. These powerful architectures have transformed a wide range of downstream applications, tackling tasks beyond human reach. In this paper, we introduce Omniwise, the first end-to-end, self-supervised fine-tuning pipeline that applies large language models (LLMs) to GPU kernel performance prediction--a novel use case in performance profiling. Omniwise is model-agnostic and lightweight, achieving strong results even with a small 3B-parameter model. It can predict key performance metrics, including memory bandwidth, cache hit rates, GFLOPs, and arithmetic intensity, directly from kernel code without the need for code execution or profiling tools. Our approach achieves over 90% of predictions within 10% relative error on GPU kernels executed on AMD MI250 and MI300X architectures. In addition to the pipeline, we develop an online inference server and a Visual Studio Code plugin that seamlessly integrate LLM-based performance prediction into developers' workflows.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20886) | **Categories:** cs.LG, cs.AI

---

### [2] [Curriculum-Guided Antifragile Reinforcement Learning for Secure UAV Deconfliction under Observation-Space Attacks](https://arxiv.org/abs/2506.21129)
*Deepak Kumar Panda, Adolfo Perrusquia, Weisi Guo*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) policies deployed in safety-critical systems, such as unmanned aerial vehicle (UAV) navigation in dynamic airspace, are vulnerable to out-ofdistribution (OOD) adversarial attacks in the observation space. These attacks induce distributional shifts that significantly degrade value estimation, leading to unsafe or suboptimal decision making rendering the existing policy fragile. To address this vulnerability, we propose an antifragile RL framework designed to adapt against curriculum of incremental adversarial perturbations. The framework introduces a simulated attacker which incrementally increases the strength of observation-space perturbations which enables the RL agent to adapt and generalize across a wider range of OOD observations and anticipate previously unseen attacks. We begin with a theoretical characterization of fragility, formally defining catastrophic forgetting as a monotonic divergence in value function distributions with increasing perturbation strength. Building on this, we define antifragility as the boundedness of such value shifts and derive adaptation conditions under which forgetting is stabilized. Our method enforces these bounds through iterative expert-guided critic alignment using Wasserstein distance minimization across incrementally perturbed observations. We empirically evaluate the approach in a UAV deconfliction scenario involving dynamic 3D obstacles. Results show that the antifragile policy consistently outperforms standard and robust RL baselines when subjected to both projected gradient descent (PGD) and GPS spoofing attacks, achieving up to 15% higher cumulative reward and over 30% fewer conflict events. These findings demonstrate the practical and theoretical viability of antifragile reinforcement learning for secure and resilient decision-making in environments with evolving threat scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21129) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [V2X-REALM: Vision-Language Model-Based Robust End-to-End Cooperative Autonomous Driving with Adaptive Long-Tail Modeling](https://arxiv.org/abs/2506.21041)
*Junwei You, Pei Li, Zhuoyu Jiang, Zilin Huang, Rui Gan, Haotian Shi, Bin Ran*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring robust planning and decision-making under rare, diverse, and visually degraded long-tail scenarios remains a fundamental challenge for autonomous driving in urban environments. This issue becomes more critical in cooperative settings, where vehicles and infrastructure jointly perceive and reason across complex environments. To address this challenge, we propose V2X-REALM, a vision-language model (VLM)-based framework with adaptive multimodal learning for robust cooperative autonomous driving under long-tail scenarios. V2X-REALM introduces three core innovations: (i) a prompt-driven long-tail scenario generation and evaluation pipeline that leverages foundation models to synthesize realistic long-tail conditions such as snow and fog across vehicle- and infrastructure-side views, enriching training diversity efficiently; (ii) a gated multi-scenario adaptive attention module that modulates the visual stream using scenario priors to recalibrate ambiguous or corrupted features; and (iii) a multi-task scenario-aware contrastive learning objective that improves multimodal alignment and promotes cross-scenario feature separability. Extensive experiments demonstrate that V2X-REALM significantly outperforms existing baselines in robustness, semantic reasoning, safety, and planning accuracy under complex, challenging driving conditions, advancing the scalability of end-to-end cooperative autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21041) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [2] [Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations](https://arxiv.org/abs/2506.21205)
*Elia Trevisan, Khaled A. Mustafa, Godert Notten, Xinwei Wang, Javier Alonso-Mora*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Deploying mobile robots safely among humans requires the motion planner to account for the uncertainty in the other agents' predicted trajectories. This remains challenging in traditional approaches, especially with arbitrarily shaped predictions and real-time constraints. To address these challenges, we propose a Dynamic Risk-Aware Model Predictive Path Integral control (DRA-MPPI), a motion planner that incorporates uncertain future motions modelled with potentially non-Gaussian stochastic predictions. By leveraging MPPI's gradient-free nature, we propose a method that efficiently approximates the joint Collision Probability (CP) among multiple dynamic obstacles for several hundred sampled trajectories in real-time via a Monte Carlo (MC) approach. This enables the rejection of samples exceeding a predefined CP threshold or the integration of CP as a weighted objective within the navigation cost function. Consequently, DRA-MPPI mitigates the freezing robot problem while enhancing safety. Real-world and simulated experiments with multiple dynamic obstacles demonstrate DRA-MPPI's superior performance compared to state-of-the-art approaches, including Scenario-based Model Predictive Control (S-MPC), Frenet planner, and vanilla MPPI.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21205) | **Categories:** cs.RO

---

### [3] [Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends](https://arxiv.org/abs/2506.20966)
*Tian-Yu Xiang, Ao-Qun Jin, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Sheng-Bin Duan, Fu-Chao Xie, Wen-Kai Wang, Si-Cheng Wang, Ling-Yun Li, Tian Tu, Zeng-Guang Hou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-language-action (VLA) models extend vision-language models (VLM) by integrating action generation modules for robotic manipulation. Leveraging strengths of VLM in vision perception and instruction understanding, VLA models exhibit promising generalization across diverse manipulation tasks. However, applications demanding high precision and accuracy reveal performance gaps without further adaptation. Evidence from multiple domains highlights the critical role of post-training to align foundational models with downstream applications, spurring extensive research on post-training VLA models. VLA model post-training aims to address the challenge of improving an embodiment's ability to interact with the environment for the given tasks, analogous to the process of humans motor skills acquisition. Accordingly, this paper reviews post-training strategies for VLA models through the lens of human motor learning, focusing on three dimensions: environments, embodiments, and tasks. A structured taxonomy is introduced aligned with human learning mechanisms: (1) enhancing environmental perception, (2) improving embodiment awareness, (3) deepening task comprehension, and (4) multi-component integration. Finally, key challenges and trends in post-training VLA models are identified, establishing a conceptual framework to guide future research. This work delivers both a comprehensive overview of current VLA model post-training methods from a human motor learning perspective and practical insights for VLA model development. (Project website: https://github.com/AoqunJin/Awesome-VLA-Post-Training)

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20966) | **Categories:** cs.RO, cs.AI

---

### [4] [STEP Planner: Constructing cross-hierarchical subgoal tree as an embodied long-horizon task planner](https://arxiv.org/abs/2506.21030)
*Zhou Tianxing, Wang Zhirui, Ao Haojia, Chen Guangyan, Xing Boyang, Cheng Jingwen, Yang Yi, Yue Yufeng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The ability to perform reliable long-horizon task planning is crucial for deploying robots in real-world environments. However, directly employing Large Language Models (LLMs) as action sequence generators often results in low success rates due to their limited reasoning ability for long-horizon embodied tasks. In the STEP framework, we construct a subgoal tree through a pair of closed-loop models: a subgoal decomposition model and a leaf node termination model. Within this framework, we develop a hierarchical tree structure that spans from coarse to fine resolutions. The subgoal decomposition model leverages a foundation LLM to break down complex goals into manageable subgoals, thereby spanning the subgoal tree. The leaf node termination model provides real-time feedback based on environmental states, determining when to terminate the tree spanning and ensuring each leaf node can be directly converted into a primitive action. Experiments conducted in both the VirtualHome WAH-NL benchmark and on real robots demonstrate that STEP achieves long-horizon embodied task completion with success rates up to 34% (WAH-NL) and 25% (real robot) outperforming SOTA methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21030) | **Categories:** cs.RO

---

### [5] [ACTLLM: Action Consistency Tuned Large Language Model](https://arxiv.org/abs/2506.21250)
*Jing Bi, Lianggong Bruce Wen, Zhang Liu, Chenliang Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces ACTLLM (Action Consistency Tuned Large Language Model), a novel approach for robot manipulation in dynamic environments. Traditional vision-based systems often struggle to learn visual representations that excel in both task execution and spatial reasoning, thereby limiting their adaptability in dynamic environments. ACTLLM addresses these challenges by harnessing language to craft structured scene descriptors, providing a uniform interface for both spatial understanding and task performance through flexible language instructions. Moreover, we introduce a novel action consistency constraint that aligns visual perception with corresponding actions, thereby enhancing the learning of actionable visual representations. Additionally, we have reformulated the Markov decision process for manipulation tasks into a multi-turn visual dialogue framework. This approach enables the modeling of long-term task execution with enhanced contextual relevance derived from the history of task execution. During our evaluation, ACTLLM excels in diverse scenarios, proving its effectiveness on challenging vision-based robot manipulation tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21250) | **Categories:** cs.RO

---

### [6] [Active Disturbance Rejection Control for Trajectory Tracking of a Seagoing USV: Design, Simulation, and Field Experiments](https://arxiv.org/abs/2506.21265)
*Jelmer van der Saag, Elia Trevisan, Wouter Falkena, Javier Alonso-Mora*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unmanned Surface Vessels (USVs) face significant control challenges due to uncertain environmental disturbances like waves and currents. This paper proposes a trajectory tracking controller based on Active Disturbance Rejection Control (ADRC) implemented on the DUS V2500. A custom simulation incorporating realistic waves and current disturbances is developed to validate the controller's performance, supported by further validation through field tests in the harbour of Scheveningen, the Netherlands, and at sea. Simulation results demonstrate that ADRC significantly reduces cross-track error across all tested conditions compared to a baseline PID controller but increases control effort and energy consumption. Field trials confirm these findings while revealing a further increase in energy consumption during sea trials compared to the baseline.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21265) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [7] [WorldVLA: Towards Autoregressive Action World Model](https://arxiv.org/abs/2506.21539)
*Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, Hao Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present WorldVLA, an autoregressive action world model that unifies action and image understanding and generation. Our WorldVLA intergrates Vision-Language-Action (VLA) model and world model in one single framework. The world model predicts future images by leveraging both action and image understanding, with the purpose of learning the underlying physics of the environment to improve action generation. Meanwhile, the action model generates the subsequent actions based on image observations, aiding in visual understanding and in turn helps visual generation of the world model. We demonstrate that WorldVLA outperforms standalone action and world models, highlighting the mutual enhancement between the world model and the action model. In addition, we find that the performance of the action model deteriorates when generating sequences of actions in an autoregressive manner. This phenomenon can be attributed to the model's limited generalization capability for action prediction, leading to the propagation of errors from earlier actions to subsequent ones. To address this issue, we propose an attention mask strategy that selectively masks prior actions during the generation of the current action, which shows significant performance improvement in the action chunk generation task.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21539) | **Categories:** cs.RO, cs.AI

---
