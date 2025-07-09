# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-10

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (6)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Chat2SPaT: A Large Language Model Based Tool for Automating Traffic Signal Control Plan Management](https://arxiv.org/abs/2507.05283)
*Yue Wang, Miao Zhou, Guijing Huang, Rui Zhuo, Chao Yi, Zhenliang Ma*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Pre-timed traffic signal control, commonly used for operating signalized intersections and coordinated arterials, requires tedious manual work for signaling plan creating and updating. When the time-of-day or day-of-week plans are utilized, one intersection is often associated with multiple plans, leading to further repetitive manual plan parameter inputting. To enable a user-friendly traffic signal control plan management process, this study proposes Chat2SPaT, a method to convert users' semi-structured and ambiguous descriptions on the signal control plan to exact signal phase and timing (SPaT) results, which could further be transformed into structured stage-based or ring-based plans to interact with intelligent transportation system (ITS) software and traffic signal controllers. With curated prompts, Chat2SPaT first leverages large language models' (LLMs) capability of understanding users' plan descriptions and reformulate the plan as a combination of phase sequence and phase attribute results in the json format. Based on LLM outputs, python scripts are designed to locate phases in a cycle, address nuances of traffic signal control, and finally assemble the complete traffic signal control plan. Within a chat, the pipeline can be utilized iteratively to conduct further plan editing. Experiments show that Chat2SPaT can generate plans with an accuracy of over 94% for both English and Chinese cases, using a test dataset with over 300 plan descriptions. As the first benchmark for evaluating LLMs' capability of understanding traffic signal control plan descriptions, Chat2SPaT provides an easy-to-use plan management pipeline for traffic practitioners and researchers, serving as a potential new building block for a more accurate and versatile application of LLMs in the field of ITS. The source codes, prompts and test dataset are openly accessible at https://github.com/yuewangits/Chat2SPaT.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05283) | **Categories:** cs.AI, cs.CL

---

### [2] [A Wireless Foundation Model for Multi-Task Prediction](https://arxiv.org/abs/2507.05938)
*Yucheng Sheng, Jiacheng Wang, Xingyu Zhou, Le Liang, Hao Ye, Shi Jin, Geoffrey Ye Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the growing complexity and dynamics of the mobile communication networks, accurately predicting key system parameters, such as channel state information (CSI), user location, and network traffic, has become essential for a wide range of physical (PHY)-layer and medium access control (MAC)-layer tasks. Although traditional deep learning (DL)-based methods have been widely applied to such prediction tasks, they often struggle to generalize across different scenarios and tasks. In response, we propose a unified foundation model for multi-task prediction in wireless networks that supports diverse prediction intervals. The proposed model enforces univariate decomposition to unify heterogeneous tasks, encodes granularity for interval awareness, and uses a causal Transformer backbone for accurate predictions. Additionally, we introduce a patch masking strategy during training to support arbitrary input lengths. After trained on large-scale datasets, the proposed foundation model demonstrates strong generalization to unseen scenarios and achieves zero-shot performance on new tasks that surpass traditional full-shot baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05938) | **Categories:** cs.AI, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Motion Generation: A Survey of Generative Approaches and Benchmarks](https://arxiv.org/abs/2507.05419)
*Aliasghar Khani, Arianna Rampini, Bruno Roy, Larasika Nadela, Noa Kaplan, Evan Atherton, Derek Cheung, Jacky Bibliowicz*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Motion generation, the task of synthesizing realistic motion sequences from various conditioning inputs, has become a central problem in computer vision, computer graphics, and robotics, with applications ranging from animation and virtual agents to human-robot interaction. As the field has rapidly progressed with the introduction of diverse modeling paradigms including GANs, autoencoders, autoregressive models, and diffusion-based techniques, each approach brings its own advantages and limitations. This growing diversity has created a need for a comprehensive and structured review that specifically examines recent developments from the perspective of the generative approach employed.   In this survey, we provide an in-depth categorization of motion generation methods based on their underlying generative strategies. Our main focus is on papers published in top-tier venues since 2023, reflecting the most recent advancements in the field. In addition, we analyze architectural principles, conditioning mechanisms, and generation settings, and compile a detailed overview of the evaluation metrics and datasets used across the literature. Our objective is to enable clearer comparisons and identify open challenges, thereby offering a timely and foundational reference for researchers and practitioners navigating the rapidly evolving landscape of motion generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05419) | **Categories:** cs.CV, cs.LG

---

### [2] [Driving as a Diagnostic Tool: Scenario-based Cognitive Assessment in Older Drivers From Driving Video](https://arxiv.org/abs/2507.05463)
*Md Zahid Hasan, Guillermo Basulto-Elias, Jun Ha Chang, Sahuna Hallmark, Matthew Rizzo, Anuj Sharma, Soumik Sarkar*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce scenario-based cognitive status identification in older drivers from Naturalistic driving videos and large vision models. In recent times, cognitive decline, including Alzheimer's disease (AD) and mild cognitive impairment (MCI), is often underdiagnosed due to the time-consuming and costly nature of current diagnostic methods. By analyzing real-world driving behavior captured through in-vehicle systems, this research aims to extract "digital fingerprints" that correlate with functional decline and clinical features of MCI and AD. Moreover, modern large vision models can draw meaningful insights from everyday driving patterns of older patients to early detect cognitive decline. We propose a framework that uses large vision models and naturalistic driving videos to analyze driver behavior, classify cognitive status and predict disease progression. We leverage the strong relationship between real-world driving behavior as an observation of the current cognitive status of the drivers where the vehicle can be utilized as a "diagnostic tool". Our method identifies early warning signs of functional impairment, contributing to proactive intervention strategies. This work enhances early detection and supports the development of scalable, non-invasive monitoring systems to mitigate the growing societal and economic burden of cognitive decline in the aging population.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05463) | **Categories:** cs.CV, cs.AI

---

### [3] [LiON-LoRA: Rethinking LoRA Fusion to Unify Controllable Spatial and Temporal Generation for Video Diffusion](https://arxiv.org/abs/2507.05678)
*Yisu Zhang, Chenjie Cao, Chaohui Yu, Jianke Zhu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video Diffusion Models (VDMs) have demonstrated remarkable capabilities in synthesizing realistic videos by learning from large-scale data. Although vanilla Low-Rank Adaptation (LoRA) can learn specific spatial or temporal movement to driven VDMs with constrained data, achieving precise control over both camera trajectories and object motion remains challenging due to the unstable fusion and non-linear scalability. To address these issues, we propose LiON-LoRA, a novel framework that rethinks LoRA fusion through three core principles: Linear scalability, Orthogonality, and Norm consistency. First, we analyze the orthogonality of LoRA features in shallow VDM layers, enabling decoupled low-level controllability. Second, norm consistency is enforced across layers to stabilize fusion during complex camera motion combinations. Third, a controllable token is integrated into the diffusion transformer (DiT) to linearly adjust motion amplitudes for both cameras and objects with a modified self-attention mechanism to ensure decoupled control. Additionally, we extend LiON-LoRA to temporal generation by leveraging static-camera videos, unifying spatial and temporal controllability. Experiments demonstrate that LiON-LoRA outperforms state-of-the-art methods in trajectory control accuracy and motion strength adjustment, achieving superior generalization with minimal training data. Project Page: https://fuchengsu.github.io/lionlora.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05678) | **Categories:** cs.CV

---

### [4] [Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models](https://arxiv.org/abs/2507.05822)
*L'ea Dubois, Klaus Schmidt, Chengyu Wang, Ji-Hoon Park, Lin Wang, Santiago Munoz*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05822) | **Categories:** cs.CV, CS, I.2.10

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Dynamic Campus Origin-Destination Mobility Prediction using Graph Convolutional Neural Network on WiFi Logs](https://arxiv.org/abs/2507.05507)
*Godwin Badu-Marfo, Bilal Farooq*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present an integrated graph-based neural networks architecture for predicting campus buildings occupancy and inter-buildings movement at dynamic temporal resolution that learns traffic flow patterns from Wi-Fi logs combined with the usage schedules within the buildings. The relative traffic flows are directly estimated from the WiFi data without assuming the occupant behaviour or preferences while maintaining individual privacy. We formulate the problem as a data-driven graph structure represented by a set of nodes (representing buildings), connected through a route of edges or links using a novel Graph Convolution plus LSTM Neural Network (GCLSTM) which has shown remarkable success in modelling complex patterns. We describe the formulation, model estimation, interpretability and examine the relative performance of our proposed model. We also present an illustrative architecture of the models and apply on real-world WiFi logs collected at the Toronto Metropolitan University campus. The results of the experiments show that the integrated GCLSTM models significantly outperform traditional pedestrian flow estimators like the Multi Layer Perceptron (MLP) and Linear Regression.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05507) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [LeAD: The LLM Enhanced Planning System Converged with End-to-end Autonomous Driving](https://arxiv.org/abs/2507.05754)
*Yuhang Zhang, Jiaqi Liu, Chengkai Xu, Peng Hang, Jian Sun*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A principal barrier to large-scale deployment of urban autonomous driving systems lies in the prevalence of complex scenarios and edge cases. Existing systems fail to effectively interpret semantic information within traffic contexts and discern intentions of other participants, consequently generating decisions misaligned with skilled drivers' reasoning patterns. We present LeAD, a dual-rate autonomous driving architecture integrating imitation learning-based end-to-end (E2E) frameworks with large language model (LLM) augmentation. The high-frequency E2E subsystem maintains real-time perception-planning-control cycles, while the low-frequency LLM module enhances scenario comprehension through multi-modal perception fusion with HD maps and derives optimal decisions via chain-of-thought (CoT) reasoning when baseline planners encounter capability limitations. Our experimental evaluation in the CARLA Simulator demonstrates LeAD's superior handling of unconventional scenarios, achieving 71 points on Leaderboard V1 benchmark, with a route completion of 93%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05754) | **Categories:** cs.RO, cs.AI

---

### [2] [CRED: Counterfactual Reasoning and Environment Design for Active Preference Learning](https://arxiv.org/abs/2507.05458)
*Yi-Shiuan Tung, Bradley Hayes, Alessandro Roncone*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: For effective real-world deployment, robots should adapt to human preferences, such as balancing distance, time, and safety in delivery routing. Active preference learning (APL) learns human reward functions by presenting trajectories for ranking. However, existing methods often struggle to explore the full trajectory space and fail to identify informative queries, particularly in long-horizon tasks. We propose CRED, a trajectory generation method for APL that improves reward estimation by jointly optimizing environment design and trajectory selection. CRED "imagines" new scenarios through environment design and uses counterfactual reasoning -- by sampling rewards from its current belief and asking "What if this reward were the true preference?" -- to generate a diverse and informative set of trajectories for ranking. Experiments in GridWorld and real-world navigation using OpenStreetMap data show that CRED improves reward learning and generalizes effectively across different environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05458) | **Categories:** cs.RO

---

### [3] [DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving](https://arxiv.org/abs/2507.05710)
*Hyeongchan Ham, Heejin Ahn*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05710) | **Categories:** cs.RO

---

### [4] [A Learning-based Planning and Control Framework for Inertia Drift Vehicles](https://arxiv.org/abs/2507.05748)
*Bei Zhou, Zhouheng Li, Lei Xie, Hongye Su, Johannes Betz*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Inertia drift is a transitional maneuver between two sustained drift stages in opposite directions, which provides valuable insights for navigating consecutive sharp corners for autonomous racing.However, this can be a challenging scenario for the drift controller to handle rapid transitions between opposing sideslip angles while maintaining accurate path tracking. Moreover, accurate drift control depends on a high-fidelity vehicle model to derive drift equilibrium points and predict vehicle states, but this is often compromised by the strongly coupled longitudinal-lateral drift dynamics and unpredictable environmental variations. To address these challenges, this paper proposes a learning-based planning and control framework utilizing Bayesian optimization (BO), which develops a planning logic to ensure a smooth transition and minimal velocity loss between inertia and sustained drift phases. BO is further employed to learn a performance-driven control policy that mitigates modeling errors for enhanced system performance. Simulation results on an 8-shape reference path demonstrate that the proposed framework can achieve smooth and stable inertia drift through sharp corners.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05748) | **Categories:** cs.RO

---

### [5] [Fast and Accurate Collision Probability Estimation for Autonomous Vehicles using Adaptive Sigma-Point Sampling](https://arxiv.org/abs/2507.06149)
*Charles Champagne Cossette, Taylor Scott Clawson, Andrew Feit*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A novel algorithm is presented for the estimation of collision probabilities between dynamic objects with uncertain trajectories, where the trajectories are given as a sequence of poses with Gaussian distributions. We propose an adaptive sigma-point sampling scheme, which ultimately produces a fast, simple algorithm capable of estimating the collision probability with a median error of 3.5%, and a median runtime of 0.21ms, when measured on an Intel Xeon Gold 6226R Processor. Importantly, the algorithm explicitly accounts for the collision probability's temporal dependence, which is often neglected in prior work and otherwise leads to an overestimation of the collision probability. Finally, the method is tested on a diverse set of relevant real-world scenarios, consisting of 400 6-second snippets of autonomous vehicle logs, where the accuracy and latency is rigorously evaluated.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06149) | **Categories:** cs.RO, cs.AI, cs.CG

---

### [6] [Learning Agile Tensile Perching for Aerial Robots from Demonstrations](https://arxiv.org/abs/2507.06172)
*Kangle Yuan, Atar Babgei, Luca Romanello, Hai-Nguyen Nguyen, Ronald Clark, Mirko Kovac, Sophie F. Armanini, Basaran Bahadir Kocer*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Perching on structures such as trees, beams, and ledges is essential for extending the endurance of aerial robots by enabling energy conservation in standby or observation modes. A tethered tensile perching mechanism offers a simple, adaptable solution that can be retrofitted to existing robots and accommodates a variety of structure sizes and shapes. However, tethered tensile perching introduces significant modelling challenges which require precise management of aerial robot dynamics, including the cases of tether slack & tension, and momentum transfer. Achieving smooth wrapping and secure anchoring by targeting a specific tether segment adds further complexity. In this work, we present a novel trajectory framework for tethered tensile perching, utilizing reinforcement learning (RL) through the Soft Actor-Critic from Demonstrations (SACfD) algorithm. By incorporating both optimal and suboptimal demonstrations, our approach enhances training efficiency and responsiveness, achieving precise control over position and velocity. This framework enables the aerial robot to accurately target specific tether segments, facilitating reliable wrapping and secure anchoring. We validate our framework through extensive simulation and real-world experiments, and demonstrate effectiveness in achieving agile and reliable trajectory generation for tensile perching.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06172) | **Categories:** cs.RO

---
