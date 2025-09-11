# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-12

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器人学 (Robotics) (7)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Exploratory Retrieval-Augmented Planning For Continual Embodied Instruction Following](https://arxiv.org/abs/2509.08222)
*Minjong Yoo, Jinwoo Jang, Wei-jin Park, Honguk Woo*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This study presents an Exploratory Retrieval-Augmented Planning (ExRAP) framework, designed to tackle continual instruction following tasks of embodied agents in dynamic, non-stationary environments. The framework enhances Large Language Models' (LLMs) embodied reasoning capabilities by efficiently exploring the physical environment and establishing the environmental context memory, thereby effectively grounding the task planning process in time-varying environment contexts. In ExRAP, given multiple continual instruction following tasks, each instruction is decomposed into queries on the environmental context memory and task executions conditioned on the query results. To efficiently handle these multiple tasks that are performed continuously and simultaneously, we implement an exploration-integrated task planning scheme by incorporating the {information-based exploration} into the LLM-based planning process. Combined with memory-augmented query evaluation, this integrated scheme not only allows for a better balance between the validity of the environmental context memory and the load of environment exploration, but also improves overall task performance. Furthermore, we devise a {temporal consistency refinement} scheme for query evaluation to address the inherent decay of knowledge in the memory. Through experiments with VirtualHome, ALFRED, and CARLA, our approach demonstrates robustness against a variety of embodied instruction following scenarios involving different instruction scales and types, and non-stationarity degrees, and it consistently outperforms other state-of-the-art LLM-based task planning approaches in terms of both goal success rate and execution efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08222) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Skeleton-based sign language recognition using a dual-stream spatio-temporal dynamic graph convolutional network](https://arxiv.org/abs/2509.08661)
*Liangjin Liu, Haoyang Zheng, Pei Zhou*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Isolated Sign Language Recognition (ISLR) is challenged by gestures that are morphologically similar yet semantically distinct, a problem rooted in the complex interplay between hand shape and motion trajectory. Existing methods, often relying on a single reference frame, struggle to resolve this geometric ambiguity. This paper introduces Dual-SignLanguageNet (DSLNet), a dual-reference, dual-stream architecture that decouples and models gesture morphology and trajectory in separate, complementary coordinate systems. Our approach utilizes a wrist-centric frame for view-invariant shape analysis and a facial-centric frame for context-aware trajectory modeling. These streams are processed by specialized networks-a topology-aware graph convolution for shape and a Finsler geometry-based encoder for trajectory-and are integrated via a geometry-driven optimal transport fusion mechanism. DSLNet sets a new state-of-the-art, achieving 93.70%, 89.97% and 99.79% accuracy on the challenging WLASL-100, WLASL-300 and LSA64 datasets, respectively, with significantly fewer parameters than competing models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08661) | **Categories:** cs.CV, cs.AI, I.2.m; I.2.0

---


## 机器人学 (Robotics) [cs.RO]
### [1] [TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals](https://arxiv.org/abs/2509.08699)
*Stefan Podgorski, Sourav Garg, Mehdi Hosseinzadeh, Lachlan Mares, Feras Dayoub, Ian Reid*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Visual navigation in robotics traditionally relies on globally-consistent 3D maps or learned controllers, which can be computationally expensive and difficult to generalize across diverse environments. In this work, we present a novel RGB-only, object-level topometric navigation pipeline that enables zero-shot, long-horizon robot navigation without requiring 3D maps or pre-trained controllers. Our approach integrates global topological path planning with local metric trajectory control, allowing the robot to navigate towards object-level sub-goals while avoiding obstacles. We address key limitations of previous methods by continuously predicting local trajectory using monocular depth and traversability estimation, and incorporating an auto-switching mechanism that falls back to a baseline controller when necessary. The system operates using foundational models, ensuring open-set applicability without the need for domain-specific fine-tuning. We demonstrate the effectiveness of our method in both simulated environments and real-world tests, highlighting its robustness and deployability. Our approach outperforms existing state-of-the-art methods, offering a more adaptable and effective solution for visual navigation in open-set environments. The source code is made publicly available: https://github.com/podgorki/TANGO.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08699) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG, cs.SY, eess.SY

---

### [2] [Mean Field Game-Based Interactive Trajectory Planning Using Physics-Inspired Unified Potential Fields](https://arxiv.org/abs/2509.08147)
*Zhen Tian, Fujiang Yuan, Chunhong Yuan, Yanhong Peng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Interactive trajectory planning in autonomous driving must balance safety, efficiency, and scalability under heterogeneous driving behaviors. Existing methods often face high computational cost or rely on external safety critics. To address this, we propose an Interaction-Enriched Unified Potential Field (IUPF) framework that fuses style-dependent benefit and risk fields through a physics-inspired variational model, grounded in mean field game theory. The approach captures conservative, aggressive, and cooperative behaviors without additional safety modules, and employs stochastic differential equations to guarantee Nash equilibrium with exponential convergence. Simulations on lane changing and overtaking scenarios show that IUPF ensures safe distances, generates smooth and efficient trajectories, and outperforms traditional optimization and game-theoretic baselines in both adaptability and computational efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08147) | **Categories:** cs.RO

---

### [3] [Diffusion-Guided Multi-Arm Motion Planning](https://arxiv.org/abs/2509.08160)
*Viraj Parimi, Brian C. Williams*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-arm motion planning is fundamental for enabling arms to complete complex long-horizon tasks in shared spaces efficiently but current methods struggle with scalability due to exponential state-space growth and reliance on large training datasets for learned models. Inspired by Multi-Agent Path Finding (MAPF), which decomposes planning into single-agent problems coupled with collision resolution, we propose a novel diffusion-guided multi-arm planner (DG-MAP) that enhances scalability of learning-based models while reducing their reliance on massive multi-arm datasets. Recognizing that collisions are primarily pairwise, we train two conditional diffusion models, one to generate feasible single-arm trajectories, and a second, to model the dual-arm dynamics required for effective pairwise collision resolution. By integrating these specialized generative models within a MAPF-inspired structured decomposition, our planner efficiently scales to larger number of arms. Evaluations against alternative learning-based methods across various team sizes demonstrate our method's effectiveness and practical applicability. Project website can be found at https://diff-mapf-mers.csail.mit.edu

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08160) | **Categories:** cs.RO, cs.AI, cs.MA

---

### [4] [Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities](https://arxiv.org/abs/2509.08302)
*Rajendramayavan Sathyam, Yueqi Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundation models are revolutionizing autonomous driving perception, transitioning the field from narrow, task-specific deep learning models to versatile, general-purpose architectures trained on vast, diverse datasets. This survey examines how these models address critical challenges in autonomous perception, including limitations in generalization, scalability, and robustness to distributional shifts. The survey introduces a novel taxonomy structured around four essential capabilities for robust performance in dynamic driving environments: generalized knowledge, spatial understanding, multi-sensor robustness, and temporal reasoning. For each capability, the survey elucidates its significance and comprehensively reviews cutting-edge approaches. Diverging from traditional method-centric surveys, our unique framework prioritizes conceptual design principles, providing a capability-driven guide for model development and clearer insights into foundational aspects. We conclude by discussing key challenges, particularly those associated with the integration of these capabilities into real-time, scalable systems, and broader deployment challenges related to computational demands and ensuring model reliability against issues like hallucinations and out-of-distribution failures. The survey also outlines crucial future research directions to enable the safe and effective deployment of foundation models in autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08302) | **Categories:** cs.RO, cs.CV

---

### [5] [PegasusFlow: Parallel Rolling-Denoising Score Sampling for Robot Diffusion Planner Flow Matching](https://arxiv.org/abs/2509.08435)
*Lei Ye, Haibo Gao, Peng Xu, Zhelin Zhang, Junqi Shan, Ao Zhang, Wei Zhang, Ruyi Zhou, Zongquan Deng, Liang Ding*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion models offer powerful generative capabilities for robot trajectory planning, yet their practical deployment on robots is hindered by a critical bottleneck: a reliance on imitation learning from expert demonstrations. This paradigm is often impractical for specialized robots where data is scarce and creates an inefficient, theoretically suboptimal training pipeline. To overcome this, we introduce PegasusFlow, a hierarchical rolling-denoising framework that enables direct and parallel sampling of trajectory score gradients from environmental interaction, completely bypassing the need for expert data. Our core innovation is a novel sampling algorithm, Weighted Basis Function Optimization (WBFO), which leverages spline basis representations to achieve superior sample efficiency and faster convergence compared to traditional methods like MPPI. The framework is embedded within a scalable, asynchronous parallel simulation architecture that supports massively parallel rollouts for efficient data collection. Extensive experiments on trajectory optimization and robotic navigation tasks demonstrate that our approach, particularly Action-Value WBFO (AVWBFO) combined with a reinforcement learning warm-start, significantly outperforms baselines. In a challenging barrier-crossing task, our method achieved a 100% success rate and was 18% faster than the next-best method, validating its effectiveness for complex terrain locomotion planning. https://masteryip.github.io/pegasusflow.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08435) | **Categories:** cs.RO

---

### [6] [Parallel, Asymptotically Optimal Algorithms for Moving Target Traveling Salesman Problems](https://arxiv.org/abs/2509.08743)
*Anoop Bhat, Geordan Gutow, Bhaskar Vundurthy, Zhongqiang Ren, Sivakumar Rathinam, Howie Choset*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The Moving Target Traveling Salesman Problem (MT-TSP) seeks an agent trajectory that intercepts several moving targets, within a particular time window for each target. In the presence of generic nonlinear target trajectories or kinematic constraints on the agent, no prior algorithm guarantees convergence to an optimal MT-TSP solution. Therefore, we introduce the Iterated Random Generalized (IRG) TSP framework. The key idea behind IRG is to alternate between randomly sampling a set of agent configuration-time points, corresponding to interceptions of targets, and finding a sequence of interception points by solving a generalized TSP (GTSP). This alternation enables asymptotic convergence to the optimum. We introduce two parallel algorithms within the IRG framework. The first algorithm, IRG-PGLNS, solves GTSPs using PGLNS, our parallelized extension of the state-of-the-art solver GLNS. The second algorithm, Parallel Communicating GTSPs (PCG), solves GTSPs corresponding to several sets of points simultaneously. We present numerical results for three variants of the MT-TSP: one where intercepting a target only requires coming within a particular distance, another where the agent is a variable-speed Dubins car, and a third where the agent is a redundant robot arm. We show that IRG-PGLNS and PCG both converge faster than a baseline based on prior work.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08743) | **Categories:** cs.RO

---

### [7] [SocialNav-SUB: Benchmarking VLMs for Scene Understanding in Social Robot Navigation](https://arxiv.org/abs/2509.08757)
*Michael J. Munje, Chen Tang, Shuijing Liu, Zichao Hu, Yifeng Zhu, Jiaxun Cui, Garrett Warnell, Joydeep Biswas, Peter Stone*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robot navigation in dynamic, human-centered environments requires socially-compliant decisions grounded in robust scene understanding. Recent Vision-Language Models (VLMs) exhibit promising capabilities such as object recognition, common-sense reasoning, and contextual understanding-capabilities that align with the nuanced requirements of social robot navigation. However, it remains unclear whether VLMs can accurately understand complex social navigation scenes (e.g., inferring the spatial-temporal relations among agents and human intentions), which is essential for safe and socially compliant robot navigation. While some recent works have explored the use of VLMs in social robot navigation, no existing work systematically evaluates their ability to meet these necessary conditions. In this paper, we introduce the Social Navigation Scene Understanding Benchmark (SocialNav-SUB), a Visual Question Answering (VQA) dataset and benchmark designed to evaluate VLMs for scene understanding in real-world social robot navigation scenarios. SocialNav-SUB provides a unified framework for evaluating VLMs against human and rule-based baselines across VQA tasks requiring spatial, spatiotemporal, and social reasoning in social robot navigation. Through experiments with state-of-the-art VLMs, we find that while the best-performing VLM achieves an encouraging probability of agreeing with human answers, it still underperforms simpler rule-based approach and human consensus baselines, indicating critical gaps in social scene understanding of current VLMs. Our benchmark sets the stage for further research on foundation models for social robot navigation, offering a framework to explore how VLMs can be tailored to meet real-world social robot navigation needs. An overview of this paper along with the code and data can be found at https://larg.github.io/socialnav-sub .

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.08757) | **Categories:** cs.RO, cs.CV

---
