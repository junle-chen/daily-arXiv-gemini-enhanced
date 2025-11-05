# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-06

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [cs.MA (1)](#cs-ma)
- [机器人学 (Robotics) (4)](#cs-ro)
- [cs.SI (1)](#cs-si)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance](https://arxiv.org/abs/2511.02119)
*Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However, participation rates among at-risk populations in the United States remain strikingly low. This gap underscores the need to understand and model the behavioral mechanisms underlying insurance decisions. Large language models (LLMs) have recently exhibited human-like intelligence across wide-ranging tasks, offering promising tools for simulating human decision-making. This study constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent, an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action, and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey data, capturing contextual information that is intractable for traditional models. The memory module supports the simulation of temporal decision evolutions, illustrated through a roller coaster life trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02119) | **Categories:** cs.AI, cs.CL

---

### [2] [Optimal-Agent-Selection: State-Aware Routing Framework for Efficient Multi-Agent Collaboration](https://arxiv.org/abs/2511.02200)
*Jingbo Wang, Sendong Zhao, Haochun Wang, Yuzheng Fan, Lizhe Zhang, Yan Liu, Ting Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The emergence of multi-agent systems powered by large language models (LLMs) has unlocked new frontiers in complex task-solving, enabling diverse agents to integrate unique expertise, collaborate flexibly, and address challenges unattainable for individual models. However, the full potential of such systems is hindered by rigid agent scheduling and inefficient coordination strategies that fail to adapt to evolving task requirements. In this paper, we propose STRMAC, a state-aware routing framework designed for efficient collaboration in multi-agent systems. Our method separately encodes interaction history and agent knowledge to power the router, which adaptively selects the most suitable single agent at each step for efficient and effective collaboration. Furthermore, we introduce a self-evolving data generation approach that accelerates the collection of high-quality execution paths for efficient system training. Experiments on challenging collaborative reasoning benchmarks demonstrate that our method achieves state-of-the-art performance, achieving up to 23.8% improvement over baselines and reducing data collection overhead by up to 90.1% compared to exhaustive search.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02200) | **Categories:** cs.AI

---

### [3] [ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning](https://arxiv.org/abs/2511.02424)
*Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Minsu Jang, Dohyung Kim, Jaehong Kim, Youngwoo Yoon*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in large language models (LLMs) have enabled significant progress in decision-making and task planning for embodied autonomous agents. However, most existing methods still struggle with complex, long-horizon tasks because they rely on a monolithic trajectory that entangles all past decisions and observations, attempting to solve the entire task in a single unified process. To address this limitation, we propose ReAcTree, a hierarchical task-planning method that decomposes a complex goal into more manageable subgoals within a dynamically constructed agent tree. Each subgoal is handled by an LLM agent node capable of reasoning, acting, and further expanding the tree, while control flow nodes coordinate the execution strategies of agent nodes. In addition, we integrate two complementary memory systems: each agent node retrieves goal-specific, subgoal-level examples from episodic memory and shares environment-specific observations through working memory. Experiments on the WAH-NL and ALFRED datasets demonstrate that ReAcTree consistently outperforms strong task-planning baselines such as ReAct across diverse LLMs. Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5 72B, nearly doubling ReAct's 31%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02424) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [iFlyBot-VLA Technical Report](https://arxiv.org/abs/2511.01914)
*Yuan Zhang, Chenyu Xue, Wenjie Xu, Chao Ji, Jiajia wu, Jia Pan*

Main category: cs.CV

TL;DR: iFlyBot-VLA 是一个大规模视觉-语言-动作模型，通过新颖的框架训练，在机器人操作任务中表现出色。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在构建一个能够理解视觉、语言并执行动作的大规模模型，以解决机器人操作任务中的挑战。

Method: 该论文提出了一种双层动作表示框架，结合了潜在动作模型和结构化离散动作令牌，并通过混合训练策略提升了 VLM 的 3D 感知和推理能力。

Result: 在 LIBERO Franka 基准测试和真实世界评估中，iFlyBot-VLA 均取得了优异的成果，并在各种操作任务中展现了竞争力。

Conclusion: 该论文提出的 iFlyBot-VLA 模型在视觉-语言-动作领域取得了显著进展，为机器人操作任务提供了一种有效的方法。

Abstract: 本文介绍了一种名为 iFlyBot-VLA 的大规模视觉-语言-动作 (VLA) 模型，该模型在一个新颖的框架下进行训练。主要贡献如下：(1) 一个在大型人类和机器人操作视频上经过充分训练的潜在动作模型；(2) 一个双层动作表示框架，在训练期间共同监督视觉-语言模型 (VLM) 和动作专家；(3) 一种混合训练策略，将机器人轨迹数据与通用问答和空间问答数据集相结合，有效增强了 VLM 主干网络的 3D 感知和推理能力。具体来说，VLM 被训练来预测两种互补形式的动作：潜在动作，源自我们预训练的跨embodiment操作数据的潜在动作模型，捕捉隐式的高级意图；以及结构化的离散动作令牌，通过连续控制信号的频域变换获得，编码显式的低级动力学。这种双重监督对齐了语言、视觉和动作的表示空间，使 VLM 能够直接促成动作生成。在 LIBERO Franka 基准测试上的实验结果证明了我们框架的优越性，而真实世界的评估进一步表明 iFlyBot-VLA 在各种具有挑战性的操作任务中取得了具有竞争力的成功率。此外，我们计划开源部分我们自建的数据集，以支持社区未来的研究。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.01914) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics](https://arxiv.org/abs/2511.02427)
*Nicolas Schuler, Lea Dewald, Nick Baldig, Jürgen Graf*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video Understanding, Scene Interpretation and Commonsense Reasoning are highly challenging tasks enabling the interpretation of visual information, allowing agents to perceive, interact with and make rational decisions in its environment. Large Language Models (LLMs) and Visual Language Models (VLMs) have shown remarkable advancements in these areas in recent years, enabling domain-specific applications as well as zero-shot open vocabulary tasks, combining multiple domains. However, the required computational complexity poses challenges for their application on edge devices and in the context of Mobile Robotics, especially considering the trade-off between accuracy and inference time. In this paper, we investigate the capabilities of state-of-the-art VLMs for the task of Scene Interpretation and Action Recognition, with special regard to small VLMs capable of being deployed to edge devices in the context of Mobile Robotics. The proposed pipeline is evaluated on a diverse dataset consisting of various real-world cityscape, on-campus and indoor scenarios. The experimental evaluation discusses the potential of these small models on edge devices, with particular emphasis on challenges, weaknesses, inherent model biases and the application of the gained information. Supplementary material is provided via the following repository: https://datahub.rz.rptu.de/hstr-csrl-public/publications/scene-interpretation-on-edge-devices/

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02427) | **Categories:** cs.CV, cs.RO

---

### [3] [Zero-Shot Multi-Animal Tracking in the Wild](https://arxiv.org/abs/2511.02591)
*Jan Frederik Meier, Timo Lüddecke*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-animal tracking is crucial for understanding animal ecology and behavior. However, it remains a challenging task due to variations in habitat, motion patterns, and species appearance. Traditional approaches typically require extensive model fine-tuning and heuristic design for each application scenario. In this work, we explore the potential of recent vision foundation models for zero-shot multi-animal tracking. By combining a Grounding Dino object detector with the Segment Anything Model 2 (SAM 2) tracker and carefully designed heuristics, we develop a tracking framework that can be applied to new datasets without any retraining or hyperparameter adaptation. Evaluations on ChimpAct, Bird Flock Tracking, AnimalTrack, and a subset of GMOT-40 demonstrate strong and consistent performance across diverse species and environments. The code is available at https://github.com/ecker-lab/SAM2-Animal-Tracking.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02591) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Dynamic Population Distribution Aware Human Trajectory Generation with Diffusion Model](https://arxiv.org/abs/2511.01929)
*Qingyue Long, Can Rong, Tong Li, Yong Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human trajectory data is crucial in urban planning, traffic engineering, and public health. However, directly using real-world trajectory data often faces challenges such as privacy concerns, data acquisition costs, and data quality. A practical solution to these challenges is trajectory generation, a method developed to simulate human mobility behaviors. Existing trajectory generation methods mainly focus on capturing individual movement patterns but often overlook the influence of population distribution on trajectory generation. In reality, dynamic population distribution reflects changes in population density across different regions, significantly impacting individual mobility behavior. Thus, we propose a novel trajectory generation framework based on a diffusion model, which integrates the dynamic population distribution constraints to guide high-fidelity generation outcomes. Specifically, we construct a spatial graph to enhance the spatial correlation of trajectories. Then, we design a dynamic population distribution aware denoising network to capture the spatiotemporal dependencies of human mobility behavior as well as the impact of population distribution in the denoising process. Extensive experiments show that the trajectories generated by our model can resemble real-world trajectories in terms of some critical statistical metrics, outperforming state-of-the-art algorithms by over 54%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.01929) | **Categories:** cs.LG, cs.AI

---

### [2] [OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning](https://arxiv.org/abs/2511.02205)
*Kevin Valencia, Thilina Balasooriya, Xihaier Luo, Shinjae Yoo, David Keetae Park*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal spatiotemporal learning on real-world experimental data is constrained by two challenges: within-modality measurements are sparse, irregular, and noisy (QA/QC artifacts) but cross-modally correlated; the set of available modalities varies across space and time, shrinking the usable record unless models can adapt to arbitrary subsets at train and test time. We propose OmniField, a continuity-aware framework that learns a continuous neural field conditioned on available modalities and iteratively fuses cross-modal context. A multimodal crosstalk block architecture paired with iterative cross-modal refinement aligns signals prior to the decoder, enabling unified reconstruction, interpolation, forecasting, and cross-modal prediction without gridding or surrogate preprocessing. Extensive evaluations show that OmniField consistently outperforms eight strong multimodal spatiotemporal baselines. Under heavy simulated sensor noise, performance remains close to clean-input levels, highlighting robustness to corrupted measurements.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02205) | **Categories:** cs.LG, cs.CV

---


## cs.MA [cs.MA]
### [1] [EvoMem: Improving Multi-Agent Planning with Dual-Evolving Memory](https://arxiv.org/abs/2511.01912)
*Wenzhe Fan, Ning Yan, Masood Mortazavi*

Main category: cs.MA

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planning has been a cornerstone of artificial intelligence for solving complex problems, and recent progress in LLM-based multi-agent frameworks have begun to extend this capability. However, the role of human-like memory within these frameworks remains largely unexplored. Understanding how agents coordinate through memory is critical for natural language planning, where iterative reasoning, constraint tracking, and error correction drive the success. Inspired by working memory model in cognitive psychology, we present EvoMem, a multi-agent framework built on a dual-evolving memory mechanism. The framework consists of three agents (Constraint Extractor, Verifier, and Actor) and two memory modules: Constraint Memory (CMem), which evolves across queries by storing task-specific rules and constraints while remains fixed within a query, and Query-feedback Memory (QMem), which evolves within a query by accumulating feedback across iterations for solution refinement. Both memory modules are reset at the end of each query session. Evaluations on trip planning, meeting planning, and calendar scheduling show consistent performance improvements, highlighting the effectiveness of EvoMem. This success underscores the importance of memory in enhancing multi-agent planning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.01912) | **Categories:** cs.MA, cs.AI, cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [TACO: Trajectory-Aware Controller Optimization for Quadrotors](https://arxiv.org/abs/2511.02060)
*Hersh Sanghvi, Spencer Folk, Vijay Kumar, Camillo Jose Taylor*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Controller performance in quadrotor trajectory tracking depends heavily on parameter tuning, yet standard approaches often rely on fixed, manually tuned parameters that sacrifice task-specific performance. We present Trajectory-Aware Controller Optimization (TACO), a framework that adapts controller parameters online based on the upcoming reference trajectory and current quadrotor state. TACO employs a learned predictive model and a lightweight optimization scheme to optimize controller gains in real time with respect to a broad class of trajectories, and can also be used to adapt trajectories to improve dynamic feasibility while respecting smoothness constraints. To enable large-scale training, we also introduce a parallelized quadrotor simulator supporting fast data collection on diverse trajectories. Experiments on a variety of trajectory types show that TACO outperforms conventional, static parameter tuning while operating orders of magnitude faster than black-box optimization baselines, enabling practical real-time deployment on a physical quadrotor. Furthermore, we show that adapting trajectories using TACO significantly reduces the tracking error obtained by the quadrotor.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02060) | **Categories:** cs.RO

---

### [2] [Stein-based Optimization of Sampling Distributions in Model Predictive Path Integral Control](https://arxiv.org/abs/2511.02015)
*Jace Aldrich, Odest Chadwicke Jenkins*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a novel method for Model Predictive Path Integral (MPPI) control that optimizes sample generation towards an optimal trajectory through Stein Variational Gradient Descent (SVGD). MPPI is traditionally reliant on randomly sampled trajectories, often by a Gaussian distribution. The result can lead to sample deprivation, under-representing the space of possible trajectories, and yield suboptimal results. Through introducing SVGD updates in between MPPI environment steps, we present Stein-Optimized Path-Integral Inference (SOPPI), an MPPI/SVGD algorithm that can dynamically update noise distributions at runtime to shape a more optimal representation without an excessive increase in computational requirements. We demonstrate the efficacy of our method systems ranging from a Cart-Pole to a two-dimensional bipedal walking task, indicating improved performance above standard MPPI across a range of hyper-parameters and demonstrate feasibility at lower particle counts. We discuss the applicability of this MPPI/SVGD method to higher degree-of-freedom systems, as well as its potential to new developments in state-of-the-art differentiable simulators.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02015) | **Categories:** cs.RO

---

### [3] [Whole-body motion planning and safety-critical control for aerial manipulation](https://arxiv.org/abs/2511.02342)
*Lin Yang, Jinwoo Lee, Domenico Campolo, H. Jin Kim, Jeonghyun Byun*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Aerial manipulation combines the maneuverability of multirotors with the dexterity of robotic arms to perform complex tasks in cluttered spaces. Yet planning safe, dynamically feasible trajectories remains difficult due to whole-body collision avoidance and the conservativeness of common geometric abstractions such as bounding boxes or ellipsoids. We present a whole-body motion planning and safety-critical control framework for aerial manipulators built on superquadrics (SQs). Using an SQ-plus-proxy representation, we model both the vehicle and obstacles with differentiable, geometry-accurate surfaces. Leveraging this representation, we introduce a maximum-clearance planner that fuses Voronoi diagrams with an equilibrium-manifold formulation to generate smooth, collision-aware trajectories. We further design a safety-critical controller that jointly enforces thrust limits and collision avoidance via high-order control barrier functions. In simulation, our approach outperforms sampling-based planners in cluttered environments, producing faster, safer, and smoother trajectories and exceeding ellipsoid-based baselines in geometric fidelity. Actual experiments on a physical aerial-manipulation platform confirm feasibility and robustness, demonstrating consistent performance across simulation and hardware settings. The video can be found at https://youtu.be/hQYKwrWf1Ak.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02342) | **Categories:** cs.RO

---

### [4] [XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations](https://arxiv.org/abs/2511.02776)
*Shichao Fan, Kun Wu, Zhengping Che, Xinhua Wang, Di Wu, Fei Liao, Ning Liu, Yixue Zhang, Zhen Zhao, Zhiyuan Xu, Meng Li, Qingjie Liu, Shanghang Zhang, Min Wan, Jian Tang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models. However, existing VLA models still face two fundamental challenges: (i) producing precise low-level actions from high-dimensional observations, (ii) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments. XR-1 introduces the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion. UVMC addresses these challenges by (i) serving as an intermediate representation between the observations and actions, and (ii) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a three-stage training paradigm: (i) self-supervised UVMC learning, (ii) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (iii) task-specific post-training. We validate XR-1 through extensive real-world experiments with more than 14,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\pi_{0.5}$, $\pi_0$, RDT, UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes. Our project is at https://xr-1-vla.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02776) | **Categories:** cs.RO

---


## cs.SI [cs.SI]
### [1] [A Unified Model for Human Mobility Generation in Natural Disasters](https://arxiv.org/abs/2511.01928)
*Qingyue Long, Huandong Wang, Qi Ryan Wang, Yong Li*

Main category: cs.SI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human mobility generation in disaster scenarios plays a vital role in resource allocation, emergency response, and rescue coordination. During disasters such as wildfires and hurricanes, human mobility patterns often deviate from their normal states, which makes the task more challenging. However, existing works usually rely on limited data from a single city or specific disaster, significantly restricting the model's generalization capability in new scenarios. In fact, disasters are highly sudden and unpredictable, and any city may encounter new types of disasters without prior experience. Therefore, we aim to develop a one-for-all model for mobility generation that can generalize to new disaster scenarios. However, building a universal framework faces two key challenges: 1) the diversity of disaster types and 2) the heterogeneity among different cities. In this work, we propose a unified model for human mobility generation in natural disasters (named UniDisMob). To enable cross-disaster generalization, we design physics-informed prompt and physics-guided alignment that leverage the underlying common patterns in mobility changes after different disasters to guide the generation process. To achieve cross-city generalization, we introduce a meta-learning framework that extracts universal patterns across multiple cities through shared parameters and captures city-specific features via private parameters. Extensive experiments across multiple cities and disaster scenarios demonstrate that our method significantly outperforms state-of-the-art baselines, achieving an average performance improvement exceeding 13%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.01928) | **Categories:** cs.SI, cs.AI

---


## eess.SY [eess.SY]
### [1] [Many-vs-Many Missile Guidance via Virtual Targets](https://arxiv.org/abs/2511.02526)
*Marc Schneider, Walter Fichter*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a novel approach to many-vs-many missile guidance using virtual targets (VTs) generated by a Normalizing Flows-based trajectory predictor. Rather than assigning n interceptors directly to m physical targets through conventional weapon target assignment algorithms, we propose a centralized strategy that constructs n VT trajectories representing probabilistic predictions of maneuvering target behavior. Each interceptor is guided toward its assigned VT using Zero-Effort-Miss guidance during midcourse flight, transitioning to Proportional Navigation guidance for terminal interception. This approach treats many-vs-many engagements as many-vs-distribution scenarios, exploiting numerical superiority (n > m) by distributing interceptors across diverse trajectory hypotheses rather than pursuing identical deterministic predictions. Monte Carlo simulations across various target-interceptor configurations (1-6 targets, 1-8 interceptors) demonstrate that the VT method matches or exceeds baseline straight-line prediction performance by 0-4.1% when n = m, with improvements increasing to 5.8-14.4% when n > m. The results confirm that probabilistic VTs enable effective exploitation of numerical superiority, significantly increasing interception probability in many-vs-many scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02526) | **Categories:** eess.SY, cs.LG, cs.RO, cs.SY

---
