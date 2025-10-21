# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-22

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [cs.DB (1)](#cs-db)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems](https://arxiv.org/abs/2510.16701)
*Ni Zhang, Zhiguang Cao, Jianan Zhou, Cong Zhang, Yew-Soon Ong*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Complex vehicle routing problems (VRPs) remain a fundamental challenge, demanding substantial expert effort for intent interpretation and algorithm design. While large language models (LLMs) offer a promising path toward automation, current approaches still rely on external intervention, which restrict autonomy and often lead to execution errors and low solution feasibility. To address these challenges, we propose an Agentic Framework with LLMs (AFL) for solving complex vehicle routing problems, achieving full automation from problem instance to solution. AFL directly extracts knowledge from raw inputs and enables self-contained code generation without handcrafted modules or external solvers. To improve trustworthiness, AFL decomposes the overall pipeline into three manageable subtasks and employs four specialized agents whose coordinated interactions enforce cross-functional consistency and logical soundness. Extensive experiments on 60 complex VRPs, ranging from standard benchmarks to practical variants, validate the effectiveness and generality of our framework, showing comparable performance against meticulously designed algorithms. Notably, it substantially outperforms existing LLM-based baselines in both code reliability and solution feasibility, achieving rates close to 100% on the evaluated benchmarks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16701) | **Categories:** cs.AI

---

### [2] [PAINT: Parallel-in-time Neural Twins for Dynamical System Reconstruction](https://arxiv.org/abs/2510.16004)
*Andreas Radler, Vincent Seyfried, Stefan Pirker, Johannes Brandstetter, Thomas Lichtenegger*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Neural surrogates have shown great potential in simulating dynamical systems, while offering real-time capabilities. We envision Neural Twins as a progression of neural surrogates, aiming to create digital replicas of real systems. A neural twin consumes measurements at test time to update its state, thereby enabling context-specific decision-making. A critical property of neural twins is their ability to remain on-trajectory, i.e., to stay close to the true system state over time. We introduce Parallel-in-time Neural Twins (PAINT), an architecture-agnostic family of methods for modeling dynamical systems from measurements. PAINT trains a generative neural network to model the distribution of states parallel over time. At test time, states are predicted from measurements in a sliding window fashion. Our theoretical analysis shows that PAINT is on-trajectory, whereas autoregressive models generally are not. Empirically, we evaluate our method on a challenging two-dimensional turbulent fluid dynamics problem. The results demonstrate that PAINT stays on-trajectory and predicts system states from sparse measurements with high fidelity. These findings underscore PAINT's potential for developing neural twins that stay on-trajectory, enabling more accurate state estimation and decision-making.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16004) | **Categories:** cs.AI, physics.flu-dyn

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](https://arxiv.org/abs/2510.15963)
*Jiani Huang, Amish Sethi, Matthew Kuo, Mayank Keoliya, Neelay Velingker, JungHo Jung, Ser-Nam Lim, Ziyang Li, Mayur Naik*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-modal large language models (MLLMs) are making rapid progress toward general-purpose embodied agents. However, current training pipelines primarily rely on high-level vision-sound-text pairs and lack fine-grained, structured alignment between pixel-level visual content and textual semantics. To overcome this challenge, we propose ESCA, a new framework for contextualizing embodied agents through structured spatial-temporal understanding. At its core is SGClip, a novel CLIP-based, open-domain, and promptable model for generating scene graphs. SGClip is trained on 87K+ open-domain videos via a neurosymbolic learning pipeline, which harnesses model-driven self-supervision from video-caption pairs and structured reasoning, thereby eliminating the need for human-labeled scene graph annotations. We demonstrate that SGClip supports both prompt-based inference and task-specific fine-tuning, excelling in scene graph generation and action localization benchmarks. ESCA with SGClip consistently improves both open-source and commercial MLLMs, achieving state-of-the-art performance across two embodied environments. Notably, it significantly reduces agent perception errors and enables open-source models to surpass proprietary baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.15963) | **Categories:** cs.CV, cs.AI, cs.LG

---

### [2] [RefAtomNet++: Advancing Referring Atomic Video Action Recognition using Semantic Retrieval based Multi-Trajectory Mamba](https://arxiv.org/abs/2510.16444)
*Kunyu Peng, Di Wen, Jia Fu, Jiamin Wu, Kailun Yang, Junwei Zheng, Ruiping Liu, Yufan Chen, Yuqian Fu, Danda Pani Paudel, Luc Van Gool, Rainer Stiefelhagen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Referring Atomic Video Action Recognition (RAVAR) aims to recognize fine-grained, atomic-level actions of a specific person of interest conditioned on natural language descriptions. Distinct from conventional action recognition and detection tasks, RAVAR emphasizes precise language-guided action understanding, which is particularly critical for interactive human action analysis in complex multi-person scenarios. In this work, we extend our previously introduced RefAVA dataset to RefAVA++, which comprises >2.9 million frames and >75.1k annotated persons in total. We benchmark this dataset using baselines from multiple related domains, including atomic action localization, video question answering, and text-video retrieval, as well as our earlier model, RefAtomNet. Although RefAtomNet surpasses other baselines by incorporating agent attention to highlight salient features, its ability to align and retrieve cross-modal information remains limited, leading to suboptimal performance in localizing the target person and predicting fine-grained actions. To overcome the aforementioned limitations, we introduce RefAtomNet++, a novel framework that advances cross-modal token aggregation through a multi-hierarchical semantic-aligned cross-attention mechanism combined with multi-trajectory Mamba modeling at the partial-keyword, scene-attribute, and holistic-sentence levels. In particular, scanning trajectories are constructed by dynamically selecting the nearest visual spatial tokens at each timestep for both partial-keyword and scene-attribute levels. Moreover, we design a multi-hierarchical semantic-aligned cross-attention strategy, enabling more effective aggregation of spatial and temporal tokens across different semantic hierarchies. Experiments show that RefAtomNet++ establishes new state-of-the-art results. The dataset and code are released at https://github.com/KPeng9510/refAVA2.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16444) | **Categories:** cs.CV, cs.MM, cs.RO, eess.IV

---


## cs.DB [cs.DB]
### [1] [Comprehending Spatio-temporal Data via Cinematic Storytelling using Large Language Models](https://arxiv.org/abs/2510.17301)
*Panos Kalnis. Shuo Shang, Christian S. Jensen*

Main category: cs.DB

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Spatio-temporal data captures complex dynamics across both space and time, yet traditional visualizations are complex, require domain expertise and often fail to resonate with broader audiences. Here, we propose MapMuse, a storytelling-based framework for interpreting spatio-temporal datasets, transforming them into compelling, narrative-driven experiences. We utilize large language models and employ retrieval augmented generation (RAG) and agent-based techniques to generate comprehensive stories. Drawing on principles common in cinematic storytelling, we emphasize clarity, emotional connection, and audience-centric design. As a case study, we analyze a dataset of taxi trajectories. Two perspectives are presented: a captivating story based on a heat map that visualizes millions of taxi trip endpoints to uncover urban mobility patterns; and a detailed narrative following a single long taxi journey, enriched with city landmarks and temporal shifts. By portraying locations as characters and movement as plot, we argue that data storytelling drives insight, engagement, and action from spatio-temporal information. The case study illustrates how MapMuse can bridge the gap between data complexity and human understanding. The aim of this short paper is to provide a glimpse to the potential of the cinematic storytelling technique as an effective communication tool for spatio-temporal data, as well as to describe open problems and opportunities for future research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17301) | **Categories:** cs.DB, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [FUSE-Traffic: Fusion of Unstructured and Structured Data for Event-aware Traffic Forecasting](https://arxiv.org/abs/2510.16053)
*Chenyang Yu, Xinpeng Xie, Yan Huang, Chenxi Qiu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate traffic forecasting is a core technology for building Intelligent Transportation Systems (ITS), enabling better urban resource allocation and improved travel experiences. With growing urbanization, traffic congestion has intensified, highlighting the need for reliable and responsive forecasting models. In recent years, deep learning, particularly Graph Neural Networks (GNNs), has emerged as the mainstream paradigm in traffic forecasting. GNNs can effectively capture complex spatial dependencies in road network topology and dynamic temporal evolution patterns in traffic flow data. Foundational models such as STGCN and GraphWaveNet, along with more recent developments including STWave and D2STGNN, have achieved impressive performance on standard traffic datasets. These approaches incorporate sophisticated graph convolutional structures and temporal modeling mechanisms, demonstrating particular effectiveness in capturing and forecasting traffic patterns characterized by periodic regularities. To address this challenge, researchers have explored various ways to incorporate event information. Early attempts primarily relied on manually engineered event features. For instance, some approaches introduced manually defined incident effect scores or constructed specific subgraphs for different event-induced traffic conditions. While these methods somewhat enhance responsiveness to specific events, their core drawback lies in a heavy reliance on domain experts' prior knowledge, making generalization to diverse and complex unknown events difficult, and low-dimensional manual features often lead to the loss of rich semantic details.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16053) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [T3 Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with Temporal Logic](https://arxiv.org/abs/2510.16767)
*Jia Li, Guoxiang Zhao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Translating natural language instructions into executable motion plans is a fundamental challenge in robotics. Traditional approaches are typically constrained by their reliance on domain-specific expertise to customize planners, and often struggle with spatio-temporal couplings that usually lead to infeasible motions or discrepancies between task planning and motion execution. Despite the proficiency of Large Language Models (LLMs) in high-level semantic reasoning, hallucination could result in infeasible motion plans. In this paper, we introduce the T3 Planner, an LLM-enabled robotic motion planning framework that self-corrects it output with formal methods. The framework decomposes spatio-temporal task constraints via three cascaded modules, each of which stimulates an LLM to generate candidate trajectory sequences and examines their feasibility via a Signal Temporal Logic (STL) verifier until one that satisfies complex spatial, temporal, and logical constraints is found.Experiments across different scenarios show that T3 Planner significantly outperforms the baselines. The required reasoning can be distilled into a lightweight Qwen3-4B model that enables efficient deployment. All supplementary materials are accessible at https://github.com/leeejia/T3_Planner.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16767) | **Categories:** cs.RO

---

### [2] [SimpleVSF: VLM-Scoring Fusion for Trajectory Prediction of End-to-End Autonomous Driving](https://arxiv.org/abs/2510.17191)
*Peiru Zheng, Yun Zhao, Zhan Gong, Hong Zhu, Shaohua Wu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end autonomous driving has emerged as a promising paradigm for achieving robust and intelligent driving policies. However, existing end-to-end methods still face significant challenges, such as suboptimal decision-making in complex scenarios. In this paper,we propose SimpleVSF (Simple VLM-Scoring Fusion), a novel framework that enhances end-to-end planning by leveraging the cognitive capabilities of Vision-Language Models (VLMs) and advanced trajectory fusion techniques. We utilize the conventional scorers and the novel VLM-enhanced scorers. And we leverage a robust weight fusioner for quantitative aggregation and a powerful VLM-based fusioner for qualitative, context-aware decision-making. As the leading approach in the ICCV 2025 NAVSIM v2 End-to-End Driving Challenge, our SimpleVSF framework demonstrates state-of-the-art performance, achieving a superior balance between safety, comfort, and efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17191) | **Categories:** cs.RO, cs.AI

---

### [3] [DiffVLA++: Bridging Cognitive Reasoning and End-to-End Driving through Metric-Guided Alignment](https://arxiv.org/abs/2510.17148)
*Yu Gao, Yiru Wang, Anqing Jiang, Heng Yuwen, Wang Shuo, Sun Hao, Wang Jijun*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Conventional end-to-end (E2E) driving models are effective at generating physically plausible trajectories, but often fail to generalize to long-tail scenarios due to the lack of essential world knowledge to understand and reason about surrounding environments. In contrast, Vision-Language-Action (VLA) models leverage world knowledge to handle challenging cases, but their limited 3D reasoning capability can lead to physically infeasible actions. In this work we introduce DiffVLA++, an enhanced autonomous driving framework that explicitly bridges cognitive reasoning and E2E planning through metric-guided alignment. First, we build a VLA module directly generating semantically grounded driving trajectories. Second, we design an E2E module with a dense trajectory vocabulary that ensures physical feasibility. Third, and most critically, we introduce a metric-guided trajectory scorer that guides and aligns the outputs of the VLA and E2E modules, thereby integrating their complementary strengths. The experiment on the ICCV 2025 Autonomous Grand Challenge leaderboard shows that DiffVLA++ achieves EPDMS of 49.12.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17148) | **Categories:** cs.RO, cs.CV

---

### [4] [HumanMPC - Safe and Efficient MAV Navigation among Humans](https://arxiv.org/abs/2510.17525)
*Simon Schaefer, Helen Oleynikova, Sandra Hirche, Stefan Leutenegger*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe and efficient robotic navigation among humans is essential for integrating robots into everyday environments. Most existing approaches focus on simplified 2D crowd navigation and fail to account for the full complexity of human body dynamics beyond root motion. We present HumanMPC, a Model Predictive Control (MPC) framework for 3D Micro Air Vehicle (MAV) navigation among humans that combines theoretical safety guarantees with data-driven models for realistic human motion forecasting. Our approach introduces a novel twist to reachability-based safety formulation that constrains only the initial control input for safety while modeling its effects over the entire planning horizon, enabling safe yet efficient navigation. We validate HumanMPC in both simulated experiments using real human trajectories and in the real-world, demonstrating its effectiveness across tasks ranging from goal-directed navigation to visual servoing for human tracking. While we apply our method to MAVs in this work, it is generic and can be adapted by other platforms. Our results show that the method ensures safety without excessive conservatism and outperforms baseline approaches in both efficiency and reliability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17525) | **Categories:** cs.RO

---

### [5] [SPOT: Sensing-augmented Trajectory Planning via Obstacle Threat Modeling](https://arxiv.org/abs/2510.16308)
*Chi Zhang, Xian Huang, Wei Dong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: UAVs equipped with a single depth camera encounter significant challenges in dynamic obstacle avoidance due to limited field of view and inevitable blind spots. While active vision strategies that steer onboard cameras have been proposed to expand sensing coverage, most existing methods separate motion planning from sensing considerations, resulting in less effective and delayed obstacle response. To address this limitation, we introduce SPOT (Sensing-augmented Planning via Obstacle Threat modeling), a unified planning framework for observation-aware trajectory planning that explicitly incorporates sensing objectives into motion optimization. At the core of our method is a Gaussian Process-based obstacle belief map, which establishes a unified probabilistic representation of both recognized (previously observed) and potential obstacles. This belief is further processed through a collision-aware inference mechanism that transforms spatial uncertainty and trajectory proximity into a time-varying observation urgency map. By integrating urgency values within the current field of view, we define differentiable objectives that enable real-time, observation-aware trajectory planning with computation times under 10 ms. Simulation and real-world experiments in dynamic, cluttered, and occluded environments show that our method detects potential dynamic obstacles 2.8 seconds earlier than baseline approaches, increasing dynamic obstacle visibility by over 500\%, and enabling safe navigation through cluttered, occluded environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16308) | **Categories:** cs.RO

---

### [6] [Advancing Off-Road Autonomous Driving: The Large-Scale ORAD-3D Dataset and Comprehensive Benchmarks](https://arxiv.org/abs/2510.16500)
*Chen Min, Jilin Mei, Heng Zhai, Shuai Wang, Tong Sun, Fanjie Kong, Haoyang Li, Fangyuan Mao, Fuyang Liu, Shuo Wang, Yiming Nie, Qi Zhu, Liang Xiao, Dawei Zhao, Yu Hu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A major bottleneck in off-road autonomous driving research lies in the scarcity of large-scale, high-quality datasets and benchmarks. To bridge this gap, we present ORAD-3D, which, to the best of our knowledge, is the largest dataset specifically curated for off-road autonomous driving. ORAD-3D covers a wide spectrum of terrains, including woodlands, farmlands, grasslands, riversides, gravel roads, cement roads, and rural areas, while capturing diverse environmental variations across weather conditions (sunny, rainy, foggy, and snowy) and illumination levels (bright daylight, daytime, twilight, and nighttime). Building upon this dataset, we establish a comprehensive suite of benchmark evaluations spanning five fundamental tasks: 2D free-space detection, 3D occupancy prediction, rough GPS-guided path planning, vision-language model-driven autonomous driving, and world model for off-road environments. Together, the dataset and benchmarks provide a unified and robust resource for advancing perception and planning in challenging off-road scenarios. The dataset and code will be made publicly available at https://github.com/chaytonmin/ORAD-3D.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16500) | **Categories:** cs.RO

---

### [7] [DIV-Nav: Open-Vocabulary Spatial Relationships for Multi-Object Navigation](https://arxiv.org/abs/2510.16518)
*Jesús Ortega-Peimbert, Finn Lukas Busch, Timon Homberger, Quantao Yang, Olov Andersson*

Main category: cs.RO

TL;DR: 该论文提出了一种名为DIV-Nav的实时导航系统，通过分解复杂空间约束的自然语言指令，利用语义地图和大型视觉语言模型，实现了对复杂自由文本查询的有效对象导航。


<details>
  <summary>Details</summary>
Motivation: 现有零样本对象导航通常只适用于简单的对象名称查询，而本文旨在解决带有空间关系的复杂自由文本查询的对象导航问题。

Method: 该方法通过以下步骤实现：1) 将带有复杂空间约束的自然语言指令分解为语义地图上更简单的对象级别查询；2) 计算各个语义置信图的交集，以识别所有对象共存的区域；3) 通过大型视觉语言模型验证发现的对象是否符合原始的复杂空间约束。

Result: 在MultiON基准测试和Boston Dynamics Spot机器人上的真实部署实验验证了该系统的有效性。

Conclusion: DIV-Nav系统能够有效地处理带有复杂空间关系的自由文本查询，并指导机器人进行对象导航。

Abstract: 开放词汇语义地图和物体导航的进步使机器人能够对其环境进行知情的搜索，以寻找任意物体。然而，这种零样本物体导航通常是为简单的查询而设计的，例如“电视”或“蓝色地毯”这样的物体名称。在这里，我们考虑具有空间关系的更复杂的自由文本查询，例如“在桌子上找到遥控器”，同时仍然利用语义地图的鲁棒性。我们提出DIV-Nav，这是一个实时导航系统，通过一系列松弛有效地解决了这个问题：i) 将具有复杂空间约束的自然语言指令分解为语义地图上更简单的对象级别查询，ii) 计算各个语义置信图的交集，以识别所有对象共存的区域，iii) 通过LVLM验证发现的对象是否符合原始的复杂空间约束。我们进一步研究了如何调整在线语义地图的前沿探索目标，以更有效地指导搜索过程，以满足这种空间搜索查询。我们通过在MultiON基准测试和在Jetson Orin AGX上使用Boston Dynamics Spot机器人的真实部署进行的大量实验来验证我们的系统。更多详细信息和视频请访问https://anonsub42.github.io/reponame/。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16518) | **Categories:** cs.RO, cs.AI

---

### [8] [C-Free-Uniform: A Map-Conditioned Trajectory Sampler for Model Predictive Path Integral Control](https://arxiv.org/abs/2510.16905)
*Yukang Cao, Rahul Moorthy, O. Goktug Poyrazoglu, Volkan Isler*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory sampling is a key component of sampling-based control mechanisms. Trajectory samplers rely on control input samplers, which generate control inputs u from a distribution p(u | x) where x is the current state. We introduce the notion of Free Configuration Space Uniformity (C-Free-Uniform for short) which has two key features: (i) it generates a control input distribution so as to uniformly sample the free configuration space, and (ii) in contrast to previously introduced trajectory sampling mechanisms where the distribution p(u | x) is independent of the environment, C-Free-Uniform is explicitly conditioned on the current local map. Next, we integrate this sampler into a new Model Predictive Path Integral (MPPI) Controller, CFU-MPPI. Experiments show that CFU-MPPI outperforms existing methods in terms of success rate in challenging navigation tasks in cluttered polygonal environments while requiring a much smaller sampling budget.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.16905) | **Categories:** cs.RO

---

### [9] [Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey](https://arxiv.org/abs/2510.17111)
*Weifan Guan, Qinghao Hu, Aosheng Li, Jian Cheng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17111) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [10] [High-Level Multi-Robot Trajectory Planning And Spurious Behavior Detection](https://arxiv.org/abs/2510.17261)
*Fernando Salanova, Jesús Roche, Cristian Mahuela, Eduardo Montijano*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The reliable execution of high-level missions in multi-robot systems with heterogeneous agents, requires robust methods for detecting spurious behaviors. In this paper, we address the challenge of identifying spurious executions of plans specified as a Linear Temporal Logic (LTL) formula, as incorrect task sequences, violations of spatial constraints, timing inconsis- tencies, or deviations from intended mission semantics. To tackle this, we introduce a structured data generation framework based on the Nets-within-Nets (NWN) paradigm, which coordinates robot actions with LTL-derived global mission specifications. We further propose a Transformer-based anomaly detection pipeline that classifies robot trajectories as normal or anomalous. Experi- mental evaluations show that our method achieves high accuracy (91.3%) in identifying execution inefficiencies, and demonstrates robust detection capabilities for core mission violations (88.3%) and constraint-based adaptive anomalies (66.8%). An ablation experiment of the embedding and architecture was carried out, obtaining successful results where our novel proposition performs better than simpler representations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17261) | **Categories:** cs.RO, cs.LG

---
