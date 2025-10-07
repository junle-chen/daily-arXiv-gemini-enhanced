# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-08

## 目录

- [人工智能 (Artificial Intelligence) (5)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Towards Policy-Compliant Agents: Learning Efficient Guardrails For Policy Violation Detection](https://arxiv.org/abs/2510.03485)
*Xiaofei Wen, Wenjie Jacky Mo, Yanan Xie, Peng Qi, Muhao Chen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous web agents need to operate under externally imposed or human-specified policies while generating long-horizon trajectories. However, little work has examined whether these trajectories comply with such policies, or whether policy violations persist across different contexts such as domains (e.g., shopping or coding websites) and subdomains (e.g., product search and order management in shopping). To address this gap, we introduce PolicyGuardBench, a benchmark of about 60k examples for detecting policy violations in agent trajectories. From diverse agent runs, we generate a broad set of policies and create both within subdomain and cross subdomain pairings with violation labels. In addition to full-trajectory evaluation, PolicyGuardBench also includes a prefix-based violation detection task where models must anticipate policy violations from truncated trajectory prefixes rather than complete sequences. Using this dataset, we train PolicyGuard-4B, a lightweight guardrail model that delivers strong detection accuracy across all tasks while keeping inference efficient. Notably, PolicyGuard-4B generalizes across domains and preserves high accuracy on unseen settings. Together, PolicyGuardBench and PolicyGuard-4B provide the first comprehensive framework for studying policy compliance in web agent trajectories, and show that accurate and generalizable guardrails are feasible at small scales.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03485) | **Categories:** cs.AI, I.2.7

---

### [2] [Bridging the Gap Between Multimodal Foundation Models and World Models](https://arxiv.org/abs/2510.03727)
*Xuehai He*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humans understand the world through the integration of multiple sensory modalities, enabling them to perceive, reason about, and imagine dynamic physical processes. Inspired by this capability, multimodal foundation models (MFMs) have emerged as powerful tools for multimodal understanding and generation. However, today's MFMs fall short of serving as effective world models. They lack the essential ability such as perform counterfactual reasoning, simulate dynamics, understand the spatiotemporal information, control generated visual outcomes, and perform multifaceted reasoning. We investigates what it takes to bridge the gap between multimodal foundation models and world models. We begin by improving the reasoning capabilities of MFMs through discriminative tasks and equipping MFMs with structured reasoning skills, such as causal inference, counterfactual thinking, and spatiotemporal reasoning, enabling them to go beyond surface correlations and understand deeper relationships within visual and textual data. Next, we explore generative capabilities of multimodal foundation models across both image and video modalities, introducing new frameworks for structured and controllable generation. Our approaches incorporate scene graphs, multimodal conditioning, and multimodal alignment strategies to guide the generation process, ensuring consistency with high-level semantics and fine-grained user intent. We further extend these techniques to controllable 4D generation, enabling interactive, editable, and morphable object synthesis over time and space.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03727) | **Categories:** cs.AI, cs.CL, cs.CV, cs.LG

---

### [3] [Zephyrus: An Agentic Framework for Weather Science](https://arxiv.org/abs/2510.04017)
*Sumanth Varambally, Marshall Fisher, Jas Thakker, Yiwei Chen, Zhirui Xia, Yasaman Jafari, Ruijia Niu, Manas Jain, Veeramakali Vignesh Manivannan, Zachary Novack, Luyu Han, Srikar Eranky, Salva Rühling Cachay, Taylor Berg-Kirkpatrick, Duncan Watson-Parris, Yi-An Ma, Rose Yu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundation models for weather science are pre-trained on vast amounts of structured numerical data and outperform traditional weather forecasting systems. However, these models lack language-based reasoning capabilities, limiting their utility in interactive scientific workflows. Large language models (LLMs) excel at understanding and generating text but cannot reason about high-dimensional meteorological datasets. We bridge this gap by building a novel agentic framework for weather science. Our framework includes a Python code-based environment for agents (ZephyrusWorld) to interact with weather data, featuring tools like an interface to WeatherBench 2 dataset, geoquerying for geographical masks from natural language, weather forecasting, and climate simulation capabilities. We design Zephyrus, a multi-turn LLM-based weather agent that iteratively analyzes weather datasets, observes results, and refines its approach through conversational feedback loops. We accompany the agent with a new benchmark, ZephyrusBench, with a scalable data generation pipeline that constructs diverse question-answer pairs across weather-related tasks, from basic lookups to advanced forecasting, extreme event detection, and counterfactual reasoning. Experiments on this benchmark demonstrate the strong performance of Zephyrus agents over text-only baselines, outperforming them by up to 35 percentage points in correctness. However, on harder tasks, Zephyrus performs similarly to text-only baselines, highlighting the challenging nature of our benchmark and suggesting promising directions for future work.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04017) | **Categories:** cs.AI, cs.LG, physics.ao-ph

---

### [4] [Constructing coherent spatial memory in LLM agents through graph rectification](https://arxiv.org/abs/2510.04195)
*Puzhen Zhang, Xuyang Chen, Yu Feng, Yuhan Jiang, Liqiu Meng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Given a map description through global traversal navigation instructions (e.g., visiting each room sequentially with action signals such as north, west, etc.), an LLM can often infer the implicit spatial layout of the environment and answer user queries by providing a shortest path from a start to a destination (for instance, navigating from the lobby to a meeting room via the hall and elevator). However, such context-dependent querying becomes incapable as the environment grows much longer, motivating the need for incremental map construction that builds a complete topological graph from stepwise observations. We propose a framework for LLM-driven construction and map repair, designed to detect, localize, and correct structural inconsistencies in incrementally constructed navigation graphs. Central to our method is the Version Control, which records the full history of graph edits and their source observations, enabling fine-grained rollback, conflict tracing, and repair evaluation. We further introduce an Edge Impact Score to prioritize minimal-cost repairs based on structural reachability, path usage, and conflict propagation. To properly evaluate our approach, we create a refined version of the MANGO benchmark dataset by systematically removing non-topological actions and inherent structural conflicts, providing a cleaner testbed for LLM-driven construction and map repair. Our approach significantly improves map correctness and robustness, especially in scenarios with entangled or chained inconsistencies. Our results highlight the importance of introspective, history-aware repair mechanisms for maintaining coherent spatial memory in LLM agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04195) | **Categories:** cs.AI

---

### [5] [AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](https://arxiv.org/abs/2510.04206)
*Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, Rui Lu, Hongning Wang, Jie Tang, Yuxiao Dong*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in large language models (LLMs) have sparked growing interest in building generalist agents that can learn through online interactions. However, applying reinforcement learning (RL) to train LLM agents in multi-turn, multi-task settings remains challenging due to lack of scalable infrastructure and stable training algorithms. In this work, we present the AgentRL framework for scalable multi-turn, multi-task agentic RL training. On the infrastructure side, AgentRL features a fully-asynchronous generation-training pipeline for efficient multi-turn RL. To support heterogeneous environment development in multi-task RL, we design a unified function-call based API interface, containerized environment development, and a centralized controller. On the algorithm side, we propose cross-policy sampling to encourage model exploration in multi-turn settings and task advantage normalization to stabilize multi-task training. Experiments show that AgentRL, trained on open LLMs across five agentic tasks, significantly outperforms GPT-5, Clause-Sonnet-4, DeepSeek-R1, and other open-source LLM agents. Multi-task training with AgentRL matches the best results among all task-specific models. AgentRL is open-sourced at https://github.com/THUDM/AgentRL. The algorithm and framework are adopted in building \textsc{\href{https://autoglm.zhipuai.cn}{AutoGLM}}.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04206) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [A Comprehensive Review on Artificial Intelligence Empowered Solutions for Enhancing Pedestrian and Cyclist Safety](https://arxiv.org/abs/2510.03314)
*Shucheng Zhang, Yan Shi, Bingzhang Wang, Yuang Zhang, Muhammad Monjurul Karim, Kehua Chen, Chenxi Liu, Mehrdad Nasri, Yinhai Wang*

Main category: cs.CV

TL;DR: 本文综述了基于视觉人工智能的弱势道路使用者（VRU）安全感知系统的最新进展，重点关注了检测、跟踪、轨迹预测和意图识别等核心任务。


<details>
  <summary>Details</summary>
Motivation: 在动态城市环境中，传统的基于基础设施的措施不足以确保弱势道路使用者（VRU）的安全。人工智能在视觉感知和推理方面的进步为主动和情境感知的VRU保护提供了新的机会。

Method: 本文系统地回顾了近五年基于视觉人工智能的VRU安全感知系统的研究进展，重点关注了检测与分类、跟踪与重识别、轨迹预测以及意图识别与预测这四个核心任务。

Result: 本文总结了数据、模型和部署等角度的四个主要开放挑战，旨在为下一代VRU安全感知系统的开发提供参考。

Conclusion: 通过将视觉人工智能的进步与实际应用相结合，本文旨在为开发下一代增强VRU安全性的感知系统提供基础参考。

Abstract: 确保弱势道路使用者（VRU）如行人及骑自行车者的安全，仍然是一项关键的全球性挑战，因为传统的基于基础设施的措施在动态城市环境中常常显得不足。人工智能（AI），特别是在视觉感知和推理方面的最新进展，为主动和情境感知的VRU保护开启了新的机遇。然而，现有关于人工智能在VRU应用方面的调查主要集中在检测上，对其他基于视觉的任务覆盖有限，而这些任务对于全面理解和保护VRU至关重要。本文对近五年基于摄像头的VRU安全人工智能感知系统的最新进展进行了最先进的综述，并着重关注了新兴的研究趋势。我们系统地考察了四个核心任务，即检测与分类、跟踪与重识别、轨迹预测以及意图识别与预测，这些任务共同构成了人工智能赋能的主动解决方案的支柱，以在智能交通系统中保护VRU。为了指导未来的研究，我们从数据、模型和部署的角度强调了四个主要的开放挑战。通过将视觉人工智能的进步与实际应用相结合，本综述旨在为开发下一代增强VRU安全性的感知系统提供基础参考。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03314) | **Categories:** cs.CV, cs.AI

---

### [2] [SketchPlan: Diffusion Based Drone Planning From Human Sketches](https://arxiv.org/abs/2510.03545)
*Sixten Norelius, Aaron O. Feldman, Mac Schwager*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose SketchPlan, a diffusion-based planner that interprets 2D hand-drawn sketches over depth images to generate 3D flight paths for drone navigation. SketchPlan comprises two components: a SketchAdapter that learns to map the human sketches to projected 2D paths, and DiffPath, a diffusion model that infers 3D trajectories from 2D projections and a first person view depth image. Our model achieves zero-shot sim-to-real transfer, generating accurate and safe flight paths in previously unseen real-world environments. To train the model, we build a synthetic dataset of 32k flight paths using a diverse set of photorealistic 3D Gaussian Splatting scenes. We automatically label the data by computing 2D projections of the 3D flight paths onto the camera plane, and use this to train the DiffPath diffusion model. However, since real human 2D sketches differ significantly from ideal 2D projections, we additionally label 872 of the 3D flight paths with real human sketches and use this to train the SketchAdapter to infer the 2D projection from the human sketch. We demonstrate SketchPlan's effectiveness in both simulated and real-world experiments, and show through ablations that training on a mix of human labeled and auto-labeled data together with a modular design significantly boosts its capabilities to correctly interpret human intent and infer 3D paths. In real-world drone tests, SketchPlan achieved 100\% success in low/medium clutter and 40\% in unseen high-clutter environments, outperforming key ablations by 20-60\% in task completion.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03545) | **Categories:** cs.CV, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [VIFO: Visual Feature Empowered Multivariate Time Series Forecasting with Cross-Modal Fusion](https://arxiv.org/abs/2510.03244)
*Yanlong Wang, Hang Yu, Jian Xu, Fei Ma, Hongkang Zhang, Tongtong Feng, Zijian Zhang, Shao-Lun Huang, Danny Dongning Sun, Xiao-Ping Zhang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large time series foundation models often adopt channel-independent architectures to handle varying data dimensions, but this design ignores crucial cross-channel dependencies. Concurrently, existing multimodal approaches have not fully exploited the power of large vision models (LVMs) to interpret spatiotemporal data. Additionally, there remains significant unexplored potential in leveraging the advantages of information extraction from different modalities to enhance time series forecasting performance. To address these gaps, we propose the VIFO, a cross-modal forecasting model. VIFO uniquely renders multivariate time series into image, enabling pre-trained LVM to extract complex cross-channel patterns that are invisible to channel-independent models. These visual features are then aligned and fused with representations from the time series modality. By freezing the LVM and training only 7.45% of its parameters, VIFO achieves competitive performance on multiple benchmarks, offering an efficient and effective solution for capturing cross-variable relationships in

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03244) | **Categories:** cs.LG, cs.AI, cs.CV

---

### [2] [StructPrune: Structured Global Pruning asymptotics with $\mathcal{O}(\sqrt{N})$ GPU Memory](https://arxiv.org/abs/2510.03246)
*Xinyuan Song, Guangji Bai, Liang Zhao*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Pruning is critical for scaling large language models (LLMs). Global pruning achieves strong performance but requires $\mathcal{O}(N)$ memory, which is infeasible for billion-parameter models. Local pruning reduces GPU memory usage to that of a single layer by pruning layers independently, but it neglects inter-layer dependencies and often leads to suboptimal performance in high-sparsity regimes. Unlike unstructured pruning, structured pruning produces regular sparsity patterns that align well with GPU kernels and library optimizations, making it more hardware-efficient. However, structured pruning typically relies on global pruning, since structured patterns are more prone to severe performance degradation under local optimization. To jointly achieve structured pruning and the memory efficiency of local pruning, we propose a divide-and-conquer strategy that decomposes the global pruning problem into coordinated subproblems across different modules, each of which fits within limited GPU memory. Building on this idea, we design \textbf{STRUPRUNE}, an ADMM-based framework that integrates structured sparsity into the pruning process, combining the memory efficiency of local pruning with the hardware compatibility of structured methods. We derive a closed-form analytical solution for structured pruning masks that provides an explicit rule for layer-wise sparsity allocation, and further develop an energy-based asymptotic framework yielding a softmax-form allocation scheme that simplifies optimization while adapting to heterogeneous layer importance. Experiments demonstrate that STRUPRUNE matches the perplexity of global structured pruning while reducing memory cost from $\mathcal{O}(N)$ to $\mathcal{O}(\sqrt{N})$, enabling practical deployment at the billion-parameter scale.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03246) | **Categories:** cs.LG, cs.AI

---

### [3] [Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](https://arxiv.org/abs/2510.03253)
*Heyang Gao, Zexu Sun, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) as autonomous agents are increasingly tasked with solving complex, long-horizon problems. Aligning these agents via preference-based offline methods like Direct Preference Optimization (DPO) is a promising direction, yet it faces a critical granularity mismatch. Trajectory-level DPO provides a signal that is too coarse for precise credit assignment, while step-level DPO is often too myopic to capture the value of multi-step behaviors. To resolve this challenge, we introduce Hierarchical Preference Learning (HPL), a hierarchical framework that optimizes LLM agents by leveraging preference signals at multiple, synergistic granularities. While HPL incorporates trajectory- and step-level DPO for global and local policy stability, its core innovation lies in group-level preference optimization guided by a dual-layer curriculum. Our approach first decomposes expert trajectories into semantically coherent action groups and then generates contrasting suboptimal groups to enable preference learning at a fine-grained, sub-task level. Then, instead of treating all preference pairs equally, HPL introduces a curriculum scheduler that organizes the learning process from simple to complex. This curriculum is structured along two axes: the group length, representing sub-task complexity, and the sample difficulty, defined by the reward gap between preferred and dispreferred action groups. Experiments on three challenging agent benchmarks show that HPL outperforms existing state-of-the-art methods. Our analyses demonstrate that the hierarchical DPO loss effectively integrates preference signals across multiple granularities, while the dual-layer curriculum is crucial for enabling the agent to solve a wide range of tasks, from simple behaviors to complex multi-step sequences.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03253) | **Categories:** cs.LG, cs.AI, I.2.7

---

### [4] [SciTS: Scientific Time Series Understanding and Generation with LLMs](https://arxiv.org/abs/2510.03255)
*Wen Wu, Ziyang Zhang, Liwei Liu, Xuenan Xu, Junlin Liu, Ke Fan, Qitan Lv, Jimin Zhuang, Chen Zhang, Zheqi Yuan, Siyuan Hou, Tianyi Lin, Kai Chen, Bowen Zhou, Chao Zhang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The scientific reasoning ability of large language models (LLMs) has recently attracted significant attention. Time series, as a fundamental modality in scientific data, presents unique challenges that are often overlooked in current multimodal LLMs, which either encode numerical sequences as text or convert them into images. Such approaches may be insufficient for comprehensive scientific time series understanding and generation. Existing unified time series models typically specialise in either forecasting or analysis, and their effectiveness on non-periodic, heterogeneous scientific signals remains unclear. To address these gaps, we introduce SciTS, a benchmark spanning 12 scientific domains and 43 tasks, with over 50k+ instances, both univariate and multivariate signals ranging from $10^0$ to $10^7$ in length and up to 10~MHz in frequency. We benchmark 17 models, including text-only LLMs, multimodal LLMs, and unified time series models, and find that general-purpose LLMs exhibit stronger generalisability than specialised time series models, while representing time series as text or images limits their performance due to excessively long sequences and loss of numerical precision, respectively. We then introduce TimeOmni, a framework that equips LLMs with the ability to understand and generate time series while remaining compatible with general-purpose LLM training. This work fills a gap in both dedicated benchmarks and modelling frameworks for scientific time series, paving the way for LLMs to understand and generate complex temporal scientific data.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03255) | **Categories:** cs.LG, cs.AI

---

### [5] [PT$^2$-LLM: Post-Training Ternarization for Large Language Models](https://arxiv.org/abs/2510.03267)
*Xianglong Yan, Chengzhu Bao, Zhiteng Li, Tianao Zhang, Kaicheng Yang, Haotong Qin, Ruobing Xie, Xingwu Sun, Yulun Zhang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have shown impressive capabilities across diverse tasks, but their large memory and compute demands hinder deployment. Ternarization has gained attention as a promising compression technique, delivering substantial size reduction and high computational efficiency. However, its potential in the post-training quantization (PTQ) setting remains underexplored, due to the challenge of training-free parameter optimization and the quantization difficulty posed by outliers and dispersed weights. To address these issues, we propose PT$^2$-LLM, a post-training ternarization framework tailored for LLMs. At its core is an Asymmetric Ternary Quantizer equipped with a two-stage refinement pipeline: (1) Iterative Ternary Fitting (ITF), which alternates between optimal ternary grid construction and flexible rounding to minimize quantization error, and (2) Activation-aware Grid Alignment (AGA), which further refines the ternary grid to better match full-precision outputs. In addition, we propose a plug-and-play Structural Similarity-based Reordering (SSR) strategy that leverages inter-column structural similarity to ease quantization and mitigate outlier effects, further enhancing overall performance. Extensive experiments demonstrate that PT$^2$-LLM delivers competitive performance against state-of-the-art (SOTA) 2-bit PTQ methods with lower memory cost, while also accelerating both prefill and decoding to achieve end-to-end speedup. The code and models will be available at https://github.com/XIANGLONGYAN/PT2-LLM.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03267) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Digital-Twin Evaluation for Proactive Human-Robot Collision Avoidance via Prediction-Guided A-RRT*](https://arxiv.org/abs/2510.03496)
*Vadivelan Murugesan, Rajasundaram Mathiazhagan, Sanjana Joshi, Aliasghar Arab*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human-robot collaboration requires precise prediction of human motion over extended horizons to enable proactive collision avoidance. Unlike existing planners that rely solely on kinodynamic models, we present a prediction-driven safe planning framework that leverages granular, joint-by-joint human motion forecasting validated in a physics-based digital twin. A capsule-based artificial potential field (APF) converts these granular predictions into collision risk metrics, triggering an Adaptive RRT* (A-RRT*) planner when thresholds are exceeded. The depth camera is used to extract 3D skeletal poses and a convolutional neural network-bidirectional long short-term memory (CNN-BiLSTM) model to predict individual joint trajectories ahead of time. A digital twin model integrates real-time human posture prediction placed in front of a simulated robot to evaluate motions and physical contacts. The proposed method enables validation of planned trajectories ahead of time and bridging potential latency gaps in updating planned trajectories in real-time. In 50 trials, our method achieved 100% proactive avoidance with > 250 mm clearance and sub-2 s replanning, demonstrating superior precision and reliability compared to existing kinematic-only planners through the integration of predictive human modeling with digital twin validation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03496) | **Categories:** cs.RO

---

### [2] [Trajectory prediction for heterogeneous agents: A performance analysis on small and imbalanced datasets](https://arxiv.org/abs/2510.03776)
*Tiago Rodrigues de Almeida, Yufei Zhu, Andrey Rudenko, Tomasz P. Kucner, Johannes A. Stork, Martin Magnusson, Achim J. Lilienthal*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robots and other intelligent systems navigating in complex dynamic environments should predict future actions and intentions of surrounding agents to reach their goals efficiently and avoid collisions. The dynamics of those agents strongly depends on their tasks, roles, or observable labels. Class-conditioned motion prediction is thus an appealing way to reduce forecast uncertainty and get more accurate predictions for heterogeneous agents. However, this is hardly explored in the prior art, especially for mobile robots and in limited data applications. In this paper, we analyse different class-conditioned trajectory prediction methods on two datasets. We propose a set of conditional pattern-based and efficient deep learning-based baselines, and evaluate their performance on robotics and outdoors datasets (TH\"OR-MAGNI and Stanford Drone Dataset). Our experiments show that all methods improve accuracy in most of the settings when considering class labels. More importantly, we observe that there are significant differences when learning from imbalanced datasets, or in new environments where sufficient data is not available. In particular, we find that deep learning methods perform better on balanced datasets, but in applications with limited data, e.g., cold start of a robot in a new environment, or imbalanced classes, pattern-based methods may be preferable.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03776) | **Categories:** cs.RO, cs.LG

---

### [3] [NoTVLA: Narrowing of Dense Action Trajectories for Generalizable Robot Manipulation](https://arxiv.org/abs/2510.03895)
*Zheng Huang, Mingyu Liu, Xiaoyi Lin, Muzhi Zhu, Canyu Zhao, Zongze Du, Xiaoman Li, Yiduo Jia, Hao Zhong, Hao Chen, Chunhua Shen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models represent a pivotal advance in embodied intelligence, yet they confront critical barriers to real-world deployment, most notably catastrophic forgetting. This issue stems from their overreliance on continuous action sequences or action chunks, which inadvertently create isolated data silos that disrupt knowledge retention across tasks. To tackle these challenges, we propose the Narrowing of Trajectory VLA (NoTVLA) framework: a novel approach that narrows its focus to sparse trajectories, thereby avoiding the catastrophic forgetting associated with dense trajectory fine-tuning. A key innovation of NoTVLA lies in its trajectory planning strategy: instead of centering on the target object's trajectory, it leverages temporal compression and spatial reasoning pruning specifically for the robot end effector's trajectory. Furthermore, training is conducted using these sparse trajectories rather than dense action trajectories, an optimization that delivers remarkable practical advantages with better performance in zero-shot. In multi-task evaluation scenarios, NoTVLA achieves superior performance and generalization compared to pi0 while operating under two critical constraints: it uses over an order of magnitude less computing power than pi0 and requires no wrist-mounted camera. This design ensures that NoTVLA's operational accuracy closely approximates that of single-task expert models. Crucially, it also preserves the model's inherent language capabilities, enabling zero-shot generalization in specific scenarios, supporting unified model deployment across multiple robot platforms, and fostering a degree of generalization even when perceiving tasks from novel perspectives.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03895) | **Categories:** cs.RO, cs.CV

---

### [4] [From Shadow to Light: Toward Safe and Efficient Policy Learning Across MPC, DeePC, RL, and LLM Agents](https://arxiv.org/abs/2510.04076)
*Amin Vahidi-Moghaddam, Sayed Pedram Haeri Boroujeni, Iman Jebellat, Ehsan Jebellat, Niloufar Mehrabi, Zhaojian Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: One of the main challenges in modern control applications, particularly in robot and vehicle motion control, is achieving accurate, fast, and safe movement. To address this, optimal control policies have been developed to enforce safety while ensuring high performance. Since basic first-principles models of real systems are often available, model-based controllers are widely used. Model predictive control (MPC) is a leading approach that optimizes performance while explicitly handling safety constraints. However, obtaining accurate models for complex systems is difficult, which motivates data-driven alternatives. ML-based MPC leverages learned models to reduce reliance on hand-crafted dynamics, while reinforcement learning (RL) can learn near-optimal policies directly from interaction data. Data-enabled predictive control (DeePC) goes further by bypassing modeling altogether, directly learning safe policies from raw input-output data. Recently, large language model (LLM) agents have also emerged, translating natural language instructions into structured formulations of optimal control problems. Despite these advances, data-driven policies face significant limitations. They often suffer from slow response times, high computational demands, and large memory needs, making them less practical for real-world systems with fast dynamics, limited onboard computing, or strict memory constraints. To address this, various technique, such as reduced-order modeling, function-approximated policy learning, and convex relaxations, have been proposed to reduce computational complexity. In this paper, we present eight such approaches and demonstrate their effectiveness across real-world applications, including robotic arms, soft robots, and vehicle motion control.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04076) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [5] [PAD-TRO: Projection-Augmented Diffusion for Direct Trajectory Optimization](https://arxiv.org/abs/2510.04436)
*Jushan Chen, Santiago Paternain*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently, diffusion models have gained popularity and attention in trajectory optimization due to their capability of modeling multi-modal probability distributions. However, addressing nonlinear equality constraints, i.e, dynamic feasi- bility, remains a great challenge in diffusion-based trajectory optimization. Recent diffusion-based trajectory optimization frameworks rely on a single-shooting style approach where the denoised control sequence is applied to forward propagate the dynamical system, which cannot explicitly enforce constraints on the states and frequently leads to sub-optimal solutions. In this work, we propose a novel direct trajectory optimization approach via model-based diffusion, which directly generates a sequence of states. To ensure dynamic feasibility, we propose a gradient-free projection mechanism that is incorporated into the reverse diffusion process. Our results show that, compared to a recent state-of-the-art baseline, our approach leads to zero dynamic feasibility error and approximately 4x higher success rate in a quadrotor waypoint navigation scenario involving dense static obstacles.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04436) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [6] [Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer](https://arxiv.org/abs/2510.03342)
*Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, Michael Bloesch, Konstantinos Bousmalis, Philemon Brakel, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Christine Chan, Oscar Chang, London Chappellet-Volpini, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, Adrian Collister, David B. D'Ambrosio, Sudeep Dasari, Todor Davchev, Meet Kirankumar Dave, Coline Devin, Norman Di Palo, Tianli Ding, Carl Doersch, Adil Dostmohamed, Yilun Du, Debidatta Dwibedi, Sathish Thoppay Egambaram, Michael Elabd, Tom Erez, Xiaolin Fang, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Ruiqi Gao, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Oliver Groth, Agrim Gupta, Roland Hafner, Steven Hansen, Leonard Hasenclever, Sam Haves, Nicolas Heess, Brandon Hernaez, Alex Hofer, Jasmine Hsu, Lu Huang, Sandy H. Huang, Atil Iscen, Mithun George Jacob, Deepali Jain, Sally Jesmonth, Abhishek Jindal, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Matija Kecman, J. Chase Kew, Donnie Kim, Frank Kim, Junkyung Kim, Thomas Kipf, Sean Kirmani, Ksenia Konyushkova, Li Yang Ku, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Tuan Anh Le, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Guy Lever, Jacky Liang, Li-Heng Lin, Fangchen Liu, Shangbang Long, Caden Lu, Sharath Maddineni, Anirudha Majumdar, Kevis-Kokitsi Maninis, Andrew Marmon, Sergio Martinez, Assaf Hurwitz Michaely, Niko Milonopoulos, Joss Moore, Robert Moreno, Michael Neunert, Francesco Nori, Joy Ortiz, Kenneth Oslund, Carolina Parada, Emilio Parisotto, Amaris Paryag, Acorn Pooley, Thomas Power, Alessio Quaglino, Haroon Qureshi, Rajkumar Vasudeva Raju, Helen Ran, Dushyant Rao, Kanishka Rao, Isaac Reid, David Rendleman, Krista Reymann, Miguel Rivas, Francesco Romano, Yulia Rubanova, Peter Pastor Sampedro, Pannag R Sanketi, Dhruv Shah, Mohit Sharma, Kathryn Shea, Mohit Shridhar, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Rachel Sterneck, Ian Storz, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Saran Tunyasuvunakool, Jake Varley, Grace Vesom, Giulia Vezzani, Maria Bauza Villalonga, Oriol Vinyals, René Wagner, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Chengda Wu, Markus Wulfmeier, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Jimmy Yan, Sherry Yang, Skye Yang, Yuxiang Yang, Hiu Hong Yu, Wenhao Yu, Wentao Yuan, Yuan Yuan, Jingwei Zhang, Tingnan Zhang, Zhiyuan Zhang, Allan Zhou, Guangyao Zhou, Yuxiang Zhou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: General-purpose robots need a deep understanding of the physical world, advanced reasoning, and general and dexterous control. This report introduces the latest generation of the Gemini Robotics model family: Gemini Robotics 1.5, a multi-embodiment Vision-Language-Action (VLA) model, and Gemini Robotics-ER 1.5, a state-of-the-art Embodied Reasoning (ER) model. We are bringing together three major innovations. First, Gemini Robotics 1.5 features a novel architecture and a Motion Transfer (MT) mechanism, which enables it to learn from heterogeneous, multi-embodiment robot data and makes the VLA more general. Second, Gemini Robotics 1.5 interleaves actions with a multi-level internal reasoning process in natural language. This enables the robot to "think before acting" and notably improves its ability to decompose and execute complex, multi-step tasks, and also makes the robot's behavior more interpretable to the user. Third, Gemini Robotics-ER 1.5 establishes a new state-of-the-art for embodied reasoning, i.e., for reasoning capabilities that are critical for robots, such as visual and spatial understanding, task planning, and progress estimation. Together, this family of models takes us a step towards an era of physical agents-enabling robots to perceive, think and then act so they can solve complex multi-step tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03342) | **Categories:** cs.RO

---

### [7] [Safety-Oriented Dynamic Path Planning for Automated Vehicles](https://arxiv.org/abs/2510.03640)
*Mostafa Emam, Matthias Gerdts*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring safety in autonomous vehicles necessitates advanced path planning and obstacle avoidance capabilities, particularly in dynamic environments. This paper introduces a bi-level control framework that efficiently augments road boundaries by incorporating time-dependent grid projections of obstacle movements, thus enabling precise and adaptive path planning. The main control loop utilizes Nonlinear Model Predictive Control (NMPC) for real-time path optimization, wherein homotopy-based constraint relaxation is employed to improve the solvability of the optimal control problem (OCP). Furthermore, an independent backup loop runs concurrently to provide safe fallback trajectories when an optimal trajectory cannot be computed by the main loop within a critical time frame, thus enhancing safety and real-time performance. Our evaluation showcases the benefits of the proposed methods in various driving scenarios, highlighting the real-time applicability and robustness of our approach. Overall, the framework represents a significant step towards safer and more reliable autonomous driving in complex and dynamic environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.03640) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [8] [SITCOM: Scaling Inference-Time COMpute for VLAs](https://arxiv.org/abs/2510.04041)
*Ayudh Saxena, Harsh Shah, Sandeep Routray, Rishi Rajesh Shah, Esha Pahwa*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning robust robotic control policies remains a major challenge due to the high cost of collecting labeled data, limited generalization to unseen environments, and difficulties in planning over long horizons. While Vision-Language-Action (VLA) models offer a promising solution by grounding natural language instructions into single-step control commands, they often lack mechanisms for lookahead and struggle with compounding errors in dynamic tasks. In this project, we introduce Scaling Inference-Time COMpute for VLAs (SITCOM), a framework that augments any pretrained VLA with model-based rollouts and reward-based trajectory selection, inspired by Model Predictive Control algorithm. SITCOM leverages a learned dynamics model to simulate multi-step action rollouts to select the best candidate plan for real-world execution, transforming one-shot VLAs into robust long-horizon planners. We develop an efficient transformer-based dynamics model trained on large-scale BridgeV2 data and fine-tuned on SIMPLER environments to bridge the Real2Sim gap, and score candidate rollouts using rewards from simulator. Through comprehensive evaluation across multiple tasks and settings in the SIMPLER environment, we demonstrate that SITCOM when combined with a good reward function can significantly improve task completion rate from 48% to 72% using trained dynamics model.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04041) | **Categories:** cs.RO

---

### [9] [Zenbo Patrol: A Social Assistive Robot Based on Multimodal Deep Learning for Real-time Illegal Parking Recognition and Notification](https://arxiv.org/abs/2510.04190)
*Jian-jie Zheng, Chih-kai Yang, Po-han Chen, Lyn Chao-ling Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In the study, the social robot act as a patrol to recognize and notify illegal parking in real-time. Dual-model pipeline method and large multimodal model were compared, and the GPT-4o multimodal model was adopted in license plate recognition without preprocessing. For moving smoothly on a flat ground, the robot navigated in a simulated parking lot in the experiments. The robot changes angle view of the camera automatically to capture the images around with the format of license plate number. From the captured images of the robot, the numbers on the plate are recognized through the GPT-4o model, and identifies legality of the numbers. When an illegal parking is detected, the robot sends Line messages to the system manager immediately. The contribution of the work is that a novel multimodal deep learning method has validated with high accuracy in license plate recognition, and a social assistive robot is also provided for solving problems in a real scenario, and can be applied in an indoor parking lot.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04190) | **Categories:** cs.RO

---

### [10] [ContextVLA: Vision-Language-Action Model with Amortized Multi-Frame Context](https://arxiv.org/abs/2510.04246)
*Huiwon Jang, Sihyun Yu, Heeseung Kwon, Hojin Jeon, Younggyo Seo, Jinwoo Shin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Leveraging temporal context is crucial for success in partially observable robotic tasks. However, prior work in behavior cloning has demonstrated inconsistent performance gains when using multi-frame observations. In this paper, we introduce ContextVLA, a policy model that robustly improves robotic task performance by effectively leveraging multi-frame observations. Our approach is motivated by the key observation that Vision-Language-Action models (VLA), i.e., policy models built upon a Vision-Language Model (VLM), more effectively utilize multi-frame observations for action generation. This suggests that VLMs' inherent temporal understanding capability enables them to extract more meaningful context from multi-frame observations. However, the high dimensionality of video inputs introduces significant computational overhead, making VLA training and inference inefficient. To address this, ContextVLA compresses past observations into a single context token, allowing the policy to efficiently leverage temporal context for action generation. Our experiments show that ContextVLA consistently improves over single-frame VLAs and achieves the benefits of full multi-frame training but with reduced training and inference times.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.04246) | **Categories:** cs.RO, cs.AI

---
