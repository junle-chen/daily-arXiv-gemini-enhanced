# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-12

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (11)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [From Prompts to Power: Measuring the Energy Footprint of LLM Inference](https://arxiv.org/abs/2511.05597)
*Francisco Caravaca, Ángel Cuevas, Rubén Cuevas*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid expansion of Large Language Models (LLMs) has introduced unprecedented energy demands, extending beyond training to large-scale inference workloads that often dominate total lifecycle consumption. Deploying these models requires energy-intensive GPU infrastructure, and in some cases has even prompted plans to power data centers with nuclear energy. Despite this growing relevance, systematic analyses of inference energy consumption remain limited. In this work, we present a large-scale measurement-based study comprising over 32,500 measurements across 21 GPU configurations and 155 model architectures, from small open-source models to frontier systems. Using the vLLM inference engine, we quantify energy usage at the prompt level and identify how architectural and operational factors shape energy demand. Building on these insights, we develop a predictive model that accurately estimates inference energy consumption across unseen architectures and hardware, and implement it as a browser extension to raise awareness of the environmental impact of generative AI.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05597) | **Categories:** cs.AI, cs.LG

---

### [2] [ROAR: Robust Accident Recognition and Anticipation for Autonomous Driving](https://arxiv.org/abs/2511.06226)
*Xingcheng Liu, Yanchen Guan, Haicheng Liao, Zhengbing He, Zhenning Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate accident anticipation is essential for enhancing the safety of autonomous vehicles (AVs). However, existing methods often assume ideal conditions, overlooking challenges such as sensor failures, environmental disturbances, and data imperfections, which can significantly degrade prediction accuracy. Additionally, previous models have not adequately addressed the considerable variability in driver behavior and accident rates across different vehicle types. To overcome these limitations, this study introduces ROAR, a novel approach for accident detection and prediction. ROAR combines Discrete Wavelet Transform (DWT), a self adaptive object aware module, and dynamic focal loss to tackle these challenges. The DWT effectively extracts features from noisy and incomplete data, while the object aware module improves accident prediction by focusing on high-risk vehicles and modeling the spatial temporal relationships among traffic agents. Moreover, dynamic focal loss mitigates the impact of class imbalance between positive and negative samples. Evaluated on three widely used datasets, Dashcam Accident Dataset (DAD), Car Crash Dataset (CCD), and AnAn Accident Detection (A3D), our model consistently outperforms existing baselines in key metrics such as Average Precision (AP) and mean Time to Accident (mTTA). These results demonstrate the model's robustness in real-world conditions, particularly in handling sensor degradation, environmental noise, and imbalanced data distributions. This work offers a promising solution for reliable and accurate accident anticipation in complex traffic environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06226) | **Categories:** cs.AI

---

### [3] [AUTO-Explorer: Automated Data Collection for GUI Agent](https://arxiv.org/abs/2511.06417)
*Xiangwu Guo, Difei Gao, Mike Zheng Shou*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in GUI agents have significantly expanded their ability to interpret natural language commands to manage software interfaces. However, acquiring GUI data remains a significant challenge. Existing methods often involve designing automated agents that browse URLs from the Common Crawl, using webpage HTML to collect screenshots and corresponding annotations, including the names and bounding boxes of UI elements. However, this method is difficult to apply to desktop software or some newly launched websites not included in the Common Crawl. While we expect the model to possess strong generalization capabilities to handle this, it is still crucial for personalized scenarios that require rapid and perfect adaptation to new software or websites. To address this, we propose an automated data collection method with minimal annotation costs, named Auto-Explorer. It incorporates a simple yet effective exploration mechanism that autonomously parses and explores GUI environments, gathering data efficiently. Additionally, to assess the quality of exploration, we have developed the UIXplore benchmark. This benchmark creates environments for explorer agents to discover and save software states. Using the data gathered, we fine-tune a multimodal large language model (MLLM) and establish a GUI element grounding testing set to evaluate the effectiveness of the exploration strategies. Our experiments demonstrate the superior performance of Auto-Explorer, showing that our method can quickly enhance the capabilities of an MLLM in explored software.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06417) | **Categories:** cs.AI

---

### [4] [GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](https://arxiv.org/abs/2511.06471)
*Jingtao Tang, Hang Ma*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We study GCS-TSP, a new variant of the Traveling Salesman Problem (TSP) defined over a Graph of Convex Sets (GCS) -- a powerful representation for trajectory planning that decomposes the configuration space into convex regions connected by a sparse graph. In this setting, edge costs are not fixed but depend on the specific trajectory selected through each convex region, making classical TSP methods inapplicable. We introduce GHOST, a hierarchical framework that optimally solves the GCS-TSP by combining combinatorial tour search with convex trajectory optimization. GHOST systematically explores tours on a complete graph induced by the GCS, using a novel abstract-path-unfolding algorithm to compute admissible lower bounds that guide best-first search at both the high level (over tours) and the low level (over feasible GCS paths realizing the tour). These bounds provide strong pruning power, enabling efficient search while avoiding unnecessary convex optimization calls. We prove that GHOST guarantees optimality and present a bounded-suboptimal variant for time-critical scenarios. Experiments show that GHOST is orders-of-magnitude faster than unified mixed-integer convex programming baselines for simple cases and uniquely handles complex trajectory planning problems involving high-order continuity constraints and an incomplete GCS.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06471) | **Categories:** cs.AI, cs.RO

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Token Is All You Need: Cognitive Planning through Sparse Intent Alignment](https://arxiv.org/abs/2511.05540)
*Shiyao Sang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Unlike world-model approaches that rely on computationally intensive future scene generation or vision-language-action (VLA) systems constrained by Markov assumptions, we show that a minimal set of semantically rich tokens is sufficient for effective planning. Experiments on the nuPlan benchmark (720 scenarios, over 11,000 samples) using perception-informed BEV representations yield three key findings: (1) even without future prediction, our sparse representation achieves 0.548 m ADE, comparable to or surpassing prior methods reporting around 0.75 m on nuScenes; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.479 m, a 12.6% improvement over current-state baselines; and (3) explicit reconstruction loss offers no benefit and may degrade performance under reliable perception inputs. Notably, we observe the emergence of temporal fuzziness, where the model adaptively attends to task-relevant semantics rather than aligning rigidly to fixed timestamps, providing a cognitive advantage for planning under uncertainty. Our "token is all you need" principle marks a paradigm shift from reconstructing the world to understanding it, laying a foundation for cognitively inspired systems that plan through imagination rather than reaction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05540) | **Categories:** cs.CV, cs.AI, cs.LG, cs.RO

---

### [2] [EVLP:Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning](https://arxiv.org/abs/2511.05553)
*Xinyan Cai, Shiguang Wu, Dafeng Chi, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Qiang Guan*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In complex embodied long-horizon manipulation tasks, effective task decomposition and execution require synergistic integration of textual logical reasoning and visual-spatial imagination to ensure efficient and accurate operation. Current methods fail to adopt a unified generation framework for multimodal planning, lead to inconsistent in multimodal planning. To address this challenge, we present \textbf{EVLP (Embodied Vision-Language Planner)}, an innovative multimodal unified generation framework that jointly models linguistic reasoning and visual generation. Our approach achieves multimodal planning for long-horizon tasks through a novel training pipeline incorporating dynamic pretraining and reinforced alignment. Our core innovations consist of three key components: \textbf{1) Unified Multimodal Generation Framework}: For understanding, We integrate semantic information with spatial features to provide comprehensive visual perception. For generation, we directly learn the joint distribution of discrete images for one-step visual synthesis, enabling coordinated language-visual modeling through learnable cross-modal attention mechanisms. \textbf{2) Dynamic Perception Pretraining}: We propose a bidirectional dynamic alignment strategy employing inverse dynamics tasks and forward dynamics tasks, effectively strengthening multimodal correlations within a unified feature space. \textbf{3) Reinforced Supervised Fine-Tuning}: While conducting instruction-based fine-tuning in the unified generation space, we construct a reinforce loss to align the spatial logic between textual actions and generated images, enabling the model to acquire spatio-awared multimodal planning capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05553) | **Categories:** cs.CV, cs.AI

---

### [3] [Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](https://arxiv.org/abs/2511.05609)
*Ziying Li, Xuequan Lu, Xinkui Zhao, Guanjie Cheng, Shuiguang Deng, Jianwei Yin*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in optimization-based text-to-3D generation heavily rely on distilling knowledge from pre-trained text-to-image diffusion models using techniques like Score Distillation Sampling (SDS), which often introduce artifacts such as over-saturation and over-smoothing into the generated 3D assets. In this paper, we address this essential problem by formulating the generation process as learning an optimal, direct transport trajectory between the distribution of the current rendering and the desired target distribution, thereby enabling high-quality generation with smaller Classifier-free Guidance (CFG) values. At first, we theoretically establish SDS as a simplified instance of the Schrödinger Bridge framework. We prove that SDS employs the reverse process of an Schrödinger Bridge, which, under specific conditions (e.g., a Gaussian noise as one end), collapses to SDS's score function of the pre-trained diffusion model. Based upon this, we introduce Trajectory-Centric Distillation (TraCe), a novel text-to-3D generation framework, which reformulates the mathematically trackable framework of Schrödinger Bridge to explicitly construct a diffusion bridge from the current rendering to its text-conditioned, denoised target, and trains a LoRA-adapted model on this trajectory's score dynamics for robust 3D optimization. Comprehensive experiments demonstrate that TraCe consistently achieves superior quality and fidelity to state-of-the-art techniques.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05609) | **Categories:** cs.CV, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [CoPRIS: Efficient and Stable Reinforcement Learning via Concurrency-Controlled Partial Rollout with Importance Sampling](https://arxiv.org/abs/2511.05589)
*Zekai Qu, Yinxu Pan, Ao Sun, Chaojun Xiao, Xu Han*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) post-training has become a trending paradigm for enhancing the capabilities of large language models (LLMs). Most existing RL systems for LLMs operate in a fully synchronous manner, where training must wait for the rollout of an entire batch to complete. This design leads to severe inefficiencies, as extremely long trajectories can stall the entire rollout process and leave many GPUs idle. To address this issue, we propose Concurrency- Controlled Partial Rollout with Importance Sampling (CoPRIS), which mitigates long-tail inefficiencies by maintaining a fixed number of concurrent rollouts, early-terminating once sufficient samples are collected, and reusing unfinished trajectories in subsequent rollouts. To mitigate the impact of off-policy trajectories, we introduce Cross-stage Importance Sampling Correction, which concatenates buffered log probabilities from the previous policy with those recomputed under the current policy for importance sampling correction. Experiments on challenging mathematical reasoning benchmarks show that CoPRIS achieves up to 1.94x faster training while maintaining comparable or superior performance to synchronous RL systems. The code of CoPRIS is available at https://github.com/777pomingzi/CoPRIS.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05589) | **Categories:** cs.LG, cs.AI

---

### [2] [MARAuder's Map: Motion-Aware Real-time Activity Recognition with Layout-Based Trajectories](https://arxiv.org/abs/2511.05773)
*Zishuai Liu, Weihang You, Jin Lu, Fei Dou*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ambient sensor-based human activity recognition (HAR) in smart homes remains challenging due to the need for real-time inference, spatially grounded reasoning, and context-aware temporal modeling. Existing approaches often rely on pre-segmented, within-activity data and overlook the physical layout of the environment, limiting their robustness in continuous, real-world deployments. In this paper, we propose MARAuder's Map, a novel framework for real-time activity recognition from raw, unsegmented sensor streams. Our method projects sensor activations onto the physical floorplan to generate trajectory-aware, image-like sequences that capture the spatial flow of human movement. These representations are processed by a hybrid deep learning model that jointly captures spatial structure and temporal dependencies. To enhance temporal awareness, we introduce a learnable time embedding module that encodes contextual cues such as hour-of-day and day-of-week. Additionally, an attention-based encoder selectively focuses on informative segments within each observation window, enabling accurate recognition even under cross-activity transitions and temporal ambiguity. Extensive experiments on multiple real-world smart home datasets demonstrate that our method outperforms strong baselines, offering a practical solution for real-time HAR in ambient sensor environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05773) | **Categories:** cs.LG, cs.CV

---


## 机器人学 (Robotics) [cs.RO]
### [1] [From Words to Safety: Language-Conditioned Safety Filtering for Robot Navigation](https://arxiv.org/abs/2511.05889)
*Zeyuan Feng, Haimingyue Zhang, Somil Bansal*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: As robots become increasingly integrated into open-world, human-centered environments, their ability to interpret natural language instructions and adhere to safety constraints is critical for effective and trustworthy interaction. Existing approaches often focus on mapping language to reward functions instead of safety specifications or address only narrow constraint classes (e.g., obstacle avoidance), limiting their robustness and applicability. We propose a modular framework for language-conditioned safety in robot navigation. Our framework is composed of three core components: (1) a large language model (LLM)-based module that translates free-form instructions into structured safety specifications, (2) a perception module that grounds these specifications by maintaining object-level 3D representations of the environment, and (3) a model predictive control (MPC)-based safety filter that enforces both semantic and geometric constraints in real time. We evaluate the effectiveness of the proposed framework through both simulation studies and hardware experiments, demonstrating that it robustly interprets and enforces diverse language-specified constraints across a wide range of environments and scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05889) | **Categories:** cs.RO

---

### [2] [OpenVLN: Open-world aerial Vision-Language Navigation](https://arxiv.org/abs/2511.06182)
*Peican Lin, Gan Sun, Chenxi Liu, Fazeng Li, Weihong Ren, Yang Cong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-language models (VLMs) have been widely-applied in ground-based vision-language navigation (VLN). However, the vast complexity of outdoor aerial environments compounds data acquisition challenges and imposes long-horizon trajectory planning requirements on Unmanned Aerial Vehicles (UAVs), introducing novel complexities for aerial VLN. To address these challenges, we propose a data-efficient Open-world aerial Vision-Language Navigation (i.e., OpenVLN) framework, which could execute language-guided flight with limited data constraints and enhance long-horizon trajectory planning capabilities in complex aerial environments. Specifically, we reconfigure a reinforcement learning framework to optimize the VLM for UAV navigation tasks, which can efficiently fine-tune VLM by using rule-based policies under limited training data. Concurrently, we introduce a long-horizon planner for trajectory synthesis that dynamically generates precise UAV actions via value-based rewards. To the end, we conduct sufficient navigation experiments on the TravelUAV benchmark with dataset scaling across diverse reward settings. Our method demonstrates consistent performance gains of up to 4.34% in Success Rate, 6.19% in Oracle Success Rate, and 4.07% in Success weighted by Path Length over baseline methods, validating its deployment efficacy for long-horizon UAV navigation in complex aerial environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06182) | **Categories:** cs.RO

---

### [3] [Multi-Agent AI Framework for Road Situation Detection and C-ITS Message Generation](https://arxiv.org/abs/2511.06892)
*Kailin Tong, Selim Solmaz, Kenan Mujkic, Gottfried Allmer, Bo Leng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Conventional road-situation detection methods achieve strong performance in predefined scenarios but fail in unseen cases and lack semantic interpretation, which is crucial for reliable traffic recommendations. This work introduces a multi-agent AI framework that combines multimodal large language models (MLLMs) with vision-based perception for road-situation monitoring. The framework processes camera feeds and coordinates dedicated agents for situation detection, distance estimation, decision-making, and Cooperative Intelligent Transport System (C-ITS) message generation. Evaluation is conducted on a custom dataset of 103 images extracted from 20 videos of the TAD dataset. Both Gemini-2.0-Flash and Gemini-2.5-Flash were evaluated. The results show 100\% recall in situation detection and perfect message schema correctness; however, both models suffer from false-positive detections and have reduced performance in terms of number of lanes, driving lane status and cause code. Surprisingly, Gemini-2.5-Flash, though more capable in general tasks, underperforms Gemini-2.0-Flash in detection accuracy and semantic understanding and incurs higher latency (Table II). These findings motivate further work on fine-tuning specialized LLMs or MLLMs tailored for intelligent transportation applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06892) | **Categories:** cs.RO

---

### [4] [Lite VLA: Efficient Vision-Language-Action Control on CPU-Bound Edge Robots](https://arxiv.org/abs/2511.05642)
*Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The deployment of artificial intelligence models at the edge is increasingly critical for autonomous robots operating in GPS-denied environments where local, resource-efficient reasoning is essential. This work demonstrates the feasibility of deploying small Vision-Language Models (VLMs) on mobile robots to achieve real-time scene understanding and reasoning under strict computational constraints. Unlike prior approaches that separate perception from mobility, the proposed framework enables simultaneous movement and reasoning in dynamic environments using only on-board hardware. The system integrates a compact VLM with multimodal perception to perform contextual interpretation directly on embedded hardware, eliminating reliance on cloud connectivity. Experimental validation highlights the balance between computational efficiency, task accuracy, and system responsiveness. Implementation on a mobile robot confirms one of the first successful deployments of small VLMs for concurrent reasoning and mobility at the edge. This work establishes a foundation for scalable, assured autonomy in applications such as service robotics, disaster response, and defense operations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05642) | **Categories:** cs.RO, cs.AR, cs.CV, eess.SY

---

### [5] [Fair and Safe: A Real-Time Hierarchical Control Framework for Intersections](https://arxiv.org/abs/2511.05886)
*Lei Shi, Yongju Kim, Xinzhi Zhong, Wissam Kontar, Qichao Liu, Soyoung Ahn*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring fairness in the coordination of connected and automated vehicles at intersections is essential for equitable access, social acceptance, and long-term system efficiency, yet it remains underexplored in safety-critical, real-time traffic control. This paper proposes a fairness-aware hierarchical control framework that explicitly integrates inequity aversion into intersection management. At the top layer, a centralized allocation module assigns control authority (i.e., selects a single vehicle to execute its trajectory) by maximizing a utility that accounts for waiting time, urgency, control history, and velocity deviation. At the bottom layer, the authorized vehicle executes a precomputed trajectory using a Linear Quadratic Regulator (LQR) and applies a high-order Control Barrier Function (HOCBF)-based safety filter for real-time collision avoidance. Simulation results across varying traffic demands and demand distributions demonstrate that the proposed framework achieves near-perfect fairness, eliminates collisions, reduces average delay, and maintains real-time feasibility. These results highlight that fairness can be systematically incorporated without sacrificing safety or performance, enabling scalable and equitable coordination for future autonomous traffic systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05886) | **Categories:** cs.RO

---

### [6] [10 Open Challenges Steering the Future of Vision-Language-Action Models](https://arxiv.org/abs/2511.05936)
*Soujanya Poria, Navonil Majumder, Chia-Yu Hung, Amir Ali Bagherzadeh, Chuan Li, Kenneth Kwok, Ziwei Wang, Cheston Tan, Jiajun Wu, David Hsu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05936) | **Categories:** cs.RO, cs.AI

---

### [7] [ExpReS-VLA: Specializing Vision-Language-Action Models Through Experience Replay and Retrieval](https://arxiv.org/abs/2511.06202)
*Shahram Najam Syed, Yatharth Ahuja, Arthur Jakobsson, Jeff Ichnowski*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action models such as OpenVLA show impressive zero-shot generalization across robotic manipulation tasks but often fail to adapt efficiently to new deployment environments. In many real-world applications, consistent high performance on a limited set of tasks is more important than broad generalization. We propose ExpReS-VLA, a method for specializing pre-trained VLA models through experience replay and retrieval while preventing catastrophic forgetting. ExpReS-VLA stores compact feature representations from the frozen vision backbone instead of raw image-action pairs, reducing memory usage by approximately 97 percent. During deployment, relevant past experiences are retrieved using cosine similarity and used to guide adaptation, while prioritized experience replay emphasizes successful trajectories. We also introduce Thresholded Hybrid Contrastive Loss, which enables learning from both successful and failed attempts. On the LIBERO simulation benchmark, ExpReS-VLA improves success rates from 82.6 to 93.1 percent on spatial reasoning tasks and from 61 to 72.3 percent on long-horizon tasks. On physical robot experiments with five manipulation tasks, it reaches 98 percent success on both seen and unseen settings, compared to 84.7 and 32 percent for naive fine-tuning. Adaptation takes 31 seconds using 12 demonstrations on a single RTX 5090 GPU, making the approach practical for real robot deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06202) | **Categories:** cs.RO

---

### [8] [A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving](https://arxiv.org/abs/2511.06496)
*Keke Long, Jiacheng Guo, Tianyun Zhang, Hongkai Yu, Xiaopeng Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision Language Models (VLMs) are increasingly used in autonomous driving to help understand traffic scenes, but they sometimes produce hallucinations, which are false details not grounded in the visual input. Detecting and mitigating hallucinations is challenging when ground-truth references are unavailable and model internals are inaccessible. This paper proposes a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access. By constructing a sentence-embedding matrix and decomposing it into a low-rank consensus component and a sparse residual, we use the residual magnitude to rank captions: selecting the one with the smallest residual as the most hallucination-free. Experiments on the NuScenes dataset demonstrate that our approach achieves 87% selection accuracy in identifying hallucination-free captions, representing a 19% improvement over the unfiltered baseline and a 6-10% improvement over multi-agent debate method. The sorting produced by sparse error magnitudes shows strong correlation with human judgments of hallucinations, validating our scoring mechanism. Additionally, our method, which can be easily parallelized, reduces inference time by 51-67% compared to debate approaches, making it practical for real-time autonomous driving applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06496) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [9] [CoFineLLM: Conformal Finetuning of LLMs for Language-Instructed Robot Planning](https://arxiv.org/abs/2511.06575)
*Jun Wang, Yevgeniy Vorobeychik, Yiannis Kantaros*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have recently emerged as planners for language-instructed agents, generating sequences of actions to accomplish natural language tasks. However, their reliability remains a challenge, especially in long-horizon tasks, since they often produce overconfident yet wrong outputs. Conformal Prediction (CP) has been leveraged to address this issue by wrapping LLM outputs into prediction sets that contain the correct action with a user-defined confidence. When the prediction set is a singleton, the planner executes that action; otherwise, it requests help from a user. This has led to LLM-based planners that can ensure plan correctness with a user-defined probability. However, as LLMs are trained in an uncertainty-agnostic manner, without awareness of prediction sets, they tend to produce unnecessarily large sets, particularly at higher confidence levels, resulting in frequent human interventions limiting autonomous deployment. To address this, we introduce CoFineLLM (Conformal Finetuning for LLMs), the first CP-aware finetuning framework for LLM-based planners that explicitly reduces prediction-set size and, in turn, the need for user interventions. We evaluate our approach on multiple language-instructed robot planning problems and show consistent improvements over uncertainty-aware and uncertainty-agnostic finetuning baselines in terms of prediction-set size, and help rates. Finally, we demonstrate robustness of our method to out-of-distribution scenarios in hardware experiments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06575) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [10] [SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation](https://arxiv.org/abs/2511.06754)
*Taisei Hanyu, Nhat Chung, Huy Le, Toan Nguyen, Yuki Ikebe, Anthony Gunderman, Duy Nguyen Ho Minh, Khoa Vo, Tung Kieu, Kashu Yamazaki, Chase Rainwater, Anh Nguyen, Ngan Le*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Inspired by how humans reason over discrete objects and their relationships, we explore whether compact object-centric and object-relation representations can form a foundation for multitask robotic manipulation. Most existing robotic multitask models rely on dense embeddings that entangle both object and background cues, raising concerns about both efficiency and interpretability. In contrast, we study object-relation-centric representations as a pathway to more structured, efficient, and explainable visuomotor control. Our contributions are two-fold. First, we introduce LIBERO+, a fine-grained benchmark dataset designed to enable and evaluate object-relation reasoning in robotic manipulation. Unlike prior datasets, LIBERO+ provides object-centric annotations that enrich demonstrations with box- and mask-level labels as well as instance-level temporal tracking, supporting compact and interpretable visuomotor representations. Second, we propose SlotVLA, a slot-attention-based framework that captures both objects and their relations for action decoding. It uses a slot-based visual tokenizer to maintain consistent temporal object representations, a relation-centric decoder to produce task-relevant embeddings, and an LLM-driven module that translates these embeddings into executable actions. Experiments on LIBERO+ demonstrate that object-centric slot and object-relation slot representations drastically reduce the number of required visual tokens, while providing competitive generalization. Together, LIBERO+ and SlotVLA provide a compact, interpretable, and effective foundation for advancing object-relation-centric robotic manipulation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06754) | **Categories:** cs.RO, cs.CV

---

### [11] [Vision-Aided Online A* Path Planning for Efficient and Safe Navigation of Service Robots](https://arxiv.org/abs/2511.06801)
*Praveen Kumar, Tushar Sandhan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The deployment of autonomous service robots in human-centric environments is hindered by a critical gap in perception and planning. Traditional navigation systems rely on expensive LiDARs that, while geometrically precise, are seman- tically unaware, they cannot distinguish a important document on an office floor from a harmless piece of litter, treating both as physically traversable. While advanced semantic segmentation exists, no prior work has successfully integrated this visual intelligence into a real-time path planner that is efficient enough for low-cost, embedded hardware. This paper presents a frame- work to bridge this gap, delivering context-aware navigation on an affordable robotic platform. Our approach centers on a novel, tight integration of a lightweight perception module with an online A* planner. The perception system employs a semantic segmentation model to identify user-defined visual constraints, enabling the robot to navigate based on contextual importance rather than physical size alone. This adaptability allows an operator to define what is critical for a given task, be it sensitive papers in an office or safety lines in a factory, thus resolving the ambiguity of what to avoid. This semantic perception is seamlessly fused with geometric data. The identified visual constraints are projected as non-geometric obstacles onto a global map that is continuously updated from sensor data, enabling robust navigation through both partially known and unknown environments. We validate our framework through extensive experiments in high-fidelity simulations and on a real-world robotic platform. The results demonstrate robust, real-time performance, proving that a cost- effective robot can safely navigate complex environments while respecting critical visual cues invisible to traditional planners.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.06801) | **Categories:** cs.RO

---
