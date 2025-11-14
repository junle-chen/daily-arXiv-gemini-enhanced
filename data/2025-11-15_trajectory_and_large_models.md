# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-15

## 目录

- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [cs.DB (1)](#cs-db)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (4)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction](https://arxiv.org/abs/2511.10203)
*Stephane Da Silva Martins, Emanuel Aldea, Sylvie Le Hégarat-Mascle*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10203) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [Social LSTM with Dynamic Occupancy Modeling for Realistic Pedestrian Trajectory Prediction](https://arxiv.org/abs/2511.09735)
*Ahmed Alia, Mohcine Chraibi, Armin Seyfried*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In dynamic and crowded environments, realistic pedestrian trajectory prediction remains a challenging task due to the complex nature of human motion and the mutual influences among individuals. Deep learning models have recently achieved promising results by implicitly learning such patterns from 2D trajectory data. However, most approaches treat pedestrians as point entities, ignoring the physical space that each person occupies. To address these limitations, this paper proposes a novel deep learning model that enhances the Social LSTM with a new Dynamic Occupied Space loss function. This loss function guides Social LSTM in learning to avoid realistic collisions without increasing displacement error across different crowd densities, ranging from low to high, in both homogeneous and heterogeneous density settings. Such a function achieves this by combining the average displacement error with a new collision penalty that is sensitive to scene density and individual spatial occupancy. For efficient training and evaluation, five datasets were generated from real pedestrian trajectories recorded during the Festival of Lights in Lyon 2022. Four datasets represent homogeneous crowd conditions -- low, medium, high, and very high density -- while the fifth corresponds to a heterogeneous density distribution. The experimental findings indicate that the proposed model not only lowers collision rates but also enhances displacement prediction accuracy in each dataset. Specifically, the model achieves up to a 31% reduction in the collision rate and reduces the average displacement error and the final displacement error by 5% and 6%, respectively, on average across all datasets compared to the baseline. Moreover, the proposed model consistently outperforms several state-of-the-art deep learning models across most test sets.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.09735) | **Categories:** cs.CV, cs.AI

---

### [3] [From Street to Orbit: Training-Free Cross-View Retrieval via Location Semantics and LLM Guidance](https://arxiv.org/abs/2511.09820)
*Jeongho Min, Dongyoung Kim, Jaehyup Lee*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Cross-view image retrieval, particularly street-to-satellite matching, is a critical task for applications such as autonomous navigation, urban planning, and localization in GPS-denied environments. However, existing approaches often require supervised training on curated datasets and rely on panoramic or UAV-based images, which limits real-world deployment. In this paper, we present a simple yet effective cross-view image retrieval framework that leverages a pretrained vision encoder and a large language model (LLM), requiring no additional training. Given a monocular street-view image, our method extracts geographic cues through web-based image search and LLM-based location inference, generates a satellite query via geocoding API, and retrieves matching tiles using a pretrained vision encoder (e.g., DINOv2) with PCA-based whitening feature refinement. Despite using no ground-truth supervision or finetuning, our proposed method outperforms prior learning-based approaches on the benchmark dataset under zero-shot settings. Moreover, our pipeline enables automatic construction of semantically aligned street-to-satellite datasets, which is offering a scalable and cost-efficient alternative to manual annotation. All source codes will be made publicly available at https://jeonghomin.github.io/street2orbit.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.09820) | **Categories:** cs.CV, cs.AI

---

### [4] [HCC-3D: Hierarchical Compensatory Compression for 98% 3D Token Reduction in Vision-Language Models](https://arxiv.org/abs/2511.09883)
*Liheng Zhang, Jin Wang, Hui Li, Bingfeng Zhang, Weifeng Liu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: 3D understanding has drawn significant attention recently, leveraging Vision-Language Models (VLMs) to enable multi-modal reasoning between point cloud and text data. Current 3D-VLMs directly embed the 3D point clouds into 3D tokens, following large 2D-VLMs with powerful reasoning capabilities. However, this framework has a great computational cost limiting its application, where we identify that the bottleneck lies in processing all 3D tokens in the Large Language Model (LLM) part. This raises the question: how can we reduce the computational overhead introduced by 3D tokens while preserving the integrity of their essential information? To address this question, we introduce Hierarchical Compensatory Compression (HCC-3D) to efficiently compress 3D tokens while maintaining critical detail retention. Specifically, we first propose a global structure compression (GSC), in which we design global queries to compress all 3D tokens into a few key tokens while keeping overall structural information. Then, to compensate for the information loss in GSC, we further propose an adaptive detail mining (ADM) module that selectively recompresses salient but under-attended features through complementary scoring. Extensive experiments demonstrate that HCC-3D not only achieves extreme compression ratios (approximately 98%) compared to previous 3D-VLMs, but also achieves new state-of-the-art performance, showing the great improvements on both efficiency and performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.09883) | **Categories:** cs.CV

---


## cs.DB [cs.DB]
### [1] [CityVerse: A Unified Data Platform for Multi-Task Urban Computing with Large Language Models](https://arxiv.org/abs/2511.10418)
*Yaqiao Zhu, Hongkai Wen, Mark Birkin, Man Luo*

Main category: cs.DB

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) show remarkable potential for urban computing, from spatial reasoning to predictive analytics. However, evaluating LLMs across diverse urban tasks faces two critical challenges: lack of unified platforms for consistent multi-source data access and fragmented task definitions that hinder fair comparison. To address these challenges, we present CityVerse, the first unified platform integrating multi-source urban data, capability-based task taxonomy, and dynamic simulation for systematic LLM evaluation in urban contexts. CityVerse provides: 1) coordinate-based Data APIs unifying ten categories of urban data-including spatial features, temporal dynamics, demographics, and multi-modal imagery-with over 38 million curated records; 2) Task APIs organizing 43 urban computing tasks into a four-level cognitive hierarchy: Perception, Spatial Understanding, Reasoning and Prediction, and Decision and Interaction, enabling standardized evaluation across capability levels; 3) an interactive visualization frontend supporting real-time data retrieval, multi-layer display, and simulation replay for intuitive exploration and validation. We validate the platform's effectiveness through evaluations on mainstream LLMs across representative tasks, demonstrating its capability to support reproducible and systematic assessment. CityVerse provides a reusable foundation for advancing LLMs and multi-task approaches in the urban computing domain.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10418) | **Categories:** cs.DB

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Out-of-Distribution Generalization with a SPARC: Racing 100 Unseen Vehicles with a Single Policy](https://arxiv.org/abs/2511.09737)
*Bram Grooten, Patrick MacAlpine, Kaushik Subramanian, Peter Stone, Peter R. Wurman*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generalization to unseen environments is a significant challenge in the field of robotics and control. In this work, we focus on contextual reinforcement learning, where agents act within environments with varying contexts, such as self-driving cars or quadrupedal robots that need to operate in different terrains or weather conditions than they were trained for. We tackle the critical task of generalizing to out-of-distribution (OOD) settings, without access to explicit context information at test time. Recent work has addressed this problem by training a context encoder and a history adaptation module in separate stages. While promising, this two-phase approach is cumbersome to implement and train. We simplify the methodology and introduce SPARC: single-phase adaptation for robust control. We test SPARC on varying contexts within the high-fidelity racing simulator Gran Turismo 7 and wind-perturbed MuJoCo environments, and find that it achieves reliable and robust OOD generalization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.09737) | **Categories:** cs.LG, cs.AI

---

### [2] [Hail to the Thief: Exploring Attacks and Defenses in Decentralised GRPO](https://arxiv.org/abs/2511.09780)
*Nikolay Blagoev, Oğuzhan Ersoy, Lydia Yiyu Chen*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralised training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralised GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to 100% in as few as 50 iterations. We propose two ways to defend against these attacks, depending on whether all users train the same model or different models. We show that these defenses can achieve stop rates of up to 100%, making the attack impossible.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.09780) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [LongComp: Long-Tail Compositional Zero-Shot Generalization for Robust Trajectory Prediction](https://arxiv.org/abs/2511.10411)
*Benjamin Stoler, Jonathan Francis, Jean Oh*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Methods for trajectory prediction in Autonomous Driving must contend with rare, safety-critical scenarios that make reliance on real-world data collection alone infeasible. To assess robustness under such conditions, we propose new long-tail evaluation settings that repartition datasets to create challenging out-of-distribution (OOD) test sets. We first introduce a safety-informed scenario factorization framework, which disentangles scenarios into discrete ego and social contexts. Building on analogies to compositional zero-shot image-labeling in Computer Vision, we then hold out novel context combinations to construct challenging closed-world and open-world settings. This process induces OOD performance gaps in future motion prediction of 5.0% and 14.7% in closed-world and open-world settings, respectively, relative to in-distribution performance for a state-of-the-art baseline. To improve generalization, we extend task-modular gating networks to operate within trajectory prediction models, and develop an auxiliary, difficulty-prediction head to refine internal representations. Our strategies jointly reduce the OOD performance gaps to 2.8% and 11.5% in the two settings, respectively, while still improving in-distribution performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10411) | **Categories:** cs.RO

---

### [2] [Learning a Thousand Tasks in a Day](https://arxiv.org/abs/2511.10110)
*Kamil Dreczkowski, Pietro Vitiello, Vitalis Vosylius, Edward Johns*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humans are remarkably efficient at learning tasks from demonstrations, but today's imitation learning methods for robot manipulation often require hundreds or thousands of demonstrations per task. We investigate two fundamental priors for improving learning efficiency: decomposing manipulation trajectories into sequential alignment and interaction phases, and retrieval-based generalisation. Through 3,450 real-world rollouts, we systematically study this decomposition. We compare different design choices for the alignment and interaction phases, and examine generalisation and scaling trends relative to today's dominant paradigm of behavioural cloning with a single-phase monolithic policy. In the few-demonstrations-per-task regime (<10 demonstrations), decomposition achieves an order of magnitude improvement in data efficiency over single-phase learning, with retrieval consistently outperforming behavioural cloning for both alignment and interaction. Building on these insights, we develop Multi-Task Trajectory Transfer (MT3), an imitation learning method based on decomposition and retrieval. MT3 learns everyday manipulation tasks from as little as a single demonstration each, whilst also generalising to novel object instances. This efficiency enables us to teach a robot 1,000 distinct everyday tasks in under 24 hours of human demonstrator time. Through 2,200 additional real-world rollouts, we reveal MT3's capabilities and limitations across different task families. Videos of our experiments can be found on at https://www.robot-learning.uk/learning-1000-tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10110) | **Categories:** cs.RO

---

### [3] [nuPlan-R: A Closed-Loop Planning Benchmark for Autonomous Driving via Reactive Multi-Agent Simulation](https://arxiv.org/abs/2511.10403)
*Mingxing Peng, Ruoyu Yao, Xusen Guo, Jun Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in closed-loop planning benchmarks have significantly improved the evaluation of autonomous vehicles. However, existing benchmarks still rely on rule-based reactive agents such as the Intelligent Driver Model (IDM), which lack behavioral diversity and fail to capture realistic human interactions, leading to oversimplified traffic dynamics. To address these limitations, we present nuPlan-R, a new reactive closed-loop planning benchmark that integrates learning-based reactive multi-agent simulation into the nuPlan framework. Our benchmark replaces the rule-based IDM agents with noise-decoupled diffusion-based reactive agents and introduces an interaction-aware agent selection mechanism to ensure both realism and computational efficiency. Furthermore, we extend the benchmark with two additional metrics to enable a more comprehensive assessment of planning performance. Extensive experiments demonstrate that our reactive agent model produces more realistic, diverse, and human-like traffic behaviors, leading to a benchmark environment that better reflects real-world interactive driving. We further reimplement a collection of rule-based, learning-based, and hybrid planning approaches within our nuPlan-R benchmark, providing a clearer reflection of planner performance in complex interactive scenarios and better highlighting the advantages of learning-based planners in handling complex and dynamic scenarios. These results establish nuPlan-R as a new standard for fair, reactive, and realistic closed-loop planning evaluation. We will open-source the code for the new benchmark.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10403) | **Categories:** cs.RO, cs.AI

---

### [4] [Optimizing the flight path for a scouting Uncrewed Aerial Vehicle](https://arxiv.org/abs/2511.10598)
*Raghav Adhikari, Sachet Khatiwada, Suman Poudel*

Main category: cs.RO

TL;DR: 本文提出了一种基于优化的无人机路径规划方法，用于灾后环境侦察，旨在通过优化无人机飞行高度，最大化传感器覆盖范围并最小化数据不确定性。


<details>
  <summary>Details</summary>
Motivation: 灾后环境的非结构化特性给救援车辆的路径规划带来了挑战。

Method: 提出了一种基于优化的方法，用于规划无人机的路径，以使其在最佳高度飞行，从而最大化传感器覆盖范围并最小化数据不确定性。

Result: 通过优化无人机飞行高度，最大化传感器覆盖范围并最小化数据不确定性。

Conclusion: 本文研究表明，在灾后环境中使用无人机进行侦察是可行的，并且可以通过优化无人机路径来提高效率。

Abstract: 灾后环境带来了独特的导航挑战。其中一个挑战是非结构化的环境，这使得为救援车辆规划路径变得困难。我们建议在此类场景中使用无人机（UAV）来执行环境侦察。为了实现这一目标，我们提出了一种基于优化的方法来规划无人机的路径，使其在最佳高度飞行，无人机的传感器可以覆盖最大面积并以最小的不确定性收集数据。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10598) | **Categories:** cs.RO, eess.SY

---


## eess.SY [eess.SY]
### [1] [Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction](https://arxiv.org/abs/2511.10586)
*Omid Mirzaeedodangeh, Eliot Shekhtman, Nikolai Matni, Lars Lindemann*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe planning of an autonomous agent in interactive environments -- such as the control of a self-driving vehicle among pedestrians and human-controlled vehicles -- poses a major challenge as the behavior of the environment is unknown and reactive to the behavior of the autonomous agent. This coupling gives rise to interaction-driven distribution shifts where the autonomous agent's control policy may change the environment's behavior, thereby invalidating safety guarantees in existing work. Indeed, recent works have used conformal prediction (CP) to generate distribution-free safety guarantees using observed data of the environment. However, CP's assumption on data exchangeability is violated in interactive settings due to a circular dependency where a control policy update changes the environment's behavior, and vice versa. To address this gap, we propose an iterative framework that robustly maintains safety guarantees across policy updates by quantifying the potential impact of a planned policy update on the environment's behavior. We realize this via adversarially robust CP where we perform a regular CP step in each episode using observed data under the current policy, but then transfer safety guarantees across policy updates by analytically adjusting the CP result to account for distribution shifts. This adjustment is performed based on a policy-to-trajectory sensitivity analysis, resulting in a safe, episodic open-loop planner. We further conduct a contraction analysis of the system providing conditions under which both the CP results and the policy updates are guaranteed to converge. We empirically demonstrate these safety and convergence guarantees on a two-dimensional car-pedestrian case study. To the best of our knowledge, these are the first results that provide valid safety guarantees in such interactive settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10586) | **Categories:** eess.SY, cs.RO

---
