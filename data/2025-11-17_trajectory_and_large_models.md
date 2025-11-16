# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-17

## 目录

- [计算机视觉 (Computer Vision) (7)](#cs-cv)
- [cs.DB (1)](#cs-db)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (4)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction](https://arxiv.org/abs/2511.10203)
*Stephane Da Silva Martins, Emanuel Aldea, Sylvie Le Hégarat-Mascle*

Main category: cs.CV

TL;DR: VISTA提出了一种递归目标条件Transformer，用于多智能体轨迹预测，能够有效减少碰撞并提高预测精度。


<details>
  <summary>Details</summary>
Motivation: 现有多智能体轨迹预测方法难以同时捕捉智能体的长期目标和细粒度的社交互动，导致预测结果不真实。

Method: VISTA结合了交叉注意力融合模块、社交Token注意力机制和成对注意力图，实现多智能体轨迹预测。

Result: 在MADRAS和SDD数据集上，VISTA取得了state-of-the-art的准确率，并显著减少了碰撞。

Conclusion: VISTA能够生成符合社交规范、感知目标且可解释的轨迹，使其在安全攸关的自主系统中具有应用前景。

Abstract: 多智能体轨迹预测对于在密集、交互环境中运行的自主系统至关重要。现有的方法通常无法联合捕捉智能体的长期目标和他们的细粒度社交互动，这导致了不切实际的多智能体未来。我们提出了VISTA，一个用于多智能体轨迹预测的递归目标条件Transformer。VISTA结合了（i）一个交叉注意力融合模块，它将长时程意图与过去运动整合，（ii）一个社交token注意力机制，用于跨智能体的灵活交互建模，以及（iii）成对注意力图，使得社交影响模式在推理时可解释。我们的模型将单智能体目标条件预测转化为一个连贯的多智能体预测框架。除了标准的位移指标外，我们还评估轨迹碰撞率作为联合真实性的度量。在高密度的MADRAS基准测试和SDD上，VISTA实现了state-of-the-art的准确率，并大幅减少了碰撞。在MADRAS上，它将强基线的平均碰撞率从2.14%降低到0.03%，在SDD上，它实现了零碰撞，同时提高了ADE、FDE和minFDE。这些结果表明，VISTA生成了符合社会规范、感知目标且可解释的轨迹，使其在安全关键的自主系统中具有应用前景。

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

### [4] [Robust Object Detection with Pseudo Labels from VLMs using Per-Object Co-teaching](https://arxiv.org/abs/2511.09955)
*Uday Bhaskar, Rishabh Bhattacharya, Avinash Patel, Sarthak Khoche, Praveen Anil Kulkarni, Naresh Manwani*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundation models, especially vision-language models (VLMs), offer compelling zero-shot object detection for applications like autonomous driving, a domain where manual labelling is prohibitively expensive. However, their detection latency and tendency to hallucinate predictions render them unsuitable for direct deployment. This work introduces a novel pipeline that addresses this challenge by leveraging VLMs to automatically generate pseudo-labels for training efficient, real-time object detectors. Our key innovation is a per-object co-teaching-based training strategy that mitigates the inherent noise in VLM-generated labels. The proposed per-object coteaching approach filters noisy bounding boxes from training instead of filtering the entire image. Specifically, two YOLO models learn collaboratively, filtering out unreliable boxes from each mini-batch based on their peers' per-object loss values. Overall, our pipeline provides an efficient, robust, and scalable approach to train high-performance object detectors for autonomous driving, significantly reducing reliance on costly human annotation. Experimental results on the KITTI dataset demonstrate that our method outperforms a baseline YOLOv5m model, achieving a significant mAP@0.5 boost ($31.12\%$ to $46.61\%$) while maintaining real-time detection latency. Furthermore, we show that supplementing our pseudo-labelled data with a small fraction of ground truth labels ($10\%$) leads to further performance gains, reaching $57.97\%$ mAP@0.5 on the KITTI dataset. We observe similar performance improvements for the ACDC and BDD100k datasets.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.09955) | **Categories:** cs.CV

---

### [5] [AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](https://arxiv.org/abs/2511.10017)
*Xinyi Wang, Xun Yang, Yanlong Xu, Yuchen Wu, Zhen Li, Na Zhao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective human-agent collaboration in physical environments requires understanding not only what to act upon, but also where the actionable elements are and how to interact with them. Existing approaches often operate at the object level or disjointedly handle fine-grained affordance reasoning, lacking coherent, instruction-driven grounding and reasoning. In this work, we introduce a new task: Fine-grained 3D Embodied Reasoning, which requires an agent to predict, for each referenced affordance element in a 3D scene, a structured triplet comprising its spatial location, motion type, and motion axis, based on a task instruction. To solve this task, we propose AffordBot, a novel framework that integrates Multimodal Large Language Models (MLLMs) with a tailored chain-of-thought (CoT) reasoning paradigm. To bridge the gap between 3D input and 2D-compatible MLLMs, we render surround-view images of the scene and project 3D element candidates into these views, forming a rich visual representation aligned with the scene geometry. Our CoT pipeline begins with an active perception stage, prompting the MLLM to select the most informative viewpoint based on the instruction, before proceeding with step-by-step reasoning to localize affordance elements and infer plausible interaction motions. Evaluated on the SceneFun3D dataset, AffordBot achieves state-of-the-art performance, demonstrating strong generalization and physically grounded reasoning with only 3D point cloud input and MLLMs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10017) | **Categories:** cs.CV

---

### [6] [Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation](https://arxiv.org/abs/2511.10020)
*Yuxin Jiang, Wei Luo, Hui Zhang, Qiyu Chen, Haiming Yao, Weiming Shen, Yunkang Cao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose Anomagic, a zero-shot anomaly generation method that produces semantically coherent anomalies without requiring any exemplar anomalies. By unifying both visual and textual cues through a crossmodal prompt encoding scheme, Anomagic leverages rich contextual information to steer an inpainting-based generation pipeline. A subsequent contrastive refinement strategy enforces precise alignment between synthesized anomalies and their masks, thereby bolstering downstream anomaly detection accuracy. To facilitate training, we introduce AnomVerse, a collection of 12,987 anomaly-mask-caption triplets assembled from 13 publicly available datasets, where captions are automatically generated by multimodal large language models using structured visual prompts and template-based textual hints. Extensive experiments demonstrate that Anomagic trained on AnomVerse can synthesize more realistic and varied anomalies than prior methods, yielding superior improvements in downstream anomaly detection. Furthermore, Anomagic can generate anomalies for any normal-category image using user-defined prompts, establishing a versatile foundation model for anomaly generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10020) | **Categories:** cs.CV, cs.AI

---

### [7] [SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](https://arxiv.org/abs/2511.10091)
*Qilang Ye, Yu Zhou, Lian He, Jie Zhang, Xuanming Guo, Jiayu Zhang, Mingkui Tan, Weicheng Xie, Yue Sun, Tao Tan, Xiaochen Yuan, Ghada Khoriba, Zitong Yu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) hold rich implicit knowledge and powerful transferability. In this paper, we explore the combination of LLMs with the human skeleton to perform action classification and description. However, when treating LLM as a recognizer, two questions arise: 1) How can LLMs understand skeleton? 2) How can LLMs distinguish among actions? To address these problems, we introduce a novel paradigm named learning Skeleton representation with visUal-motion knowledGe for Action Recognition (SUGAR). In our pipeline, we first utilize off-the-shelf large-scale video models as a knowledge base to generate visual, motion information related to actions. Then, we propose to supervise skeleton learning through this prior knowledge to yield discrete representations. Finally, we use the LLM with untouched pre-training weights to understand these representations and generate the desired action targets and descriptions. Notably, we present a Temporal Query Projection (TQP) module to continuously model the skeleton signals with long sequences. Experiments on several skeleton-based action classification benchmarks demonstrate the efficacy of our SUGAR. Moreover, experiments on zero-shot scenarios show that SUGAR is more versatile than linear-based methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10091) | **Categories:** cs.CV

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

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Post-disaster situations pose unique navigation challenges. One of those challenges is the unstructured nature of the environment, which makes it hard to layout paths for rescue vehicles. We propose the use of Uncrewed Aerial Vehicle (UAV) in such scenario to perform reconnaissance across the environment. To accomplish this, we propose an optimization-based approach to plan a path for the UAV at optimal height where the sensors of the UAV can cover the most area and collect data with minimum uncertainty.

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
