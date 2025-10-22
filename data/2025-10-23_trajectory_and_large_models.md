# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-23

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (3)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [PlanU: Large Language Model Decision Making through Planning under Uncertainty](https://arxiv.org/abs/2510.18442)
*Ziwei Deng, Mian Deng, Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei, Cheng Wang, Siqi Shen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) are increasingly being explored across a range of decision-making tasks. However, LLMs sometimes struggle with decision-making tasks under uncertainty that are relatively easy for humans, such as planning actions in stochastic environments. The adoption of LLMs for decision-making is impeded by uncertainty challenges, such as LLM uncertainty and environmental uncertainty. LLM uncertainty arises from the stochastic sampling process inherent to LLMs. Most LLM-based Decision-Making (LDM) approaches address LLM uncertainty through multiple reasoning chains or search trees. However, these approaches overlook environmental uncertainty, which leads to poor performance in environments with stochastic state transitions. Some recent LDM approaches deal with uncertainty by forecasting the probability of unknown variables. However, they are not designed for multi-step decision-making tasks that require interaction with the environment. To address uncertainty in LLM decision-making, we introduce PlanU, an LLM-based planning method that captures uncertainty within Monte Carlo Tree Search (MCTS). PlanU models the return of each node in the MCTS as a quantile distribution, which uses a set of quantiles to represent the return distribution. To balance exploration and exploitation during tree search, PlanU introduces an Upper Confidence Bounds with Curiosity (UCC) score which estimates the uncertainty of MCTS nodes. Through extensive experiments, we demonstrate the effectiveness of PlanU in LLM-based decision-making tasks under uncertainty.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18442) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [HouseTour: A Virtual Real Estate A(I)gent](https://arxiv.org/abs/2510.18054)
*Ata Çelen, Marc Pollefeys, Daniel Barath, Iro Armeni*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce HouseTour, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18054) | **Categories:** cs.CV, cs.CL

---

### [2] [SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving](https://arxiv.org/abs/2510.18123)
*Xiangbo Gao, Tzu-Hsiang Lin, Ruojing Song, Yuheng Wu, Kuan-Ru Huang, Zicheng Jin, Fangzhou Lin, Shinan Liu, Zhengzhong Tu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Collaborative driving systems leverage vehicle-to-everything (V2X) communication across multiple agents to enhance driving safety and efficiency. Traditional V2X systems take raw sensor data, neural features, or perception results as communication media, which face persistent challenges, including high bandwidth demands, semantic loss, and interoperability issues. Recent advances investigate natural language as a promising medium, which can provide semantic richness, decision-level reasoning, and human-machine interoperability at significantly lower bandwidth. Despite great promise, this paradigm shift also introduces new vulnerabilities within language communication, including message loss, hallucinations, semantic manipulation, and adversarial attacks. In this work, we present the first systematic study of full-stack safety and security issues in natural-language-based collaborative driving. Specifically, we develop a comprehensive taxonomy of attack strategies, including connection disruption, relay/replay interference, content spoofing, and multi-connection forgery. To mitigate these risks, we introduce an agentic defense pipeline, which we call SafeCoop, that integrates a semantic firewall, language-perception consistency checks, and multi-source consensus, enabled by an agentic transformation function for cross-frame spatial alignment. We systematically evaluate SafeCoop in closed-loop CARLA simulation across 32 critical scenarios, achieving 69.15% driving score improvement under malicious attacks and up to 67.32% F1 score for malicious detection. This study provides guidance for advancing research on safe, secure, and trustworthy language-driven collaboration in transportation systems. Our project page is https://xiangbogaobarry.github.io/SafeCoop.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18123) | **Categories:** cs.CV, cs.AI, cs.CL, cs.RO

---

### [3] [SAVANT: Semantic Analysis with Vision-Augmented Anomaly deTection](https://arxiv.org/abs/2510.18034)
*Roberto Brusnicki, David Pop, Yuan Gao, Mattia Piccinini, Johannes Betz*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous driving systems remain critically vulnerable to the long-tail of rare, out-of-distribution scenarios with semantic anomalies. While Vision Language Models (VLMs) offer promising reasoning capabilities, naive prompting approaches yield unreliable performance and depend on expensive proprietary models, limiting practical deployment. We introduce SAVANT (Semantic Analysis with Vision-Augmented Anomaly deTection), a structured reasoning framework that achieves high accuracy and recall in detecting anomalous driving scenarios from input images through layered scene analysis and a two-phase pipeline: structured scene description extraction followed by multi-modal evaluation. Our approach transforms VLM reasoning from ad-hoc prompting to systematic analysis across four semantic layers: Street, Infrastructure, Movable Objects, and Environment. SAVANT achieves 89.6% recall and 88.0% accuracy on real-world driving scenarios, significantly outperforming unstructured baselines. More importantly, we demonstrate that our structured framework enables a fine-tuned 7B parameter open-source model (Qwen2.5VL) to achieve 90.8% recall and 93.8% accuracy - surpassing all models evaluated while enabling local deployment at near-zero cost. By automatically labeling over 9,640 real-world images with high accuracy, SAVANT addresses the critical data scarcity problem in anomaly detection and provides a practical path toward reliable, accessible semantic monitoring for autonomous systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18034) | **Categories:** cs.CV, cs.AI, cs.RO, I.2.9; I.4.8

---

### [4] [VelocityNet: Real-Time Crowd Anomaly Detection via Person-Specific Velocity Analysis](https://arxiv.org/abs/2510.18187)
*Fatima AlGhamdi, Omar Alharbi, Abdullah Aldwyish, Raied Aljadaany, Muhammad Kamran J Khan, Huda Alamri*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Detecting anomalies in crowded scenes is challenging due to severe inter-person occlusions and highly dynamic, context-dependent motion patterns. Existing approaches often struggle to adapt to varying crowd densities and lack interpretable anomaly indicators. To address these limitations, we introduce VelocityNet, a dual-pipeline framework that combines head detection and dense optical flow to extract person-specific velocities. Hierarchical clustering categorizes these velocities into semantic motion classes (halt, slow, normal, and fast), and a percentile-based anomaly scoring system measures deviations from learned normal patterns. Experiments demonstrate the effectiveness of our framework in real-time detection of diverse anomalous motion patterns within densely crowded environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18187) | **Categories:** cs.CV, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [SPACeR: Self-Play Anchoring with Centralized Reference Models](https://arxiv.org/abs/2510.18060)
*Wei-Jer Chang, Akshay Rangesh, Kevin Joseph, Matthew Strong, Masayoshi Tomizuka, Yihan Hu, Wei Zhan*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Developing autonomous vehicles (AVs) requires not only safety and efficiency, but also realistic, human-like behaviors that are socially aware and predictable. Achieving this requires sim agent policies that are human-like, fast, and scalable in multi-agent settings. Recent progress in imitation learning with large diffusion-based or tokenized models has shown that behaviors can be captured directly from human driving data, producing realistic policies. However, these models are computationally expensive, slow during inference, and struggle to adapt in reactive, closed-loop scenarios. In contrast, self-play reinforcement learning (RL) scales efficiently and naturally captures multi-agent interactions, but it often relies on heuristics and reward shaping, and the resulting policies can diverge from human norms. We propose SPACeR, a framework that leverages a pretrained tokenized autoregressive motion model as a centralized reference policy to guide decentralized self-play. The reference model provides likelihood rewards and KL divergence, anchoring policies to the human driving distribution while preserving RL scalability. Evaluated on the Waymo Sim Agents Challenge, our method achieves competitive performance with imitation-learned policies while being up to 10x faster at inference and 50x smaller in parameter size than large generative models. In addition, we demonstrate in closed-loop ego planning evaluation tasks that our sim agents can effectively measure planner quality with fast and scalable traffic simulation, establishing a new paradigm for testing autonomous driving policies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18060) | **Categories:** cs.LG, cs.AI, cs.RO, I.2.9; I.2.6

---

### [2] [Automated Algorithm Design for Auto-Tuning Optimizers](https://arxiv.org/abs/2510.17899)
*Floris-Jan Willemsen, Niki van Stein, Ben van Werkhoven*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Automatic performance tuning (auto-tuning) is essential for optimizing high-performance applications, where vast and irregular parameter spaces make manual exploration infeasible. Traditionally, auto-tuning relies on well-established optimization algorithms such as evolutionary algorithms, annealing methods, or surrogate model-based optimizers to efficiently find near-optimal configurations. However, designing effective optimizers remains challenging, as no single method performs best across all tuning tasks.   In this work, we explore a new paradigm: using large language models (LLMs) to automatically generate optimization algorithms tailored to auto-tuning problems. We introduce a framework that prompts LLMs with problem descriptions and search-space characteristics results to produce specialized optimization strategies, which are iteratively examined and improved.   These generated algorithms are evaluated on four real-world auto-tuning applications across six hardware platforms and compared against the state-of-the-art in optimization algorithms of two contemporary auto-tuning frameworks. The evaluation demonstrates that providing additional application- and search space-specific information in the generation stage results in an average performance improvement of 30.7\% and 14.6\%, respectively. In addition, our results show that LLM-generated optimizers can rival, and in various cases outperform, existing human-designed algorithms, with our best-performing generated optimization algorithms achieving, on average, 72.4\% improvement over state-of-the-art optimizers for auto-tuning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17899) | **Categories:** cs.LG, cs.AI, cs.NE

---


## 机器人学 (Robotics) [cs.RO]
### [1] [EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](https://arxiv.org/abs/2510.18546)
*Zebin Yang, Sunjian Zheng, Tong Xie, Tianshi Xu, Bo Yu, Fan Wang, Jie Tang, Shaoshan Liu, Meng Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Object-goal navigation (ObjNav) tasks an agent with navigating to the location of a specific object in an unseen environment. Embodied agents equipped with large language models (LLMs) and online constructed navigation maps can perform ObjNav in a zero-shot manner. However, existing agents heavily rely on giant LLMs on the cloud, e.g., GPT-4, while directly switching to small LLMs, e.g., LLaMA3.2-11b, suffer from significant success rate drops due to limited model capacity for understanding complex navigation maps, which prevents deploying ObjNav on local devices. At the same time, the long prompt introduced by the navigation map description will cause high planning latency on local devices. In this paper, we propose EfficientNav to enable on-device efficient LLM-based zero-shot ObjNav. To help the smaller LLMs better understand the environment, we propose semantics-aware memory retrieval to prune redundant information in navigation maps. To reduce planning latency, we propose discrete memory caching and attention-based memory clustering to efficiently save and re-use the KV cache. Extensive experimental results demonstrate that EfficientNav achieves 11.1% improvement in success rate on HM3D benchmark over GPT-4-based baselines, and demonstrates 6.7x real-time latency reduction and 4.7x end-to-end latency reduction over GPT-4 planner. Our code will be released soon.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18546) | **Categories:** cs.RO, cs.AI

---

### [2] [Studying the Effects of Robot Intervention on School Shooters in Virtual Reality](https://arxiv.org/abs/2510.17948)
*Christopher A McClurg, Alan R Wagner*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We advance the understanding of robotic intervention in high-risk scenarios by examining their potential to distract and impede a school shooter. To evaluate this concept, we conducted a virtual reality study with 150 university participants role-playing as a school shooter. Within the simulation, an autonomous robot predicted the shooter's movements and positioned itself strategically to interfere and distract. The strategy the robot used to approach the shooter was manipulated -- either moving directly in front of the shooter (aggressive) or maintaining distance (passive) -- and the distraction method, ranging from no additional cues (low), to siren and lights (medium), to siren, lights, and smoke to impair visibility (high). An aggressive, high-distraction robot reduced the number of victims by 46.6% relative to a no-robot control. This outcome underscores both the potential of robotic intervention to enhance safety and the pressing ethical questions surrounding their use in school environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.17948) | **Categories:** cs.RO, cs.AI, cs.CY, cs.HC

---

### [3] [Sharing the Load: Distributed Model-Predictive Control for Precise Multi-Rover Cargo Transport](https://arxiv.org/abs/2510.18766)
*Alexander Krawciw, Sven Lilge, Luka Antonyshyn, Timothy D. Barfoot*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: For autonomous cargo transportation, teams of mobile robots can provide more operational flexibility than a single large robot. In these scenarios, precision in both inter-vehicle distance and path tracking is key. With this motivation, we develop a distributed model-predictive controller (MPC) for multi-vehicle cargo operations that builds on the precise path-tracking of lidar teach and repeat. To carry cargo, a following vehicle must maintain a Euclidean distance offset from a lead vehicle regardless of the path curvature. Our approach uses a shared map to localize the robots relative to each other without GNSS or direct observations. We compare our approach to a centralized MPC and a baseline approach that directly measures the inter-vehicle distance. The distributed MPC shows equivalent nominal performance to the more complex centralized MPC. Using a direct measurement of the relative distance between the leader and follower shows improved tracking performance in close-range scenarios but struggles with long-range offsets. The operational flexibility provided by distributing the computation makes it well suited for real deployments. We evaluate four types of convoyed path trackers with over 10 km of driving in a coupled convoy. With convoys of two and three rovers, the proposed distributed MPC method works in real-time to allow map-based convoying to maintain maximum spacing within 20 cm of the target in various conditions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.18766) | **Categories:** cs.RO

---
