# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-27

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Building a Foundation Model for Trajectory from Scratch](https://arxiv.org/abs/2511.20610)
*Gaspard Merten, Mahmoud Sakr, Gilles Dejaegere*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundation models are transformative in artificial intelligence, but building them from scratch, especially for mobility trajectories, is not yet clear or documented. This tutorial bridges this gap by demonstrating the steps and code of a minimal implementation of a trajectory-focused foundation model starting from GPT-2. Through a concise, step-by-step, code-driven process, we demonstrate adapting GPT-2 for spatiotemporal data. We then review and compare representative trajectory foundation models, such as TrajFM and TrajGPT, highlighting their architectural innovations and differences. Additionally, we introduce complementary techniques from related domains, like TimesFM's patching approach. Targeted at researchers and practitioners, this tutorial aims to explain the concepts and terminology of foundation models, at the implementation level. We find it timely and indispensable to create this educational material in order to support the SIGSPATIAL community in building and evaluating mobility foundation models, enhancing both research clarity and peer-review effectiveness in mobility AI.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20610) | **Categories:** cs.AI

---

### [2] [Fara-7B: An Efficient Agentic Model for Computer Use](https://arxiv.org/abs/2511.19663)
*Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, Andrew Zhao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19663) | **Categories:** cs.AI, cs.CL, cs.CV

---

### [3] [Using Wearable Devices to Improve Chronic PainTreatment among Patients with Opioid Use Disorder](https://arxiv.org/abs/2511.19577)
*Abhay Goyal, Navin Kumar, Kimberly DiMeola, Rafael Trujillo, Soorya Ram Shimgekar, Christian Poellabauer, Pi Zonooz, Ermonda Gjoni-Markaj, Declan Barry, Lynn Madden*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Chronic pain (CP) and opioid use disorder (OUD) are common and interrelated chronic medical conditions. Currently, there is a paucity of evidence-based integrated treatments for CP and OUD among individuals receiving medication for opioid use disorder (MOUD). Wearable devices have the potential to monitor complex patient information and inform treatment development for persons with OUD and CP, including pain variability (e.g., exacerbations of pain or pain spikes) and clinical correlates (e.g., perceived stress). However, the application of large language models (LLMs) with wearable data for understanding pain spikes, remains unexplored. Consequently, the aim of this pilot study was to examine the clinical correlates of pain spikes using a range of AI approaches. We found that machine learning models achieved relatively high accuracy (>0.7) in predicting pain spikes, while LLMs were limited in providing insights on pain spikes. Real-time monitoring through wearable devices, combined with advanced AI models, could facilitate early detection of pain spikes and support personalized interventions that may help mitigate the risk of opioid relapse, improve adherence to MOUD, and enhance the integration of CP and OUD care. Given overall limited LLM performance, these findings highlight the need to develop LLMs which can provide actionable insights in the OUD/CP context.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19577) | **Categories:** cs.AI, cs.HC

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Reasoning-VLA: A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving](https://arxiv.org/abs/2511.19912)
*Dapeng Zhang, Zhenlong Yuan, Zhangquan Chen, Chih-Ting Liao, Yinda Chen, Fei Shen, Qingguo Zhou, Tat-Seng Chua*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models have recently shown strong decision-making capabilities in autonomous driving. However, existing VLAs often struggle with achieving efficient inference and generalizing to novel autonomous vehicle configurations and driving scenarios. In this paper, we propose Reasoning-VLA, a general and fast action-generation VLA framework. The proposed model employs a set of learnable action queries, initialized via Gaussian sampling from ground-truth trajectories within the training corpus. These learnable queries interact with reasoning-enhanced vision-language features to generate continuous action trajectories in parallel. To promote robust generalization, we consolidate eight publicly available autonomous driving datasets into a standardized, Chain-of-Thought reasoning-based, and easy-to-use data format for model training. Leveraging both supervised learning and reinforcement learning fine-tuning, extensive empirical evaluations across multiple benchmarks demonstrate that Reasoning-VLA achieves state-of-the-art performance, superior generalization capability, and the excellent inference speed reported to date.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19912) | **Categories:** cs.CV, cs.RO

---

### [2] [GigaWorld-0: World Models as Data Engine to Empower Embodied AI](https://arxiv.org/abs/2511.19861)
*GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, Qiuping Deng, Siting Wang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yankai Wang, Yu Cao, Yifan Chang, Yuan Xu, Yun Ye, Yang Wang, Yukun Zhou, Zhengyuan Zhang, Zhehao Dong, Zheng Zhu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19861) | **Categories:** cs.CV, cs.RO

---

### [3] [Map-World: Masked Action planning and Path-Integral World Model for Autonomous Driving](https://arxiv.org/abs/2511.20156)
*Bin Hu, Zijian Lu, Haicheng Liao, Chengran Yuan, Bin Rao, Yongkang Li, Guofa Li, Zhiyong Cui, Cheng-zhong Xu, Zhenning Li*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Motion planning for autonomous driving must handle multiple plausible futures while remaining computationally efficient. Recent end-to-end systems and world-model-based planners predict rich multi-modal trajectories, but typically rely on handcrafted anchors or reinforcement learning to select a single best mode for training and control. This selection discards information about alternative futures and complicates optimization. We propose MAP-World, a prior-free multi-modal planning framework that couples masked action planning with a path-weighted world model. The Masked Action Planning (MAP) module treats future ego motion as masked sequence completion: past waypoints are encoded as visible tokens, future waypoints are represented as mask tokens, and a driving-intent path provides a coarse scaffold. A compact latent planning state is expanded into multiple trajectory queries with injected noise, yielding diverse, temporally consistent modes without anchor libraries or teacher policies. A lightweight world model then rolls out future BEV semantics conditioned on each candidate trajectory. During training, semantic losses are computed as an expectation over modes, using trajectory probabilities as discrete path weights, so the planner learns from the full distribution of plausible futures instead of a single selected path. On NAVSIM, our method matches anchor-based approaches and achieves state-of-the-art performance among world-model-based methods, while avoiding reinforcement learning and maintaining real-time inference latency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20156) | **Categories:** cs.CV, cs.RO

---

### [4] [BRIC: Bridging Kinematic Plans and Physical Control at Test Time](https://arxiv.org/abs/2511.20431)
*Dohun Lim, Minji Kim, Jaewoon Lim, Sungchan Kim*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose BRIC, a novel test-time adaptation (TTA) framework that enables long-term human motion generation by resolving execution discrepancies between diffusion-based kinematic motion planners and reinforcement learning-based physics controllers. While diffusion models can generate diverse and expressive motions conditioned on text and scene context, they often produce physically implausible outputs, leading to execution drift during simulation. To address this, BRIC dynamically adapts the physics controller to noisy motion plans at test time, while preserving pre-trained skills via a loss function that mitigates catastrophic forgetting. In addition, BRIC introduces a lightweight test-time guidance mechanism that steers the diffusion model in the signal space without updating its parameters. By combining both adaptation strategies, BRIC ensures consistent and physically plausible long-term executions across diverse environments in an effective and efficient manner. We validate the effectiveness of BRIC on a variety of long-term tasks, including motion composition, obstacle avoidance, and human-scene interaction, achieving state-of-the-art performance across all tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20431) | **Categories:** cs.CV, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Hidden markov model to predict tourists visited place](https://arxiv.org/abs/2511.19465)
*Theo Demessance, Chongke Bi, Sonia Djebali, Guillaume Guerard*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Nowadays, social networks are becoming a popular way of analyzing tourist behavior, thanks to the digital traces left by travelers during their stays on these networks. The massive amount of data generated; by the propensity of tourists to share comments and photos during their trip; makes it possible to model their journeys and analyze their behavior. Predicting the next movement of tourists plays a key role in tourism marketing to understand demand and improve decision support. In this paper, we propose a method to understand and to learn tourists' movements based on social network data analysis to predict future movements. The method relies on a machine learning grammatical inference algorithm. A major contribution in this paper is to adapt the grammatical inference algorithm to the context of big data. Our method produces a hidden Markov model representing the movements of a group of tourists. The hidden Markov model is flexible and editable with new data. The capital city of France, Paris is selected to demonstrate the efficiency of the proposed methodology.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19465) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Unifying Perception and Action: A Hybrid-Modality Pipeline with Implicit Visual Chain-of-Thought for Robotic Action Generation](https://arxiv.org/abs/2511.19859)
*Xiangkai Ma, Lekai Xing, Han Zhang, Wenzhong Li, Sanglu Lu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models built upon Chain-of-Thought (CoT) have achieved remarkable success in advancing general-purpose robotic agents, owing to its significant perceptual comprehension. Recently, since text-only CoT struggles to adequately capture scene details in complex spatial environments, a highly promising strategy involves leveraging visual priors to guide robotic action generation. Nevertheless, these strategies face two inherent challenges: (i) a modality gap between visual observations and low-level actions, and (ii) unstable training due to competing objectives between visual prediction and action generation. To address these challenges, we propose a Vision-Integrated Trajectory Alignment (VITA) framework that learns a shared discrete latent space for vision and action, enabling joint modeling of perception and motor control. VITA introduces a implicit visual CoT: autoregressively generated tokens is simultaneously decoded into future frames predictions and robot actions, thereby internalizing visual dynamics as an inductive bias for motion planning. Extensive experiments on simulated and real-world environments demonstrate state-of-the-art performance. VITA improves 14.5\%, 9.6\% and 12.1\% over existing baselines on CALVIN, LIBERO and SimplerEnv. Furthermore, VITA attains an average success rate of 80.5\% across six real-world tasks, demonstrating its potential as a generalist robotic manipulation model.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19859) | **Categories:** cs.RO

---

### [2] [CoC-VLA: Delving into Adversarial Domain Transfer for Explainable Autonomous Driving via Chain-of-Causality Visual-Language-Action Model](https://arxiv.org/abs/2511.19914)
*Dapeng Zhang, Fei Shen, Rui Zhao, Yinda Chen, Peng Zhi, Chenyang Li, Rui Zhou, Qingguo Zhou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous driving represents a prominent application of artificial intelligence. Recent approaches have shifted from focusing solely on common scenarios to addressing complex, long-tail situations such as subtle human behaviors, traffic accidents, and non-compliant driving patterns. Given the demonstrated capabilities of large language models (LLMs) in understanding visual and natural language inputs and following instructions, recent methods have integrated LLMs into autonomous driving systems to enhance reasoning, interpretability, and performance across diverse scenarios. However, existing methods typically rely either on real-world data, which is suitable for industrial deployment, or on simulation data tailored to rare or hard case scenarios. Few approaches effectively integrate the complementary advantages of both data sources. To address this limitation, we propose a novel VLM-guided, end-to-end adversarial transfer framework for autonomous driving that transfers long-tail handling capabilities from simulation to real-world deployment, named CoC-VLA. The framework comprises a teacher VLM model, a student VLM model, and a discriminator. Both the teacher and student VLM models utilize a shared base architecture, termed the Chain-of-Causality Visual-Language Model (CoC VLM), which integrates temporal information via an end-to-end text adapter. This architecture supports chain-of-thought reasoning to infer complex driving logic. The teacher and student VLM models are pre-trained separately on simulated and real-world datasets. The discriminator is trained adversarially to facilitate the transfer of long-tail handling capabilities from simulated to real-world environments by the student VLM model, using a novel backpropagation strategy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19914) | **Categories:** cs.RO

---

### [3] [Discover, Learn, and Reinforce: Scaling Vision-Language-Action Pretraining with Diverse RL-Generated Trajectories](https://arxiv.org/abs/2511.19528)
*Rushuai Yang, Zhiyuan Feng, Tianxiang Zhang, Kaixin Wang, Chuheng Zhang, Li Zhao, Xiu Su, Yi Chen, Jiang Bian*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Scaling vision-language-action (VLA) model pre-training requires large volumes of diverse, high-quality manipulation trajectories. Most current data is obtained via human teleoperation, which is expensive and difficult to scale. Reinforcement learning (RL) methods learn useful skills through autonomous exploration, making them a viable approach for generating data. However, standard RL training collapses to a narrow execution pattern, limiting its utility for large-scale pre-training. We propose Discover, Lea rn and Reinforce (DLR), an information-theoretic pattern discovery framework that generates multiple distinct, high-success behavioral patterns for VLA pretraining. Empirically, DLR generates a markedly more diverse trajectory corpus on LIBERO. Specifically, it learns multiple distinct, high-success strategies for the same task where standard RL discovers only one, and hence it covers substantially broader regions of the state-action space. When adapted to unseen downstream task suites, VLA models pretrained on our diverse RL data surpass counterparts trained on equal-sized standard RL datasets. Moreover, DLR exhibits positive data-scaling behavior that single-pattern RL lacks. These results position multi-pattern RL as a practical, scalable data engine for embodied foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19528) | **Categories:** cs.RO, cs.AI

---

### [4] [Flow-Based Path Planning for Multiple Homogenous UAVs for Outdoor Formation-Flying](https://arxiv.org/abs/2511.19653)
*Mahmud Suhaimi Ibrahim, Shantanu Rahman, Muhammad Samin Hasan, Minhaj Uddin Ahmad, Abdullah Abrar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Collision-free path planning is the most crucial component in multi-UAV formation-flying (MFF). We use unlabeled homogenous quadcopters (UAVs) to demonstrate the use of a flow network to create complete (inter-UAV) collision-free paths. This procedure has three main parts: 1) Creating a flow network graph from physical GPS coordinates, 2) Finding a path of minimum cost (least distance) using any graph-based path-finding algorithm, and 3) Implementing the Ford-Fulkerson Method to find the paths with the maximum flow (no collision). Simulations of up to 64 UAVs were conducted for various formations, followed by a practical experiment with 3 quadcopters for testing physical plausibility and feasibility. The results of these tests show the efficacy of this method's ability to produce safe, collision-free paths.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19653) | **Categories:** cs.RO

---

### [5] [Development of a Testbed for Autonomous Vehicles: Integrating MPC Control with Monocular Camera Lane Detection](https://arxiv.org/abs/2511.19655)
*Shantanu Rahman, Nayeb Hasin, Mainul Islam*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous vehicles are becoming popular day by day not only for autonomous road traversal but also for industrial automation, farming and military. Most of the standard vehicles follow the Ackermann style steering mechanism. This has become to de facto standard for large and long faring vehicles. The local planner of an autonomous vehicle controls the low-level vehicle movement upon which the vehicle will perform its motor actuation. In our work, we focus on autonomous vehicles in road and perform experiments to analyze the effect of low-level controllers in the simulation and a real environment. To increase the precision and stability of trajectory tracking in autonomous cars, a novel method that combines lane identification with Model Predictive Control (MPC) is presented. The research focuses on camera-equipped autonomous vehicles and uses methods like edge recognition, sliding window-based straight-line identification for lane line extraction, and dynamic region of interest (ROI) extraction. Next, to follow the identified lane line, an MPC built on a bicycle vehicle dynamics model is created. A single-lane road simulation model is built using ROS Gazebo and tested in order to verify the controller's performance. The root mean square error between the optimal tracking trajectory and the target trajectory was reduced by 27.65% in the simulation results, demonstrating the high robustness and flexibility of the developed controller.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19655) | **Categories:** cs.RO

---

### [6] [Multi-Agent gatekeeper: Safe Flight Planning and Formation Control for Urban Air Mobility](https://arxiv.org/abs/2511.19691)
*Thomas Marshall Vielmetti, Devansh R Agrawal, Dimitra Panagou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present Multi-Agent gatekeeper, a framework that provides provable safety guarantees for leader-follower formation control in cluttered 3D environments. Existing methods face a trad-off: online planners and controllers lack formal safety guarantees, while offline planners lack adaptability to changes in the number of agents or desired formation. To address this gap, we propose a hybrid architecture where a single leader tracks a pre-computed, safe trajectory, which serves as a shared trajectory backup set for all follower agents. Followers execute a nominal formation-keeping tracking controller, and are guaranteed to remain safe by always possessing a known-safe backup maneuver along the leader's path. We formally prove this method ensures collision avoidance with both static obstacles and other agents. The primary contributions are: (1) the multi-agent gatekeeper algorithm, which extends our single-agent gatekeeper framework to multi-agent systems; (2) the trajectory backup set for provably safe inter-agent coordination for leader-follower formation control; and (3) the first application of the gatekeeper framework in a 3D environment. We demonstrate our approach in a simulated 3D urban environment, where it achieved a 100% collision-avoidance success rate across 100 randomized trials, significantly outperforming baseline CBF and NMPC methods. Finally, we demonstrate the physical feasibility of the resulting trajectories on a team of quadcopters.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19691) | **Categories:** cs.RO

---

### [7] [Hibikino-Musashi@Home 2025 Team Description Paper](https://arxiv.org/abs/2511.20180)
*Ryohei Kobayashi, Kosei Isomoto, Kosei Yamao, Soma Fumoto, Koshun Arimura, Naoki Yamaguchi, Akinobu Mizutani, Tomoya Shiba, Kouki Kimizuka, Yuta Ohno, Ryo Terashima, Hiromasa Yamaguchi, Tomoaki Fujino, Ryoga Maruno, Wataru Yoshimura, Kazuhito Mine, Tang Phu Thien Nhan, Yuga Yano, Yuichiro Tanaka, Takeshi Nishida, Takashi Morie, Hakaru Tamukoh*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper provides an overview of the techniques employed by Hibikino-Musashi@Home, which intends to participate in the domestic standard platform league. The team developed a dataset generator for training a robot vision system and an open-source development environment running on a Human Support Robot simulator. The large-language-model-powered task planner selects appropriate primitive skills to perform the task requested by the user. Moreover, the team has focused on research involving brain-inspired memory models for adaptation to individual home environments. This approach aims to provide intuitive and personalized assistance. Additionally, the team contributed to the reusability of the navigation system developed by Pumas in RoboCup2024. The team aimed to design a home service robot to assist humans in their homes and continuously attend competitions to evaluate and improve the developed system.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20180) | **Categories:** cs.RO

---

### [8] [Improved adaptive wind driven optimization algorithm for real-time path planning](https://arxiv.org/abs/2511.20394)
*Shiqian Liu, Azlan Mohd Zain, Le-le Mao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently, path planning has achieved remarkable progress in enhancing global search capability and convergence accuracy through heuristic and learning-inspired optimization frameworks. However, real-time adaptability in dynamic environments remains a critical challenge for autonomous navigation, particularly when robots must generate collision-free, smooth, and efficient trajectories under complex constraints. By analyzing the difficulties in dynamic path planning, the Wind Driven Optimization (WDO) algorithm emerges as a promising framework owing to its physically interpretable search dynamics. Motivated by these observations, this work revisits the WDO principle and proposes a variant formulation, Multi-hierarchical adaptive wind driven optimization(MAWDO), that improves adaptability and robustness in time-varying environments. To mitigate instability and premature convergence, a hierarchical-guidance mechanism divides the population into multiple groups guided by individual, regional, and global leaders to balance exploration and exploitation. Extensive evaluations on sixteen benchmark functions show that MAWDO achieves superior optimization accuracy, convergence stability, and adaptability over state-of-the art metaheuristics. In dynamic path planning, MAWDO shortens the path length to 469.28 pixels, improving over Multi-strategy ensemble wind driven optimization(MEWDO), Adaptive wind driven optimization(AWDO) and WDO by 3.51\%, 11.63\% and 14.93\%, and achieves the smallest optimality gap (1.01) with smoothness 0.71 versus 13.50 and 15.67 for AWDO and WDO, leading to smoother, shorter, and collision-free trajectories that confirm its effectiveness for real-time path planning in complex environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20394) | **Categories:** cs.RO

---

### [9] [Safe and Stable Neural Network Dynamical Systems for Robot Motion Planning](https://arxiv.org/abs/2511.20593)
*Allen Emmanuel Binny, Mahathi Anand, Hugo T. M. Kussaba, Lingyun Chen, Shreenabh Agrawal, Fares J. Abu-Dakka, Abdalla Swikir*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning safe and stable robot motions from demonstrations remains a challenge, especially in complex, nonlinear tasks involving dynamic, obstacle-rich environments. In this paper, we propose Safe and Stable Neural Network Dynamical Systems S$^2$-NNDS, a learning-from-demonstration framework that simultaneously learns expressive neural dynamical systems alongside neural Lyapunov stability and barrier safety certificates. Unlike traditional approaches with restrictive polynomial parameterizations, S$^2$-NNDS leverages neural networks to capture complex robot motions providing probabilistic guarantees through split conformal prediction in learned certificates. Experimental results on various 2D and 3D datasets -- including LASA handwriting and demonstrations recorded kinesthetically from the Franka Emika Panda robot -- validate S$^2$-NNDS effectiveness in learning robust, safe, and stable motions from potentially unsafe demonstrations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20593) | **Categories:** cs.RO, eess.SY

---

### [10] [Reinforcing Action Policies by Prophesying](https://arxiv.org/abs/2511.20633)
*Jiahui Zhang, Ze Huang, Chun Gu, Zipei Ma, Li Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) policies excel in aligning language, perception, and robot control. However, most VLAs are trained purely by imitation, which overfits to demonstrations, and is brittle under distribution shift. Reinforcement learning (RL) directly optimizes task reward and thus addresses this misalignment, but real-robot interaction is expensive and conventional simulators are hard to engineer and transfer. We address both data efficiency and optimization stability in VLA post-training via a learned world model and an RL procedure tailored to flow-based action heads. Specifically, we introduce Prophet, a unified action-to-video robot actuation pretrained across large-scale, heterogeneous robot data to learn reusable action-outcome dynamics. It is able to few-shot adapt to new robots, objects, and environments, yielding a rollout-ready simulator. Upon Prophet, we reinforce action policies with Flow-action-GRPO (FA-GRPO), which adapts Flow-GRPO to operate on VLA actions, and with FlowScale, a stepwise reweighting that rescales per-step gradients in the flow head. Together, Prophet, FA-GRPO, and FlowScale constitute ProphRL, a practical, data- and compute-efficient path to VLA post-training. Experiments show 5-17% success gains on public benchmarks and 24-30% gains on real robots across different VLA variants.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20633) | **Categories:** cs.RO

---
