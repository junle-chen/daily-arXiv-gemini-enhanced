# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-01

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](https://arxiv.org/abs/2509.23589)
*Shu Liu, Wenlin Chen, Weihao Li, Zheng Wang, Lijin Yang, Jianing Huang, Yipin Zhang, Zhongzhan Huang, Ze Cheng, Hao Yang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion-based planners have shown great promise for autonomous driving due to their ability to capture multi-modal driving behaviors. However, guiding these models effectively in reactive, closed-loop environments remains a significant challenge. Simple conditioning often fails to provide sufficient guidance in complex and dynamic driving scenarios. Recent work attempts to use typical expert driving behaviors (i.e., anchors) to guide diffusion models but relies on a truncated schedule, which introduces theoretical inconsistencies and can compromise performance. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach provides a principled diffusion framework that effectively translates anchors into fine-grained trajectory plans, appropriately responding to varying traffic conditions. Our planner is compatible with efficient ODE solvers, a critical factor for real-time autonomous driving deployment. We achieve state-of-the-art performance on the Bench2Drive benchmark, improving the success rate by 5% over prior arts.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23589) | **Categories:** cs.AI, cs.CV, cs.LG

---

### [2] [Multiplayer Nash Preference Optimization](https://arxiv.org/abs/2509.23102)
*Fang Wu, Xu Huang, Weihao Xuan, Zhiwei Zhang, Yijia Xiao, Guancheng Wan, Xiaomin Li, Bing Hu, Peng Xia, Jure Leskovec, Yejin Choi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning from human feedback (RLHF) has emerged as the standard paradigm for aligning large language models (LLMs) with human preferences. However, reward-based methods built on the Bradley-Terry assumption struggle to capture the non-transitive and heterogeneous nature of real-world preferences. To address this, recent studies have reframed alignment as a two-player Nash game, giving rise to Nash learning from human feedback (NLHF). While this perspective has inspired algorithms such as INPO, ONPO, and EGPO with strong theoretical and empirical guarantees, they remain fundamentally restricted to two-player interactions, creating a single-opponent bias that fails to capture the full complexity of realistic preference structures. In this work, we introduce Multiplayer Nash Preference Optimization (MNPO), a novel framework that generalizes NLHF to the multiplayer regime. It formulates alignment as an $n$-player game, where each policy competes against a population of opponents while being regularized toward a reference model. Our framework establishes well-defined Nash equilibria in multiplayer settings and extends the concept of duality gap to quantify approximation quality. We demonstrate that MNPO inherits the equilibrium guarantees of two-player methods while enabling richer competitive dynamics and improved coverage of diverse preference structures. Through comprehensive empirical evaluation, we show that MNPO consistently outperforms existing NLHF baselines on instruction-following benchmarks, achieving superior alignment quality under heterogeneous annotator conditions and mixed-policy evaluation scenarios. Together, these results establish MNPO as a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences. Code is available at https://github.com/smiles724/MNPO.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23102) | **Categories:** cs.AI, cs.CL

---

### [3] [ViTSP: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems](https://arxiv.org/abs/2509.23465)
*Zhuoli Yin, Yi Ding, Reem Khir, Hua Cai*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Solving Traveling Salesman Problem (TSP) is NP-hard yet fundamental for wide real-world applications. Classical exact methods face challenges in scaling, and heuristic methods often require domain-specific parameter calibration. While learning-based approaches have shown promise, they suffer from poor generalization and limited scalability due to fixed training data. This work proposes ViTSP, a novel framework that leverages pre-trained vision language models (VLMs) to visually guide the solution process for large-scale TSPs. The VLMs function to identify promising small-scale subproblems from a visualized TSP instance, which are then efficiently optimized using an off-the-shelf solver to improve the global solution. ViTSP bypasses the dedicated model training at the user end while maintaining effectiveness across diverse instances. Experiments on real-world TSP instances ranging from 1k to 88k nodes demonstrate that ViTSP consistently achieves solutions with average optimality gaps below 0.2%, outperforming existing learning-based methods. Under the same runtime budget, it surpasses the best-performing heuristic solver, LKH-3, by reducing its gaps by 12% to 100%, particularly on very-large-scale instances with more than 10k nodes. Our framework offers a new perspective in hybridizing pre-trained generative models and operations research solvers in solving combinatorial optimization problems, with practical implications for integration into more complex logistics systems. The code is available at https://anonymous.4open.science/r/ViTSP_codes-6683.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23465) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [ARSS: Taming Decoder-only Autoregressive Visual Generation for View Synthesis From Single View](https://arxiv.org/abs/2509.23008)
*Wenbin Teng, Gonglin Chen, Haiwei Chen, Yajie Zhao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite their exceptional generative quality, diffusion models have limited applicability to world modeling tasks, such as novel view generation from sparse inputs. This limitation arises because diffusion models generate outputs in a non-causal manner, often leading to distortions or inconsistencies across views, and making it difficult to incrementally adapt accumulated knowledge to new queries. In contrast, autoregressive (AR) models operate in a causal fashion, generating each token based on all previously generated tokens. In this work, we introduce \textbf{ARSS}, a novel framework that leverages a GPT-style decoder-only AR model to generate novel views from a single image, conditioned on a predefined camera trajectory. We employ a video tokenizer to map continuous image sequences into discrete tokens and propose a camera encoder that converts camera trajectories into 3D positional guidance. Then to enhance generation quality while preserving the autoregressive structure, we propose a autoregressive transformer module that randomly permutes the spatial order of tokens while maintaining their temporal order. Extensive qualitative and quantitative experiments on public datasets demonstrate that our method performs comparably to, or better than, state-of-the-art view synthesis approaches based on diffusion models. Our code will be released upon paper acceptance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23008) | **Categories:** cs.CV

---

### [2] [Planning with Unified Multimodal Models](https://arxiv.org/abs/2509.23014)
*Yihao Sun, Zhilong Zhang, Yang Yu, Pierre-Luc Bacon*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the powerful reasoning capabilities of large language models (LLMs) and vision-language models (VLMs), many recent works have explored using them for decision-making. However, most of these approaches rely solely on language-based reasoning, which limits their ability to reason and make informed decisions. Recently, a promising new direction has emerged with unified multimodal models (UMMs), which support both multimodal inputs and outputs. We believe such models have greater potential for decision-making by enabling reasoning through generated visual content. To this end, we propose Uni-Plan, a planning framework built on UMMs. Within this framework, a single model simultaneously serves as the policy, dynamics model, and value function. In addition, to avoid hallucinations in dynamics predictions, we present a novel approach self-discriminated filtering, where the generative model serves as a self-discriminator to filter out invalid dynamics predictions. Experiments on long-horizon planning tasks show that Uni-Plan substantially improves success rates compared to VLM-based methods, while also showing strong data scalability, requiring no expert demonstrations and achieving better performance under the same training-data size. This work lays a foundation for future research in reasoning and decision-making with UMMs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23014) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Physically Plausible Multi-System Trajectory Generation and Symmetry Discovery](https://arxiv.org/abs/2509.23003)
*Jiayin Liu, Yulong Yang, Vineet Bansal, Christine Allen-Blanchette*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: From metronomes to celestial bodies, mechanics underpins how the world evolves in time and space. With consideration of this, a number of recent neural network models leverage inductive biases from classical mechanics to encourage model interpretability and ensure forecasted states are physical. However, in general, these models are designed to capture the dynamics of a single system with fixed physical parameters, from state-space measurements of a known configuration space. In this paper we introduce Symplectic Phase Space GAN (SPS-GAN) which can capture the dynamics of multiple systems, and generalize to unseen physical parameters from. Moreover, SPS-GAN does not require prior knowledge of the system configuration space. In fact, SPS-GAN can discover the configuration space structure of the system from arbitrary measurement types (e.g., state-space measurements, video frames). To achieve physically plausible generation, we introduce a novel architecture which embeds a Hamiltonian neural network recurrent module in a conditional GAN backbone. To discover the structure of the configuration space, we optimize the conditional time-series GAN objective with an additional physically motivated term to encourages a sparse representation of the configuration space. We demonstrate the utility of SPS-GAN for trajectory prediction, video generation and symmetry discovery. Our approach captures multiple systems and achieves performance on par with supervised models designed for single systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23003) | **Categories:** cs.LG, cs.AI, cs.SY, eess.SY

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Open-Vocabulary Spatio-Temporal Scene Graph for Robot Perception and Teleoperation Planning](https://arxiv.org/abs/2509.23107)
*Yi Wang, Zeyu Xue, Mujie Liu, Tongqin Zhang, Yan Hu, Zhou Zhao, Chenguang Yang, Zhenyu Lu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Teleoperation via natural-language reduces operator workload and enhances safety in high-risk or remote settings. However, in dynamic remote scenes, transmission latency during bidirectional communication creates gaps between remote perceived states and operator intent, leading to command misunderstanding and incorrect execution. To mitigate this, we introduce the Spatio-Temporal Open-Vocabulary Scene Graph (ST-OVSG), a representation that enriches open-vocabulary perception with temporal dynamics and lightweight latency annotations. ST-OVSG leverages LVLMs to construct open-vocabulary 3D object representations, and extends them into the temporal domain via Hungarian assignment with our temporal matching cost, yielding a unified spatio-temporal scene graph. A latency tag is embedded to enable LVLM planners to retrospectively query past scene states, thereby resolving local-remote state mismatches caused by transmission delays. To further reduce redundancy and highlight task-relevant cues, we propose a task-oriented subgraph filtering strategy that produces compact inputs for the planner. ST-OVSG generalizes to novel categories and enhances planning robustness against transmission latency without requiring fine-tuning. Experiments show that our method achieves 74 percent node accuracy on the Replica benchmark, outperforming ConceptGraph. Notably, in the latency-robustness experiment, the LVLM planner assisted by ST-OVSG achieved a planning success rate of 70.5 percent.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23107) | **Categories:** cs.RO, cs.AI

---

### [2] [Ask, Reason, Assist: Decentralized Robot Collaboration via Language and Logic](https://arxiv.org/abs/2509.23506)
*Dan BW Choe, Sundhar Vinodh Sangeetha, Steven Emanuel, Chih-Yuan Chiu, Samuel Coogan, Shreyas Kousik*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Increased robot deployment, such as in warehousing, has revealed a need for seamless collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To address this challenge, we propose a novel decentralized framework that enables robots to request and provide help. The process begins when a robot detects a conflict and uses a Large Language Model (LLM) to decide whether external assistance is required. If so, it crafts and broadcasts a natural language (NL) help request. Potential helper robots reason over the request and respond with offers of assistance, including information about the effect on their ongoing tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar, ensuring syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot selects a helper by reasoning over the expected increase in system-level total task completion time. We evaluated our framework through experiments comparing different helper-selection strategies and found that considering multiple offers allows the requester to minimize added makespan. Our approach significantly outperforms heuristics such as selecting the nearest available candidate helper robot, and achieves performance comparable to a centralized "Oracle" baseline but without heavy information demands.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23506) | **Categories:** cs.RO

---

### [3] [ReSeFlow: Rectifying SE(3)-Equivariant Policy Learning Flows](https://arxiv.org/abs/2509.22695)
*Zhitao Wang, Yanke Wang, Jiangtao Wen, Roberto Horowitz, Yuxing Han*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic manipulation in unstructured environments requires the generation of robust and long-horizon trajectory-level policy with conditions of perceptual observations and benefits from the advantages of SE(3)-equivariant diffusion models that are data-efficient. However, these models suffer from the inference time costs. Inspired by the inference efficiency of rectified flows, we introduce the rectification to the SE(3)-diffusion models and propose the ReSeFlow, i.e., Rectifying SE(3)-Equivariant Policy Learning Flows, providing fast, geodesic-consistent, least-computational policy generation. Crucially, both components employ SE(3)-equivariant networks to preserve rotational and translational symmetry, enabling robust generalization under rigid-body motions. With the verification on the simulated benchmarks, we find that the proposed ReSeFlow with only one inference step can achieve better performance with lower geodesic distance than the baseline methods, achieving up to a 48.5% error reduction on the painting task and a 21.9% reduction on the rotating triangle task compared to the baseline's 100-step inference. This method takes advantages of both SE(3) equivariance and rectified flow and puts it forward for the real-world application of generative policy learning models with the data and inference efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22695) | **Categories:** cs.RO, cs.CV

---

### [4] [Self-driving cars: Are we there yet?](https://arxiv.org/abs/2509.22754)
*Merve Atasever, Zhuochen Liu, Qingpei Li, Akshay Hitendra Shah, Hans Walker, Jyotirmoy V. Deshmukh, Rahul Jain*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous driving remains a highly active research domain that seeks to enable vehicles to perceive dynamic environments, predict the future trajectories of traffic agents such as vehicles, pedestrians, and cyclists and plan safe and efficient future motions. To advance the field, several competitive platforms and benchmarks have been established to provide standardized datasets and evaluation protocols. Among these, leaderboards by the CARLA organization and nuPlan and the Waymo Open Dataset have become leading benchmarks for assessing motion planning algorithms. Each offers a unique dataset and challenging planning problems spanning a wide range of driving scenarios and conditions. In this study, we present a comprehensive comparative analysis of the motion planning methods featured on these three leaderboards. To ensure a fair and unified evaluation, we adopt CARLA leaderboard v2.0 as our common evaluation platform and modify the selected models for compatibility. By highlighting the strengths and weaknesses of current approaches, we identify prevailing trends, common challenges, and suggest potential directions for advancing motion planning research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22754) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [5] [DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes](https://arxiv.org/abs/2509.22937)
*Trent Weiss, Amar Kulkarni, Madhur Behl*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A significant challenge in autonomous racing is to generate overtaking maneuvers. Racing agents must execute these maneuvers on complex racetracks with little room for error. Optimization techniques and graph-based methods have been proposed, but these methods often rely on oversimplified assumptions for collision-avoidance and dynamic constraints. In this work, we present an approach to trajectory synthesis based on an extension of the Differential Bayesian Filtering framework. Our approach for collision-free trajectory synthesis frames the problem as one of Bayesian Inference over the space of Composite Bezier Curves. Our method is derivative-free, does not require a spherical approximation of the vehicle footprint, linearization of constraints, or simplifying upper bounds on collision avoidance. We conduct a closed-loop analysis of DBF-MA and find it successfully overtakes an opponent in 87% of tested scenarios, outperforming existing methods in autonomous overtaking.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22937) | **Categories:** cs.RO

---

### [6] [LAGEA: Language Guided Embodied Agents for Robotic Manipulation](https://arxiv.org/abs/2509.23155)
*Abdul Monaf Chowdhury, Akm Moshiur Rahman Mazumder, Rabeya Akter, Safaeid Hossain Arib*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic manipulation benefits from foundation models that describe goals, but today's agents still lack a principled way to learn from their own mistakes. We ask whether natural language can serve as feedback, an error reasoning signal that helps embodied agents diagnose what went wrong and correct course. We introduce LAGEA (Language Guided Embodied Agents), a framework that turns episodic, schema-constrained reflections from a vision language model (VLM) into temporally grounded guidance for reinforcement learning. LAGEA summarizes each attempt in concise language, localizes the decisive moments in the trajectory, aligns feedback with visual state in a shared representation, and converts goal progress and feedback agreement into bounded, step-wise shaping rewardswhose influence is modulated by an adaptive, failure-aware coefficient. This design yields dense signals early when exploration needs direction and gracefully recedes as competence grows. On the Meta-World MT10 embodied manipulation benchmark, LAGEA improves average success over the state-of-the-art (SOTA) methods by 9.0% on random goals and 5.3% on fixed goals, while converging faster. These results support our hypothesis: language, when structured and grounded in time, is an effective mechanism for teaching robots to self-reflect on mistakes and make better choices. Code will be released soon.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23155) | **Categories:** cs.RO

---

### [7] [Leave No Observation Behind: Real-time Correction for VLA Action Chunks](https://arxiv.org/abs/2509.23224)
*Kohei Sendai, Maxime Alvarez, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA's action chunk. The module combines the latest observation, the predicted action from VLA (base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model's competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic Kinetix task suite (12 tasks) and LIBERO Spatial, our method yields consistent success rate improvements across increasing delays and execution horizons (+23% point and +7% point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23224) | **Categories:** cs.RO, cs.AI, cs.CV, cs.SY, eess.SY

---

### [8] [Preventing Robotic Jailbreaking via Multimodal Domain Adaptation](https://arxiv.org/abs/2509.23281)
*Francesco Marchiori, Rohan Sinha, Christopher Agia, Alexander Robey, George J. Pappas, Mauro Conti, Marco Pavone*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) and Vision-Language Models (VLMs) are increasingly deployed in robotic environments but remain vulnerable to jailbreaking attacks that bypass safety mechanisms and drive unsafe or physically harmful behaviors in the real world. Data-driven defenses such as jailbreak classifiers show promise, yet they struggle to generalize in domains where specialized datasets are scarce, limiting their effectiveness in robotics and other safety-critical contexts. To address this gap, we introduce J-DAPT, a lightweight framework for multimodal jailbreak detection through attention-based fusion and domain adaptation. J-DAPT integrates textual and visual embeddings to capture both semantic intent and environmental grounding, while aligning general-purpose jailbreak datasets with domain-specific reference data. Evaluations across autonomous driving, maritime robotics, and quadruped navigation show that J-DAPT boosts detection accuracy to nearly 100% with minimal overhead. These results demonstrate that J-DAPT provides a practical defense for securing VLMs in robotic applications. Additional materials are made available at: https://j-dapt.github.io.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23281) | **Categories:** cs.RO, I.2.6; I.2.9

---

### [9] [RAVEN: Resilient Aerial Navigation via Open-Set Semantic Memory and Behavior Adaptation](https://arxiv.org/abs/2509.23563)
*Seungchan Kim, Omar Alama, Dmytro Kurdydyk, John Keller, Nikhil Keetha, Wenshan Wang, Yonatan Bisk, Sebastian Scherer*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Aerial outdoor semantic navigation requires robots to explore large, unstructured environments to locate target objects. Recent advances in semantic navigation have demonstrated open-set object-goal navigation in indoor settings, but these methods remain limited by constrained spatial ranges and structured layouts, making them unsuitable for long-range outdoor search. While outdoor semantic navigation approaches exist, they either rely on reactive policies based on current observations, which tend to produce short-sighted behaviors, or precompute scene graphs offline for navigation, limiting adaptability to online deployment. We present RAVEN, a 3D memory-based, behavior tree framework for aerial semantic navigation in unstructured outdoor environments. It (1) uses a spatially consistent semantic voxel-ray map as persistent memory, enabling long-horizon planning and avoiding purely reactive behaviors, (2) combines short-range voxel search and long-range ray search to scale to large environments, (3) leverages a large vision-language model to suggest auxiliary cues, mitigating sparsity of outdoor targets. These components are coordinated by a behavior tree, which adaptively switches behaviors for robust operation. We evaluate RAVEN in 10 photorealistic outdoor simulation environments over 100 semantic tasks, encompassing single-object search, multi-class, multi-instance navigation and sequential task changes. Results show RAVEN outperforms baselines by 85.25% in simulation and demonstrate its real-world applicability through deployment on an aerial robot in outdoor field tests.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.23563) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---
