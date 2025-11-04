# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-05

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Aligning LLM agents with human learning and adjustment behavior: a dual agent approach](https://arxiv.org/abs/2511.00993)
*Tianming Liu, Jirong Yang, Yafeng Yin, Manzi Li, Linghao Wang, Zheng Zhu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective modeling of how human travelers learn and adjust their travel behavior from interacting with transportation systems is critical for system assessment and planning. However, this task is also difficult due to the complex cognition and decision-making involved in such behavior. Recent research has begun to leverage Large Language Model (LLM) agents for this task. Building on this, we introduce a novel dual-agent framework that enables continuous learning and alignment between LLM agents and human travelers on learning and adaptation behavior from online data streams. Our approach involves a set of LLM traveler agents, equipped with a memory system and a learnable persona, which serve as simulators for human travelers. To ensure behavioral alignment, we introduce an LLM calibration agent that leverages the reasoning and analytical capabilities of LLMs to train the personas of these traveler agents. Working together, this dual-agent system is designed to track and align the underlying decision-making mechanisms of travelers and produce realistic, adaptive simulations. Using a real-world dataset from a day-to-day route choice experiment, we show our approach significantly outperforms existing LLM-based methods in both individual behavioral alignment and aggregate simulation accuracy. Furthermore, we demonstrate that our method moves beyond simple behavioral mimicry to capture the evolution of underlying learning processes, a deeper alignment that fosters robust generalization. Overall, our framework provides a new approach for creating adaptive and behaviorally realistic agents to simulate travelers' learning and adaptation that can benefit transportation simulation and policy analysis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00993) | **Categories:** cs.AI, cs.LG

---

### [2] [Advancing Cognitive Science with LLMs](https://arxiv.org/abs/2511.00206)
*Dirk U. Wulff, Rui Mata*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Cognitive science faces ongoing challenges in knowledge synthesis and conceptual clarity, in part due to its multifaceted and interdisciplinary nature. Recent advances in artificial intelligence, particularly the development of large language models (LLMs), offer tools that may help to address these issues. This review examines how LLMs can support areas where the field has historically struggled, including establishing cross-disciplinary connections, formalizing theories, developing clear measurement taxonomies, achieving generalizability through integrated modeling frameworks, and capturing contextual and individual variation. We outline the current capabilities and limitations of LLMs in these domains, including potential pitfalls. Taken together, we conclude that LLMs can serve as tools for a more integrative and cumulative cognitive science when used judiciously to complement, rather than replace, human expertise.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00206) | **Categories:** cs.AI, cs.CL

---

### [3] [Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries](https://arxiv.org/abs/2511.00710)
*Minghe Shen, Zhuo Zhi, Chonghan Liu, Shuo Xing, Zhengzhong Tu, Che Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While Vision-Language Models (VLMs) post-trained with Reinforcement Learning (RL) show impressive general reasoning, their evaluation is often confined to language-dominant tasks (e.g., math). This raises a critical question: can RL post-training truly extend the inherent capability boundary of a base VLM, particularly for visual-centric spatial tasks where it initially fails? To investigate this, we introduce Ariadne, a framework utilizing synthetic mazes for multi-step spatial reasoning where task difficulty (e.g., path length, turns) is precisely controlled. We leverage this controllable environment to train VLMs using Reinforcement Learning with Verified Rewards (RLVR) in a difficulty-aware curriculum. Surprisingly, post-RLVR training, the VLM achieves over 50% accuracy on a problem set where the base model scored 0%, demonstrating that our approach expands the model's initial capability boundary. To assess real-world viability, we evaluate out-of-distribution (OOD) generalization on practical benchmarks. Despite training only on synthetic maze samples, Ariadne achieves significant zero-shot improvements, averaging 16% on MapBench (e.g., museum navigation) and 24% on ReasonMap (subway transfer tasks). These results confirm that our method not only broadens the model's fundamental limits but also enhances its generalization to real-world spatial reasoning. We acknowledge our study is limited to the post-training phase, given the opaqueness of pre-training data, and hope our research motivates further work on specialized, capability-extending alignment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00710) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Object-Aware 4D Human Motion Generation](https://arxiv.org/abs/2511.00248)
*Shurui Gui, Deep Anil Patel, Xiner Li, Martin Renqiang Min*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in video diffusion models have enabled the generation of high-quality videos. However, these videos still suffer from unrealistic deformations, semantic violations, and physical inconsistencies that are largely rooted in the absence of 3D physical priors. To address these challenges, we propose an object-aware 4D human motion generation framework grounded in 3D Gaussian representations and motion diffusion priors. With pre-generated 3D humans and objects, our method, Motion Score Distilled Interaction (MSDI), employs the spatial and prompt semantic information in large language models (LLMs) and motion priors through the proposed Motion Diffusion Score Distillation Sampling (MSDS). The combination of MSDS and LLMs enables our spatial-aware motion optimization, which distills score gradients from pre-trained motion diffusion models, to refine human motion while respecting object and semantic constraints. Unlike prior methods requiring joint training on limited interaction datasets, our zero-shot approach avoids retraining and generalizes to out-of-distribution object aware human motions. Experiments demonstrate that our framework produces natural and physically plausible human motions that respect 3D spatial context, offering a scalable solution for realistic 4D generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00248) | **Categories:** cs.CV, cs.GR

---

### [2] [Generative human motion mimicking through feature extraction in denoising diffusion settings](https://arxiv.org/abs/2511.00011)
*Alexander Okupnik, Johannes Schneider, Kyriakos Flouris*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent success with large language models has sparked a new wave of verbal human-AI interaction. While such models support users in a variety of creative tasks, they lack the embodied nature of human interaction. Dance, as a primal form of human expression, is predestined to complement this experience. To explore creative human-AI interaction exemplified by dance, we build an interactive model based on motion capture (MoCap) data. It generates an artificial other by partially mimicking and also "creatively" enhancing an incoming sequence of movement data. It is the first model, which leverages single-person motion data and high level features in order to do so and, thus, it does not rely on low level human-human interaction data. It combines ideas of two diffusion models, motion inpainting, and motion style transfer to generate movement representations that are both temporally coherent and responsive to a chosen movement reference. The success of the model is demonstrated by quantitatively assessing the convergence of the feature distribution of the generated samples and the test set which serves as simulating the human performer. We show that our generations are first steps to creative dancing with AI as they are both diverse showing various deviations from the human partner while appearing realistic.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00011) | **Categories:** cs.CV, cs.AI, cs.HC

---

### [3] [World Simulation with Video Foundation Models for Physical AI](https://arxiv.org/abs/2511.00062)
*NVIDIA, :, Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, Prithvijit Chattopadhyay, Mike Chen, Yongxin Chen, Yu Chen, Shuai Cheng, Yin Cui, Jenna Diamond, Yifan Ding, Jiaojiao Fan, Linxi Fan, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Ruiyuan Gao, Yunhao Ge, Jinwei Gu, Aryaman Gupta, Siddharth Gururani, Imad El Hanafi, Ali Hassani, Zekun Hao, Jacob Huffman, Joel Jang, Pooya Jannaty, Jan Kautz, Grace Lam, Xuan Li, Zhaoshuo Li, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Yen-Chen Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Yifan Lu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Seungjun Nah, Yashraj Narang, Abhijeet Panaskar, Lindsey Pavao, Trung Pham, Morteza Ramezanali, Fitsum Reda, Scott Reed, Xuanchi Ren, Haonan Shao, Yue Shen, Stella Shi, Shuran Song, Bartosz Stefaniak, Shangkun Sun, Shitao Tang, Sameena Tasmeen, Lyne Tchapmi, Wei-Cheng Tseng, Jibin Varghese, Andrew Z. Wang, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Jiashu Xu, Dinghao Yang, Xiaodong Yang, Haotian Ye, Seonghyeon Ye, Xiaohui Zeng, Jing Zhang, Qinsheng Zhang, Kaiwen Zheng, Andrew Zhu, Yuke Zhu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce [Cosmos-Predict2.5], the latest generation of the Cosmos World Foundation Models for Physical AI. Built on a flow-based architecture, [Cosmos-Predict2.5] unifies Text2World, Image2World, and Video2World generation in a single model and leverages [Cosmos-Reason1], a Physical AI vision-language model, to provide richer text grounding and finer control of world simulation. Trained on 200M curated video clips and refined with reinforcement learning-based post-training, [Cosmos-Predict2.5] achieves substantial improvements over [Cosmos-Predict1] in video quality and instruction alignment, with models released at 2B and 14B scales. These capabilities enable more reliable synthetic data generation, policy evaluation, and closed-loop simulation for robotics and autonomous systems. We further extend the family with [Cosmos-Transfer2.5], a control-net style framework for Sim2Real and Real2Real world translation. Despite being 3.5$\times$ smaller than [Cosmos-Transfer1], it delivers higher fidelity and robust long-horizon video generation. Together, these advances establish [Cosmos-Predict2.5] and [Cosmos-Transfer2.5] as versatile tools for scaling embodied intelligence. To accelerate research and deployment in Physical AI, we release source code, pretrained checkpoints, and curated benchmarks under the NVIDIA Open Model License at https://github.com/nvidia-cosmos/cosmos-predict2.5 and https://github.com/nvidia-cosmos/cosmos-transfer2.5. We hope these open resources lower the barrier to adoption and foster innovation in building the next generation of embodied intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00062) | **Categories:** cs.CV, cs.AI, cs.LG, cs.RO

---

### [4] [Chain of Time: In-Context Physical Simulation with Image Generation Models](https://arxiv.org/abs/2511.00110)
*YingQiao Wang, Eric Bigelow, Boyi Li, Tomer Ullman*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose a novel cognitively-inspired method to improve and interpret physical simulation in vision-language models. Our ``Chain of Time" method involves generating a series of intermediate images during a simulation, and it is motivated by in-context reasoning in machine learning, as well as mental simulation in humans. Chain of Time is used at inference time, and requires no additional fine-tuning. We apply the Chain-of-Time method to synthetic and real-world domains, including 2-D graphics simulations and natural 3-D videos. These domains test a variety of particular physical properties, including velocity, acceleration, fluid dynamics, and conservation of momentum. We found that using Chain-of-Time simulation substantially improves the performance of a state-of-the-art image generation model. Beyond examining performance, we also analyzed the specific states of the world simulated by an image model at each time step, which sheds light on the dynamics underlying these simulations. This analysis reveals insights that are hidden from traditional evaluations of physical reasoning, including cases where an image generation model is able to simulate physical properties that unfold over time, such as velocity, gravity, and collisions. Our analysis also highlights particular cases where the image generation model struggles to infer particular physical parameters from input images, despite being capable of simulating relevant physical processes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00110) | **Categories:** cs.CV, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [FLoRA: Fused forward-backward adapters for parameter efficient fine-tuning and reducing inference-time latencies of LLMs](https://arxiv.org/abs/2511.00050)
*Dhananjaya Gowda, Seoha Song, Junhyun Lee, Harshith Goka*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: As the large language models (LLMs) grow in size each day, efficient training and fine-tuning has never been as important as nowadays. This resulted in the great interest in parameter efficient fine-tuning (PEFT), and effective methods including low-rank adapters (LoRA) has emerged. Although the various PEFT methods have been studied extensively in the recent years, the greater part of the subject remains unexplored with the huge degree of freedom. In this paper, we propose FLoRA, a family of fused forward-backward adapters (FFBA) for parameter-efficient fine-tuning of LLMs on downstream tasks. The FFBA combine ideas from the popular LoRA and parallel adapters to improve the overall fine-tuning accuracies. At the same time, latencies are minimized by fusing the forward and backward adapters into existing projection layers of the base model. Experimental results show that the proposed FFB adapters perform significantly better than the popularly used LoRA in both accuracy and latency for a similar parameter budget.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00050) | **Categories:** cs.LG, cs.AI

---

### [2] [Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features](https://arxiv.org/abs/2511.00126)
*Lu Bowen*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent deep trajectory predictors (e.g., Jiang et al., 2023; Zhou et al., 2022) have achieved strong average accuracy but remain unreliable in complex long-tail driving scenarios. These limitations reveal the weakness of the prevailing "one-model-fits-all" paradigm, particularly in safety-critical urban contexts where simpler physics-based models can occasionally outperform advanced networks (Kalman, 1960). To bridge this gap, we propose a dynamic multi-expert gating framework that adaptively selects the most reliable trajectory predictor among a physics-informed LSTM, a Transformer, and a fine-tuned GameFormer on a per-sample basis.   Our method leverages internal model signals (meta-features) such as stability and uncertainty (Gal and Ghahramani, 2016), which we demonstrate to be substantially more informative than geometric scene descriptors. To the best of our knowledge, this is the first work to formulate trajectory expert selection as a pairwise-ranking problem over internal model signals (Burges et al., 2005), directly optimizing decision quality without requiring post-hoc calibration.   Evaluated on the nuPlan-mini dataset (Caesar et al., 2021) with 1,287 samples, our LLM-enhanced tri-expert gate achieves a Final Displacement Error (FDE) of 2.567 m, representing a 9.5 percent reduction over GameFormer (2.835 m), and realizes 57.8 percent of the oracle performance bound. In open-loop simulations, after trajectory horizon alignment, the same configuration reduces FDE on left-turn scenarios by approximately 10 percent, demonstrating consistent improvements across both offline validation and open-loop evaluation. These results indicate that adaptive hybrid systems enhance trajectory reliability in safety-critical autonomous driving, providing a practical pathway beyond static single-model paradigms.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00126) | **Categories:** cs.LG, cs.AI

---

### [3] [SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](https://arxiv.org/abs/2511.00054)
*Gio Huh, Dhruv Sheth, Rayhan Zirvi, Frank Xiao*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While Vision-Language Models (VLMs) excel in many areas, they struggle with complex spatial reasoning, which requires problem decomposition and strategic tool use. Fine-tuning smaller, more deployable models offers an efficient path to strong performance, but this is hampered by a major bottleneck: the absence of high-quality, step-by-step reasoning data. To address this data-efficiency gap, we introduce SpatialTraceGen, a framework to distill the reasoning processes of a large teacher model into a high-quality dataset of multi-hop, multi-tool reasoning traces. A key innovation is our automated Verifier, which scalably ensures the fidelity of each reasoning step, providing a cost-effective alternative to manual human annotation. On the CLEVR-Humans benchmark, this verifier-guided process improves the average quality score of traces by 17\% while reducing quality variance by over 40\%. SpatialTraceGen delivers a dataset of expert traces, providing the structured, step-by-step examples of tool use necessary for effective fine-tuning and sample-efficient offline reinforcement learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00054) | **Categories:** cs.LG, cs.AI

---

### [4] [Automated Discovery of Conservation Laws via Hybrid Neural ODE-Transformers](https://arxiv.org/abs/2511.00102)
*Vivan Doshi*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The discovery of conservation laws is a cornerstone of scientific progress. However, identifying these invariants from observational data remains a significant challenge. We propose a hybrid framework to automate the discovery of conserved quantities from noisy trajectory data. Our approach integrates three components: (1) a Neural Ordinary Differential Equation (Neural ODE) that learns a continuous model of the system's dynamics, (2) a Transformer that generates symbolic candidate invariants conditioned on the learned vector field, and (3) a symbolic-numeric verifier that provides a strong numerical certificate for the validity of these candidates. We test our framework on canonical physical systems and show that it significantly outperforms baselines that operate directly on trajectory data. This work demonstrates the robustness of a decoupled learn-then-search approach for discovering mathematical principles from imperfect data.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00102) | **Categories:** cs.LG, cs.AI

---

### [5] [Analysis of Line Break prediction models for detecting defensive breakthrough in football](https://arxiv.org/abs/2511.00121)
*Shoma Yagi, Jun Ichikawa, Genki Ichinose*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In football, attacking teams attempt to break through the opponent's defensive line to create scoring opportunities. This action, known as a Line Break, is a critical indicator of offensive effectiveness and tactical performance, yet previous studies have mainly focused on shots or goal opportunities rather than on how teams break the defensive line. In this study, we develop a machine learning model to predict Line Breaks using event and tracking data from the 2023 J1 League season. The model incorporates 189 features, including player positions, velocities, and spatial configurations, and employs an XGBoost classifier to estimate the probability of Line Breaks. The proposed model achieved high predictive accuracy, with an AUC of 0.982 and a Brier score of 0.015. Furthermore, SHAP analysis revealed that factors such as offensive player speed, gaps in the defensive line, and offensive players' spatial distributions significantly contribute to the occurrence of Line Breaks. Finally, we found a moderate positive correlation between the predicted probability of being Line-Broken and the number of shots and crosses conceded at the team level. These results suggest that Line Breaks are closely linked to the creation of scoring opportunities and provide a quantitative framework for understanding tactical dynamics in football.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00121) | **Categories:** cs.LG, physics.soc-ph, stat.AP

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail](https://arxiv.org/abs/2511.00088)
*NVIDIA, :, Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, Marco Pavone*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. To address this, we introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning to enhance decision-making in complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a Vision-Language Model pre-trained for Physical AI applications, with a diffusion-based trajectory decoder that generates dynamically feasible plans in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to optimize reasoning quality via large reasoning model feedback and enforce reasoning-action consistency. Evaluation shows AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in off-road rate and 25% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% as measured by a large reasoning model critic and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. We plan to release AR1 models and a subset of the CoC in a future update.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00088) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [2] [Don't Just Search, Understand: Semantic Path Planning Agent for Spherical Tensegrity Robots in Unknown Environments](https://arxiv.org/abs/2511.01236)
*Junwen Zhang, Changyue Liu, Pengqi Fu, Xiang Guo, Ye Shi, Xudong Liang, Zhijian Wang, Hanzhi Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Endowed with inherent dynamical properties that grant them remarkable ruggedness and adaptability, spherical tensegrity robots stand as prototypical examples of hybrid softrigid designs and excellent mobile platforms. However, path planning for these robots in unknown environments presents a significant challenge, requiring a delicate balance between efficient exploration and robust planning. Traditional path planners, which treat the environment as a geometric grid, often suffer from redundant searches and are prone to failure in complex scenarios due to their lack of semantic understanding. To overcome these limitations, we reframe path planning in unknown environments as a semantic reasoning task. We introduce a Semantic Agent for Tensegrity robots (SATPlanner) driven by a Large Language Model (LLM). SATPlanner leverages high-level environmental comprehension to generate efficient and reliable planning strategies.At the core of SATPlanner is an Adaptive Observation Window mechanism, inspired by the "fast" and "slow" thinking paradigms of LLMs. This mechanism dynamically adjusts the perceptual field of the agent: it narrows for rapid traversal of open spaces and expands to reason about complex obstacle configurations. This allows the agent to construct a semantic belief of the environment, enabling the search space to grow only linearly with the path length (O(L)) while maintaining path quality. We extensively evaluate SATPlanner in 1,000 simulation trials, where it achieves a 100% success rate, outperforming other real-time planning algorithms. Critically, SATPlanner reduces the search space by 37.2% compared to the A* algorithm while achieving comparable, near-optimal path lengths. Finally, the practical feasibility of SATPlanner is validated on a physical spherical tensegrity robot prototype.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.01236) | **Categories:** cs.RO

---

### [3] [When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage](https://arxiv.org/abs/2511.00783)
*Jingzehua Xu, Weihang Zhang, Yangyang Li, Hongmiaoyi Zhang, Guanwen Xie, Jiwei Tang, Shuai Zhang, Yi Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00783) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [4] [Real-Time Learning of Predictive Dynamic Obstacle Models for Robotic Motion Planning](https://arxiv.org/abs/2511.00814)
*Stella Kombo, Masih Haseli, Skylar Wei, Joel W. Burdick*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous systems often must predict the motions of nearby agents from partial and noisy data. This paper asks and answers the question: "can we learn, in real-time, a nonlinear predictive model of another agent's motions?" Our online framework denoises and forecasts such dynamics using a modified sliding-window Hankel Dynamic Mode Decomposition (Hankel-DMD). Partial noisy measurements are embedded into a Hankel matrix, while an associated Page matrix enables singular-value hard thresholding (SVHT) to estimate the effective rank. A Cadzow projection enforces structured low-rank consistency, yielding a denoised trajectory and local noise variance estimates. From this representation, a time-varying Hankel-DMD lifted linear predictor is constructed for multi-step forecasts. The residual analysis provides variance-tracking signals that can support downstream estimators and risk-aware planning. We validate the approach in simulation under Gaussian and heavy-tailed noise, and experimentally on a dynamic crane testbed. Results show that the method achieves stable variance-aware denoising and short-horizon prediction suitable for integration into real-time control frameworks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00814) | **Categories:** cs.RO, cs.LG, cs.SY, eess.SY, 93C41, 93E11, 37M10, I.2.9; I.2.6; I.2.8

---

### [5] [Embodied Cognition Augmented End2End Autonomous Driving](https://arxiv.org/abs/2511.01334)
*Ling Niu, Xiaoji Zheng, Han Wang, Chen Zheng, Ziyuan Yang, Bokui Chen, Jiangtao Gong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In recent years, vision-based end-to-end autonomous driving has emerged as a new paradigm. However, popular end-to-end approaches typically rely on visual feature extraction networks trained under label supervision. This limited supervision framework restricts the generality and applicability of driving models. In this paper, we propose a novel paradigm termed $E^{3}AD$, which advocates for comparative learning between visual feature extraction networks and the general EEG large model, in order to learn latent human driving cognition for enhancing end-to-end planning. In this work, we collected a cognitive dataset for the mentioned contrastive learning process. Subsequently, we investigated the methods and potential mechanisms for enhancing end-to-end planning with human driving cognition, using popular driving models as baselines on publicly available autonomous driving datasets. Both open-loop and closed-loop tests are conducted for a comprehensive evaluation of planning performance. Experimental results demonstrate that the $E^{3}AD$ paradigm significantly enhances the end-to-end planning performance of baseline models. Ablation studies further validate the contribution of driving cognition and the effectiveness of comparative learning process. To the best of our knowledge, this is the first work to integrate human driving cognition for improving end-to-end autonomous driving planning. It represents an initial attempt to incorporate embodied cognitive data into end-to-end autonomous driving, providing valuable insights for future brain-inspired autonomous driving systems. Our code will be made available at Github

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.01334) | **Categories:** cs.RO, cs.AI, cs.HC, 68T45

---

### [6] [STRIDER: Navigation via Instruction-Aligned Structural Decision Space Optimization](https://arxiv.org/abs/2511.00033)
*Diqi He, Xuehao Gao, Hao Li, Junwei Han, Dingwen Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) task requires agents to navigate previously unseen 3D environments using natural language instructions, without any scene-specific training. A critical challenge in this setting lies in ensuring agents' actions align with both spatial structure and task intent over long-horizon execution. Existing methods often fail to achieve robust navigation due to a lack of structured decision-making and insufficient integration of feedback from previous actions. To address these challenges, we propose STRIDER (Instruction-Aligned Structural Decision Space Optimization), a novel framework that systematically optimizes the agent's decision space by integrating spatial layout priors and dynamic task feedback. Our approach introduces two key innovations: 1) a Structured Waypoint Generator that constrains the action space through spatial structure, and 2) a Task-Alignment Regulator that adjusts behavior based on task progress, ensuring semantic alignment throughout navigation. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate that STRIDER significantly outperforms strong SOTA across key metrics; in particular, it improves Success Rate (SR) from 29% to 35%, a relative gain of 20.7%. Such results highlight the importance of spatially constrained decision-making and feedback-guided execution in improving navigation fidelity for zero-shot VLN-CE.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00033) | **Categories:** cs.RO, cs.AI

---

### [7] [Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World](https://arxiv.org/abs/2511.00041)
*Yingzhao Jian, Zhongan Wang, Yi Yang, Hehe Fan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humanoid agents often struggle to handle flexible and diverse interactions in open environments. A common solution is to collect massive datasets to train a highly capable model, but this approach can be prohibitively expensive. In this paper, we explore an alternative solution: empowering off-the-shelf Vision-Language Models (VLMs, such as GPT-4) to control humanoid agents, thereby leveraging their strong open-world generalization to mitigate the need for extensive data collection. To this end, we present \textbf{BiBo} (\textbf{B}uilding humano\textbf{I}d agent \textbf{B}y \textbf{O}ff-the-shelf VLMs). It consists of two key components: (1) an \textbf{embodied instruction compiler}, which enables the VLM to perceive the environment and precisely translate high-level user instructions (e.g., {\small\itshape ``have a rest''}) into low-level primitive commands with control parameters (e.g., {\small\itshape ``sit casually, location: (1, 2), facing: 90$^\circ$''}); and (2) a diffusion-based \textbf{motion executor}, which generates human-like motions from these commands, while dynamically adapting to physical feedback from the environment. In this way, BiBo is capable of handling not only basic interactions but also diverse and complex motions. Experiments demonstrate that BiBo achieves an interaction task success rate of 90.2\% in open environments, and improves the precision of text-guided motion execution by 16.3\% over prior methods. The code will be made publicly available.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00041) | **Categories:** cs.RO, cs.AI

---

### [8] [Maestro: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots](https://arxiv.org/abs/2511.00917)
*Junyao Shi, Rujia Yang, Kaitian Chao, Selina Bingqing Wan, Yifei Shao, Jiahui Lei, Jianing Qian, Long Le, Pratik Chaudhari, Kostas Daniilidis, Chuan Wen, Dinesh Jayaraman*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Today's best-explored routes towards generalist robots center on collecting ever larger "observations-in actions-out" robotics datasets to train large end-to-end models, copying a recipe that has worked for vision-language models (VLMs). We pursue a road less traveled: building generalist policies directly around VLMs by augmenting their general capabilities with specific robot capabilities encapsulated in a carefully curated set of perception, planning, and control modules. In Maestro, a VLM coding agent dynamically composes these modules into a programmatic policy for the current task and scenario. Maestro's architecture benefits from a streamlined closed-loop interface without many manually imposed structural constraints, and a comprehensive and diverse tool repertoire. As a result, it largely surpasses today's VLA models for zero-shot performance on challenging manipulation skills. Further, Maestro is easily extensible to incorporate new modules, easily editable to suit new embodiments such as a quadruped-mounted arm, and even easily adapts from minimal real-world experiences through local code edits.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00917) | **Categories:** cs.RO, cs.AI

---

### [9] [Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation](https://arxiv.org/abs/2511.00933)
*Xiangyu Shi, Zerui Li, Yanyuan Qiao, Qi Wu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in Vision-and-Language Navigation in Continuous Environments (VLN-CE) have leveraged multimodal large language models (MLLMs) to achieve zero-shot navigation. However, existing methods often rely on panoramic observations and two-stage pipelines involving waypoint predictors, which introduce significant latency and limit real-world applicability. In this work, we propose Fast-SmartWay, an end-to-end zero-shot VLN-CE framework that eliminates the need for panoramic views and waypoint predictors. Our approach uses only three frontal RGB-D images combined with natural language instructions, enabling MLLMs to directly predict actions. To enhance decision robustness, we introduce an Uncertainty-Aware Reasoning module that integrates (i) a Disambiguation Module for avoiding local optima, and (ii) a Future-Past Bidirectional Reasoning mechanism for globally coherent planning. Experiments on both simulated and real-robot environments demonstrate that our method significantly reduces per-step latency while achieving competitive or superior performance compared to panoramic-view baselines. These results demonstrate the practicality and effectiveness of Fast-SmartWay for real-world zero-shot embodied navigation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.00933) | **Categories:** cs.RO, cs.CV

---
