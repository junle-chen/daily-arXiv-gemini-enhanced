# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-20

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Cognitive-Aligned Spatio-Temporal Large Language Models For Next Point-of-Interest Prediction](https://arxiv.org/abs/2510.14702)
*Penglong Zhai, Jie Li, Fanyi Di, Yue Liu, Yifang Yuan, Jie Huang, Peng Wu, Sicong Wang, Mingyang Yin, Tingting Hu, Yao Xu, Xin Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The next point-of-interest (POI) recommendation task aims to predict the users' immediate next destinations based on their preferences and historical check-ins, holding significant value in location-based services. Recently, large language models (LLMs) have shown great potential in recommender systems, which treat the next POI prediction in a generative manner. However, these LLMs, pretrained primarily on vast corpora of unstructured text, lack the native understanding of structured geographical entities and sequential mobility patterns required for next POI prediction tasks. Moreover, in industrial-scale POI prediction applications, incorporating world knowledge and alignment of human cognition, such as seasons, weather conditions, holidays, and users' profiles (such as habits, occupation, and preferences), can enhance the user experience while improving recommendation performance. To address these issues, we propose CoAST (Cognitive-Aligned Spatial-Temporal LLMs), a framework employing natural language as an interface, allowing for the incorporation of world knowledge, spatio-temporal trajectory patterns, profiles, and situational information. Specifically, CoAST mainly comprises of 2 stages: (1) Recommendation Knowledge Acquisition through continued pretraining on the enriched spatial-temporal trajectory data of the desensitized users; (2) Cognitive Alignment to align cognitive judgments with human preferences using enriched training data through Supervised Fine-Tuning (SFT) and a subsequent Reinforcement Learning (RL) phase. Extensive offline experiments on various real-world datasets and online experiments deployed in "Guess Where You Go" of AMAP App homepage demonstrate the effectiveness of CoAST.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14702) | **Categories:** cs.AI

---

### [2] [Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction](https://arxiv.org/abs/2510.14319)
*Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Model based multi-agent systems (MAS) excel at collaborative problem solving but remain brittle to cascading errors: a single faulty step can propagate across agents and disrupt the trajectory. In this paper, we present MASC, a metacognitive framework that endows MAS with real-time, unsupervised, step-level error detection and self-correction. MASC rethinks detection as history-conditioned anomaly scoring via two complementary designs: (1) Next-Execution Reconstruction, which predicts the embedding of the next step from the query and interaction history to capture causal consistency, and (2) Prototype-Guided Enhancement, which learns a prototype prior over normal-step embeddings and uses it to stabilize reconstruction and anomaly scoring under sparse context (e.g., early steps). When an anomaly step is flagged, MASC triggers a correction agent to revise the acting agent's output before information flows downstream. On the Who&When benchmark, MASC consistently outperforms all baselines, improving step-level error detection by up to 8.47% AUC-ROC ; When plugged into diverse MAS frameworks, it delivers consistent end-to-end gains across architectures, confirming that our metacognitive monitoring and targeted correction can mitigate error propagation with minimal overhead.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14319) | **Categories:** cs.AI

---

### [3] [ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](https://arxiv.org/abs/2509.26255)
*Yichao Liang, Dat Nguyen, Cambridge Yang, Tianyang Li, Joshua B. Tenenbaum, Carl Edward Rasmussen, Adrian Weller, Zenna Tavares, Tom Silver, Kevin Ellis*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Long-horizon embodied planning is challenging because the world does not only change through an agent's actions: exogenous processes (e.g., water heating, dominoes cascading) unfold concurrently with the agent's actions. We propose a framework for abstract world models that jointly learns (i) symbolic state representations and (ii) causal processes for both endogenous actions and exogenous mechanisms. Each causal process models the time course of a stochastic cause-effect relation. We learn these world models from limited data via variational Bayesian inference combined with LLM proposals. Across five simulated tabletop robotics environments, the learned models enable fast planning that generalizes to held-out tasks with more objects and more complex goals, outperforming a range of baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.26255) | **Categories:** cs.AI, cs.CV, cs.LG, cs.RO

---

### [4] [RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning](https://arxiv.org/abs/2510.14828)
*Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14828) | **Categories:** cs.AI, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion Models](https://arxiv.org/abs/2510.14615)
*Edward Sandra, Lander Vanroye, Dries Dirckx, Ruben Cartuyvels, Jan Swevers, Wilm Decré*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Classical methods in robot motion planning, such as sampling-based and optimization-based methods, often struggle with scalability towards higher-dimensional state spaces and complex environments. Diffusion models, known for their capability to learn complex, high-dimensional and multi-modal data distributions, provide a promising alternative when applied to motion planning problems and have already shown interesting results. However, most of the current approaches train their model for a single environment, limiting their generalization to environments not seen during training. The techniques that do train a model for multiple environments rely on a specific camera to provide the model with the necessary environmental information and therefore always require that sensor. To effectively adapt to diverse scenarios without the need for retraining, this research proposes Context-Aware Motion Planning Diffusion (CAMPD). CAMPD leverages a classifier-free denoising probabilistic diffusion model, conditioned on sensor-agnostic contextual information. An attention mechanism, integrated in the well-known U-Net architecture, conditions the model on an arbitrary number of contextual parameters. CAMPD is evaluated on a 7-DoF robot manipulator and benchmarked against state-of-the-art approaches on real-world tasks, showing its ability to generalize to unseen environments and generate high-quality, multi-modal trajectories, at a fraction of the time required by existing methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14615) | **Categories:** cs.RO

---

### [2] [Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](https://arxiv.org/abs/2510.14827)
*Yufei Zhu, Shih-Min Yang, Andrey Rudenko, Tomasz P. Kucner, Achim J. Lilienthal, Martin Magnusson*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe and efficient robot operation in complex human environments can benefit from good models of site-specific motion patterns. Maps of Dynamics (MoDs) provide such models by encoding statistical motion patterns in a map, but existing representations use discrete spatial sampling and typically require costly offline construction. We propose a continuous spatio-temporal MoD representation based on implicit neural functions that directly map coordinates to the parameters of a Semi-Wrapped Gaussian Mixture Model. This removes the need for discretization and imputation for unevenly sampled regions, enabling smooth generalization across both space and time. Evaluated on a large public dataset with long-term real-world people tracking data, our method achieves better accuracy of motion representation and smoother velocity distributions in sparse regions while still being computationally efficient, compared to available baselines. The proposed approach demonstrates a powerful and efficient way of modeling complex human motion patterns.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14827) | **Categories:** cs.RO

---

### [3] [STITCHER: Constrained Trajectory Planning in Known Environments with Real-Time Motion Primitive Search](https://arxiv.org/abs/2510.14893)
*Helene J. Levy, Brett T. Lopez*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous high-speed navigation through large, complex environments requires real-time generation of agile trajectories that are dynamically feasible, collision-free, and satisfy state or actuator constraints. Modern trajectory planning techniques primarily use numerical optimization, as they enable the systematic computation of high-quality, expressive trajectories that satisfy various constraints. However, stringent requirements on computation time and the risk of numerical instability can limit the use of optimization-based planners in safety-critical scenarios. This work presents an optimization-free planning framework called STITCHER that stitches short trajectory segments together with graph search to compute long-range, expressive, and near-optimal trajectories in real-time. STITCHER outperforms modern optimization-based planners through our innovative planning architecture and several algorithmic developments that make real-time planning possible. Extensive simulation testing is performed to analyze the algorithmic components that make up STITCHER, along with a thorough comparison with two state-of-the-art optimization planners. Simulation tests show that safe trajectories can be created within a few milliseconds for paths that span the entirety of two 50 m x 50 m environments. Hardware tests with a custom quadrotor verify that STITCHER can produce trackable paths in real-time while respecting nonconvex constraints, such as limits on tilt angle and motor forces, which are otherwise hard to include in optimization-based planners.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14893) | **Categories:** cs.RO

---

### [4] [A Diffusion-Refined Planner with Reinforcement Learning Priors for Confined-Space Parking](https://arxiv.org/abs/2510.14000)
*Mingyang Jiang, Yueyuan Li, Jiaru Zhang, Songan Zhang, Ming Yang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The growing demand for parking has increased the need for automated parking planning methods that can operate reliably in confined spaces. In restricted and complex environments, high-precision maneuvers are required to achieve a high success rate in planning, yet existing approaches often rely on explicit action modeling, which faces challenges when accurately modeling the optimal action distribution. In this paper, we propose DRIP, a diffusion-refined planner anchored in reinforcement learning (RL) prior action distribution, in which an RL-pretrained policy provides prior action distributions to regularize the diffusion training process. During the inference phase the denoising process refines these coarse priors into more precise action distributions. By steering the denoising trajectory through the reinforcement learning prior distribution during training, the diffusion model inherits a well-informed initialization, resulting in more accurate action modeling, a higher planning success rate, and reduced inference steps. We evaluate our approach across parking scenarios with varying degrees of spatial constraints. Experimental results demonstrate that our method significantly improves planning performance in confined-space parking environments while maintaining strong generalization in common scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14000) | **Categories:** cs.RO

---

### [5] [Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming](https://arxiv.org/abs/2510.14063)
*Nan Li, Jiming Ren, Haris Miller, Samuel Coogan, Karen M. Feigh, Ye Zhao*

Main category: cs.RO

TL;DR: 本文提出了一种名为OATH的自适应障碍物感知任务分配与规划方法，通过引入新颖的障碍物感知任务分配策略，提升了异构机器人团队在复杂环境下的任务执行效率和适应性。


<details>
  <summary>Details</summary>
Motivation: 多智能体任务分配与规划（MATP）在可扩展性、空间推理以及在多障碍环境中的适应性方面面临挑战。

Method: 提出了OATH框架，包括自适应Halton序列地图和集群-拍卖-选择框架，并利用LLM解释人类指令以实时指导规划。

Result: 在NVIDIA Isaac Sim中的实验表明，OATH在任务分配质量、可扩展性、动态变化适应性和整体执行性能方面均优于现有MATP基线。

Conclusion: OATH框架通过其障碍物感知策略和集成机制，有效地提升了异构机器人在复杂环境中的任务分配和规划能力。

Abstract: 多智能体任务分配与规划（MATP）越来越受到关注，但在可扩展性、空间推理以及在多障碍环境中的适应性方面仍然具有挑战性。为了应对这些挑战，我们提出了一种名为OATH的自适应障碍物感知任务分配与规划方法，用于异构机器人团队，该方法通过引入一种新颖的障碍物感知策略来改进MATP。首先，我们开发了一种自适应Halton序列地图，这是Halton采样与MATP中障碍物感知自适应的首次已知应用，它可以根据障碍物分布调整采样密度。其次，我们提出了一个集群-拍卖-选择框架，该框架将障碍物感知聚类与加权拍卖和集群内任务选择相结合。这些机制共同实现了异构机器人之间的有效协调，同时保持了可扩展性和接近最优的分配性能。此外，我们的框架利用LLM来解释人类指令并实时直接指导规划器。我们在NVIDIA Isaac Sim中验证了OATH，结果表明，与最先进的MATP基线相比，OATH在任务分配质量、可扩展性、动态变化适应性和整体执行性能方面都有显著提高。项目网站请访问https://llm-oath.github.io/。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14063) | **Categories:** cs.RO

---

### [6] [Learning Human-Humanoid Coordination for Collaborative Object Carrying](https://arxiv.org/abs/2510.14293)
*Yushi Du, Yixuan Li, Baoxiong Jia, Yutang Lin, Pei Zhou, Wei Liang, Yanchao Yang, Siyuan Huang*

Main category: cs.RO

TL;DR: 该论文提出了一种仅使用本体感受的强化学习方法COLA，实现了顺应性人-人形机器人协作，通过预测物体运动模式和人类意图来保持负载平衡。


<details>
  <summary>Details</summary>
Motivation: 由于人形机器人复杂的全身动力学，顺应性人-人形机器人协作尚未得到充分探索。该论文旨在解决这一问题，实现人形机器人在医疗、家政和制造等领域的协作应用。

Method: 该论文提出了一种名为COLA的本体感受强化学习方法，该方法在单一策略中结合了领导者和跟随者行为。该模型在闭环环境中进行训练，通过动态物体交互来隐式预测物体运动模式和人类意图，从而实现顺应性协作，并通过协调的轨迹规划来保持负载平衡。

Result: 仿真实验表明，与基线方法相比，该模型可减少人类24.7%的体力消耗，同时保持物体稳定性。真实世界的实验验证了在不同物体类型（盒子、桌子、担架等）和运动模式（直线、转弯、爬坡）下的鲁棒协作搬运。包含23名参与者的人体用户研究证实，与基线模型相比，平均提高了27.4%。

Conclusion: 该方法无需外部传感器或复杂的交互模型，即可实现顺应性人-人形机器人协作搬运，为实际部署提供了一种实用的解决方案。

Abstract: 人与人形机器人的协作在医疗保健、家庭帮助和制造业中显示出巨大的潜力。虽然顺应性机器人与人的协作已在机器人手臂上得到广泛发展，但由于人形机器人复杂的全身动力学，顺应性的人与人形机器人协作在很大程度上仍未得到探索。在本文中，我们提出了一种仅使用本体感受的强化学习方法COLA，该方法在单一策略中结合了领导者和跟随者行为。该模型在具有动态对象交互的闭环环境中进行训练，以隐式预测对象运动模式和人类意图，从而实现顺应性协作，以通过协调的轨迹规划来保持负载平衡。我们通过全面的模拟器和真实世界的协作搬运任务实验评估了我们的方法，证明了我们的模型在各种地形和对象上的有效性、泛化性和鲁棒性。仿真实验表明，与基线方法相比，我们的模型可减少人类24.7%的体力消耗，同时保持物体稳定性。真实世界的实验验证了在不同物体类型（盒子、桌子、担架等）和运动模式（直线、转弯、爬坡）下的鲁棒协作搬运。包含23名参与者的人体用户研究证实，与基线模型相比，平均提高了27.4%。我们的方法无需外部传感器或复杂的交互模型，即可实现顺应性人-人形机器人协作搬运，为实际部署提供了一种实用的解决方案。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14293) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [7] [Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning](https://arxiv.org/abs/2510.14300)
*Weijie Shen, Yitian Liu, Yuhao Wu, Zhixuan Liang, Sijia Gu, Dehui Wang, Tian Nian, Lei Xu, Yusen Qin, Jiangmiao Pang, Xinping Guan, Xiaokang Yang, Yao Mu*

Main category: cs.RO

TL;DR: AdaMoE通过继承预训练VLA模型权重并引入解耦技术，实现了在机器人操作任务中性能和效率的提升。


<details>
  <summary>Details</summary>
Motivation: 现有VLA模型训练成本高昂，且机器人数据稀缺；同时，实时控制需要在模型容量和计算效率之间取得平衡。

Method: 提出AdaMoE，一种MoE架构，通过将前馈层替换为稀疏激活的MoE层来扩展动作专家，并采用解耦技术将专家选择与专家权重分离。

Result: 在LIBERO上性能提升1.8%，在RoboTwin上提升9.3%，在真实世界实验中提升21.5%。

Conclusion: 通过协作式专家利用，AdaMoE能够在保持计算效率的同时实现卓越的性能。

Abstract: 视觉-语言-动作（VLA）模型正在经历快速发展，并在机器人操作任务中展现出令人鼓舞的能力。然而，扩展VLA模型面临着几个关键挑战：（1）从头开始训练新的VLA模型需要大量的计算资源和广泛的数据集。鉴于目前机器人数据的稀缺性，在扩展过程中充分利用预训练的VLA模型权重变得尤为重要。（2）实时控制需要仔细平衡模型容量和计算效率。为了应对这些挑战，我们提出了AdaMoE，一种混合专家（MoE）架构，它继承了密集VLA模型的预训练权重，并通过将前馈层替换为稀疏激活的MoE层来扩展动作专家。AdaMoE采用了一种解耦技术，通过与传统路由器一起工作的独立尺度适配器将专家选择与专家权重分离。这使得专家能够基于任务相关性进行选择，同时以独立控制的权重做出贡献，从而实现协作式专家利用，而不是赢者通吃的模式。我们的方法表明，专业知识不必垄断。相反，通过协作式专家利用，我们可以在保持计算效率的同时实现卓越的性能。AdaMoE在关键基准测试中始终优于基线模型，在LIBERO上实现了1.8%的性能提升，在RoboTwin上实现了9.3%的性能提升。最重要的是，真实世界实验中高达21.5%的提升验证了其在机器人操作任务中的实际有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14300) | **Categories:** cs.RO, cs.AI

---

### [8] [GOPLA: Generalizable Object Placement Learning via Synthetic Augmentation of Human Arrangement](https://arxiv.org/abs/2510.14627)
*Yao Zhong, Hanzhi Chen, Simon Schaefer, Anran Zhang, Stefan Leutenegger*

Main category: cs.RO

TL;DR: GOPLA提出了一种分层框架，通过学习人类演示来推广物体放置，利用多模态大型语言模型生成结构化计划，并通过扩散规划器生成放置姿势，显著提高了物体放置的成功率。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在解决机器人物体放置任务中的语义偏好（常识物体关系）和几何可行性（避碰）问题。

Method: 该方法提出了一种分层框架GOPLA，它利用多模态大型语言模型将人类指令和视觉输入转换为结构化计划，并通过空间映射器将其转换为具有几何常识的3D可供性地图，最后使用扩散规划器生成放置姿势。

Result: 实验结果表明，GOPLA方法在物体放置成功率上比第二名提高了30.04个百分点，并在定位精度和物理合理性方面表现出强大的泛化能力。

Conclusion: 该论文证明了GOPLA框架在各种真实世界机器人放置场景中的有效性和泛化能力。

Abstract: 机器人有望成为智能助手，帮助人类进行日常家居整理。在这种场景下，一个核心挑战是物体放置任务，这需要对语义偏好（例如，常识性物体关系）和几何可行性（例如，避碰）进行推理。我们提出了GOPLA，一个分层框架，它从增强的人类演示中学习可推广的物体放置。一个多模态大型语言模型将人类指令和视觉输入翻译成指定成对物体关系的结构化计划。然后，这些计划通过空间映射器被转换为具有几何常识的3D可供性地图，而基于扩散的规划器生成由测试时成本引导的放置姿势，同时考虑多计划分布和避碰。为了克服数据稀缺问题，我们引入了一个可扩展的管道，将人类放置演示扩展为多样化的合成训练数据。大量实验表明，我们的方法在放置成功率方面比第二名提高了30.04个百分点，在定位精度和物理合理性方面进行了评估，证明了在各种真实世界机器人放置场景中的强大泛化能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14627) | **Categories:** cs.RO, cs.CV

---

### [9] [When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks](https://arxiv.org/abs/2510.14677)
*Steffen Hagedorn, Luka Donkov, Aron Distelzweig, Alexandru P. Condurache*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planner evaluation in closed-loop simulation often uses rule-based traffic agents, whose simplistic and passive behavior can hide planner deficiencies and bias rankings. Widely used IDM agents simply follow a lead vehicle and cannot react to vehicles in adjacent lanes, hindering tests of complex interaction capabilities. We address this issue by integrating the state-of-the-art learned traffic agent model SMART into nuPlan. Thus, we are the first to evaluate planners under more realistic conditions and quantify how conclusions shift when narrowing the sim-to-real gap. Our analysis covers 14 recent planners and established baselines and shows that IDM-based simulation overestimates planning performance: nearly all scores deteriorate. In contrast, many planners interact better than previously assumed and even improve in multi-lane, interaction-heavy scenarios like lane changes or turns. Methods trained in closed-loop demonstrate the best and most stable driving performance. However, when reaching their limits in augmented edge-case scenarios, all learned planners degrade abruptly, whereas rule-based planners maintain reasonable basic behavior. Based on our results, we suggest SMART-reactive simulation as a new standard closed-loop benchmark in nuPlan and release the SMART agents as a drop-in alternative to IDM at https://github.com/shgd95/InteractiveClosedLoop.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14677) | **Categories:** cs.RO, cs.AI, cs.LG, cs.MA

---

### [10] [From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance](https://arxiv.org/abs/2510.14952)
*Zhe Li, Cheng Chi, Yangyang Wei, Boan Zhu, Yibo Peng, Tao Huang, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang, Chang Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Natural language offers a natural interface for humanoid robots, but existing language-guided humanoid locomotion pipelines remain cumbersome and unreliable. They typically decode human motion, retarget it to robot morphology, and then track it with a physics-based controller. However, this multi-stage process is prone to cumulative errors, introduces high latency, and yields weak coupling between semantics and control. These limitations call for a more direct pathway from language to action, one that eliminates fragile intermediate stages. Therefore, we present RoboGhost, a retargeting-free framework that directly conditions humanoid policies on language-grounded motion latents. By bypassing explicit motion decoding and retargeting, RoboGhost enables a diffusion-based policy to denoise executable actions directly from noise, preserving semantic intent and supporting fast, reactive control. A hybrid causal transformer-diffusion motion generator further ensures long-horizon consistency while maintaining stability and diversity, yielding rich latent representations for precise humanoid behavior. Extensive experiments demonstrate that RoboGhost substantially reduces deployment latency, improves success rates and tracking accuracy, and produces smooth, semantically aligned locomotion on real humanoids. Beyond text, the framework naturally extends to other modalities such as images, audio, and music, providing a general foundation for vision-language-action humanoid systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14952) | **Categories:** cs.RO, cs.CV

---
