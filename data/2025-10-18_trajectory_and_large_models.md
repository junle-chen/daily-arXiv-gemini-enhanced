# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-18

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Cognitive-Aligned Spatio-Temporal Large Language Models For Next Point-of-Interest Prediction](https://arxiv.org/abs/2510.14702)
*Penglong Zhai, Jie Li, Fanyi Di, Yue Liu, Yifang Yuan, Jie Huang, Peng Wu, Sicong Wang, Mingyang Yin, Tingting Hu, Yao Xu, Xin Li*

Main category: cs.AI

TL;DR: 本文提出了一种名为CoAST的框架，它利用自然语言作为接口，融合世界知识、时空轨迹模式、用户画像和情境信息，以提升下一兴趣点推荐的性能。


<details>
  <summary>Details</summary>
Motivation: 现有大型语言模型（LLMs）在下一兴趣点（POI）推荐任务中表现出潜力，但缺乏对结构化地理实体和时序移动模式的理解，且难以有效融入世界知识和人类认知。

Method: CoAST框架包含两个阶段：1）通过在富含时空轨迹数据的脱敏用户数据上进行持续预训练，获取推荐知识；2）通过监督微调（SFT）和强化学习（RL）阶段，利用富含的训练数据对齐认知判断与人类偏好。

Result: 在多个真实世界数据集上的大量离线实验以及在高德地图App首页“猜你去哪儿”中部署的在线实验表明，CoAST框架是有效的。

Conclusion: CoAST框架能够有效融合世界知识和人类认知，提升下一兴趣点推荐的性能。

Abstract: 下一兴趣点（POI）推荐任务旨在根据用户的偏好和历史签到记录预测用户接下来的目的地，这在基于位置的服务中具有重要价值。最近，大型语言模型（LLMs）在推荐系统中展现出巨大的潜力，它们以生成的方式处理下一POI预测。然而，这些LLMs主要在大量的非结构化文本语料库上进行预训练，缺乏对结构化地理实体和顺序移动模式的本地理解，而这些对于下一POI预测任务是必需的。此外，在工业规模的POI预测应用中，融入世界知识和人类认知，例如季节、天气条件、节假日和用户画像（例如习惯、职业和偏好），可以增强用户体验，同时提高推荐性能。为了解决这些问题，我们提出了CoAST（认知对齐的时空LLMs），这是一个采用自然语言作为接口的框架，允许融入世界知识、时空轨迹模式、用户画像和情境信息。具体来说，CoAST主要包括两个阶段：（1）通过在脱敏用户的丰富时空轨迹数据上进行持续预训练，获取推荐知识；（2）认知对齐，通过监督微调（SFT）和随后的强化学习（RL）阶段，使用丰富的训练数据将认知判断与人类偏好对齐。在各种真实世界数据集上的大量离线实验以及在高德地图App首页“猜你去哪儿”中部署的在线实验证明了CoAST的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14702) | **Categories:** cs.AI

---

### [2] [Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction](https://arxiv.org/abs/2510.14319)
*Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen*

Main category: cs.AI

TL;DR: MASC框架通过实时、无监督的错误检测和自我纠正，提高了基于大型语言模型的多智能体系统（MAS）的鲁棒性。


<details>
  <summary>Details</summary>
Motivation: 解决大型语言模型多智能体系统在协同问题解决中易受级联错误影响的问题。

Method: 提出MASC框架，通过下一步执行重构和原型引导增强两种互补设计，实现基于历史条件异常评分的实时错误检测，并触发纠正智能体。

Result: 在Who&When基准测试中，MASC持续优于所有基线，步骤级错误检测的AUC-ROC提高了高达8.47%，并在不同MAS框架中实现了持续的端到端增益。

Conclusion: MASC的元认知监控和有针对性的纠正能够以最小的开销减轻错误传播。

Abstract: 基于大型语言模型的多智能体系统（MAS）在协同问题解决方面表现出色，但仍然容易受到级联错误的影响：单个错误步骤可能会在智能体之间传播并扰乱轨迹。在本文中，我们提出了MASC，这是一个元认知框架，它赋予MAS实时、无监督的步骤级错误检测和自我纠正能力。MASC将检测重新定义为通过两种互补设计进行历史条件异常评分：（1）下一步执行重构，它从查询和交互历史记录预测下一步的嵌入，以捕获因果一致性；（2）原型引导增强，它学习正常步骤嵌入的原型先验，并使用它来稳定稀疏上下文（例如，早期步骤）下的重构和异常评分。当标记异常步骤时，MASC会触发一个纠正智能体来修改执行智能体的输出，然后再将信息向下游传递。在Who&When基准测试中，MASC始终优于所有基线，步骤级错误检测的AUC-ROC提高了高达8.47%；当插入到不同的MAS框架中时，它可以在各种架构中提供一致的端到端增益，这证实了我们的元认知监控和有针对性的纠正可以以最小的开销减轻错误传播。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14319) | **Categories:** cs.AI

---

### [3] [ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](https://arxiv.org/abs/2509.26255)
*Yichao Liang, Dat Nguyen, Cambridge Yang, Tianyang Li, Joshua B. Tenenbaum, Carl Edward Rasmussen, Adrian Weller, Zenna Tavares, Tom Silver, Kevin Ellis*

Main category: cs.AI

TL;DR: 该论文提出了一种抽象世界模型框架，用于联合学习内生动作和外生机制的符号状态表示和因果过程，从而实现长程具身规划。


<details>
  <summary>Details</summary>
Motivation: 长程具身规划面临挑战，因为世界变化不仅来自智能体的动作，还包括外生过程。

Method: 通过变分贝叶斯推断结合LLM提议，从有限数据中学习世界模型，该模型包含符号状态表示以及内生动作和外生机制的因果过程。

Result: 在五个模拟桌面机器人环境中，学习到的模型能够进行快速规划，并推广到具有更多对象和更复杂目标的保留任务，优于一系列基线方法。

Conclusion: 该研究提出了一种有效的抽象世界模型学习方法，可以提升长程具身规划的泛化能力和效率。

Abstract: 长时程具身规划面临着巨大的挑战，因为世界的变化不仅仅来自于智能体的动作，还包括与智能体动作并发发生的外生过程（例如，水加热、多米诺骨牌倒塌）。我们提出了一个抽象世界模型的框架，该框架联合学习（i）符号状态表示和（ii）内生动作和外生机制的因果过程。每个因果过程都对随机因果关系的时间过程进行建模。我们通过变分贝叶斯推理结合LLM提议，从有限的数据中学习这些世界模型。在五个模拟的桌面机器人环境中，学习到的模型能够实现快速规划，并推广到具有更多对象和更复杂目标的预留任务，优于一系列基线方法。

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


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Contrastive Diffusion Alignment: Learning Structured Latents for Controllable Generation](https://arxiv.org/abs/2510.14190)
*Ruchi Sandilya, Sumaira Perez, Charles Lynch, Lindsay Victoria, Benjamin Zebley, Derrick Matthew Buchanan, Mahendra T. Bhati, Nolan Williams, Timothy J. Spellman, Faith M. Gunning, Conor Liston, Logan Grosenick*

Main category: cs.LG

TL;DR: ConDA框架通过在扩散模型嵌入中应用对比学习，对齐潜在空间几何结构与系统动力学，实现可控生成。


<details>
  <summary>Details</summary>
Motivation: 扩散模型的潜在空间缺乏明确的组织结构，难以进行可解释的控制。

Method: 提出ConDA框架，在扩散模型嵌入中应用对比学习，以对比方式构建潜在空间。

Result: 在流体动力学、神经钙成像、神经刺激和面部表情等多个基准测试中，ConDA生成了可解释的潜在表示，并提高了可控性。

Conclusion: 扩散模型的潜在空间编码了与动力学相关的结构，但需要组织潜在空间并在潜在流形上进行遍历才能利用这种结构。

Abstract: 扩散模型在生成方面表现出色，但它们的潜在空间并没有被明确地组织起来以实现可解释的控制。我们引入了ConDA（对比扩散对齐），这是一个在扩散嵌入中应用对比学习的框架，用于将潜在几何结构与系统动力学对齐。受到最近进展的启发，这些进展表明对比目标可以恢复更解耦和结构化的表示，ConDA组织扩散潜在空间，使得遍历方向反映了底层动力学因素。在这个对比结构化的空间中，ConDA支持非线性轨迹遍历，从而支持忠实的插值、外推和可控生成。在流体动力学、神经钙成像、治疗性神经刺激和面部表情等基准测试中，与线性遍历和基于条件的基线相比，ConDA产生了具有改进的可控性的可解释潜在表示。这些结果表明，扩散潜在空间编码了与动力学相关的结构，但利用这种结构需要在潜在组织和沿潜在流形的遍历。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14190) | **Categories:** cs.LG

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

### [2] [GOPLA: Generalizable Object Placement Learning via Synthetic Augmentation of Human Arrangement](https://arxiv.org/abs/2510.14627)
*Yao Zhong, Hanzhi Chen, Simon Schaefer, Anran Zhang, Stefan Leutenegger*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robots are expected to serve as intelligent assistants, helping humans with everyday household organization. A central challenge in this setting is the task of object placement, which requires reasoning about both semantic preferences (e.g., common-sense object relations) and geometric feasibility (e.g., collision avoidance). We present GOPLA, a hierarchical framework that learns generalizable object placement from augmented human demonstrations. A multi-modal large language model translates human instructions and visual inputs into structured plans that specify pairwise object relationships. These plans are then converted into 3D affordance maps with geometric common sense by a spatial mapper, while a diffusion-based planner generates placement poses guided by test-time costs, considering multi-plan distributions and collision avoidance. To overcome data scarcity, we introduce a scalable pipeline that expands human placement demonstrations into diverse synthetic training data. Extensive experiments show that our approach improves placement success rates by 30.04 percentage points over the runner-up, evaluated on positioning accuracy and physical plausibility, demonstrating strong generalization across a wide range of real-world robotic placement scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14627) | **Categories:** cs.RO, cs.CV

---

### [3] [Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](https://arxiv.org/abs/2510.14827)
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

### [4] [STITCHER: Constrained Trajectory Planning in Known Environments with Real-Time Motion Primitive Search](https://arxiv.org/abs/2510.14893)
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

### [5] [A Diffusion-Refined Planner with Reinforcement Learning Priors for Confined-Space Parking](https://arxiv.org/abs/2510.14000)
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

### [6] [Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming](https://arxiv.org/abs/2510.14063)
*Nan Li, Jiming Ren, Haris Miller, Samuel Coogan, Karen M. Feigh, Ye Zhao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-Agent Task Assignment and Planning (MATP) has attracted growing attention but remains challenging in terms of scalability, spatial reasoning, and adaptability in obstacle-rich environments. To address these challenges, we propose OATH: Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming, which advances MATP by introducing a novel obstacle-aware strategy for task assignment. First, we develop an adaptive Halton sequence map, the first known application of Halton sampling with obstacle-aware adaptation in MATP, which adjusts sampling density based on obstacle distribution. Second, we propose a cluster-auction-selection framework that integrates obstacle-aware clustering with weighted auctions and intra-cluster task selection. These mechanisms jointly enable effective coordination among heterogeneous robots while maintaining scalability and near-optimal allocation performance. In addition, our framework leverages an LLM to interpret human instructions and directly guide the planner in real time. We validate OATH in NVIDIA Isaac Sim, showing substantial improvements in task assignment quality, scalability, adaptability to dynamic changes, and overall execution performance compared to state-of-the-art MATP baselines. A project website is available at https://llm-oath.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14063) | **Categories:** cs.RO

---

### [7] [Learning Human-Humanoid Coordination for Collaborative Object Carrying](https://arxiv.org/abs/2510.14293)
*Yushi Du, Yixuan Li, Baoxiong Jia, Yutang Lin, Pei Zhou, Wei Liang, Yanchao Yang, Siyuan Huang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human-humanoid collaboration shows significant promise for applications in healthcare, domestic assistance, and manufacturing. While compliant robot-human collaboration has been extensively developed for robotic arms, enabling compliant human-humanoid collaboration remains largely unexplored due to humanoids' complex whole-body dynamics. In this paper, we propose a proprioception-only reinforcement learning approach, COLA, that combines leader and follower behaviors within a single policy. The model is trained in a closed-loop environment with dynamic object interactions to predict object motion patterns and human intentions implicitly, enabling compliant collaboration to maintain load balance through coordinated trajectory planning. We evaluate our approach through comprehensive simulator and real-world experiments on collaborative carrying tasks, demonstrating the effectiveness, generalization, and robustness of our model across various terrains and objects. Simulation experiments demonstrate that our model reduces human effort by 24.7%. compared to baseline approaches while maintaining object stability. Real-world experiments validate robust collaborative carrying across different object types (boxes, desks, stretchers, etc.) and movement patterns (straight-line, turning, slope climbing). Human user studies with 23 participants confirm an average improvement of 27.4% compared to baseline models. Our method enables compliant human-humanoid collaborative carrying without requiring external sensors or complex interaction models, offering a practical solution for real-world deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14293) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [8] [Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning](https://arxiv.org/abs/2510.14300)
*Weijie Shen, Yitian Liu, Yuhao Wu, Zhixuan Liang, Sijia Gu, Dehui Wang, Tian Nian, Lei Xu, Yusen Qin, Jiangmiao Pang, Xinping Guan, Xiaokang Yang, Yao Mu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models are experiencing rapid development and demonstrating promising capabilities in robotic manipulation tasks. However, scaling up VLA models presents several critical challenges: (1) Training new VLA models from scratch demands substantial computational resources and extensive datasets. Given the current scarcity of robot data, it becomes particularly valuable to fully leverage well-pretrained VLA model weights during the scaling process. (2) Real-time control requires carefully balancing model capacity with computational efficiency. To address these challenges, We propose AdaMoE, a Mixture-of-Experts (MoE) architecture that inherits pretrained weights from dense VLA models, and scales up the action expert by substituting the feedforward layers into sparsely activated MoE layers. AdaMoE employs a decoupling technique that decouples expert selection from expert weighting through an independent scale adapter working alongside the traditional router. This enables experts to be selected based on task relevance while contributing with independently controlled weights, allowing collaborative expert utilization rather than winner-takes-all dynamics. Our approach demonstrates that expertise need not monopolize. Instead, through collaborative expert utilization, we can achieve superior performance while maintaining computational efficiency. AdaMoE consistently outperforms the baseline model across key benchmarks, delivering performance gains of 1.8% on LIBERO and 9.3% on RoboTwin. Most importantly, a substantial 21.5% improvement in real-world experiments validates its practical effectiveness for robotic manipulation tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14300) | **Categories:** cs.RO, cs.AI

---

### [9] [When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks](https://arxiv.org/abs/2510.14677)
*Steffen Hagedorn, Luka Donkov, Aron Distelzweig, Alexandru P. Condurache*

Main category: cs.RO

TL;DR: 本文通过将先进的智能交通智能体模型SMART集成到nuPlan中，首次在更真实的条件下评估规划器，并量化了缩小模拟与现实差距时结论的变化。


<details>
  <summary>Details</summary>
Motivation: 闭环仿真中的规划器评估通常使用基于规则的交通智能体，但其简单和被动的行为可能会掩盖规划器的缺陷并导致排名偏差。

Method: 将最先进的智能交通智能体模型SMART集成到nuPlan中。

Result: 基于IDM的仿真高估了规划性能，几乎所有分数都有所下降；相比之下，许多规划器的交互性能比之前假设的要好，甚至在多车道、交互性强的场景（如变道或转弯）中有所提高。

Conclusion: 基于本文的结果，作者建议将SMART反应式仿真作为nuPlan中新的标准闭环基准，并发布SMART智能体作为IDM的替代品。

Abstract: 闭环仿真中的规划器评估通常使用基于规则的交通智能体，但其简单和被动的行为可能会掩盖规划器的缺陷并导致排名偏差。 广泛使用的IDM智能体只是跟随前车，无法对相邻车道的车辆做出反应，这阻碍了对复杂交互能力的测试。 我们通过将最先进的智能交通智能体模型SMART集成到nuPlan中来解决这个问题。 因此，我们是第一个在更真实的条件下评估规划器，并量化缩小模拟与现实差距时结论如何变化的人。 我们的分析涵盖了14个最新的规划器和已建立的基线，结果表明，基于IDM的仿真高估了规划性能：几乎所有分数都有所下降。 相比之下，许多规划器的交互性能比之前假设的要好，甚至在多车道、交互性强的场景（如变道或转弯）中有所提高。 在闭环中训练的方法表现出最好和最稳定的驾驶性能。 然而，当在增强的边缘情况下达到极限时，所有学习的规划器都会突然退化，而基于规则的规划器则保持合理的基线行为。 基于我们的结果，我们建议将SMART反应式仿真作为nuPlan中新的标准闭环基准，并发布SMART智能体作为IDM的直接替代品，地址为https://github.com/shgd95/InteractiveClosedLoop。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14677) | **Categories:** cs.RO, cs.AI, cs.LG, cs.MA

---

### [10] [From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance](https://arxiv.org/abs/2510.14952)
*Zhe Li, Cheng Chi, Yangyang Wei, Boan Zhu, Yibo Peng, Tao Huang, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang, Chang Xu*

Main category: cs.RO

TL;DR: RoboGhost是一个无需重定向的框架，它直接基于语言驱动的运动潜在空间来调节人形机器人的策略。


<details>
  <summary>Details</summary>
Motivation: 现有的语言引导的人形机器人运动流程繁琐且不可靠，存在累积误差、高延迟以及语义和控制之间的弱耦合等问题。

Method: RoboGhost通过绕过显式的运动解码和重定向，使基于扩散的策略能够直接从噪声中去噪可执行的动作，并采用混合因果变换器-扩散运动生成器来确保长时程一致性。

Result: 大量实验表明，RoboGhost显著降低了部署延迟，提高了成功率和跟踪精度，并在真实的人形机器人上产生了平滑且语义对齐的运动。

Conclusion: RoboGhost框架不仅限于文本，还可以自然地扩展到其他模态，如图像、音频和音乐，为视觉-语言-动作人形机器人系统提供了一个通用的基础。

Abstract: 自然语言为人形机器人提供了一种自然的交互界面，但现有的语言引导的人形机器人运动流程仍然繁琐且不可靠。它们通常解码人类运动，将其重定向到机器人形态，然后通过基于物理的控制器进行跟踪。然而，这种多阶段过程容易产生累积误差，引入高延迟，并导致语义和控制之间的弱耦合。这些限制需要一种更直接的从语言到行动的途径，消除脆弱的中间阶段。因此，我们提出了RoboGhost，一个无需重定向的框架，它直接基于语言驱动的运动潜在空间来调节人形机器人的策略。通过绕过显式的运动解码和重定向，RoboGhost使基于扩散的策略能够直接从噪声中去噪可执行的动作，从而保留语义意图并支持快速、反应式控制。混合因果变换器-扩散运动生成器进一步确保了长时程一致性，同时保持稳定性和多样性，从而为精确的人形机器人行为产生丰富的潜在表示。大量实验表明，RoboGhost显著降低了部署延迟，提高了成功率和跟踪精度，并在真实的人形机器人上产生了平滑且语义对齐的运动。除了文本之外，该框架还可以自然地扩展到其他模态，如图像、音频和音乐，为视觉-语言-动作人形机器人系统提供了一个通用的基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.14952) | **Categories:** cs.RO, cs.CV

---
