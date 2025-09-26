# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-27

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (11)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning](https://arxiv.org/abs/2509.20754)
*Yufan Mao, Hanjing Ye, Wenlong Dong, Chengjie Zhang, Hong Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Navigating complex environments requires robots to effectively store observations as memories and leverage them to answer human queries about spatial locations, which is a critical yet underexplored research challenge. While prior work has made progress in constructing robotic memory, few have addressed the principled mechanisms needed for efficient memory retrieval and integration. To bridge this gap, we propose Meta-Memory, a large language model (LLM)-driven agent that constructs a high-density memory representation of the environment. The key innovation of Meta-Memory lies in its capacity to retrieve and integrate relevant memories through joint reasoning over semantic and spatial modalities in response to natural language location queries, thereby empowering robots with robust and accurate spatial reasoning capabilities. To evaluate its performance, we introduce SpaceLocQA, a large-scale dataset encompassing diverse real-world spatial question-answering scenarios. Experimental results show that Meta-Memory significantly outperforms state-of-the-art methods on both the SpaceLocQA and the public NaVQA benchmarks. Furthermore, we successfully deployed Meta-Memory on real-world robotic platforms, demonstrating its practical utility in complex environments. Project page: https://itsbaymax.github.io/meta-memory.github.io/ .

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20754) | **Categories:** cs.AI, cs.RO

---

### [2] [ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective](https://arxiv.org/abs/2509.21134)
*Yiwen Zhang, Ziang Chen, Fanqi Kong, Yizhe Huang, Xue Feng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21134) | **Categories:** cs.AI, cs.MA

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Beyond the Individual: Introducing Group Intention Forecasting with SHOT Dataset](https://arxiv.org/abs/2509.20715)
*Ruixu Zhang, Yuran Wang, Xinyi Hu, Chaoyu Mai, Wenxuan Liu, Danni Xu, Xian Zhong, Zheng Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Intention recognition has traditionally focused on individual intentions, overlooking the complexities of collective intentions in group settings. To address this limitation, we introduce the concept of group intention, which represents shared goals emerging through the actions of multiple individuals, and Group Intention Forecasting (GIF), a novel task that forecasts when group intentions will occur by analyzing individual actions and interactions before the collective goal becomes apparent. To investigate GIF in a specific scenario, we propose SHOT, the first large-scale dataset for GIF, consisting of 1,979 basketball video clips captured from 5 camera views and annotated with 6 types of individual attributes. SHOT is designed with 3 key characteristics: multi-individual information, multi-view adaptability, and multi-level intention, making it well-suited for studying emerging group intentions. Furthermore, we introduce GIFT (Group Intention ForecasTer), a framework that extracts fine-grained individual features and models evolving group dynamics to forecast intention emergence. Experimental results confirm the effectiveness of SHOT and GIFT, establishing a strong foundation for future research in group intention forecasting. The dataset is available at https://xinyi-hu.github.io/SHOT_DATASET.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20715) | **Categories:** cs.CV, cs.AI

---

### [2] [SimDiff: Simulator-constrained Diffusion Model for Physically Plausible Motion Generation](https://arxiv.org/abs/2509.20927)
*Akihisa Watanabe, Jiawei Ren, Li Siyao, Yichen Peng, Erwin Wu, Edgar Simo-Serra*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generating physically plausible human motion is crucial for applications such as character animation and virtual reality. Existing approaches often incorporate a simulator-based motion projection layer to the diffusion process to enforce physical plausibility. However, such methods are computationally expensive due to the sequential nature of the simulator, which prevents parallelization. We show that simulator-based motion projection can be interpreted as a form of guidance, either classifier-based or classifier-free, within the diffusion process. Building on this insight, we propose SimDiff, a Simulator-constrained Diffusion Model that integrates environment parameters (e.g., gravity, wind) directly into the denoising process. By conditioning on these parameters, SimDiff generates physically plausible motions efficiently, without repeated simulator calls at inference, and also provides fine-grained control over different physical coefficients. Moreover, SimDiff successfully generalizes to unseen combinations of environmental parameters, demonstrating compositional generalization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20927) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Beyond Visual Similarity: Rule-Guided Multimodal Clustering with explicit domain rules](https://arxiv.org/abs/2509.20501)
*Kishor Datta Gupta, Mohd Ariful Haque, Marufa Kamal, Ahmed Rafi Hasan, Md. Mahfuzur Rahman, Roy George*

Main category: cs.LG

TL;DR: DARTVAE 通过将领域知识规则融入变分自编码器，实现了更具语义和可解释性的聚类。


<details>
  <summary>Details</summary>
Motivation: 传统聚类技术仅依赖输入数据的相似性，无法捕捉领域相关的结构或语义约束。

Method: DARTVAE 将显式规则、语义表示和数据驱动特征嵌入到统一的潜在空间中，并通过损失函数中的规则一致性和违反惩罚来强制执行约束。

Result: 在飞机和汽车数据集上的实验表明，规则引导的聚类产生了更具操作意义和可解释性的聚类，同时提高了传统聚类指标。

Conclusion: DARTVAE 通过结合规则编码和学习到的表示，实现了比纯数据驱动模型更有意义和一致的聚类结果，突出了约束引导的多模态聚类在复杂、知识密集型环境中的效用。

Abstract: 传统的聚类技术通常只依赖于输入数据的相似性，这限制了它们在许多领域中捕捉结构或语义约束的能力。我们引入了领域感知规则触发变分自编码器 (DARTVAE)，这是一个规则引导的多模态聚类框架，它将特定领域的约束直接整合到表示学习过程中。DARTVAE 通过将显式规则、语义表示和数据驱动的特征嵌入到一个统一的潜在空间中，同时通过损失函数中的规则一致性和违反惩罚来强制执行约束，从而扩展了 VAE 架构。与仅依赖视觉相似性或将规则应用为事后过滤器的传统聚类方法不同，DARTVAE 将规则视为第一类学习信号。这些规则由 LLM 生成，构建成知识图谱，并通过结合了重构、KL 散度、一致性和违反惩罚的损失函数来强制执行。在飞机和汽车数据集上的实验表明，规则引导的聚类产生了更具操作意义和可解释性的聚类，例如，隔离无人机、统一隐形飞机或将 SUV 与轿车分离，同时提高了传统聚类指标。然而，该框架面临着一些挑战：LLM 生成的规则可能会产生幻觉或冲突，过多的规则有过度拟合的风险，并且扩展到复杂领域会增加计算和一致性方面的困难。通过将规则编码与学习到的表示相结合，DARTVAE 实现了比纯数据驱动模型更有意义和一致的聚类结果，突出了约束引导的多模态聚类在复杂、知识密集型环境中的效用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20501) | **Categories:** cs.LG, cs.CV

---

### [2] [CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](https://arxiv.org/abs/2509.20712)
*Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}ontrolling \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20712) | **Categories:** cs.LG, cs.CL

---

### [3] [CaTS-Bench: Can Language Models Describe Numeric Time Series?](https://arxiv.org/abs/2509.20823)
*Luca Zhou, Pratham Yashwante, Marshall Fisher, Alessio Sampieri, Zihao Zhou, Fabio Galasso, Rose Yu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time series captioning, the task of describing numeric time series in natural language, requires numerical reasoning, trend interpretation, and contextual understanding. Existing benchmarks, however, often rely on synthetic data or overly simplistic captions, and typically neglect metadata and visual representations. To close this gap, we introduce CaTS-Bench, the first large-scale, real-world benchmark for Context-aware Time Series captioning. CaTS-Bench is derived from 11 diverse datasets reframed as captioning and Q&A tasks, comprising roughly 465k training and 105k test timestamps. Each sample includes a numeric series segment, contextual metadata, a line-chart image, and a caption. A key contribution of this work is the scalable pipeline used to generate reference captions: while most references are produced by an oracle LLM and verified through factual checks, human indistinguishability studies, and diversity analyses, we also provide a human-revisited subset of 579 test captions, refined from LLM outputs to ensure accuracy and human-like style. Beyond captioning, CaTS-Bench offers 460 multiple-choice questions targeting deeper aspects of time series reasoning. We further propose new tailored evaluation metrics and benchmark leading VLMs, highlighting both their strengths and persistent limitations. Together, these contributions establish CaTS-Bench and its captioning pipeline as a reliable and extensible foundation for future research at the intersection of time series analysis and foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20823) | **Categories:** cs.LG, cs.AI, cs.CV

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting](https://arxiv.org/abs/2509.20499)
*Boqi Li, Siyuan Li, Weiyi Wang, Anran Li, Zhong Cao, Henry X. Liu*

Main category: cs.RO

TL;DR: 本文提出了一种结合航点预测器和多模态大型语言模型的零样本框架，用于解决连续环境下的视觉语言导航任务，并在R2R-CE和RxR-CE数据集上取得了最先进的零样本性能。


<details>
  <summary>Details</summary>
Motivation: 视觉语言导航（VLN）是具身智能体的关键任务，在连续环境中面临着自然语言理解、环境感知和低级动作规划的挑战。

Method: 该方法结合了一个简化的航点预测器和一个多模态大型语言模型（MLLM），航点预测器在抽象障碍物地图上生成线性可达的航点，并将其整合到动态更新的拓扑图中，通过编码空间结构和探索历史来促进探索和纠错。

Result: 在R2R-CE和RxR-CE数据集上取得了最先进的零样本性能，成功率分别为41%和36%，超过了之前的最先进方法。

Conclusion: 该研究提出的零样本框架有效地解决了连续环境下的视觉语言导航任务，并在实验中取得了显著的性能提升。

Abstract: 随着基础模型和机器人技术的飞速发展，视觉语言导航（VLN）已成为具身智能体的关键任务，具有广泛的实际应用。本文研究了连续环境下的VLN，这是一个特别具有挑战性的环境，智能体必须联合解释自然语言指令，感知其周围环境，并规划低级动作。我们提出了一个零样本框架，该框架集成了简化的但有效的航点预测器和多模态大型语言模型（MLLM）。该预测器在抽象障碍物地图上运行，生成线性可达的航点，这些航点被合并到具有显式访问记录的动态更新的拓扑图中。图和访问信息被编码到提示中，从而能够对空间结构和探索历史进行推理，以鼓励探索并使MLLM具备用于纠错的本地路径规划。在R2R-CE和RxR-CE上进行的大量实验表明，我们的方法实现了最先进的零样本性能，成功率分别为41％和36％，优于先前的最先进方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20499) | **Categories:** cs.RO, cs.AI

---

### [2] [Digital Twin-Guided Robot Path Planning: A Beta-Bernoulli Fusion with Large Language Model as a Sensor](https://arxiv.org/abs/2509.20709)
*Mani Amani, Reza Akhavian*

Main category: cs.RO

TL;DR: 该论文提出了一种新颖的框架，通过 Beta-Bernoulli 贝叶斯融合将自然语言指令与 BIM 导出的语义地图融合，从而引导机器人在建筑环境中沿更安全、更符合上下文的路径移动。


<details>
  <summary>Details</summary>
Motivation: 近年来，将自然语言 (NL) 提示集成到机器人任务规划中引起了人们的极大兴趣。在建筑领域，建筑信息模型 (BIM) 封装了丰富的环境 NL 描述。

Method: 该方法通过将 LLM 解释为传感器，将每个障碍物的设计时排斥系数视为 Beta(alpha, beta) 随机变量，并将 LLM 返回的危险分数作为伪计数来更新 alpha 和 beta，从而通过 Beta-Bernoulli 贝叶斯融合来融合 NL 指令和 BIM 导出的语义地图。

Result: 仿真结果表明，这种 Beta-Bernoulli 融合在路径鲁棒性和有效性方面产生了定性和定量的改进。

Conclusion: 通过根据从用户提示推断的情感和上下文调整增益，该方法引导机器人在更安全、更符合上下文的路径上移动。这提供了一种数值稳定的方法，可以链接来自建筑工人和工头的多个自然命令和提示，从而在规划的同时提供灵活性，以便集成到任何学习或经典 AI 框架中。

Abstract: 近年来，将自然语言 (NL) 提示集成到机器人任务规划中引起了人们的极大兴趣。在建筑领域，建筑信息模型 (BIM) 封装了丰富的环境 NL 描述。我们提出了一种新颖的框架，通过 Beta-Bernoulli 贝叶斯融合将 NL 指令与 BIM 导出的语义地图融合，方法是将 LLM 解释为传感器：每个障碍物的设计时排斥系数被视为 Beta(alpha, beta) 随机变量，并且 LLM 返回的危险分数被合并为伪计数以更新 alpha 和 beta。由此产生的后验均值产生一个连续的、上下文感知的排斥增益，该增益增强了基于欧几里德距离的势场以用于成本启发式。通过根据从用户提示推断的情感和上下文调整增益，我们的方法引导机器人在更安全、更符合上下文的路径上移动。这提供了一种数值稳定的方法，可以链接来自建筑工人和工头的多个自然命令和提示，从而在规划的同时提供灵活性，以便集成到任何学习或经典 AI 框架中。仿真结果表明，这种 Beta-Bernoulli 融合在路径鲁棒性和有效性方面产生了定性和定量的改进。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20709) | **Categories:** cs.RO

---

### [3] [Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation](https://arxiv.org/abs/2509.20623)
*Satyajeet Das, Darren Chiu, Zhehui Huang, Lars Lindemann, Gaurav S. Sukhatme*

Main category: cs.RO

TL;DR: 该论文提出了一种名为潜在激活编辑（LAE）的框架，用于在不修改预训练策略权重的情况下，在推理时改进多旋翼飞行器的导航安全性。


<details>
  <summary>Details</summary>
Motivation: 现有强化学习策略在复杂环境中易发生碰撞，重新训练成本高且可能降低已学技能。

Method: 该框架通过在线分类器检测不良行为状态，并使用激活编辑模块选择性地修改激活，从而引导策略转向更安全的模式。通过训练潜在碰撞世界模型来预测未来的预碰撞激活，从而促使更早和更谨慎的避障反应。

Result: 仿真和真实Crazyflie实验表明，LAE显著减少了碰撞（与未编辑的基线相比，累积碰撞减少了近90%），并大大增加了无碰撞轨迹的比例，同时保持了任务完成。

Conclusion: LAE是一种轻量级范例，适用于资源受限的硬件，可用于已学习机器人策略的部署后改进。

Abstract: 强化学习在复杂领域（如协调和导航多个四旋翼飞行器）中取得了显著进展。然而，即使是经过良好训练的策略，在障碍物丰富的环境中仍然容易发生碰撞。通过重新训练或微调来解决这些不常发生但至关重要的安全问题，成本高昂，并且有降低先前学习技能的风险。受到大型语言模型中的激活引导和计算机视觉中的潜在编辑的启发，我们引入了一种推理时潜在激活编辑（LAE）框架，该框架可以在不修改其权重或架构的情况下改进预训练策略的行为。该框架分两个阶段运行：（i）在线分类器监控中间激活，以检测与不良行为相关的状态，以及（ii）激活编辑模块，该模块选择性地修改标记的激活，以将策略转移到更安全的模式。在这项工作中，我们专注于提高多四旋翼飞行器导航的安全性。我们假设放大策略对风险的内部感知可以诱导更安全的行为。我们通过训练潜在碰撞世界模型来预测未来的预碰撞激活来实现这一想法，从而促使更早和更谨慎的避障反应。广泛的模拟和真实的Crazyflie实验表明，LAE在统计上显着减少了碰撞（与未编辑的基线相比，累积碰撞减少了近90％），并大大增加了无碰撞轨迹的比例，同时保持了任务完成。更广泛地说，我们的结果表明，LAE是一种轻量级范例，在资源受限的硬件上可行，可用于已学习机器人策略的部署后改进。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20623) | **Categories:** cs.RO

---

### [4] [Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation](https://arxiv.org/abs/2509.20681)
*Wei-Teng Chu, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20681) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [5] [Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework](https://arxiv.org/abs/2509.20705)
*Reza Akhavian, Mani Amani, Johannes Mootz, Robert Ashe, Behrad Beheshti*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20705) | **Categories:** cs.RO

---

### [6] [SLAM-Free Visual Navigation with Hierarchical Vision-Language Perception and Coarse-to-Fine Semantic Topological Planning](https://arxiv.org/abs/2509.20739)
*Guoyang Zhao, Yudong Li, Weiqing Qi, Kai Zhang, Bonan Liu, Kai Chen, Haoang Li, Jun Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Conventional SLAM pipelines for legged robot navigation are fragile under rapid motion, calibration demands, and sensor drift, while offering limited semantic reasoning for task-driven exploration. To deal with these issues, we propose a vision-only, SLAM-free navigation framework that replaces dense geometry with semantic reasoning and lightweight topological representations. A hierarchical vision-language perception module fuses scene-level context with object-level cues for robust semantic inference. And a semantic-probabilistic topological map supports coarse-to-fine planning: LLM-based global reasoning for subgoal selection and vision-based local planning for obstacle avoidance. Integrated with reinforcement-learning locomotion controllers, the framework is deployable across diverse legged robot platforms. Experiments in simulation and real-world settings demonstrate consistent improvements in semantic accuracy, planning quality, and navigation success, while ablation studies further showcase the necessity of both hierarchical perception and fine local planning. This work introduces a new paradigm for SLAM-free, vision-language-driven navigation, shifting robotic exploration from geometry-centric mapping to semantics-driven decision making.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20739) | **Categories:** cs.RO, cs.CV

---

### [7] [MTRDrive: Memory-Tool Synergistic Reasoning for Robust Autonomous Driving in Corner Cases](https://arxiv.org/abs/2509.20843)
*Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, Diange Yang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language Models(VLMs) have demonstrated significant potential for end-to-end autonomous driving, yet a substantial gap remains between their current capabilities and the reliability necessary for real-world deployment. A critical challenge is their fragility, characterized by hallucinations and poor generalization in out-of-distribution (OOD) scenarios. To bridge this gap, we introduce MTRDrive, a novel framework that integrates procedural driving experiences with a dynamic toolkit to enhance generalization and proactive decision-making.   MTRDrive addresses these limitations through a closed-loop system that combines a memory-based experience retrieval mechanism with dynamic toolkits. This synergy enables the model to interact more effectively with its environment, improving both reasoning and decision-making capabilities with the help of our memory-tool synergistic reasoning. Additionally, we introduce a new benchmark based on complex Roadwork construction scenarios to rigorously evaluate zero-shot generalization.   Extensive experiments demonstrate the superior effectiveness of our approach. On the public NAVSIM benchmark, our 3B-parameter MTRDrive model achieves an exceptional PDMS of 88.3 without chain-of-thought and sets a state-of-the-art performance bar on high-level planning, with a driving metric score of 79.8\% and a planning accuracy of 82.6\%. Rigorous zero-shot evaluation on the new Roadwork-VLM benchmark shows a strong ability to reason robustly in unseen scenarios, achieving a driving metric score of 80.2\%. These results highlight MTRDrive's potential to advance autonomous driving toward safer and more reliable systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20843) | **Categories:** cs.RO

---

### [8] [Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement](https://arxiv.org/abs/2509.20938)
*Jianbo Zhao, Taiyu Ban, Xiangjie Li, Xingtai Gui, Hangning Zhou, Lei Liu, Hongwei Zhao, Bin Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The inherent sequential modeling capabilities of autoregressive models make them a formidable baseline for end-to-end planning in autonomous driving. Nevertheless, their performance is constrained by a spatio-temporal misalignment, as the planner must condition future actions on past sensory data. This creates an inconsistent worldview, limiting the upper bound of performance for an otherwise powerful approach. To address this, we propose a Time-Invariant Spatial Alignment (TISA) module that learns to project initial environmental features into a consistent ego-centric frame for each future time step, effectively correcting the agent's worldview without explicit future scene prediction. In addition, we employ a kinematic action prediction head (i.e., acceleration and yaw rate) to ensure physically feasible trajectories. Finally, we introduce a multi-objective post-training stage using Direct Preference Optimization (DPO) to move beyond pure imitation. Our approach provides targeted feedback on specific driving behaviors, offering a more fine-grained learning signal than the single, overall objective used in standard DPO. Our model achieves a state-of-the-art 89.8 PDMS on the NAVSIM dataset among autoregressive models. The video document is available at https://tisa-dpo-e2e.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20938) | **Categories:** cs.RO, cs.CV

---

### [9] [KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models](https://arxiv.org/abs/2509.21027)
*Sibo Li, Qianyue Hao, Yu Shang, Yong Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic world models are a promising paradigm for forecasting future environment states, yet their inference speed and the physical plausibility of generated trajectories remain critical bottlenecks, limiting their real-world applications. This stems from the redundancy of the prevailing frame-to-frame generation approach, where the model conducts costly computation on similar frames, as well as neglecting the semantic importance of key transitions. To address this inefficiency, we propose KeyWorld, a framework that improves text-conditioned robotic world models by concentrating transformers computation on a few semantic key frames while employing a lightweight convolutional model to fill the intermediate frames. Specifically, KeyWorld first identifies significant transitions by iteratively simplifying the robot's motion trajectories, obtaining the ground truth key frames. Then, a DiT model is trained to reason and generate these physically meaningful key frames from textual task descriptions. Finally, a lightweight interpolator efficiently reconstructs the full video by inpainting all intermediate frames. Evaluations on the LIBERO benchmark demonstrate that KeyWorld achieves a 5.68$\times$ acceleration compared to the frame-to-frame generation baseline, and focusing on the motion-aware key frames further contributes to the physical validity of the generated videos, especially on complex tasks. Our approach highlights a practical path toward deploying world models in real-time robotic control and other domains requiring both efficient and effective world models. Code is released at https://anonymous.4open.science/r/Keyworld-E43D.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21027) | **Categories:** cs.RO, cs.CV

---

### [10] [Cross-Modal Instructions for Robot Motion Generation](https://arxiv.org/abs/2509.21107)
*William Barron, Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Teaching robots novel behaviors typically requires motion demonstrations via teleoperation or kinaesthetic teaching, that is, physically guiding the robot. While recent work has explored using human sketches to specify desired behaviors, data collection remains cumbersome, and demonstration datasets are difficult to scale. In this paper, we introduce an alternative paradigm, Learning from Cross-Modal Instructions, where robots are shaped by demonstrations in the form of rough annotations, which can contain free-form text labels, and are used in lieu of physical motion. We introduce the CrossInstruct framework, which integrates cross-modal instructions as examples into the context input to a foundational vision-language model (VLM). The VLM then iteratively queries a smaller, fine-tuned model, and synthesizes the desired motion over multiple 2D views. These are then subsequently fused into a coherent distribution over 3D motion trajectories in the robot's workspace. By incorporating the reasoning of the large VLM with a fine-grained pointing model, CrossInstruct produces executable robot behaviors that generalize beyond the environment of in the limited set of instruction examples. We then introduce a downstream reinforcement learning pipeline that leverages CrossInstruct outputs to efficiently learn policies to complete fine-grained tasks. We rigorously evaluate CrossInstruct on benchmark simulation tasks and real hardware, demonstrating effectiveness without additional fine-tuning and providing a strong initialization for policies subsequently refined via reinforcement learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21107) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [11] [Human-like Navigation in a World Built for Humans](https://arxiv.org/abs/2509.21189)
*Bhargav Chandaka, Gloria X. Wang, Haozhe Chen, Henry Che, Albert J. Zhai, Shenlong Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: When navigating in a man-made environment they haven't visited before--like an office building--humans employ behaviors such as reading signs and asking others for directions. These behaviors help humans reach their destinations efficiently by reducing the need to search through large areas. Existing robot navigation systems lack the ability to execute such behaviors and are thus highly inefficient at navigating within large environments. We present ReasonNav, a modular navigation system which integrates these human-like navigation skills by leveraging the reasoning capabilities of a vision-language model (VLM). We design compact input and output abstractions based on navigation landmarks, allowing the VLM to focus on language understanding and reasoning. We evaluate ReasonNav on real and simulated navigation tasks and show that the agent successfully employs higher-order reasoning to navigate efficiently in large, complex buildings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21189) | **Categories:** cs.RO, cs.AI, cs.CV

---
