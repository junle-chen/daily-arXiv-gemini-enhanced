# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-20

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算语言学 (Computation and Language) (1)](#cs-cl)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (7)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [(P)rior(D)yna(F)low: A Priori Dynamic Workflow Construction via Multi-Agent Collaboration](https://arxiv.org/abs/2509.14547)
*Yi Lin, Lujin Zhao, Yijie Shi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent studies have shown that carefully designed workflows coordinating large language models(LLMs) significantly enhance task-solving capabilities compared to using a single model. While an increasing number of works focus on autonomous workflow construction, most existing approaches rely solely on historical experience, leading to limitations in efficiency and adaptability. We argue that while historical experience is valuable, workflow construction should also flexibly respond to the unique characteristics of each task. To this end, we propose an a priori dynamic framework for automated workflow construction. Our framework first leverages Q-table learning to optimize the decision space, guiding agent decisions and enabling effective use of historical experience. At the same time, agents evaluate the current task progress and make a priori decisions regarding the next executing agent, allowing the system to proactively select the more suitable workflow structure for each given task. Additionally, we incorporate mechanisms such as cold-start initialization, early stopping, and pruning to further improve system efficiency. Experimental evaluations on four benchmark datasets demonstrate the feasibility and effectiveness of our approach. Compared to state-of-the-art baselines, our method achieves an average improvement of 4.05%, while reducing workflow construction and inference costs to only 30.68%-48.31% of those required by existing methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14547) | **Categories:** cs.AI

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
*Hai Huang, Yann LeCun, Randall Balestriero*

Main category: cs.CL

TL;DR: 本文提出了LLM-JEPA，一种基于联合嵌入预测架构（JEPA）的大语言模型训练方法，它在预训练和微调方面均优于标准LLM训练目标。


<details>
  <summary>Details</summary>
Motivation: 现有大语言模型训练依赖于输入空间重建和生成能力，但视觉领域的经验表明，嵌入空间训练目标（如JEPA）更优越。本文旨在探索是否能将视觉领域的训练技巧应用于语言模型，并解决为语言模型设计JEPA式目标的挑战。

Method: 本文提出了LLM-JEPA，一种基于JEPA的大语言模型解决方案，适用于微调和预训练。

Result: LLM-JEPA在多个数据集（NL-RX, GSM8K, Spider, RottenTomatoes）和多种模型（Llama3, OpenELM, Gemma2, Olmo）上均显著优于标准LLM训练目标，并且具有更强的鲁棒性，不易过拟合。

Conclusion: LLM-JEPA是一种有效的大语言模型训练方法，通过在嵌入空间进行训练，可以显著提高模型性能并增强鲁棒性。

Abstract: 大型语言模型（LLM）的预训练、微调和评估依赖于输入空间重建和生成能力。然而，在视觉领域已经观察到，嵌入空间训练目标，例如使用联合嵌入预测架构（JEPA），远优于其输入空间对应方法。语言和视觉之间训练方式的这种不匹配引出了一个自然的问题：{\em 语言训练方法能否从视觉方法中学习一些技巧？} 缺乏JEPA风格的LLM证明了为语言设计此类目标的挑战。在这项工作中，我们提出了朝这个方向迈出的第一步，我们开发了LLM-JEPA，这是一种基于JEPA的LLM解决方案，既适用于微调也适用于预训练。到目前为止，LLM-JEPA能够在模型中显著优于标准LLM训练目标，同时对过度拟合具有鲁棒性。这些发现在众多数据集（NL-RX、GSM8K、Spider、RottenTomatoes）和来自Llama3、OpenELM、Gemma2和Olmo系列的各种模型中观察到。代码：https://github.com/rbalestr-lab/llm-jepa。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14252) | **Categories:** cs.CL, cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DiffVL: Diffusion-Based Visual Localization on 2D Maps via BEV-Conditioned GPS Denoising](https://arxiv.org/abs/2509.14565)
*Li Gao, Hongyang Sun, Liu Liu, Yunhao Li, Yang Cai*

Main category: cs.CV

TL;DR: 该论文提出DiffVL，一个利用扩散模型将视觉定位重新定义为GPS去噪任务的框架，实现了在标准地图上亚米级的定位精度。


<details>
  <summary>Details</summary>
Motivation: 现有视觉定位方法依赖于高精地图，但其构建和维护成本高昂，限制了可扩展性。而基于标准地图的方法忽略了普遍存在的噪声GPS信号。该论文旨在解决在标准地图上利用噪声GPS实现高精度视觉定位的问题。

Method: 该论文提出DiffVL框架，该框架利用扩散模型，通过视觉BEV特征和标准地图对噪声GPS轨迹进行条件处理，隐式地编码真实姿态分布，并通过迭代扩散细化来恢复真实姿态。

Result: 在多个数据集上的实验表明，DiffVL方法相比于BEV匹配的基线方法，实现了最先进的精度，并且可以在不依赖高精地图的情况下实现亚米级的定位精度。

Conclusion: 该论文证明了扩散模型可以通过将噪声GPS作为生成先验，实现可扩展的定位，从而实现了从传统的基于匹配的方法的范式转变。

Abstract: 精确的视觉定位对于自动驾驶至关重要，但现有方法面临一个根本的困境：虽然高精地图提供了高精度的定位参考，但其高昂的构建和维护成本限制了可扩展性，这推动了对像OpenStreetMap这样的标准地图的研究。目前基于标准地图的方法主要侧重于图像和地图之间的鸟瞰图（BEV）匹配，忽略了普遍存在的信号噪声GPS。虽然GPS很容易获得，但它在城市环境中会受到多径误差的影响。我们提出了DiffVL，这是第一个使用扩散模型将视觉定位重新定义为GPS去噪任务的框架。我们的关键见解是，当以视觉BEV特征和标准地图为条件时，噪声GPS轨迹隐式地编码了真实的姿态分布，可以通过迭代扩散细化来恢复。与先前的BEV匹配方法（例如，OrienterNet）或基于Transformer的配准方法不同，DiffVL通过联合建模GPS、标准地图和视觉信号来学习反转GPS噪声扰动，从而在不依赖高精地图的情况下实现亚米级的精度。在多个数据集上的实验表明，与BEV匹配的基线方法相比，我们的方法实现了最先进的精度。至关重要的是，我们的工作证明了扩散模型可以通过将噪声GPS视为生成先验来实现可扩展的定位，从而实现了从传统的基于匹配的方法的范式转变。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14565) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [STEP: Structured Training and Evaluation Platform for benchmarking trajectory prediction models](https://arxiv.org/abs/2509.14801)
*Julian F. Schumann, Anna Mészáros, Jens Kober, Arkady Zgonnikov*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While trajectory prediction plays a critical role in enabling safe and effective path-planning in automated vehicles, standardized practices for evaluating such models remain underdeveloped. Recent efforts have aimed to unify dataset formats and model interfaces for easier comparisons, yet existing frameworks often fall short in supporting heterogeneous traffic scenarios, joint prediction models, or user documentation. In this work, we introduce STEP -- a new benchmarking framework that addresses these limitations by providing a unified interface for multiple datasets, enforcing consistent training and evaluation conditions, and supporting a wide range of prediction models. We demonstrate the capabilities of STEP in a number of experiments which reveal 1) the limitations of widely-used testing procedures, 2) the importance of joint modeling of agents for better predictions of interactions, and 3) the vulnerability of current state-of-the-art models against both distribution shifts and targeted attacks by adversarial agents. With STEP, we aim to shift the focus from the ``leaderboard'' approach to deeper insights about model behavior and generalization in complex multi-agent settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14801) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [AEGIS: Automated Error Generation and Identification for Multi-Agent Systems](https://arxiv.org/abs/2509.14295)
*Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng*

Main category: cs.RO

TL;DR: 本文提出了AEGIS，一个用于自动生成和识别多智能体系统错误的新框架，通过系统地注入可控和可追踪的错误来创建丰富的故障数据集。


<details>
  <summary>Details</summary>
Motivation: 多智能体系统（MAS）变得越来越自主和复杂，理解它们的错误模式对于确保其可靠性和安全性至关重要。然而，缺乏具有精确的、带有ground-truth错误标签的大规模多样化数据集严重阻碍了这方面的研究。

Method: 通过将可控和可追踪的错误系统地注入到最初成功的轨迹中，使用基于LLM的上下文感知自适应操纵器执行诸如prompt注入和响应损坏等复杂攻击，以诱导特定的、预定义的错误模式，从而创建一个丰富的真实故障数据集。

Result: 通过探索用于错误识别任务的三种不同的学习范式：监督微调、强化学习和对比学习，证明了数据集的价值。综合实验表明，在AEGIS数据上训练的模型在所有三种学习范式中都取得了显著的改进。值得注意的是，我们的一些微调模型的性能与大一个数量级的专有系统相比具有竞争力或更优越。

Conclusion: 本文验证了自动数据生成框架是开发更健壮和可解释的多智能体系统的关键资源。

Abstract: 随着多智能体系统（MAS）变得越来越自主和复杂，理解它们的错误模式对于确保其可靠性和安全性至关重要。然而，缺乏具有精确的、带有ground-truth错误标签的大规模多样化数据集严重阻碍了这方面的研究。为了解决这个瓶颈，我们介绍了AEGIS，这是一个用于多智能体系统的自动错误生成和识别的新框架。通过将可控和可追踪的错误系统地注入到最初成功的轨迹中，我们创建了一个丰富的真实故障数据集。这是通过使用基于LLM的上下文感知自适应操纵器来实现的，该操纵器执行诸如prompt注入和响应损坏等复杂攻击，以诱导特定的、预定义的错误模式。我们通过探索用于错误识别任务的三种不同的学习范式：监督微调、强化学习和对比学习，证明了我们数据集的价值。我们全面的实验表明，在AEGIS数据上训练的模型在所有三种学习范式中都取得了显著的改进。值得注意的是，我们的一些微调模型的性能与大一个数量级的专有系统相比具有竞争力或更优越，从而验证了我们的自动数据生成框架是开发更健壮和可解释的多智能体系统的关键资源。我们的项目网站是https://kfq20.github.io/AEGIS-Website。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14295) | **Categories:** cs.RO

---

### [2] [CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks](https://arxiv.org/abs/2509.14380)
*Seoyeon Choi, Kanghyun Ryu, Jonghoon Ock, Negar Mehr*

Main category: cs.RO

TL;DR: CRAFT框架利用大型语言模型分解复杂多机器人协作任务，并通过视觉语言模型优化奖励函数，从而实现高效的多智能体强化学习。


<details>
  <summary>Details</summary>
Motivation: 多智能体强化学习应用于机器人领域面临高维连续动作空间、复杂奖励设计和非平稳环境等挑战；人类通过分阶段课程学习复杂协作，受此启发，CRAFT框架旨在利用大型模型作为“教练”来指导多机器人协作。

Method: CRAFT框架首先使用大型语言模型将长时程协作任务分解为子任务序列，然后使用大型语言模型生成奖励函数来训练每个子任务，并通过视觉语言模型引导的奖励优化循环来改进它们。

Result: 在多足机器人导航和双臂操作任务上的评估表明，CRAFT框架能够学习复杂的协作行为；此外，还在真实硬件实验中验证了多足机器人导航策略。

Conclusion: CRAFT框架利用大型模型分解任务和优化奖励，为多智能体强化学习在机器人领域的应用提供了一种有效方法。

Abstract: 多智能体强化学习（MARL）为学习多智能体系统中的协作提供了一个强大的框架。然而，由于高维连续联合动作空间、复杂的奖励设计以及分散设置中固有的非平稳转换，将MARL应用于机器人仍然具有挑战性。另一方面，人类通过分阶段的课程学习复杂的协作，其中长期的行为是逐步建立在更简单的技能之上的。受此启发，我们提出了CRAFT：使用基础模型自主进行教练式强化学习，用于多机器人协作任务，该框架利用基础模型的推理能力来充当多机器人协作的“教练”。CRAFT利用大型语言模型（LLM）的规划能力，自动将长时程协作任务分解为子任务序列。接下来，CRAFT使用LLM生成的奖励函数训练每个子任务，并通过视觉语言模型（VLM）引导的奖励优化循环来改进它们。我们在多足机器人导航和双臂操作任务上评估CRAFT，证明了它学习复杂协作行为的能力。此外，我们在真实的硬件实验中验证了多足机器人导航策略。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14380) | **Categories:** cs.RO

---

### [3] [SimCoachCorpus: A naturalistic dataset with language and trajectories for embodied teaching](https://arxiv.org/abs/2509.14548)
*Emily Sumner, Deepak E. Gopinath, Laporsha Dees, Patricio Reyes Gomez, Xiongyi Cui, Andrew Silva, Jean Costa, Allison Morgan, Mariah Schrum, Tiffany L. Chen, Avinash Balachandran, Guy Rosman*

Main category: cs.RO

TL;DR: 本文提出了 SimCoachCorpus 数据集，该数据集包含赛车模拟器驾驶数据，旨在研究在指导和非指导下运动技能习得过程中的交互现象。


<details>
  <summary>Details</summary>
Motivation: 缺乏能够捕捉人们如何通过口头指导随时间推移获得具身技能的数据集。

Method: 收集了 29 名人类在赛车模拟器中驾驶的数据，其中 15 名参与者接受了专业驾驶教练的一对一指导，14 名参与者在没有指导的情况下驾驶。数据集包含车辆状态、输入、地图和锥形地标等具身特征，并与专业教练的同步口头指导以及每圈结束时的额外反馈相结合。此外，还提供了每个并发反馈话语的指导类别注释、学生对指导建议的依从性评级以及参与者的自我报告认知负荷和情绪状态。

Result: 该数据集包含超过 20,000 条并发反馈话语、超过 400 条最终反馈话语以及超过 40 小时的车辆驾驶数据。通过在上下文学习、模仿学习和主题建模中的应用展示了该数据集的有效性。

Conclusion: SimCoachCorpus 数据集可用于研究运动学习动力学、探索语言现象以及训练教学计算模型。

Abstract: 为了弥补语言和物理行为深度交织的领域中缺乏数据集的问题，本文介绍了 SimCoachCorpus：一个独特的赛车模拟器驾驶数据集，用于研究在指导和非指导下运动技能习得过程中的丰富交互现象。该数据集包含 29 名人类在赛车模拟器中驾驶的数据，其中 15 名参与者接受了专业驾驶教练的一对一指导，14 名参与者在没有指导的情况下驾驶。数据集包含车辆状态、输入、地图和锥形地标等具身特征，并与专业教练的同步口头指导以及每圈结束时的额外反馈相结合。此外，还提供了每个并发反馈话语的指导类别注释、学生对指导建议的依从性评级以及参与者的自我报告认知负荷和情绪状态。该数据集包含超过 20,000 条并发反馈话语、超过 400 条最终反馈话语以及超过 40 小时的车辆驾驶数据。该数据集可用于研究运动学习动力学、探索语言现象以及训练教学计算模型。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14548) | **Categories:** cs.RO, cs.HC

---

### [4] [FlowDrive: Energy Flow Field for End-to-End Autonomous Driving](https://arxiv.org/abs/2509.14303)
*Hao Jiang, Zhipeng Zhang, Yu Gao, Zhigang Sun, Yiru Wang, Yuwen Heng, Shuo Wang, Jinhao Chai, Zhuo Chen, Hao Zhao, Hao Sun, Xi Zhang, Anqing Jiang, Chuan Hu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in end-to-end autonomous driving leverage multi-view images to construct BEV representations for motion planning. In motion planning, autonomous vehicles need considering both hard constraints imposed by geometrically occupied obstacles (e.g., vehicles, pedestrians) and soft, rule-based semantics with no explicit geometry (e.g., lane boundaries, traffic priors). However, existing end-to-end frameworks typically rely on BEV features learned in an implicit manner, lacking explicit modeling of risk and guidance priors for safe and interpretable planning. To address this, we propose FlowDrive, a novel framework that introduces physically interpretable energy-based flow fields-including risk potential and lane attraction fields-to encode semantic priors and safety cues into the BEV space. These flow-aware features enable adaptive refinement of anchor trajectories and serve as interpretable guidance for trajectory generation. Moreover, FlowDrive decouples motion intent prediction from trajectory denoising via a conditional diffusion planner with feature-level gating, alleviating task interference and enhancing multimodal diversity. Experiments on the NAVSIM v2 benchmark demonstrate that FlowDrive achieves state-of-the-art performance with an EPDMS of 86.3, surpassing prior baselines in both safety and planning quality. The project is available at https://astrixdrive.github.io/FlowDrive.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14303) | **Categories:** cs.RO, cs.AI

---

### [5] [GestOS: Advanced Hand Gesture Interpretation via Large Language Models to control Any Type of Robot](https://arxiv.org/abs/2509.14412)
*Artem Lykov, Oleg Kobzarev, Dzmitry Tsetserukou*

Main category: cs.RO

TL;DR: GestOS提出了一种基于手势的操作系统，通过LLM推理实现对异构机器人团队的语义化和动态任务分配。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在解决现有手势控制系统无法有效控制异构机器人团队的问题，这些系统通常将手势映射到固定命令或单智能体动作，缺乏灵活性和适应性。

Method: 该系统结合了轻量级视觉感知和大型语言模型（LLM）推理，将手势转换为结构化文本描述，LLM用于推断意图并生成机器人特定命令。机器人选择模块确保每个手势触发的任务与最合适的代理实时匹配。

Result: GestOS实现了上下文感知和自适应控制，无需用户明确指定目标或命令。

Conclusion: GestOS通过将手势交互从识别提升到智能编排，支持在动态环境中与机器人系统进行可扩展、灵活且用户友好的协作。

Abstract: 我们提出了GestOS，一个基于手势的操作系统，用于对异构机器人团队进行高级控制。与先前将手势映射到固定命令或单智能体动作的系统不同，GestOS在语义上解释手势，并根据其能力、当前状态和支持的指令集在多个机器人之间动态分配任务。该系统结合了轻量级视觉感知与大型语言模型（LLM）推理：手部姿势被转换为结构化文本描述，LLM使用这些描述来推断意图并生成机器人特定的命令。机器人选择模块确保每个手势触发的任务在实时匹配到最合适的代理。这种架构实现了上下文感知、自适应控制，而无需用户明确指定目标或命令。通过将手势交互从识别提升到智能编排，GestOS支持在动态环境中与机器人系统进行可扩展、灵活且用户友好的协作。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14412) | **Categories:** cs.RO

---

### [6] [Rethinking Reference Trajectories in Agile Drone Racing: A Unified Reference-Free Model-Based Controller via MPPI](https://arxiv.org/abs/2509.14726)
*Fangguo Zhao, Xin Guan, Shuo Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While model-based controllers have demonstrated remarkable performance in autonomous drone racing, their performance is often constrained by the reliance on pre-computed reference trajectories. Conventional approaches, such as trajectory tracking, demand a dynamically feasible, full-state reference, whereas contouring control relaxes this requirement to a geometric path but still necessitates a reference. Recent advancements in reinforcement learning (RL) have revealed that many model-based controllers optimize surrogate objectives, such as trajectory tracking, rather than the primary racing goal of directly maximizing progress through gates. Inspired by these findings, this work introduces a reference-free method for time-optimal racing by incorporating this gate progress objective, derived from RL reward shaping, directly into the Model Predictive Path Integral (MPPI) formulation. The sampling-based nature of MPPI makes it uniquely capable of optimizing the discontinuous and non-differentiable objective in real-time. We also establish a unified framework that leverages MPPI to systematically and fairly compare three distinct objective functions with a consistent dynamics model and parameter set: classical trajectory tracking, contouring control, and the proposed gate progress objective. We compare the performance of these three objectives when solved via both MPPI and a traditional gradient-based solver. Our results demonstrate that the proposed reference-free approach achieves competitive racing performance, rivaling or exceeding reference-based methods. Videos are available at https://zhaofangguo.github.io/racing_mppi/

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14726) | **Categories:** cs.RO

---

### [7] [CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids](https://arxiv.org/abs/2509.14935)
*Punith Reddy Vanteddu, Davide Gorbani, Giuseppe L'Erario, Hosameldin Awadalla Omer Mohamed, Fabio Bergonti, Daniele Pucci*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a CAD-driven co-design framework for optimizing jet-powered aerial humanoid robots to execute dynamically constrained trajectories. Starting from the iRonCub-Mk3 model, a Design of Experiments (DoE) approach is used to generate 5,000 geometrically varied and mechanically feasible designs by modifying limb dimensions, jet interface geometry (e.g., angle and offset), and overall mass distribution. Each model is constructed through CAD assemblies to ensure structural validity and compatibility with simulation tools. To reduce computational cost and enable parameter sensitivity analysis, the models are clustered using K-means, with representative centroids selected for evaluation. A minimum-jerk trajectory is used to assess flight performance, providing position and velocity references for a momentum-based linearized Model Predictive Control (MPC) strategy. A multi-objective optimization is then conducted using the NSGA-II algorithm, jointly exploring the space of design centroids and MPC gain parameters. The objectives are to minimize trajectory tracking error and mechanical energy expenditure. The framework outputs a set of flight-ready humanoid configurations with validated control parameters, offering a structured method for selecting and implementing feasible aerial humanoid designs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14935) | **Categories:** cs.RO

---
