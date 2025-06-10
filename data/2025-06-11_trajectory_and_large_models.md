# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-11

## 目录

- [人工智能 (Artificial Intelligence) (9)](#cs-ai)
- [计算语言学 (Computation and Language) (3)](#cs-cl)
- [计算机视觉 (Computer Vision) (15)](#cs-cv)
- [cs.CY (2)](#cs-cy)
- [cs.DC (1)](#cs-dc)
- [cs.GR (1)](#cs-gr)
- [机器学习 (Machine Learning) (9)](#cs-lg)
- [cs.MA (2)](#cs-ma)
- [机器人学 (Robotics) (18)](#cs-ro)
- [统计机器学习 (Machine Learning Statistics) (1)](#stat-ml)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Large Language Models and Their Applications in Roadway Safety and Mobility Enhancement: A Comprehensive Review](https://arxiv.org/abs/2506.06301)
*Muhammad Monjurul Karim, Yan Shi, Shucheng Zhang, Bingzhang Wang, Mehrdad Nasri, Yinhai Wang*

Main category: cs.AI

TL;DR: 本文综述了大型语言模型（LLM）在道路安全和移动性方面的应用，强调了其潜力和挑战，并提出了未来的研究方向。


<details>
  <summary>Details</summary>
Motivation: 现代交通系统面临道路安全和移动性的挑战，需要创新的分析框架。

Method: 通过架构、训练、提示和多模态策略调整LLM，以弥合与交通独特的时空和物理数据的“模态差距”。

Result: LLM在交通流量预测、信号控制、碰撞分析和驾驶员行为评估等多个应用中表现出潜力。

Conclusion: LLMs在交通安全和移动性方面具有变革潜力，但需要负责任的创新。

Abstract: 道路安全和移动性仍然是现代交通系统面临的关键挑战，需要能够解决复杂、动态和异构环境的创新分析框架。虽然传统的工程方法取得了一些进展，但现实世界交通的复杂性和动态性需要更先进的分析框架。大型语言模型（LLM）在自然语言理解、知识整合和推理方面具有前所未有的能力，代表了一种有希望的范式转变。本文全面回顾了LLM在增强道路安全和移动性方面的应用和定制。一个关键的重点是如何通过架构、训练、提示和多模态策略来调整LLM，以弥合与交通独特的时空和物理数据的“模态差距”。该综述系统地分析了LLM在移动性（例如，交通流量预测、信号控制）和安全（例如，碰撞分析、驾驶员行为评估）方面的各种应用。还研究了诸如V2X集成、特定领域的基础模型、可解释性框架和边缘计算等使能技术。尽管潜力巨大，但在固有的LLM局限性（幻觉、推理缺陷）、数据治理（隐私、偏见）、部署复杂性（sim-to-real、延迟）和严格的安全保证方面仍然存在挑战。强调了有希望的未来研究方向，包括先进的多模态融合、增强的时空推理、人机协作、持续学习以及高效、可验证系统的开发。本综述提供了当前能力、局限性和机遇的结构化路线图，强调了LLM的变革潜力，同时强调需要负责任的创新，以实现更安全、更智能的交通系统。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06301) | **Categories:** cs.AI

---

### [2] [WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making](https://arxiv.org/abs/2506.06725)
*Guillaume Levy, Cedric Colas, Pierre-Yves Oudeyer, Thomas Carta, Clement Romac*

Main category: cs.AI

TL;DR: WorldLLM框架结合贝叶斯推理和强化学习，提升了大型语言模型在特定环境中的预测能力，并能生成可解释的环境动态理论。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）具有一般的世界知识，但通常难以在结构化的、特定领域的环境（如模拟）中生成精确的预测。这些限制源于它们无法将广泛的、非结构化的理解置于特定的环境中。

Method: 该论文提出了一种名为WorldLLM的框架，该框架结合了贝叶斯推理和自主主动探索与强化学习，通过在提示中提供自然语言假设，利用LLM的上下文学习能力来指导基于LLM的世界模型的预测。通过贝叶斯推理框架迭代地完善这些假设，该框架利用第二个LLM作为给定收集证据的提议分布。使用基于好奇心驱动的强化学习策略来收集证据，该策略探索环境以找到在当前假设下基于LLM的预测模型下具有低对数似然的转换。

Result: 实验表明，WorldLLM在需要智能体操纵和组合对象的文本游戏环境中表现出有效性。该框架不仅提高了预测准确性，还生成了人类可解释的环境动态理论。

Conclusion: 该框架通过迭代优化假设和收集新证据，自主地持续改进预测，并在文本游戏环境中表现出有效性，能够提高预测准确性，并生成人类可解释的环境动态理论。

Abstract: 大型语言模型（LLM）拥有广泛的世界知识，但通常难以在结构化的、特定领域的环境（如模拟）中生成精确的预测。这些局限性源于它们无法将它们广阔的、非结构化的理解扎根于特定的环境。为了解决这个问题，我们提出了WorldLLM，该框架通过结合贝叶斯推理和自主主动探索与强化学习来增强基于LLM的世界建模。WorldLLM利用LLM的上下文学习能力，通过在提示中提供的自然语言假设来指导基于LLM的世界模型的预测。通过贝叶斯推理框架迭代地完善这些假设，该框架利用第二个LLM作为给定收集证据的提议分布。使用基于好奇心驱动的强化学习策略来收集证据，该策略探索环境以找到在当前假设下基于LLM的预测模型下具有低对数似然的转换。通过在完善假设和收集新证据之间交替，我们的框架自主地驱动预测的持续改进。我们的实验证明了WorldLLM在需要智能体操纵和组合对象的文本游戏环境中的有效性。该框架不仅提高了预测准确性，还生成了人类可解释的环境动态理论。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06725) | **Categories:** cs.AI, cs.LG

---

### [3] [Causal Graph based Event Reasoning using Semantic Relation Experts](https://arxiv.org/abs/2506.06910)
*Mahnaz Koupaee, Xueying Bai, Mudan Chen, Greg Durrett, Nathanael Chambers, Niranjan Balasubramanian*

Main category: cs.AI

TL;DR: 本文提出了一种协作式因果图生成方法，该方法无需在下游任务上进行微调，即可在预测和未来事件预测任务上实现与最先进模型相媲美的结果。


<details>
  <summary>Details</summary>
Motivation: LLMs struggle to accurately identify causal connections between events, leading to poor performance on deeper reasoning tasks like event forecasting and timeline understanding.

Method: A collaborative approach to causal graph generation is proposed, where LLMs simulate experts focusing on specific semantic relations, engaging in discussions consolidated by a final expert.

Result: The approach achieves competitive results with state-of-the-art models on forecasting and next event prediction tasks and introduces a new explainable event prediction task.

Conclusion: The proposed collaborative approach to causal graph generation, without finetuning on downstream tasks, achieves competitive results with state-of-the-art models on forecasting and next event prediction tasks.

Abstract: 理解情景中事件如何相互因果关联，对于有效地建模和推理事件至关重要。但事件推理仍然是一个困难的挑战，尽管最近取得了进展，大型语言模型（LLM）仍然难以准确识别事件之间的因果关系。这种困难导致在更深层次的推理任务（如事件预测和时间线理解）中表现不佳。为了应对这一挑战，我们研究了因果事件图（例如，A 导致 B）的生成，作为一种并行机制，以帮助 LLM 在推理过程中显式地表示因果关系。本文评估了如何生成正确的图，以及图如何帮助推理。我们提出了一种因果图生成的协作方法，其中我们使用 LLM 来模拟专注于特定语义关系的专家。专家们进行多轮讨论，然后由最终专家进行整合。然后，为了证明因果图的效用，我们在多个下游应用程序中使用它们，并且还引入了一项新的可解释的事件预测任务，该任务需要在解释中包含因果事件链。这些解释比基线生成更具信息性和连贯性。最后，我们的整体方法没有在任何下游任务上进行微调，在预测和下一个事件预测任务上都取得了与最先进模型相比具有竞争力的结果。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06910) | **Categories:** cs.AI

---

### [4] [Mathesis: Towards Formal Theorem Proving from Natural Languages](https://arxiv.org/abs/2506.07047)
*Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, Claudia Beth Ong, Austen Jeremy Sugiarto, Yaoxi Zhang, Wai Ming Tai, Huan Cao, Dongcai Lu, Jiacheng Sun, Qiang Xu, Shen Xin, Zhenguo Li*

Main category: cs.AI

TL;DR: Mathesis是一个端到端定理证明流程，通过强化学习自动形式化自然语言问题，并在高考基准测试中取得了优异成果。


<details>
  <summary>Details</summary>
Motivation: 现有基于大型语言模型的定理证明器受限于需要专家编写的形式化语句作为输入，限制了它们在自然语言表达的实际问题中的应用。

Method: Mathesis包含Mathesis-Autoformalizer和Mathesis-Prover两个模块，前者使用强化学习提高自然语言问题的形式化能力，后者从形式化语句生成形式化证明。

Result: Mathesis-Autoformalizer在Gaokao-Formal上的通过率比最佳基线高出22%，完整系统在MiniF2F上达到64%的准确率，在Gaokao-Formal上达到18%的当前最佳水平。

Conclusion: Mathesis在MiniF2F和Gaokao-Formal基准测试中表现出色，证明了其有效性。

Abstract: 本文介绍了一种名为Mathesis的端到端定理证明流程，它可以处理非正式的问题陈述。Mathesis包含Mathesis-Autoformalizer，它是第一个使用强化学习来增强自然语言问题形式化能力的自动形式化器，并由我们新颖的LeanScorer框架辅助进行细致的形式化质量评估。它还提出了一个Mathesis-Prover，可以从形式化语句生成形式化证明。为了评估端到端形式定理证明的实际应用性，我们引入了Gaokao-Formal，这是一个包含488个来自中国高考的复杂问题的基准。实验表明Mathesis的有效性，其中自动形式化器在Gaokao-Formal上的通过率比最佳基线高出22%。完整系统超过了其他模型组合，在MiniF2F上实现了64%的准确率，并在Gaokao-Formal上实现了18%的当前最佳水平。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07047) | **Categories:** cs.AI

---

### [5] [BIMgent: Towards Autonomous Building Modeling via Computer-use Agents](https://arxiv.org/abs/2506.07217)
*Zihan Deng, Changyu Du, Stavros Nousias, André Borrmann*

Main category: cs.AI

TL;DR: BIMgent 是一个基于多模态大型语言模型的代理框架，可以通过图形用户界面操作自主创建建筑模型。


<details>
  <summary>Details</summary>
Motivation: 现有的计算机使用代理主要侧重于通用桌面自动化任务，对其在高度专业化领域的应用探索有限。建筑、工程和施工（AEC）领域的 3D 建筑建模过程涉及开放式设计任务和建筑信息模型（BIM）创作软件中的复杂交互模式，而当前的研究尚未对此进行充分探讨。

Method: 本文提出了一种由多模态大型语言模型（LLM）驱动的代理框架 BIMgent，旨在通过图形用户界面（GUI）操作实现自主建筑模型创作。

Result: BIMgent 的设计质量达到合理水平，其操作成功率为 32%，而所有基线模型均未能完成任务（成功率为 0%）。

Conclusion: BIMgent 能够有效减少人工工作量，同时保留设计意图，在实际建筑建模场景中具有实际部署的潜力。

Abstract: 现有的计算机使用代理主要侧重于通用桌面自动化任务，对其在高度专业化领域的应用探索有限。特别是，建筑、工程和施工（AEC）领域的 3D 建筑建模过程涉及开放式设计任务和建筑信息模型（BIM）创作软件中的复杂交互模式，而当前的研究尚未对此进行充分探讨。在本文中，我们提出了一种由多模态大型语言模型（LLM）驱动的代理框架 BIMgent，旨在通过图形用户界面（GUI）操作实现自主建筑模型创作。BIMgent 实现了建筑建模过程的自动化，包括概念设计的多模态输入、软件特定工作流程的规划以及创作 GUI 操作的有效执行。我们在实际建筑建模任务中评估了 BIMgent，包括基于文本的概念设计生成和从现有建筑设计重建。BIMgent 的设计质量达到合理水平，其操作成功率为 32%，而所有基线模型均未能完成任务（成功率为 0%）。结果表明，BIMgent 能够有效减少人工工作量，同时保留设计意图，突显了其在实际建筑建模场景中实际部署的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07217) | **Categories:** cs.AI

---

### [6] [LLM-Enhanced Rapid-Reflex Async-Reflect Embodied Agent for Real-Time Decision-Making in Dynamically Changing Environments](https://arxiv.org/abs/2506.07223)
*Yangqing Zheng, Shunqi Mao, Dingxin Zhang, Weidong Cai*

Main category: cs.AI

TL;DR: 本文提出了时间转换机制（TCM）和快速-反应异步-反思智能体（RRARA），以解决具身智能体在动态高风险场景中决策延迟的问题，并在HAZARD基准测试中取得了优异表现。


<details>
  <summary>Details</summary>
Motivation: 在具身智能领域，大型语言模型（LLM）的演进显著增强了智能体的决策能力。因此，研究人员已经开始探索智能体在动态变化的高风险场景中的表现，即HAZARD基准测试中的火灾、洪水和风灾场景。在这些极端条件下，决策中的延迟成为一个关键但研究不足的问题。

Method: 本文提出了一种时间转换机制（TCM），将决策中的推理延迟转换为等效的模拟帧，从而在基于FPS的单一指标下对齐认知和物理成本。同时，本文提出了Rapid-Reflex Async-Reflect Agent (RRARA)，它将轻量级的LLM引导的反馈模块与基于规则的代理相结合，以实现即时反应行为和异步的实时改进。

Result: 在HAZARD上的实验表明，RRARA在延迟敏感的场景中大大优于现有的基线。

Conclusion: RRARA在HAZARD基准测试中，于对延迟敏感的场景下，显著优于现有的基线模型。

Abstract: 在具身智能领域，大型语言模型（LLM）的发展显著提升了智能体的决策能力。因此，研究人员开始探索智能体在动态变化的高风险场景（如HAZARD基准测试中的火灾、洪水和风灾）中的表现。在这些极端条件下，决策延迟成为了一个关键但研究不足的问题。本文提出了一种时间转换机制（TCM），该机制将决策过程中的推理延迟转化为等效的模拟帧数，从而在统一的基于FPS的指标下对齐认知成本和物理成本。通过使用响应延迟（RL）和延迟-动作比率（LAR）扩展HAZARD，我们提供了一个完全延迟感知的评估协议。此外，我们还提出了快速-反应异步-反思智能体（RRARA），它将轻量级的LLM引导的反馈模块与基于规则的智能体相结合，从而实现了即时反应行为和异步的实时改进。在HAZARD上的实验表明，RRARA在对延迟敏感的场景中显著优于现有的基线。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07223) | **Categories:** cs.AI

---

### [7] [Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark](https://arxiv.org/abs/2506.07896)
*Shoko Oka*

Main category: cs.AI

TL;DR: 大型语言模型在一定程度上能够应对人工智能领域的框架问题和符号接地问题。


<details>
  <summary>Details</summary>
Motivation: 探讨大型语言模型是否具备解决人工智能领域中框架问题和符号接地问题所需的认知能力。

Method: 设计了两个基准任务，分别反映了框架问题和符号接地问题的哲学核心，并在零样本条件下对13个大型语言模型进行了测试。

Result: 实验结果表明，开源模型由于模型大小、量化和指令调整的差异，性能表现出差异性，而一些闭源模型则持续获得高分。

Conclusion: 部分大型语言模型在解决框架问题和符号接地问题上表现出潜力，表明它们可能具备应对这些长期理论挑战的能力。

Abstract: 大型语言模型（LLM）的最新进展重振了围绕人工智能的哲学辩论。框架问题和符号接地问题是两个最根本的挑战，历史上一直被认为是传统符号人工智能系统中无法解决的。本研究调查了现代LLM是否具备解决这些问题所需的认知能力。为此，我设计了两个基准任务，分别反映了每个问题的哲学核心，在零样本条件下将它们应用于13个著名LLM（包括封闭和开源），并评估了模型在每次试验中的输出质量。根据多个标准对响应进行评分，包括上下文推理、语义连贯性和信息过滤。结果表明，虽然开源模型由于模型大小、量化和指令调整的差异而表现出性能差异，但一些封闭模型始终获得高分。这些发现表明，选择现代LLM可能正在获得足够的能力，从而对这些长期存在的理论挑战产生有意义且稳定的反应。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07896) | **Categories:** cs.AI, cs.CL

---

### [8] [LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement](https://arxiv.org/abs/2506.07915)
*Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo*

Main category: cs.AI

TL;DR: LUCIFER：一个利用人类情境知识，融合强化学习和大型语言模型的分层决策框架，可提升自主系统在动态环境中的探索效率和决策质量。


<details>
  <summary>Details</summary>
Motivation: 在动态环境中，先前存在的环境知识的快速过时会在智能体的内部模型与其操作环境不断变化的现实之间造成差距。先前和更新的环境估值之间的这种差异从根本上限制了自主决策的有效性。

Method: LUCIFER 整合了分层决策架构与强化学习 (RL) 和大型语言模型 (LLM)，形成统一的系统。该架构模仿了人类分解复杂任务的方式，使高级规划器能够协调专门的子代理，每个子代理都专注于不同的目标和时间上相互依赖的行动。LUCIFER 将 LLM 集成到两个协同作用的角色中：作为上下文提取器，将口头利益相关者的输入构建为领域感知表示，通过注意力空间机制影响决策，将 LLM 衍生的见解与代理的学习过程对齐；以及作为零样本探索促进器，在探索期间指导代理的行动选择过程。

Result: 我们对各种 LLM 在这两个角色中进行了基准测试，并证明 LUCIFER 提高了探索效率和决策质量，优于扁平的、以目标为条件的策略。

Conclusion: LUCIFER展示了情境驱动决策的潜力，自主系统可以利用人类情境知识来获得运营上的成功。

Abstract: 在动态环境中，预先存在的环境知识快速过时，导致智能体的内部模型与其操作环境的演变现实之间存在差距。先前环境评估与更新的环境评估之间的差异，从根本上限制了自主决策的有效性。为了弥合这一差距，人类领域利益相关者的情境偏见变得不可或缺，他们自然可以通过直接的实时观察积累见解。然而，将他们细致入微且富含情境的输入转化为自主系统的可行情报仍然是一个公开的挑战。为了解决这个问题，我们提出了 LUCIFER（用于探索和行为改进的语言理解和情境注入框架），这是一个领域无关的框架，它将分层决策架构与强化学习 (RL) 和大型语言模型 (LLM) 集成到一个统一的系统中。这种架构模仿了人类分解复杂任务的方式，使高级规划器能够协调专门的子代理，每个子代理都专注于不同的目标和时间上相互依赖的行动。与 LLM 仅限于单一角色的传统应用不同，LUCIFER 将它们集成到两个协同作用的角色中：作为情境提取器，将口头利益相关者的输入构建为领域感知表示，通过注意力空间机制影响决策，将 LLM 衍生的见解与智能体的学习过程对齐；以及作为零样本探索促进器，在探索期间指导智能体的行动选择过程。我们对各种 LLM 在这两个角色中进行了基准测试，并证明 LUCIFER 提高了探索效率和决策质量，优于扁平的、以目标为条件的策略。我们的研究结果表明了情境驱动决策的潜力，自主系统可以利用人类情境知识来实现运营上的成功。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07915) | **Categories:** cs.AI, cs.CL, cs.SY, eess.SY

---

### [9] [GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior](https://arxiv.org/abs/2506.08012)
*Penghao Wu, Shengnan Ma, Bo Wang, Jiaheng Yu, Lewei Lu, Ziwei Liu*

Main category: cs.AI

TL;DR: GUI-Reflection框架通过集成自我反思和纠错能力，提升了多模态GUI模型的自动化性能。


<details>
  <summary>Details</summary>
Motivation: 现有的GUI模型主要依赖于从几乎没有错误的离线轨迹中学习，因此缺乏反思和错误恢复能力。为了弥合这一差距，我们提出了GUI-Reflection。

Method: 我们提出了GUI-Reflection，这是一个新颖的框架，它通过专门的训练阶段将自我反思和纠错能力显式地集成到端到端多模态GUI模型中：GUI特定的预训练、离线监督微调（SFT）和在线反思调整。

Result: GUI-reflection通过完全自动化的数据生成和学习过程，无需任何人工注释，即可实现自我反思行为的出现。

Conclusion: 该框架使GUI代理具备了自我反思和纠错能力，为更强大、适应性更强和更智能的GUI自动化铺平了道路，所有数据、模型、环境和工具都将公开发布。

Abstract: 多模态大型语言模型（MLLM）在彻底改变图形用户界面（GUI）自动化方面显示出巨大的潜力。然而，现有的GUI模型主要依赖于从几乎没有错误的离线轨迹中学习，因此缺乏反思和错误恢复能力。为了弥合这一差距，我们提出了GUI-Reflection，这是一个新颖的框架，它通过专门的训练阶段将自我反思和纠错能力显式地集成到端到端多模态GUI模型中：GUI特定的预训练、离线监督微调（SFT）和在线反思调整。GUI-reflection通过完全自动化的数据生成和学习过程，无需任何人工注释，即可实现自我反思行为的出现。具体来说，1）我们首先提出了可扩展的数据管道，以从现有的成功轨迹中自动构建反思和纠错数据。虽然现有的GUI模型主要侧重于 grounding 和 UI 理解能力，但我们提出了 GUI-Reflection 任务套件来显式地学习和评估面向反思的能力。2）此外，我们构建了一个多样化且高效的环境，用于在移动设备上对GUI模型进行在线训练和数据收集。3）我们还提出了一种迭代在线反思调整算法，该算法利用所提出的环境，使模型能够不断增强其反思和纠错能力。我们的框架使GUI代理具备了自我反思和纠错能力，为更强大、适应性更强和更智能的GUI自动化铺平了道路，所有数据、模型、环境和工具都将公开发布。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08012) | **Categories:** cs.AI, cs.CV

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](https://arxiv.org/abs/2506.07424)
*Kyeonghyun Kim, Jinhee Jang, Juhwan Choi, Yoonji Lee, Kyohoon Jin, YoungBin Kim*

Main category: cs.CL

TL;DR: PiFi 通过将大型语言模型的冻结层融入小型语言模型，实现了性能与效率的提升。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型计算成本高昂，不适合资源受限的环境，而小型语言模型泛化能力较弱。PiFi 旨在弥合这一差距。

Method: PiFi 框架将大型语言模型中的一个冻结层集成到小型语言模型中，并通过微调结合后的模型来优化特定任务的性能。

Result: 实验结果表明，PiFi 在包括自然语言理解和生成在内的多个自然语言处理任务中均实现了持续的性能提升。此外，PiFi 还能有效利用大型语言模型的知识，增强对未见领域的泛化能力，并促进语言能力的迁移。

Conclusion: PiFi 通过将大型语言模型的知识迁移到小型语言模型，实现了在各种自然语言处理任务上的性能提升，并提高了泛化能力。

Abstract: 大型语言模型（LLM）以其广泛的语言知识和强大的泛化能力而闻名，但其高计算需求使其不适合资源受限的环境。相比之下，小型语言模型（SLM）的计算效率很高，但通常缺乏 LLM 的广泛泛化能力。为了弥合这一差距，我们提出了 PiFi，这是一个新颖的框架，它结合了 LLM 和 SLM 的优势，以在保持效率的同时实现高性能。PiFi 将来自 LLM 的单个冻结层集成到 SLM 中，并微调组合模型以执行特定任务，从而在不显着增加计算成本的情况下提高性能。我们表明，PiFi 在包括自然语言理解和生成在内的一系列自然语言处理任务中提供了持续的性能改进。此外，我们的研究结果表明 PiFi 能够有效地利用 LLM 知识，增强对未见领域的泛化能力，并促进语言能力的转移。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07424) | **Categories:** cs.CL, cs.AI

---

### [2] [Enhancing Decision-Making of Large Language Models via Actor-Critic](https://arxiv.org/abs/2506.06376)
*Heng Dong, Kefei Duan, Chongjie Zhang*

Main category: cs.CL

TL;DR: 本文提出了一种基于LLM的Actor-Critic框架（LAC），通过长期动作评估来有效提升LLM策略，从而在复杂决策场景中实现更优的性能。


<details>
  <summary>Details</summary>
Motivation: 现有方法在需要长期推理和与高级目标对齐的复杂决策场景中面临挑战，要么依赖于短期自回归动作生成，要么在准确模拟rollout和评估结果方面存在局限性，导致次优决策。

Method: 提出了一种基于LLM的Actor-Critic框架，称为LAC，通过计算与积极/消极结果相关的token logits来提取稳健的动作评估，并通过未来的轨迹rollout和推理来增强评估。

Result: 在包括高级决策（ALFWorld）、低级动作空间（BabyAI-Text）和大型动作空间（WebShop）在内的各种环境中进行的实验表明，该框架具有通用性，并且优于最先进的方法。值得注意的是，我们的方法使用7B/8B参数LLM实现了具有竞争力的性能，甚至在复杂的任务中优于使用GPT-4的基线方法。

Conclusion: 该研究表明，将结构化策略优化与LLM的内在知识相结合，可以提升LLM在多步骤环境中的决策能力。

Abstract: 大型语言模型（LLM）在自然语言处理任务中取得了显著进展，但它们在需要长期推理和与高级目标对齐的复杂决策场景中面临挑战。现有方法要么依赖于短期自回归动作生成，要么在准确模拟rollout和评估结果方面存在局限性，导致次优决策。本文介绍了一种新颖的基于LLM的Actor-Critic框架，称为LAC，该框架以原则性和可扩展的方式有效地改进了LLM策略，并具有长期动作评估能力。我们的方法解决了两个关键挑战：（1）通过计算与积极/消极结果相关的token logits来提取稳健的动作评估，并通过未来的轨迹rollout和推理来增强评估；（2）通过无梯度机制实现有效的策略改进。在包括高级决策（ALFWorld）、低级动作空间（BabyAI-Text）和大型动作空间（WebShop）在内的各种环境中进行的实验表明，该框架具有通用性，并且优于最先进的方法。值得注意的是，我们的方法使用7B/8B参数LLM实现了具有竞争力的性能，甚至在复杂的任务中优于使用GPT-4的基线方法。这些结果强调了将结构化策略优化与LLM的内在知识相结合，以推进多步骤环境中决策能力的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06376) | **Categories:** cs.CL, cs.AI

---

### [3] [Correlated Errors in Large Language Models](https://arxiv.org/abs/2506.07962)
*Elliot Kim, Avi Garg, Kenny Peng, Nikhil Garg*

Main category: cs.CL

TL;DR: 大型语言模型即使在架构和提供商不同时，其错误也存在高度相关性。


<details>
  <summary>Details</summary>
Motivation: 训练数据、架构和提供者的多样性被认为可以减轻 LLM 的同质性。然而，我们缺乏关于不同 LLM 是否有意义地不同的经验证据。

Method: 对超过 350 个 LLM 进行了大规模的实证评估，使用了两个流行的排行榜和一个简历筛选任务。

Result: 模型错误存在很大的相关性——在一个排行榜数据集上，当两个模型都出错时，模型有 60% 的时间是相同的。我们确定了驱动模型相关性的因素，包括共享架构和提供者。

Conclusion: 即使架构和提供者不同，更大、更准确的模型的错误也高度相关。

Abstract: 假设训练数据、架构和提供商的多样性可以减轻大型语言模型 (LLM) 的同质性。然而，我们缺乏关于不同 LLM 是否有意义地不同的经验证据。我们对总共超过 350 个 LLM 进行了大规模的实证评估，使用了两个流行的排行榜和一个简历筛选任务。我们发现模型错误存在很大的相关性——在一个排行榜数据集上，当两个模型都出错时，模型有 60% 的时间是相同的。我们确定了驱动模型相关性的因素，包括共享架构和提供商。但至关重要的是，即使架构和提供者不同，更大、更准确的模型的错误也高度相关。最后，我们展示了相关性在两个下游任务中的影响：LLM 作为评委的评估和招聘——后者反映了关于算法单一文化的理论预测。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07962) | **Categories:** cs.CL, cs.AI, cs.CY, stat.ML

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DONUT: A Decoder-Only Model for Trajectory Prediction](https://arxiv.org/abs/2506.06854)
*Markus Knoche, Daan de Geus, Bastian Leibe*

Main category: cs.CV

TL;DR: DONUT：一个decoder-only网络，用于轨迹展开，在运动预测任务上取得了state-of-the-art的结果。


<details>
  <summary>Details</summary>
Motivation: 为了使自动驾驶汽车能够预测其他智能体的运动，受decoder-only模型在语言建模方面的成功启发。

Method: 提出了一种新的decoder-only网络DONUT，用于轨迹展开，使用单一的自回归模型编码历史轨迹并预测未来轨迹，并引入了“过度预测”策略。

Result: 实验表明，decoder-only方法优于encoder-decoder基线，并在Argoverse 2单智能体运动预测基准测试中取得了新的state-of-the-art结果。

Conclusion: 提出的DONUT模型在Argoverse 2单智能体运动预测基准测试中取得了新的最先进的结果，优于encoder-decoder基线。

Abstract: 为了使自动驾驶汽车能够预测其他智能体的运动，我们提出了一种新的decoder-only网络DONUT，用于轨迹展开。与现有的encoder-decoder预测模型不同，我们使用单一的自回归模型编码历史轨迹并预测未来轨迹。这使得模型能够以一致的方式进行迭代预测，并确保模型始终获得最新的信息，从而提高性能。此外，受语言建模中多token预测的启发，我们引入了一种“过度预测”策略，该策略为网络提供了在更长的时间范围内预测轨迹的辅助任务。这使得模型能够更好地预测未来，并进一步提高性能。实验表明，我们的decoder-only方法优于encoder-decoder基线，并在Argoverse 2单智能体运动预测基准测试中取得了新的state-of-the-art结果。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06854) | **Categories:** cs.CV

---

### [2] [ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models](https://arxiv.org/abs/2506.07725)
*Shadi Hamdan, Chonghao Sima, Zetong Yang, Hongyang Li, Fatma Güney*

Main category: cs.CV

TL;DR: ETA通过异步提前预测和批量推理，使大型模型能够在自动驾驶系统中实现快速响应，并在CARLA基准测试中取得了显著的性能提升。


<details>
  <summary>Details</summary>
Motivation: 在自动驾驶系统中，如何既能利用大型模型的优势，又不牺牲推理速度是一个常见的难题。现有的双系统设计难以使大型模型对每个在线帧做出及时响应。

Method: 提出了一种名为“超前思考效率 (ETA)”的异步系统，该系统利用大模型预测未来帧，并将信息特征从过去传递到当前帧，同时使用小模型提取当前帧特征，并通过动作掩码机制整合双重特征。

Result: 在Bench2Drive CARLA Leaderboard-v2基准测试中，ETA将最先进的性能提高了 8%，驾驶分数为 69.53，同时保持了 50 毫秒的近实时推理速度。

Conclusion: ETA通过预测未来帧并进行批量推理，显著提高了自动驾驶系统的性能，并在保持近实时推理速度的同时，在CARLA Leaderboard-v2基准测试中取得了最先进的结果。

Abstract: 在自动驾驶系统中，如何在不牺牲推理速度的情况下，充分利用大型模型的优势是一个常见的难题。目前常见的解决方案是采用双系统架构，即使用小型模型进行快速、反应式的决策，而使用大型模型进行速度较慢但信息量更大的分析。现有的双系统设计通常采用并行架构，要么直接使用大型模型对每个当前帧进行推理，要么从先前存储的推理结果中检索。然而，这些方法仍然难以使大型模型对每个在线帧做出及时响应。我们的关键见解是将当前帧的密集计算转移到先前的时间步，并执行多个时间步的批量推理，以使大型模型能够迅速响应每个时间步。为了实现这种转移，我们引入了“超前思考效率 (ETA)”，这是一种异步系统，旨在：(1) 使用来自大型模型的未来预测，将信息丰富的特征从过去传播到当前帧；(2) 使用小型模型提取当前帧特征，以实现实时响应；(3) 通过强调动作关键图像区域的动作掩码机制，整合这些双重特征。在 Bench2Drive CARLA Leaderboard-v2 基准测试中进行评估，ETA 将最先进的性能提高了 8%，驾驶分数为 69.53，同时保持了 50 毫秒的近实时推理速度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07725) | **Categories:** cs.CV, cs.AI

---

### [3] [Egocentric Event-Based Vision for Ping Pong Ball Trajectory Prediction](https://arxiv.org/abs/2506.07860)
*Ivan Alberico, Marco Cannici, Giovanni Cioffi, Davide Scaramuzza*

Main category: cs.CV

TL;DR: 本文提出首个使用事件相机从自我中心视角进行实时乒乓球轨迹预测的系统。


<details>
  <summary>Details</summary>
Motivation: 传统相机在高球速下存在高延迟和运动模糊的问题，而事件相机具有更高的时间分辨率。

Method: 该系统利用眼动追踪数据实现注视视觉，只处理观察者注视区域的事件，从而提高球检测性能并显著降低计算延迟。

Result: 该检测管道的最坏情况总延迟为4.5毫秒，比基于帧的30 FPS系统（仅感知就需要66毫秒）低得多。在收集的轨迹上实现了10.81的减少因子。

Conclusion: 该论文首次提出了一种使用事件相机从自我中心视角预测乒乓球轨迹的方法。

Abstract: 本文提出了一种使用事件相机的实时自我中心乒乓球轨迹预测系统。与标准相机在高球速下存在高延迟和运动模糊的问题不同，事件相机提供更高的时间分辨率，从而实现更频繁的状态更新、更高的离群值鲁棒性以及仅在对手击球后使用较短时间窗口的精确轨迹预测。我们收集了一个乒乓球游戏序列的数据集，包括球的3D地面实况轨迹，与Meta Project Aria眼镜和事件流的传感器数据同步。我们的系统利用注视视觉，使用眼镜中的眼动追踪数据来仅处理观察者注视区域中的事件。这种受生物学启发的方提高了球的检测性能，并显着降低了计算延迟，因为它有效地将资源分配给最相关的感知区域，在收集的轨迹上实现了10.81的减少因子。我们的检测管道的最坏情况总延迟为4.5毫秒，包括计算和感知 - 显着低于基于帧的30 FPS系统，后者在最坏的情况下仅感知就需要66毫秒。最后，我们将轨迹预测模型拟合到球的估计状态，从而能够在未来进行3D轨迹预测。据我们所知，这是第一种使用事件相机从自我中心视角预测乒乓球轨迹的方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07860) | **Categories:** cs.CV

---

### [4] [Continuous-Time SO(3) Forecasting with Savitzky--Golay Neural Controlled Differential Equations](https://arxiv.org/abs/2506.06780)
*Lennart Bastian, Mohammad Rashed, Nassir Navab, Tolga Birdal*

Main category: cs.CV

TL;DR: 提出了一种新的基于神经控制微分方程的方法，用于在SO(3)上建模连续时间旋转对象动力学，实现了更准确的旋转预测。


<details>
  <summary>Details</summary>
Motivation: 在计算机视觉和机器人学中，跟踪和预测物体的旋转是至关重要的，然而，由于传感器观测可能存在噪声和稀疏性，运动模式可能受复杂动力学支配，以及应用设置可能需要长期预测，因此SO(3)外推仍然具有挑战性。

Method: 提出了一种基于Savitzky-Golay路径引导的神经控制微分方程，用于在$SO(3)$上建模连续时间旋转对象动力学。

Result: 在真实世界数据上的实验结果表明，与现有方法相比，该方法具有令人信服的预测能力。

Conclusion: 该方法在真实世界数据上表现出比现有方法更强的预测能力。

Abstract: 在计算机视觉和机器人技术中，跟踪和预测物体的旋转至关重要。然而，由于传感器观测数据可能存在噪声和稀疏性，运动模式可能由复杂的动力学控制，并且应用场景可能需要长期预测，因此SO(3)外推仍然具有挑战性。本文提出了一种新的方法，使用由Savitzky-Golay路径引导的神经控制微分方程在$SO(3)$上对连续时间旋转物体动力学进行建模。与依赖简化运动假设的现有方法不同，我们的方法学习了一个潜在的通用动态系统，该系统描述了底层物体的运动轨迹，同时尊重旋转的几何结构。在真实世界数据上的实验结果表明，与现有方法相比，该方法具有令人信服的预测能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06780) | **Categories:** cs.CV, cs.LG

---

### [5] [Reading in the Dark with Foveated Event Vision](https://arxiv.org/abs/2506.06918)
*Carl Brander, Giovanni Cioffi, Nico Messikommer, Davide Scaramuzza*

Main category: cs.CV

TL;DR: 提出了一种基于事件相机的智能眼镜OCR方法，该方法在低光照下表现出色，并显著降低了带宽需求。


<details>
  <summary>Details</summary>
Motivation: 现有的配备RGB相机的智能眼镜在低光和高速运动场景中难以感知环境，并且帧相机捕获密集图像需要大带宽和功耗，导致电池耗尽更快。这些挑战与开发能够从图像中读取文本的算法尤其相关。

Method: 提出了一种新颖的基于事件的OCR方法，该方法利用用户视线来聚焦事件流，并通过深度二进制重建和多模态LLM进行OCR。

Result: 实验结果表明，该方法能够在RGB相机难以工作的低光照环境下读取文本，同时使用的带宽比可穿戴RGB相机少2400倍。

Conclusion: 该研究表明，基于事件相机的OCR方法在低光照环境下优于传统OCR解决方案，并且比可穿戴RGB相机节省高达2400倍的带宽。

Abstract: 目前配备RGB相机的智能眼镜由于运动模糊和帧相机的有限动态范围，难以在低光和高速运动场景中感知环境。此外，使用帧相机捕获密集图像需要很大的带宽和功耗，从而导致电池消耗更快。这些挑战对于开发能够从图像中读取文本的算法尤为重要。在这项工作中，我们提出了一种用于智能眼镜的新型基于事件的光学字符识别（OCR）方法。通过使用用户的视线，我们聚焦事件流，从而显著减少约98%的带宽，同时利用事件相机在高动态和快速场景中的优势。我们提出的方法执行在合成数据上训练的深度二进制重建，并利用多模态LLM进行OCR，优于传统的OCR解决方案。我们的结果表明，该方法能够在RGB相机难以工作的低光照环境下读取文本，同时使用的带宽比可穿戴RGB相机少2400倍。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06918) | **Categories:** cs.CV, cs.RO

---

### [6] [SceneLCM: End-to-End Layout-Guided Interactive Indoor Scene Generation with Latent Consistency Model](https://arxiv.org/abs/2506.07091)
*Yangkai Lin, Jiabao Lei, Kui Jia*

Main category: cs.CV

TL;DR: SceneLCM 通过结合大型语言模型和潜在一致性模型，实现自动生成复杂、交互式室内场景。


<details>
  <summary>Details</summary>
Motivation: 现有方法在室内场景合成方面存在编辑约束僵化、物理不连贯、人工干预过多、单房间限制和材质质量欠佳等问题。

Method: SceneLCM 结合大型语言模型（LLM）进行布局设计和潜在一致性模型（LCM）进行场景优化，将场景生成分解为四个模块化流程：布局生成、家具生成、环境优化和物理编辑。

Result: 大量实验验证了 SceneLCM 相对于现有技术的优越性。

Conclusion: SceneLCM在生成复杂、交互式室内场景方面优于现有技术，具有广泛的应用潜力。

Abstract: 该项目介绍了一个名为 SceneLCM 的端到端框架，用于生成复杂、交互式的室内场景。该框架结合了大型语言模型（LLM）进行布局设计和潜在一致性模型（LCM）进行场景优化，解决了现有室内场景合成方法在编辑约束、物理连贯性、人工干预、单房间限制和材质质量方面的问题。SceneLCM 将场景生成分解为四个模块化流程：(1) 布局生成，利用 LLM 将文本描述转换为参数化蓝图，并通过迭代的程序验证机制细化布局参数；(2) 家具生成，采用一致性轨迹采样（CTS）和一致性蒸馏采样损失，形成快速、语义丰富和高质量的表示；(3) 环境优化，使用多分辨率纹理场编码场景外观，并通过 CTS 损失进行优化，引入法线感知交叉注意力解码器以保持纹理一致性；(4) 物理编辑，通过集成物理模拟实现持久的物理真实感。大量实验验证了 SceneLCM 的优越性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07091) | **Categories:** cs.CV

---

### [7] [SAP-Bench: Benchmarking Multimodal Large Language Models in Surgical Action Planning](https://arxiv.org/abs/2506.07196)
*Mengya Xu, Zhongzhen Huang, Dillan Imans, Yiru Ye, Xiaofan Zhang, Qi Dou*

Main category: cs.CV

TL;DR: SAP-Bench是一个用于评估MLLM在手术行动计划中性能的大规模数据集，揭示了现有模型在预测下一步行动方面的不足。


<details>
  <summary>Details</summary>
Motivation: 现有的基准不足以评估多模态大型语言模型执行可解释的手术行动计划的能力。

Method: 提出了MLLM-SAP框架，该框架利用MLLM从当前的手术场景和自然语言指令生成下一个行动建议，并注入了手术领域知识。

Result: SAP-Bench基准测试，来自胆囊切除术程序上下文，平均持续时间为1137.5秒，并引入了时间定位的手术行动注释，包括1,226个临床验证的行动片段（平均持续时间：68.7秒），涵盖74个程序中的五个基本手术行动。

Conclusion: 七个最先进的MLLM的评估揭示了在预测下一个行动表现方面的关键差距。

Abstract: 为了推动MLLM研究的进展，有效的评估至关重要。外科手术行动计划（SAP）任务旨在从视觉输入生成未来的行动序列，需要精确和复杂的分析能力。与数学推理不同，外科手术决策在危及生命的领域中进行，需要细致、可验证的过程来确保可靠性和患者安全。此任务需要区分原子视觉动作并协调复杂的长时程程序的能力，而当前基准对这些能力的评估不足。为了解决这一差距，我们推出了SAP-Bench，这是一个大规模、高质量的数据集，旨在使多模态大型语言模型（MLLM）能够执行可解释的外科手术行动计划。我们的SAP-Bench基准测试来自胆囊切除术程序上下文，平均持续时间为1137.5秒，并引入了时间定位的手术行动注释，包括1,226个临床验证的行动片段（平均持续时间：68.7秒），涵盖74个程序中的五个基本手术行动。该数据集提供了1,152个策略性采样的当前帧，每个帧与相应的下一个动作配对，作为多模态分析锚点。我们提出了MLLM-SAP框架，该框架利用MLLM从当前的手术场景和自然语言指令生成下一个行动建议，并注入了手术领域知识。为了评估我们数据集的有效性和当前模型的更广泛能力，我们评估了七个最先进的MLLM（例如，OpenAI-o1、GPT-4o、QwenVL2.5-72B、Claude-3.5-Sonnet、GeminiPro2.5、Step-1o和GLM-4v），并揭示了在预测下一个行动表现方面的关键差距。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07196) | **Categories:** cs.CV, cs.CL

---

### [8] [AllTracker: Efficient Dense Point Tracking at High Resolution](https://arxiv.org/abs/2506.07310)
*Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus, Katerina Fragkiadaki, Leonidas J. Guibas*

Main category: cs.CV

TL;DR: AllTracker 通过估计帧间光流场实现长程点跟踪，并在高分辨率下达到最先进的精度。


<details>
  <summary>Details</summary>
Motivation: 现有的点跟踪方法无法提供高分辨率和密集的对应关系，现有的光流方法只能将一帧对应到后续的帧，而无法对应到数百帧。

Method: 提出了一种新的架构，该架构融合了光流和点跟踪技术，在低分辨率的对应估计网格上进行迭代推理，通过 2D 卷积层在空间上传播信息，并通过像素对齐的注意力层在时间上传播信息。

Result: 该模型快速且参数效率高（1600 万个参数），并且在高分辨率下实现了最先进的点跟踪精度（即在 40G GPU 上跟踪 768x1024 像素）。

Conclusion: 该模型在多个数据集上进行训练，并在高分辨率下实现了最先进的点跟踪精度。

Abstract: 我们介绍了 AllTracker：该模型通过估计查询帧和视频中每个其他帧之间的光流场来估计长程点轨迹。与现有的点跟踪方法不同，我们的方法提供高分辨率和密集的（所有像素）对应关系场，可以将其可视化为光流图。与现有的光流方法不同，我们的方法将一帧对应到数百个后续帧，而不仅仅是下一帧。我们为此任务开发了一种新的架构，融合了光流和点跟踪技术：该模型在低分辨率的对应估计网格上进行迭代推理，通过 2D 卷积层在空间上传播信息，并通过像素对齐的注意力层在时间上传播信息。该模型快速且参数效率高（1600 万个参数），并且在高分辨率下实现了最先进的点跟踪精度（即在 40G GPU 上跟踪 768x1024 像素）。我们设计的优点是可以训练更多的数据集，而且我们发现这样做对于获得最佳性能至关重要。我们对我们的架构细节和训练方法进行了广泛的消融研究，清楚地表明了哪些细节最重要。我们的代码和模型权重可在 https://alltracker.github.io 获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07310) | **Categories:** cs.CV

---

### [9] [DINO-CoDT: Multi-class Collaborative Detection and Tracking with Vision Foundation Models](https://arxiv.org/abs/2506.07375)
*Xunjie He, Christina Dao Wen Lee, Meiling Wang, Chengran Yuan, Zefan Huang, Yufeng Yue, Marcelo H. Ang Jr*

Main category: cs.CV

TL;DR: 提出了一种多类协作检测和跟踪框架，通过全局空间注意力融合、视觉语义重识别和基于速度的自适应轨迹管理，显著提升了检测和跟踪精度。


<details>
  <summary>Details</summary>
Motivation: Existing collaborative perception methods primarily focus on the vehicle superclass, lacking effective solutions for multi-class collaborative detection and tracking, which limits their applicability in real-world scenarios.

Method: A multi-class collaborative detection and tracking framework is proposed, featuring a global spatial attention fusion (GSAF) module for detection, a tracklet RE-IDentification (REID) module leveraging visual semantics, and a velocity-based adaptive tracklet management (VATM) module.

Result: Extensive experiments on the V2X-Real and OPV2V datasets demonstrate that the proposed approach significantly outperforms existing state-of-the-art methods in both detection and tracking accuracy.

Conclusion: The proposed multi-class collaborative detection and tracking framework significantly outperforms existing methods in both detection and tracking accuracy on the V2X-Real and OPV2V datasets.

Abstract: 协作感知通过扩展感知范围和提高传感器故障的鲁棒性，在增强环境理解方面起着至关重要的作用，主要涉及协作3D检测和跟踪任务。前者侧重于在单个帧中进行对象识别，而后者则捕获随时间的连续实例轨迹。然而，目前这两个领域的研究主要集中在车辆超类上，缺乏针对多类协作检测和跟踪的有效解决方案。这种限制阻碍了它们在实际场景中的应用，因为实际场景涉及具有不同外观和运动模式的各种对象类别。为了克服这些限制，我们提出了一个为各种道路使用者量身定制的多类协作检测和跟踪框架。我们首先提出了一个带有全局空间注意力融合（GSAF）模块的检测器，增强了对不同大小对象的多尺度特征学习。接下来，我们引入了一个轨迹RE-IDentification（REID）模块，该模块利用视觉语义和视觉基础模型，以有效减少ID SWitch（IDSW）错误，特别是在涉及行人等小对象的错误匹配情况下。我们进一步设计了一个基于速度的自适应轨迹管理（VATM）模块，该模块根据对象运动动态调整跟踪间隔。在V2X-Real和OPV2V数据集上的大量实验表明，我们的方法在检测和跟踪精度方面均明显优于现有的最先进方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07375) | **Categories:** cs.CV

---

### [10] [Drive Any Mesh: 4D Latent Diffusion for Mesh Deformation from Video](https://arxiv.org/abs/2506.07489)
*Yahao Shi, Yang Liu, Yanmin Wu, Xing Liu, Chen Zhao, Jie Luo, Bin Zhou*

Main category: cs.CV

TL;DR: DriveAnyMesh 通过 4D 扩散模型，仅用单目视频就能驱动网格动画，快速生成高质量动画并兼容现代渲染引擎。


<details>
  <summary>Details</summary>
Motivation: 现有的 4D 生成技术在现代渲染引擎中面临挑战。隐式方法渲染效率低，对基于光栅化的引擎不友好，而骨骼方法需要大量的手工劳动，并且缺乏跨类别的泛化能力。动画现有的 3D 资产，而不是从头开始创建 4D 资产，需要对输入的 3D 结构有深刻的理解。

Method: 提出了一种 4D 扩散模型，该模型对潜在集合序列进行去噪，然后解码这些序列以从点云轨迹序列生成网格动画。这些潜在集合利用基于 Transformer 的变分自编码器，同时捕获 3D 形状和运动信息。通过采用时空的、基于 Transformer 的扩散模型，信息可以在多个潜在帧之间交换，从而提高生成结果的效率和泛化能力。

Result: 实验结果表明，DriveAnyMesh 能够快速生成高质量的复杂运动动画，并且与现代渲染引擎兼容。

Conclusion: DriveAnyMesh 能够快速生成高质量的复杂运动动画，并与现代渲染引擎兼容，在游戏和电影行业具有应用潜力。

Abstract: 我们提出 DriveAnyMesh，这是一种由单目视频驱动网格的方法。目前的 4D 生成技术在现代渲染引擎中遇到了挑战。隐式方法渲染效率低，对基于光栅化的引擎不友好，而骨骼方法需要大量的手工劳动，并且缺乏跨类别的泛化能力。动画现有的 3D 资产，而不是从头开始创建 4D 资产，需要对输入的 3D 结构有深刻的理解。为了解决这些挑战，我们提出了一个 4D 扩散模型，该模型对潜在集合序列进行去噪，然后解码这些序列以从点云轨迹序列生成网格动画。这些潜在集合利用基于 Transformer 的变分自编码器，同时捕获 3D 形状和运动信息。通过采用时空的、基于 Transformer 的扩散模型，信息可以在多个潜在帧之间交换，从而提高生成结果的效率和泛化能力。我们的实验结果表明，DriveAnyMesh 可以快速生成高质量的复杂运动动画，并且与现代渲染引擎兼容。这种方法在游戏和电影行业都具有应用潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07489) | **Categories:** cs.CV

---

### [11] [SpatialLM: Training Large Language Models for Structured Indoor Modeling](https://arxiv.org/abs/2506.07491)
*Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, Zihan Zhou*

Main category: cs.CV

TL;DR: SpatialLM是一个大型语言模型，通过微调开源LLM，实现了对3D点云数据的结构化场景理解，并在布局估计和3D物体检测方面取得了优秀成果。


<details>
  <summary>Details</summary>
Motivation: 旨在解决现有方法依赖于特定任务网络设计的问题，探索通用LLM在3D场景理解中的应用。

Method: 提出了SpatialLM，一个能够处理3D点云数据并生成结构化3D场景理解输出的大型语言模型。

Result: 在公共基准测试中，该模型在布局估计方面表现出最先进的性能，并在3D物体检测方面取得了有竞争力的结果。

Conclusion: 该研究展示了通过微调开源LLM，增强现代LLM空间理解能力的可行路径，为增强现实、具身机器人等应用提供了可能。

Abstract: SpatialLM是一个大型语言模型，旨在处理3D点云数据并生成结构化的3D场景理解输出，包括墙壁、门、窗户等建筑元素以及带有语义类别的定向对象框。与以往依赖于特定任务网络设计的方法不同，我们的模型遵循标准的多模态LLM架构，并直接从开源LLM进行微调。为了训练SpatialLM，我们收集了一个大规模、高质量的合成数据集，其中包含12328个室内场景（54778个房间）的点云以及真实的3D注释，并对各种建模和训练决策进行了仔细研究。在公共基准测试中，我们的模型在布局估计方面表现出最先进的性能，并在3D物体检测方面取得了有竞争力的结果。由此，我们展示了一条可行的路径，可以增强现代LLM的空间理解能力，以用于增强现实、具身机器人等应用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07491) | **Categories:** cs.CV

---

### [12] [Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency](https://arxiv.org/abs/2506.07497)
*Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang*

Main category: cs.CV

TL;DR: Genesis提出了一种统一框架，可以联合生成时空和跨模态一致的多视角驾驶视频和激光雷达序列。


<details>
  <summary>Details</summary>
Motivation: Genesis提出了一个统一的框架，用于联合生成具有时空和跨模态一致性的多视角驾驶视频和LiDAR序列。

Method: Genesis采用了一个两阶段架构，该架构集成了基于DiT的视频扩散模型与3D-VAE编码，以及一个具有基于NeRF的渲染和自适应采样的BEV感知LiDAR生成器。两种模态通过共享潜在空间直接耦合。

Result: 在nuScenes基准上的大量实验表明，Genesis在视频和LiDAR指标上都取得了最先进的性能(FVD 16.95, FID 4.24, Chamfer 0.611)，并有益于包括分割和3D检测在内的下游任务。

Conclusion: Genesis在视频和LiDAR指标上都取得了最先进的性能，并有益于下游任务，验证了生成数据的语义保真度和实用性。

Abstract: 我们提出了Genesis，这是一个统一的框架，用于联合生成具有时空和跨模态一致性的多视角驾驶视频和LiDAR序列。Genesis采用了一个两阶段架构，该架构集成了基于DiT的视频扩散模型与3D-VAE编码，以及一个具有基于NeRF的渲染和自适应采样的BEV感知LiDAR生成器。两种模态通过共享潜在空间直接耦合，从而实现跨视觉和几何域的连贯演变。为了用结构化语义指导生成，我们引入了DataCrafter，这是一个建立在视觉-语言模型上的字幕模块，它提供了场景级和实例级监督。在nuScenes基准上的大量实验表明，Genesis在视频和LiDAR指标上都取得了最先进的性能（FVD 16.95, FID 4.24, Chamfer 0.611），并有益于包括分割和3D检测在内的下游任务，验证了生成数据的语义保真度和实用性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07497) | **Categories:** cs.CV

---

### [13] [LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization](https://arxiv.org/abs/2506.07570)
*Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, Feng Zheng*

Main category: cs.CV

TL;DR: 本文提出了 OptiScene，一个基于大型语言模型并经过优化的室内布局生成模型，它优于现有方法并在交互任务中表现出潜力。


<details>
  <summary>Details</summary>
Motivation: 现有的室内布局生成方法要么依赖于封闭的LLM服务，要么受限于粗糙的关系图和有限的数据集，导致空间不一致和泛化能力不足。

Method: 提出了 OptiScene，一个基于大型语言模型的室内布局生成模型，通过两阶段训练在 3D-SynthPlace 数据集上进行微调，包括监督微调（SFT）和直接偏好优化（DPO）。

Result: OptiScene 在多个实验中优于传统的 prompt-driven 和 learning-based 基线，并在场景编辑和机器人导航等交互任务中表现出潜力。

Conclusion: OptiScene 在室内布局生成方面优于传统方法，并在交互任务中显示出潜力。

Abstract: 本文重新探讨了基于大型语言模型的室内布局生成，并提出了一个大型数据集 3D-SynthPlace，该数据集结合了通过 “GPT 合成，人工检查” 流程生成的合成布局，并从 3D-Front 数据集升级而来。3D-SynthPlace 包含近 17,000 个场景，涵盖四种常见的房间类型——卧室、客厅、厨房和浴室——并包含各种对象和高级空间注释。此外，我们还推出了 OptiScene，这是一个强大的开源 LLM，针对室内布局生成进行了优化，基于我们的 3D-SynthPlace 数据集通过我们的两阶段训练进行了微调。在预热阶段 I，我们采用监督微调 (SFT)，它被教导首先生成高级空间描述，然后有条件地预测具体的对象放置。对于强化阶段 II，为了使生成的布局更好地与人类设计偏好保持一致，我们应用了多轮直接偏好优化 (DPO)，这显着提高了布局质量和生成成功率。大量实验表明，OptiScene 优于传统的 prompt-driven 和 learning-based 基线。此外，OptiScene 在场景编辑和机器人导航等交互任务中显示出巨大的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07570) | **Categories:** cs.CV, cs.AI

---

### [14] [EgoM2P: Egocentric Multimodal Multitask Pretraining](https://arxiv.org/abs/2506.07886)
*Gen Li, Yutong Chen, Yiqian Wu, Kaifeng Zhao, Marc Pollefeys, Siyu Tang*

Main category: cs.CV

TL;DR: EgoM2P是一个用于以自我为中心的4D理解的masked建模框架，它在多个任务上表现出色，且速度更快，并将开源。


<details>
  <summary>Details</summary>
Motivation: 在以自我为中心的视觉中理解多模态信号对于增强现实、机器人和人机交互至关重要。然而，构建大规模的以自我为中心的多模态和多任务模型提出了独特的挑战，例如数据异构性、模态缺失和动态相机运动。

Method: 我们引入了一组高效的时间分词器，并提出了EgoM2P，一个masked建模框架，该框架从时间感知的多模态tokens中学习，以训练一个用于以自我为中心的4D理解的大型通用模型。

Result: EgoM2P在视线预测、以自我为中心的相机跟踪和单眼深度估计等任务上匹配或超过了专家模型，同时也是条件性以自我为中心的视频合成的生成模型。

Conclusion: EgoM2P在多个任务上与专家模型相媲美或超越，同时速度提高了一个数量级，并且将完全开源以支持社区和推进以自我为中心的视觉研究。

Abstract: 理解以自我为中心的视觉中的多模态信号，如RGB视频、深度、相机姿态和注视点，对于增强现实、机器人和人机交互等应用至关重要。这些能力使系统能够更好地理解相机佩戴者的行为、意图和周围环境。然而，构建大规模的以自我为中心的多模态和多任务模型面临着独特的挑战。以自我为中心的数据本质上是异构的，不同设备和设置之间的模态覆盖范围差异很大。为缺失的模态（如注视点或头戴式相机轨迹）生成伪标签通常是不可行的，这使得标准的监督学习方法难以扩展。此外，动态相机运动和第一人称视频的复杂时空结构对现有多模态基础模型的直接应用提出了额外的挑战。为了应对这些挑战，我们引入了一组高效的时间分词器，并提出了EgoM2P，一个masked建模框架，该框架从时间感知的多模态tokens中学习，以训练一个用于以自我为中心的4D理解的大型通用模型。这种统一的设计支持跨各种以自我为中心的感知和合成任务的多任务处理，包括注视点预测、以自我为中心的相机跟踪和以自我为中心的视频的单眼深度估计。EgoM2P也可以作为条件性以自我为中心的视频合成的生成模型。在所有这些任务中，EgoM2P与专家模型相媲美或超越，同时速度提高了一个数量级。我们将完全开源EgoM2P，以支持社区和推进以自我为中心的视觉研究。项目页面：https://egom2p.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07886) | **Categories:** cs.CV

---

### [15] [Real-time Localization of a Soccer Ball from a Single Camera](https://arxiv.org/abs/2506.07981)
*Dmitrii Vorobev, Artem Prosvetov, Karim Elhadji Daou*

Main category: cs.CV

TL;DR: 本文提出了一种计算高效的单摄像机三维足球轨迹重建方法，该方法具有厘米级精度，适用于实时广播。


<details>
  <summary>Details</summary>
Motivation: 本文提出了一种计算高效的方法，用于从单个广播摄像机实时重建三维足球轨迹。

Method: 作者提出了一种多模式状态模型，该模型具有 W 个离散模式，以显著加速优化，同时保持厘米级的精度。

Result: 在 6K 分辨率的俄罗斯足球超级联赛比赛的专有数据集上进行的大量评估表明，该方法的性能与多摄像机系统相当，而无需专门或昂贵的基础设施。

Conclusion: 该方法在专业足球环境中提供了一种实用且精确的 3D 球体追踪方法。

Abstract: 我们提出了一种计算高效的方法，用于从单个广播摄像机实时重建三维足球轨迹。与之前的工作相比，我们的方法引入了一个具有 W 个离散模式的多模式状态模型，以在保持厘米级精度的同时显著加速优化——即使在严重遮挡、运动模糊和复杂背景的情况下也是如此。该系统在标准 CPU 上运行，并实现适用于直播环境的低延迟。在 6K 分辨率的俄罗斯足球超级联赛比赛的专有数据集上进行的大量评估表明，该方法的性能与多摄像机系统相当，而无需专门或昂贵的基础设施。这项工作为在专业足球环境中进行可访问且精确的 3D 球体追踪提供了一种实用的方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07981) | **Categories:** cs.CV, cs.LG

---


## cs.CY [cs.CY]
### [1] [LLMs as World Models: Data-Driven and Human-Centered Pre-Event Simulation for Disaster Impact Assessment](https://arxiv.org/abs/2506.06355)
*Lingyao Li, Dawei Li, Zhenhui Ou, Xiaoran Xu, Jingxiao Liu, Zihui Ma, Runlong Yu, Min Deng*

Main category: cs.CY

TL;DR: 大型语言模型可以通过模拟灾害影响来加强灾前规划。


<details>
  <summary>Details</summary>
Motivation: 高效的模拟对于加强对地震等突发灾害的主动准备至关重要。大型语言模型（LLM）作为世界模型在模拟复杂场景方面显示出前景。

Method: 利用包含地理空间、社会经济、建筑和街道图像数据的多模态数据集，生成邮政编码和县级规模的修正麦卡利强度（MMI）预测。

Result: 在2014年Napa和2019年Ridgecrest地震的评估中，使用USGS的“你感觉到了吗？（DYFI）”报告证明了显著的一致性，邮政编码级别的相关性高达0.88，RMSE低至0.77。RAG和ICL等技术可以提高模拟性能，而与单独的结构化数值数据相比，视觉输入显着提高了准确性。

Conclusion: LLMs在模拟灾害影响方面具有潜力，可以帮助加强灾前规划。

Abstract: 高效的模拟对于加强对地震等突发灾害的主动准备至关重要。最近，大型语言模型（LLM）作为世界模型在模拟复杂场景方面显示出前景。本研究考察了多种LLM，以主动估计感知到的地震影响。利用包括地理空间、社会经济、建筑和街道图像数据的多模态数据集，我们的框架生成了邮政编码和县级规模的修正麦卡利强度（MMI）预测。在2014年Napa和2019年Ridgecrest地震中使用USGS的“你感觉到了吗？（DYFI）”报告进行的评估表明，与真实报告相比，邮政编码级别的相关性高达0.88，RMSE低至0.77，具有显著的一致性。RAG和ICL等技术可以提高模拟性能，而与单独的结构化数值数据相比，视觉输入显着提高了准确性。这些发现表明，LLM在模拟灾害影响方面具有潜力，可以帮助加强灾前规划。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06355) | **Categories:** cs.CY, cs.CE, cs.CL, cs.CV

---

### [2] [Evaluating Large Language Model Capabilities in Assessing Spatial Econometrics Research](https://arxiv.org/abs/2506.06377)
*Giuseppe Arbia, Luca Morandini, Vincenzo Nardelli*

Main category: cs.CY

TL;DR: 大型语言模型在评估经济研究的表面一致性方面表现出色，但在深入经济推理方面存在局限性，表明其在同行评审中可辅助，但需人工监督。


<details>
  <summary>Details</summary>
Motivation: 本文研究了大型语言模型（LLM）评估空间计量经济学中实证研究的经济合理性和理论一致性的能力。

Method: 该研究创建了来自28篇已发表论文的原始和故意修改的“反事实”摘要，并由各种大型语言模型进行评估。大型语言模型对变量选择、系数合理性和发表适用性进行了定性评估和结构化二元分类。

Result: 结果表明，虽然大型语言模型可以熟练地评估变量选择的连贯性（GPT-4o等顶级模型的总体F1得分为0.87），但在评估系数合理性和总体发表适用性等更深层次的方面时，其性能差异很大。结果还表明，大型语言模型的选择、论文的具体特征以及这两个因素之间的相互作用会显著影响评估的准确性，特别是对于细微的判断。

Conclusion: 研究结果表明，大型语言模型在评估实证研究的经济合理性和理论一致性方面表现出一定的能力，但在进行深入的经济推理方面存在局限性，表明其在同行评审中可以发挥辅助作用，但仍需要强大的人工监督。

Abstract: 本文研究了大型语言模型（LLM）评估空间计量经济学实证研究中经济合理性和理论一致性的能力。我们从2005-2024年发表的28篇论文中创建了原始的和故意修改的“反事实”摘要，并由各种LLM进行评估。LLM对变量选择、系数合理性和发表适用性进行了定性评估和结构化二元分类。结果表明，虽然LLM可以熟练地评估变量选择的连贯性（GPT-4o等顶级模型的总体F1得分为0.87），但在评估系数合理性和总体发表适用性等更深层次的方面时，其性能差异很大。结果进一步表明，LLM的选择、论文的具体特征以及这两个因素之间的相互作用显著影响评估的准确性，特别是对于细微的判断。这些发现突显了LLM目前在协助初步、更表面化的检查方面的优势，以及在执行全面、深入的经济推理方面的局限性，表明其在同行评审中可能发挥辅助作用，但仍需要强大的人工监督。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06377) | **Categories:** cs.CY, cs.LG, econ.EM, stat.CO

---


## cs.DC [cs.DC]
### [1] [Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage](https://arxiv.org/abs/2506.06472)
*Ziqi Yuan, Haoyang Zhang, Yirui Eric Zhou, Apoorve Mohan, I-Hsin Chung, Seetharami Seelam, Jian Huang*

Main category: cs.DC

TL;DR: TERAIO 是一个具有生命周期感知的张量卸载框架，它利用低成本 SSD 扩展 GPU 内存，显著提升了 LLM 的训练性能。


<details>
  <summary>Details</summary>
Motivation: 现有的 LLM 训练中，GPU 内存分配中只有一小部分张量是活跃的，而大量不活跃张量长期占用 GPU 内存，这为利用低成本的 PCIe SSD 进行张量卸载/预取提供了机会。

Method: TERAIO 框架通过在训练过程的最初几次迭代中分析张量的生命周期，准确估计每个张量在 GPU 内存中的活跃时间，并生成优化的张量卸载/预取计划，通过 GPUDirect storage 执行该计划。

Result: 实验结果表明，与 ZeRO-Offload 和 ZeRO-Infinity 等最先进的研究相比，TERAIO 将各种 LLM 的训练性能平均提高了 1.47 倍，并达到了假设无限 GPU 内存情况下理想性能的 80.7%。

Conclusion: TERAIO 通过预测张量的生命周期并优化卸载/预取计划，显著提升了 LLM 的训练性能，平均提高了 1.47 倍，并达到了无限 GPU 内存下理想性能的 80.7%。

Abstract: 本文介绍了一种新的、具有生命周期感知的张量卸载框架 TERAIO，用于使用低成本的基于 PCIe 的固态硬盘 (SSD) 进行 GPU 内存扩展。TERAIO 专为使用多个 GPU 和多个 SSD 的大型语言模型 (LLM) 训练而开发。其设计基于我们的观察，即在每个 LLM 训练迭代中，活跃张量仅占用已分配 GPU 内存的一小部分（平均 1.7%），而不活跃张量通常很大并且长时间不会使用，这为将张量卸载/预取到/从慢速 SSD 提供了充足的机会，而不会阻碍 GPU 训练过程。TERAIO 通过分析训练过程的最初几次迭代，准确估计每个张量的生命周期（在 GPU 内存中的活跃时间）。通过张量生命周期分析，TERAIO 将生成优化的张量卸载/预取计划，并通过 PyTorch 将其集成到编译后的 LLM 程序中。TERAIO 具有运行时张量迁移引擎，可通过 GPUDirect 存储执行卸载/预取计划，从而允许 GPU 和 SSD 之间直接进行张量迁移，从而缓解 CPU 瓶颈并最大限度地提高 SSD 带宽利用率。与 ZeRO-Offload 和 ZeRO-Infinity 等最先进的研究相比，我们表明 TERAIO 将各种 LLM 的训练性能平均提高了 1.47 倍，并达到了假设无限 GPU 内存情况下理想性能的 80.7%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06472) | **Categories:** cs.DC, cs.AI, cs.LG, cs.PF

---


## cs.GR [cs.GR]
### [1] [HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance](https://arxiv.org/abs/2506.07209)
*Lei Li, Angela Dai*

Main category: cs.GR

TL;DR: HOI-PAGE通过部件级的可供性推理，以零样本方式从文本生成逼真的4D人-物交互。


<details>
  <summary>Details</summary>
Motivation: 现有的工作主要关注全局的、全身-物体的运动来进行4D HOI合成，但生成逼真和多样化的HOI需要更细粒度的理解——即人体部位如何与物体部位进行交互。

Method: 提出了一种新的方法HOI-PAGE，该方法通过部件级的可供性推理，以零样本方式从文本提示中合成4D人-物交互(HOI)。

Result: 大量的实验表明，该方法是灵活的，能够生成复杂的多物体或多人交互序列，并在零样本4D HOI生成方面显著提高了真实性和文本对齐。

Conclusion: 该方法能够生成复杂的多物体或多人交互序列，并在零样本4D HOI生成方面显著提高了真实性和文本对齐。

Abstract: 我们提出了一种新的方法HOI-PAGE，该方法通过部件级的可供性推理，以零样本方式从文本提示中合成4D人-物交互(HOI)。与之前的工作主要关注全局的、全身-物体的运动来进行4D HOI合成不同，我们观察到，生成逼真和多样化的HOI需要更细粒度的理解——即人体部位如何与物体部位进行交互。因此，我们引入了部件可供性图(PAG)，这是一种从大型语言模型(llm)中提取的结构化HOI表示，它编码了细粒度的部件信息以及接触关系。然后，我们使用这些pag来指导一个三阶段的合成:首先，将输入的三维对象分解为几何部件;然后，从文本提示生成参考HOI视频，从中提取基于部件的运动约束;最后，优化4D HOI运动序列，使其不仅模仿参考动力学，而且满足部件级的接触约束。大量的实验表明，该方法是灵活的，能够生成复杂的多物体或多人交互序列，并在零样本4D HOI生成方面显著提高了真实性和文本对齐。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07209) | **Categories:** cs.GR, cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks](https://arxiv.org/abs/2506.06532)
*Zijiang Yan, Hao Zhou, Jianhua Pei, Hina Tabassum*

Main category: cs.LG

TL;DR: 本文提出了一种基于大型语言模型（LLM）的分层协作方法，用于解决多无人机在动态环境中的联合运动和通信控制问题，实验表明该方法能有效提高系统性能并降低碰撞率。


<details>
  <summary>Details</summary>
Motivation: 多无人机系统的控制和优化仍然是一个重大挑战，特别是在动态和受限环境中。本文探讨了在包含高空平台站（HAPS）的集成地面和非地面网络中运行的多个无人机的联合运动和通信控制。

Method: 提出了一种基于大型语言模型（LLM）的新型分层协作方法。在该方法中，部署在高空平台站（HAPS）上的LLM执行无人机接入控制，而每个无人机上搭载的另一个LLM处理运动规划和控制。

Result: 实验结果表明，所提出的基于LLM的协同方法实现了更高的系统奖励、更低的运营成本，并且与基线方法相比，无人机碰撞率显著降低。

Conclusion: 所提出的基于LLM的协同方法在系统奖励方面表现更好，运营成本更低，并且与基线方法相比，无人机碰撞率显著降低。

Abstract: 无人机（UAV）已在各种实际应用中得到广泛采用。然而，多无人机系统的控制和优化仍然是一个重大挑战，特别是在动态和受限环境中。这项工作探讨了在包含高空平台站（HAPS）的集成地面和非地面网络中运行的多个无人机的联合运动和通信控制。具体来说，我们考虑了一个空中高速公路场景，其中无人机必须加速、减速和变换车道以避免碰撞并维持整体交通流量。与现有研究不同，我们提出了一种基于大型语言模型（LLM）的新型分层协作方法。在我们的方法中，部署在HAPS上的LLM执行无人机接入控制，而每个无人机上搭载的另一个LLM处理运动规划和控制。这种基于LLM的框架利用嵌入在预训练模型中的丰富知识，从而能够实现高层次的战略规划和低层次的战术决策。这种知识驱动的范例为下一代3D空中高速公路系统的开发带来了巨大的潜力。实验结果表明，与基线方法相比，我们提出的基于LLM的协同方法实现了更高的系统奖励、更低的运营成本，并且无人机碰撞率显著降低。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06532) | **Categories:** cs.LG, cs.AI, cs.NI, cs.RO, cs.SY, eess.SY

---

### [2] [Breaking Data Silos: Towards Open and Scalable Mobility Foundation Models via Generative Continual Learning](https://arxiv.org/abs/2506.06694)
*Yuan Yuan, Yukun Liu, Chonghua Han, Jie Feng, Yong Li*

Main category: cs.LG

TL;DR: MoveGCL提出了一种可扩展且保护隐私的框架，用于通过生成式持续学习训练移动基础模型。


<details>
  <summary>Details</summary>
Motivation: 由于移动数据的隐私敏感性以及机构间的数据孤岛，为人类移动性构建类似的基础模型仍然具有挑战性。

Method: MoveGCL通过生成式持续学习训练移动基础模型，利用混合专家Transformer和逐层渐进适应策略。

Result: 在六个真实城市数据集上的实验表明，MoveGCL实现了与联合训练相当的性能，并且显著优于联邦学习基线。

Conclusion: MoveGCL在保护隐私的同时，实现了与联合训练相当的性能，并在六个真实城市数据集上显著优于联邦学习基线。

Abstract: 基础模型彻底改变了自然语言处理和计算机视觉等领域，实现了跨不同任务和数据集的通用学习。然而，由于移动数据的隐私敏感性以及机构间由此产生的数据孤岛，为人类移动性构建类似的模型仍然具有挑战性。为了弥合这一差距，我们提出了MoveGCL，这是一个可扩展且保护隐私的框架，用于通过生成式持续学习来训练移动基础模型。在不共享原始数据的情况下，MoveGCL通过回放从冻结的教师模型生成的合成轨迹来实现分散和渐进的模型演化，并通过定制的知识提炼策略来缓解灾难性遗忘，从而加强知识保留。为了解决移动模式的异质性问题，MoveGCL采用了一种混合专家Transformer，其中包含一种感知移动性的专家路由机制，并采用逐层渐进适应策略来稳定持续更新。在六个真实城市数据集上的实验表明，MoveGCL实现了与联合训练相当的性能，并且显著优于联邦学习基线，同时提供了强大的隐私保护。MoveGCL标志着朝着解锁移动基础模型迈出的关键一步，为基础模型时代开放、可扩展且保护隐私的模型开发提供了切实可行的蓝图。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06694) | **Categories:** cs.LG, cs.CR

---

### [3] [Towards Physics-informed Diffusion for Anomaly Detection in Trajectories](https://arxiv.org/abs/2506.06999)
*Arun Sharma, Mingzhou Yang, Majid Farhadloo, Subhankar Ghosh, Bharat Jayaprakash, Shashi Shekhar*

Main category: cs.LG

TL;DR: 提出了一种物理信息扩散模型，通过整合运动学约束来检测异常轨迹，从而提高 GPS 欺骗检测的准确性。


<details>
  <summary>Details</summary>
Motivation: 旨在发现指示可能的 GPS 欺骗的异常轨迹，以遏制国际水域的非法活动，例如未经授权的捕鱼和非法石油转移。现有的生成模型没有考虑细粒度的时空依赖性和先验物理知识，导致较高的误报率。

Method: 提出了一种物理信息扩散模型，该模型集成了运动学约束，以识别不符合物理定律的轨迹。

Result: 在海洋和城市领域的真实世界数据集上的实验结果表明，该框架在异常检测和轨迹生成方法中分别具有更高的预测精度和更低的估计误差率。

Conclusion: 所提出的物理信息扩散模型在海洋和城市领域的真实世界数据集上，实现了更高的预测精度和更低的异常检测和轨迹生成方法的估计误差率。

Abstract: 给定轨迹数据、特定领域的研究区域和用户定义的阈值，我们的目标是找到指示可能的 GPS 欺骗（例如，伪造轨迹）的异常轨迹。 这个问题对于遏制国际水域的非法活动（如未经授权的捕鱼和非法石油转移）具有重要的社会意义。 由于深度伪造生成（例如，加性噪声、伪造轨迹）中人工智能的进步，以及缺乏足够数量的标记样本进行地面实况验证，这个问题具有挑战性。 最近的文献表明，尽管数据稀疏，但使用生成模型进行异常轨迹检测已显示出可喜的结果。 然而，它们没有考虑细粒度的时空依赖性和先验物理知识，导致更高的误报率。 为了解决这些局限性，我们提出了一种物理信息扩散模型，该模型集成了运动学约束，以识别不符合物理定律的轨迹。 在海洋和城市领域的真实世界数据集上的实验结果表明，所提出的框架在异常检测和轨迹生成方法中分别具有更高的预测精度和更低的估计误差率。 我们的实现在 https://github.com/arunshar/Physics-Informed-Diffusion-Probabilistic-Model 上提供。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06999) | **Categories:** cs.LG, cs.AI, cs.CV, stat.ML

---

### [4] [Pre-trained Large Language Models Learn Hidden Markov Models In-context](https://arxiv.org/abs/2506.07298)
*Yijia Dai, Zhaolin Gao, Yahya Satter, Sarah Dean, Jennifer J. Sun*

Main category: cs.LG

TL;DR: 大型语言模型可以通过上下文学习有效地学习和预测隐马尔可夫模型生成的数据。


<details>
  <summary>Details</summary>
Motivation: Fitting Hidden Markov Models (HMMs) to real-world data remains computationally challenging

Method: Using pre-trained large language models (LLMs) to model data generated by HMMs via in-context learning (ICL)

Result: LLMs achieve predictive accuracy approaching the theoretical optimum on a diverse set of synthetic HMMs and competitive performance on real-world animal decision-making tasks

Conclusion: ICL can learn and predict HMM-generated sequences

Abstract: 隐马尔可夫模型 (HMMs) 是用于建模具有潜在马尔可夫结构的序列数据的基本工具，但将其应用于真实世界的数据在计算上仍然具有挑战性。在这项工作中，我们表明预训练的大型语言模型 (LLMs) 可以通过上下文学习 (ICL) 有效地建模由 HMM 生成的数据——即它们从提示中的示例推断模式的能力。在一组不同的合成 HMM 上，LLM 实现了接近理论最优的预测精度。我们发现了受 HMM 属性影响的新颖的缩放趋势，并为这些经验观察提供了理论猜想。我们还为科学家提供了使用 ICL 作为复杂数据诊断工具的实用指南。在真实世界的动物决策任务中，ICL 实现了与人类专家设计的模型相媲美的性能。据我们所知，这是第一个证明 ICL 可以学习和预测 HMM 生成的序列的例子——这一进步加深了我们对 LLM 中上下文学习的理解，并确立了其作为揭示复杂科学数据中隐藏结构的强大工具的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07298) | **Categories:** cs.LG, cs.AI

---

### [5] [Mobility-Aware Asynchronous Federated Learning with Dynamic Sparsification](https://arxiv.org/abs/2506.07328)
*Jintao Yan, Tan Chen, Yuxuan Sun, Zhaojun Nan, Sheng Zhou, Zhisheng Niu*

Main category: cs.LG

TL;DR: 本文提出了一种移动感知动态稀疏化（MADS）算法，以优化异步联邦学习中由于设备移动性导致的稀疏化和模型陈旧问题，从而提高模型收敛性。


<details>
  <summary>Details</summary>
Motivation: 设备移动性引入了间歇性连接，这需要梯度稀疏化并导致模型陈旧，共同影响AFL的收敛性。

Method: 提出了一个移动感知动态稀疏化（MADS）算法，该算法基于接触时间和模型陈旧度优化稀疏化程度。

Result: 推导出了闭式解，表明在低速条件下，MADS增加稀疏化程度以增强收敛性，而在高速条件下，它降低稀疏化程度以保证在有限接触时间内可靠的上传。

Conclusion: 实验结果验证了理论分析，并且提出的MADS算法在CIFAR-10图像分类准确率上提高了8.76%，在Argoverse轨迹预测数据集上的平均位移误差降低了9.46%。

Abstract: 异步联邦学习（AFL）允许多个移动设备进行分布式模型训练，允许每个设备独立更新其本地模型，而无需等待其他设备。然而，设备移动性引入了间歇性连接，这需要梯度稀疏化并导致模型陈旧，共同影响AFL的收敛性。本文建立了一个理论模型，以表征稀疏化、模型陈旧和移动性引起的接触模式之间的相互作用，以及它们对AFL收敛性的共同影响。基于该分析，我们提出了一种移动感知动态稀疏化（MADS）算法，该算法基于接触时间和模型陈旧度优化稀疏化程度。推导出了闭式解，表明在低速条件下，MADS增加稀疏化程度以增强收敛性，而在高速条件下，它降低稀疏化程度以保证在有限接触时间内可靠的上传。实验结果验证了理论分析。与最先进的基准相比，MADS算法在CIFAR-10数据集上的图像分类准确率提高了8.76%，在Argoverse轨迹预测数据集上的平均位移误差降低了9.46%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07328) | **Categories:** cs.LG

---

### [6] [Anomaly Detection and Early Warning Mechanism for Intelligent Monitoring Systems in Multi-Cloud Environments Based on LLM](https://arxiv.org/abs/2506.07407)
*Yihong Jin, Ze Yang, Juntian Liu, Xinhe Xu*

Main category: cs.LG

TL;DR: 提出了一种基于LLM的多云环境智能监控异常检测和预警机制，提高了检测精度和实时性。


<details>
  <summary>Details</summary>
Motivation: 随着多云环境的快速发展，确保智能监控系统的安全性和可靠性变得越来越重要。

Method: 提出了一种基于大规模语言模型（LLM）的多云环境智能监控系统异常检测和预警机制，创新性地引入了多层次特征提取方法，结合LLM的自然语言处理能力与传统机器学习方法。

Result: 该模型在检测精度和延迟方面明显优于传统异常检测系统，并显著提高了云基础设施的弹性和主动管理能力。

Conclusion: 实验结果表明，该模型在检测精度和延迟方面均优于传统异常检测系统，显著提高了云基础设施的弹性和主动管理能力。

Abstract: 随着多云环境的快速发展，确保智能监控系统的安全性和可靠性变得越来越重要。本文提出了一种基于大规模语言模型（LLM）的多云环境智能监控系统异常检测和预警机制。在现有监控框架的基础上，该模型创新性地引入了一种多层次特征提取方法，将LLM的自然语言处理能力与传统机器学习方法相结合，以提高异常检测的准确性，并提高实时响应效率。通过引入LLM的上下文理解能力，该模型可以动态适应不同的云服务提供商和环境，从而更有效地检测异常模式并预测潜在的故障。实验结果表明，该模型在检测精度和延迟方面均优于传统异常检测系统，并显著提高了云基础设施的弹性和主动管理能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07407) | **Categories:** cs.LG, cs.AI

---

### [7] [LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments](https://arxiv.org/abs/2506.07416)
*Jin Huang, Yuchao Jin, Le An, Josh Park*

Main category: cs.LG

TL;DR: 该论文提出了一种高效的VLM流水线，通过patch和token选择以及推测解码，显著降低了嵌入式设备上的计算开销，实现了实时部署。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在解决在机器人和自动驾驶等嵌入式设备上部署视觉-语言模型（VLM）时面临的计算开销过大的问题。

Method: 该论文提出了一种高效的视觉-语言模型（VLM）流水线，该流水线通过联合利用patch选择来过滤不相关的相机视图，token选择模块来减少LLM的输入序列长度，以及推测解码来加速token生成，从而显著降低计算开销。

Result: 在NVIDIA DRIVE Thor平台上进行的自动驾驶应用评估表明，该流水线在不影响任务准确性的前提下，实现了2.5倍的端到端延迟降低。当应用FP8后训练量化时，加速效果进一步提高到3.2倍。

Conclusion: 该论文展示了一个在资源受限环境中实现实时VLM部署的可行方案。

Abstract: 本文介绍了一种高效的视觉-语言模型（VLM）流水线，该流水线专门为部署在机器人和自动驾驶等嵌入式设备上进行了优化。该流水线通过联合利用patch选择来过滤不相关的相机视图，token选择模块来减少LLM的输入序列长度，以及推测解码来加速token生成，从而显著降低计算开销。在NVIDIA DRIVE Thor平台上进行的自动驾驶应用评估表明，我们的流水线在不影响任务准确性的前提下，实现了2.5倍的端到端延迟降低。当应用FP8后训练量化时，加速效果进一步提高到3.2倍。这些结果表明，我们的流水线是在资源受限环境中实现实时VLM部署的可行解决方案。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07416) | **Categories:** cs.LG, cs.AI

---

### [8] [Aircraft Trajectory Dataset Augmentation in Latent Space](https://arxiv.org/abs/2506.07585)
*Seokbin Yoon, Keumjin Lee*

Main category: cs.LG

TL;DR: 本文提出了一种名为ATRADA的新框架，用于生成高质量的合成飞机轨迹数据，以增强飞机轨迹数据集。


<details>
  <summary>Details</summary>
Motivation: 飞机轨迹建模在空中交通管理（ATM）中起着至关重要的作用，并且对于各种下游任务（包括冲突检测和着陆时间预测）非常重要。通过添加合成生成的轨迹数据进行数据集增强对于开发更强大的飞机轨迹模型并确保轨迹数据集足够且平衡是必要的。

Method: 该框架使用Transformer编码器学习原始轨迹数据集中的潜在模式，并使用主成分分析（PCA）和高斯混合模型（GMM）在降维空间中拟合数据点的概率分布，最后使用多层感知器（MLP）解码新样本。

Result: 实验表明，该框架能够有效地生成新的、高质量的合成飞机轨迹数据，并与多个基线的结果进行了比较。

Conclusion: 该框架能够有效地生成新的、高质量的合成飞机轨迹数据。

Abstract: 飞机轨迹建模在空中交通管理（ATM）中起着至关重要的作用，对于冲突检测和着陆时间预测等下游任务至关重要。为了开发更强大的飞机轨迹模型并确保轨迹数据集的充分性和平衡性，需要通过添加合成生成的轨迹数据来进行数据集增强。本文提出了一种名为ATRADA的飞机轨迹数据集增强新框架。在该框架中，Transformer编码器学习原始轨迹数据集中的潜在模式，并将每个数据点转换为学习到的潜在空间中的上下文向量。然后，使用主成分分析（PCA）将潜在空间中的转换数据集投影到降维空间中，并应用高斯混合模型（GMM）来拟合降维空间中数据点的概率分布。最后，从拟合的GMM中抽取新样本，将样本的维度恢复到原始维度，并使用多层感知器（MLP）对其进行解码。实验表明，该框架能够有效地生成新的、高质量的合成飞机轨迹数据，并与多个基线的结果进行了比较。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07585) | **Categories:** cs.LG

---

### [9] [Improving large language models with concept-aware fine-tuning](https://arxiv.org/abs/2506.07833)
*Michael K. Chen, Xikun Zhang, Jiaxing Huang, Dacheng Tao*

Main category: cs.LG

TL;DR: CAFT提出了一种新的多token训练方法，显著提升了LLM在各种任务中的性能，并为LLM微调带来了新的可能性。


<details>
  <summary>Details</summary>
Motivation: 现有的LLM的next-token预测范式限制了其形成连贯、高层次概念的能力，阻碍了更深层次的理解和推理。

Method: 提出了概念感知微调（CAFT）方法，通过学习跨多个token的序列，促进更强的概念感知学习。

Result: 实验表明，与传统的next-token微调方法相比，CAFT在文本摘要和蛋白质设计等多种任务中均表现出显著的改进。

Conclusion: CAFT方法在多种任务上表现出显著的改进，表明多token训练在LLM微调中具有广泛的应用前景。

Abstract: 大型语言模型（LLM）已成为现代人工智能的基石。然而，现有的next-token预测范式从根本上限制了它们形成连贯、高层次概念的能力，这成为了实现类人理解和推理的关键障碍。以“核糖核酸”这个短语为例：LLM会首先将其分解为多个token，即人为的文本片段（“核”、“糖”等），然后按顺序学习每个token，而不是将这个短语理解为一个统一的、连贯的语义实体。这种碎片化的表示阻碍了更深层次的概念理解，并最终阻碍了真正智能系统的发展。作为回应，我们引入了概念感知微调（CAFT），这是一种新颖的多token训练方法，它重新定义了LLM的微调方式。通过学习跨多个token的序列，该方法促进了更强的概念感知学习。我们的实验表明，与传统的next-token微调方法相比，CAFT在包括文本摘要等传统应用以及从头蛋白质设计等特定领域应用在内的各种任务中均表现出显著的改进。多token预测以前仅在成本高昂的预训练阶段才有可能实现；据我们所知，CAFT是第一个将多token设置引入到后训练阶段的方法，从而有效地为更广泛的从业者和研究人员社区普及了它的好处。最后，我们提出的方法出人意料的有效性表明，它对机器学习研究界具有更广泛的意义。所有代码和数据都可以在https://github.com/michaelchen-lab/caft-llm上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07833) | **Categories:** cs.LG, cs.AI, cs.CL

---


## cs.MA [cs.MA]
### [1] [Learn as Individuals, Evolve as a Team: Multi-agent LLMs Adaptation in Embodied Environments](https://arxiv.org/abs/2506.07232)
*Xinran Li, Chenjia Bai, Zijian Li, Jiakun Zheng, Ting Xiao, Jun Zhang*

Main category: cs.MA

TL;DR: LIET框架使LLM智能体能够在多智能体具身环境中学习和进化，从而提高规划和协作能力。


<details>
  <summary>Details</summary>
Motivation: 现有的基于LLM的规划算法在多智能体具身场景中的适应能力较弱。

Method: 提出了一种“个体学习，团队进化 (LIET)” 的范式，用于多智能体LLM的适应。在个体层面，LLM智能体从探索性数据集中学习局部效用函数，以更好地理解具身环境。在团队层面，LLM智能体协作并迭代地维护和更新共享的合作知识列表，以指导更有效的沟通。

Result: 在Communicative Watch-And-Help和ThreeD-World Multi-Agent Transport基准测试中，LIET优于现有基线，并表现出强大的协作规划能力。

Conclusion: LIET通过个体学习和团队进化，实现了LLM智能体在复杂环境中的有效适应和协作规划。

Abstract: 大型语言模型 (LLM) 拥有广泛的知识库和强大的推理能力，使其成为具身环境中复杂多智能体规划的有前途的工具。然而，尽管 LLM 具有先进的能力以及智能体方法的复杂模块化设计，但现有的基于 LLM 的规划算法仍然受到多智能体具身场景适应能力弱的限制。我们通过引入一个框架来解决这一限制，该框架使 LLM 智能体能够在测试之前和期间进行学习和进化，从而使他们能够获得与环境相关的知识，从而更好地进行规划和加强沟通，从而改善合作。受到多智能体强化学习中集中训练和分散执行的启发，我们提出了一种“个体学习，团队进化 (LIET)” 的范式，用于多智能体 LLM 适应。在个体层面，LLM 智能体从探索性数据集中学习局部效用函数，以更好地理解具身环境，然后在测试时查询该函数以支持知情决策。在团队层面，LLM 智能体协作并迭代地维护和更新共享的合作知识列表，基于新的经验，使用它来指导更有效的沟通。通过将个人学习与团队进化相结合，LIET 能够为 LLM 智能体实现全面而灵活的适应。我们在 Communicative Watch-And-Help 和 ThreeD-World Multi-Agent Transport 基准测试中的实验表明，LIET（使用 LLaMA 和 GPT-4o 实例化）优于现有基线，并表现出强大的协作规划能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07232) | **Categories:** cs.MA, cs.AI, cs.LG

---

### [2] [G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems](https://arxiv.org/abs/2506.07398)
*Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, Shuicheng Yan*

Main category: cs.MA

TL;DR: G-Memory为多智能体系统引入了一种分层记忆结构，有效提升了其协作能力和任务表现。


<details>
  <summary>Details</summary>
Motivation: 现有的多智能体系统记忆机制过于简单，忽略了智能体间的协作轨迹，且缺乏跨试验和智能体定制。

Method: 提出了G-Memory，一个受组织记忆理论启发的层级式、代理式多智能体系统记忆系统，通过洞察、查询和交互图管理多智能体交互。

Result: 在五个基准测试、三个大型语言模型和三个流行的多智能体框架上的大量实验表明，G-Memory在不修改原始框架的情况下，将具身行动的成功率和知识问答的准确率分别提高了高达20.89%和10.12%。

Conclusion: G-Memory通过分层记忆结构，显著提升了多智能体系统在具身行动和知识问答任务中的性能。

Abstract: 大型语言模型驱动的多智能体系统（MAS）展示了远超单个LLM智能体的认知和执行能力，但其自我进化能力仍受到不发达的记忆架构的阻碍。经过仔细检查，我们震惊地发现，目前流行的MAS记忆机制（1）过于简单，完全忽略了细致的智能体间协作轨迹，并且（2）缺乏跨试验和特定于智能体的定制，这与为单个智能体开发的富有表现力的记忆形成鲜明对比。为了弥合这一差距，我们引入了G-Memory，一个受组织记忆理论启发的MAS层级式、代理式记忆系统，它通过三层图层次结构管理冗长的MAS交互：洞察、查询和交互图。在接收到新的用户查询时，G-Memory执行双向记忆遍历，以检索$\\{textit{高层次、可推广的洞察}\\}$，使系统能够利用跨试验知识，以及$\\{textit{细粒度、浓缩的交互轨迹}\\}$，紧凑地编码先前的协作经验。在任务执行时，整个层次结构通过吸收新的协作轨迹而进化，从而培养智能体团队的逐步进化。在五个基准测试、三个LLM骨干和三个流行的MAS框架上的大量实验表明，G-Memory在不修改原始框架的情况下，将具身行动的成功率和知识问答的准确率分别提高了高达20.89%和10.12%。我们的代码可在https://github.com/bingreeky/GMemory上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07398) | **Categories:** cs.MA, cs.CL, cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [BeliefMapNav: 3D Voxel-Based Belief Map for Zero-Shot Object Navigation](https://arxiv.org/abs/2506.06487)
*Zibo Zhou, Yue Hu, Lingkai Zhang, Zonglin Li, Siheng Chen*

Main category: cs.RO

TL;DR: 本文提出了一种基于3D体素置信地图的导航系统BeliefMapNav，有效提升了零样本物体导航的成功率和效率。


<details>
  <summary>Details</summary>
Motivation: 现有的大型语言模型（LLM）和视觉-语言模型（VLM）在零样本物体导航（ZSON）中存在局限性，它们通常贪婪地选择下一个目标，缺乏对环境的全局理解和有效的空间推理能力。

Method: 提出了一种新颖的基于3D体素的置信地图，该地图估计目标在体素化3D空间中的先验存在分布。在此基础上，引入了BeliefMapNav，一个高效的导航系统，它将LLM语义推理与3D分层语义体素空间相结合，以实现精确的目标位置估计，并整合了顺序路径规划以实现高效的全局导航决策。

Result: 在HM3D、MP3D和HSSD基准测试中，BeliefMapNav取得了最先进的成功率（SR）和路径长度加权成功率（SPL），与之前最佳SR方法相比，SPL显著提高了46.4%。

Conclusion: BeliefMapNav在HM3D、MP3D和HSSD基准测试中取得了最先进的成功率（SR）和路径长度加权成功率（SPL），与之前最佳SR方法相比，SPL显著提高了46.4%，验证了其有效性和效率。

Abstract: 本文提出了一种名为BeliefMapNav的导航系统，旨在解决零样本物体导航中大型语言模型和视觉-语言模型在空间推理和全局理解方面的局限性。该系统利用3D体素化的置信地图来估计目标物体的位置分布，并结合了LLM的语义信息和实时观测数据。BeliefMapNav通过在3D空间中整合语义推理和顺序路径规划，实现了更精确的目标定位和更高效的全局导航。实验结果表明，该系统在多个基准测试中取得了显著的性能提升。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06487) | **Categories:** cs.RO

---

### [2] [Enhancing Robot Safety via MLLM-Based Semantic Interpretation of Failure Data](https://arxiv.org/abs/2506.06570)
*Aryaman Gupta, Yusuf Umut Ciftci, Somil Bansal*

Main category: cs.RO

TL;DR: 该论文提出了一种利用多模态大型语言模型自动组织机器人失败数据，从而实现可扩展的失败学习的方法。


<details>
  <summary>Details</summary>
Motivation: 随着机器人系统越来越多地集成到现实环境中，它们不可避免地会遇到导致失败的各种非结构化场景。手动分析大规模故障数据集是不切实际的。

Method: 该方法利用在互联网规模数据上训练的多模态大型语言模型 (MLLM) 的推理能力，从原始感知轨迹中推断出高级失败原因，并在未整理的失败日志中发现可解释的结构。

Result: 研究表明，发现的失败模式可以指导有针对性的数据收集以进行策略改进，从而加速代理策略和整体安全性的迭代改进。此外，这些语义集群可用于在线失败检测，为实时自适应提供轻量级但功能强大的保障。

Conclusion: 该框架通过将现实世界的失败转化为可操作和可解释的适应信号，从而增强了机器人学习和鲁棒性。

Abstract: 随着机器人系统越来越多地集成到现实环境中，从自动驾驶汽车到家庭助手，它们不可避免地会遇到导致失败的各种非结构化场景。虽然这些失败带来了安全性和可靠性方面的挑战，但它们也为改进未来的性能提供了丰富的感知数据。然而，手动分析大规模故障数据集是不切实际的。在这项工作中，我们提出了一种自动将大规模机器人失败数据组织成具有语义意义的集群的方法，从而无需人工干预即可实现可扩展的失败学习。我们的方法利用在互联网规模数据上训练的多模态大型语言模型 (MLLM) 的推理能力，从原始感知轨迹中推断出高级失败原因，并在未整理的失败日志中发现可解释的结构。这些语义集群揭示了潜在的模式和假设的失败原因，从而实现可扩展的经验学习。我们证明了发现的失败模式可以指导有针对性的数据收集以进行策略改进，从而加速代理策略和整体安全性的迭代改进。此外，我们表明这些语义集群可用于在线失败检测，为实时自适应提供轻量级但功能强大的保障。我们证明了该框架通过将现实世界的失败转化为可操作和可解释的适应信号，从而增强了机器人学习和鲁棒性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06570) | **Categories:** cs.RO

---

### [3] [DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](https://arxiv.org/abs/2506.06659)
*Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, Zuxuan Wu*

Main category: cs.RO

TL;DR: DriveSuprim通过粗到精的过滤、旋转增强和自蒸馏框架，提升了自动驾驶汽车在复杂环境中的安全性和轨迹质量。


<details>
  <summary>Details</summary>
Motivation: 在复杂的驾驶环境中，自动驾驶汽车必须安全导航。依赖于单一预测路径通常不能明确评估预测轨迹的安全性；选择方法在精确选择最佳选项和区分细微但对安全至关重要的差异方面面临优化挑战，尤其是在罕见或未充分表示的场景中。

Method: 提出了一种由粗到精的范例，用于渐进式候选过滤；一种基于旋转的增强方法，以提高在分布外场景中的鲁棒性；以及一个自蒸馏框架，以稳定训练。

Result: DriveSuprim达到了最先进的性能，在NAVSIM v1中达到93.5%的PDMS，在NAVSIM v2中达到87.1%的EPDMS，且没有额外数据。

Conclusion: DriveSuprim在多种驾驶场景中表现出卓越的安全性能，包括避撞和遵守规则，同时保持高质量的轨迹。

Abstract: 在复杂的驾驶环境中，自动驾驶汽车必须安全导航。像基于回归的方法那样依赖单一预测路径，通常不能明确评估预测轨迹的安全性。选择方法通过生成和评估多个轨迹候选，并预测每个轨迹的安全得分来解决这个问题，但它们在精确选择最佳选项和区分细微但对安全至关重要的差异方面面临优化挑战，尤其是在罕见或未充分表示的场景中。我们提出了DriveSuprim来克服这些挑战，并通过一种由粗到精的范例进行渐进式候选过滤，一种基于旋转的增强方法来提高在分布外场景中的鲁棒性，以及一个自蒸馏框架来稳定训练，从而推进了基于选择的范例。DriveSuprim达到了最先进的性能，在NAVSIM v1中达到93.5%的PDMS，在NAVSIM v2中达到87.1%的EPDMS，且没有额外数据，这表明了其卓越的安全关键能力，包括避撞和遵守规则，同时在各种驾驶场景中保持高质量的轨迹。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06659) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [4] [Generalized Trajectory Scoring for End-to-end Multimodal Planning](https://arxiv.org/abs/2506.06664)
*Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, Jose M. Alvarez*

Main category: cs.RO

TL;DR: GTRS 结合粗粒度和细粒度轨迹评估，为端到端多模态规划提供了一个统一的框架。


<details>
  <summary>Details</summary>
Motivation: 现有的轨迹评分器在静态轨迹集和动态轨迹集上都存在泛化性问题。静态词汇表提供有效的粗粒度离散化，但难以进行细粒度调整，而动态提议提供详细的精度，但无法捕获更广泛的轨迹分布。

Method: GTRS 包含三个创新点：(1) 基于扩散的轨迹生成器，用于生成多样化的细粒度提议；(2) 词汇泛化技术，通过在超密集轨迹集上使用 dropout 正则化训练评分器，使其能够在较小的子集上进行鲁棒的推理；(3) 传感器增强策略，增强了域外泛化能力，同时结合了关键轨迹判别的细化训练。

Result: GTRS 作为 Navsim v2 挑战赛的获胜解决方案，即使在传感器输入不佳的情况下也表现出卓越的性能，接近依赖于地面实况感知的特权方法。

Conclusion: GTRS通过结合粗粒度和细粒度的轨迹评估，在端到端多模态规划中实现了卓越的性能，尤其是在传感器输入不佳的情况下。

Abstract: 端到端多模态规划是自动驾驶中一个很有前景的范例，它可以通过不同的轨迹候选方案进行决策。一个关键的组成部分是一个强大的轨迹评分器，能够从这些候选方案中选择最佳轨迹。虽然最近的轨迹评分器侧重于对大型静态轨迹集或小型动态生成的轨迹集进行评分，但两种方法在泛化方面都面临着重大限制。静态词汇表提供有效的粗粒度离散化，但难以进行细粒度调整，而动态提议提供详细的精度，但无法捕获更广泛的轨迹分布。为了克服这些挑战，我们提出了 GTRS（广义轨迹评分），这是一个统一的端到端多模态规划框架，它结合了粗粒度和细粒度的轨迹评估。GTRS 包含三个互补的创新点：(1) 基于扩散的轨迹生成器，用于生成多样化的细粒度提议；(2) 词汇泛化技术，通过在超密集轨迹集上使用 dropout 正则化训练评分器，使其能够在较小的子集上进行鲁棒的推理；(3) 传感器增强策略，增强了域外泛化能力，同时结合了关键轨迹判别的细化训练。作为 Navsim v2 挑战赛的获胜解决方案，GTRS 即使在传感器输入不佳的情况下也表现出卓越的性能，接近依赖于地面实况感知的特权方法。代码将在 https://github.com/NVlabs/GTRS 提供。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06664) | **Categories:** cs.RO, cs.CV

---

### [5] [SpikePingpong: High-Frequency Spike Vision-based Robot Learning for Precise Striking in Table Tennis Game](https://arxiv.org/abs/2506.06690)
*Hao Wang, Chengkai Hou, Xianglong Li, Yankai Fu, Chenxuan Li, Ning Chen, Gaole Dai, Jiaming Liu, Tiejun Huang, Shanghang Zhang*

Main category: cs.RO

TL;DR: SpikePingpong结合脉冲视觉和模仿学习，显著提升了机器人乒乓球的精度和策略水平。


<details>
  <summary>Details</summary>
Motivation: 在现实世界中控制高速物体仍然是机器人领域一个具有挑战性的前沿问题。乒乓球作为这个问题的理想试验台，需要快速拦截快速移动的球并精确调整它们的轨迹。这项任务提出了两个基本挑战：它需要一个高精度的视觉系统，能够准确预测球的轨迹；并且需要智能的战略规划，以确保将球精确放置到目标区域。

Method: 该系统集成了两个关键模块：SONIC，一个基于脉冲相机的模块，通过补偿空气阻力和摩擦等实际不确定性，实现了毫米级的球拍接触预测精度；IMPACT，一个战略规划模块，能够准确地将球放置到目标区域。

Result: SpikePingpong在30厘米精度目标区域的成功率达到了91%，在更具挑战性的20厘米精度任务中达到了71%，分别超过了之前的最先进方法38%和37%。

Conclusion: SpikePingpong系统通过整合基于脉冲的视觉和模仿学习，显著提高了机器人乒乓球的精度和策略性，为高速动态任务中的机器人控制提供了新的研究视角。

Abstract: 本文介绍了一种名为SpikePingpong的新型系统，该系统集成了基于脉冲的视觉和模仿学习，用于高精度机器人乒乓球。我们的方法引入了两个关键尝试，直接解决了上述挑战：SONIC，一个基于脉冲相机的模块，通过补偿空气阻力和摩擦等实际不确定性，实现了毫米级的球拍接触预测精度；IMPACT，一个战略规划模块，能够准确地将球放置到目标区域。该系统利用一个20 kHz的脉冲相机进行高时间分辨率的球跟踪，并结合高效的神经网络模型进行实时轨迹校正和击球规划。实验结果表明，SpikePingpong在30厘米精度目标区域的成功率达到了91%，在更具挑战性的20厘米精度任务中达到了71%，分别超过了之前的最先进方法38%和37%。这些显著的性能改进使得能够稳健地实施复杂的战术游戏策略，为高速动态任务中的机器人控制提供了新的研究视角。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06690) | **Categories:** cs.RO, cs.CV

---

### [6] [Multimodal Spatial Language Maps for Robot Navigation and Manipulation](https://arxiv.org/abs/2506.06862)
*Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard*

Main category: cs.RO

TL;DR: 本文提出了一种多模态空间语言地图，它融合了预训练的多模态特征与环境的 3D 重建，实现了零样本空间和多模态目标导航。


<details>
  <summary>Details</summary>
Motivation: 以往的方法与环境地图脱节，缺乏几何地图的空间精度，或者忽略了视觉之外的额外模态信息。

Method: 提出了多模态空间语言地图，它将预训练的多模态特征与环境的 3D 重建融合。

Result: 在模拟和现实环境中的实验表明，多模态空间语言地图能够实现零样本空间和多模态目标导航，并在模糊场景中将召回率提高 50%。

Conclusion: 多模态空间语言地图能够实现零样本空间和多模态目标导航，并在模糊场景中将召回率提高 50%。

Abstract: 本文提出了一种多模态空间语言地图，作为一种空间地图表示，它将预训练的多模态特征与环境的 3D 重建融合。我们使用标准的探索方法自主构建这些地图。我们展示了地图的两个实例，即视觉-语言地图 (VLMaps) 及其通过添加音频信息扩展到音频-视觉-语言地图 (AVLMaps)。当与大型语言模型 (LLM) 结合使用时，VLMaps 可以 (i) 将自然语言命令转换为直接位于地图中的开放词汇空间目标（例如，“沙发和电视之间”），并且 (ii) 可以在不同的机器人形态之间共享，以按需生成量身定制的障碍物地图。在上述功能的基础上，AVLMaps 通过融合来自预训练的多模态基础模型的特征，引入了统一的 3D 空间表示，集成了音频、视觉和语言线索，从而扩展了 VLMaps。这使机器人能够将多模态目标查询（例如，文本、图像或音频片段）定位到空间位置以进行导航。此外，在模糊环境中，结合不同的感官输入显着增强了目标消歧能力。在模拟和真实环境中的实验表明，我们的多模态空间语言地图能够实现零样本空间和多模态目标导航，并在模糊场景中将召回率提高 50%。这些功能扩展到移动机器人和桌面机械手，支持在视觉、音频和空间线索的引导下进行导航和交互。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06862) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG, cs.SD, eess.AS

---

### [7] [Taking Flight with Dialogue: Enabling Natural Language Control for PX4-based Drone Agent](https://arxiv.org/abs/2506.07509)
*Shoon Kit Lim, Melissa Jia Ying Chong, Jing Huey Khor, Ting Yang Ling*

Main category: cs.RO

TL;DR: 提出了一个开源的自主无人机自然语言控制框架，并在实际平台上进行了验证。


<details>
  <summary>Details</summary>
Motivation: 为了普及自主无人机的自然语言控制，解决现有无人机多模态视觉语言系统依赖闭源模型的问题。

Method: 提出了一个开源的代理框架，集成了PX4飞控、ROS 2中间件和本地托管模型。

Result: 对四个大型语言模型家族的命令生成和三个视觉语言模型家族的场景理解进行了基准测试。

Conclusion: 通过开源框架和本地模型，实现了自主无人机的自然语言控制，并在仿真和实际平台上进行了性能评估。

Abstract: 近年来，具身和物理人工智能（AI）的进展主要集中在人形和轮式机器人等地面平台上，而对空中机器人的研究相对较少。同时，最先进的无人机（UAV）多模态视觉语言系统通常依赖于只有资源充足的组织才能访问的闭源模型。为了普及自主无人机的自然语言控制，我们提出了一个开源的代理框架，该框架集成了基于PX4的飞行控制、机器人操作系统2（ROS 2）中间件，并使用Ollama本地托管模型。我们在仿真和定制四轴飞行器平台上评估了性能，对四个大型语言模型（LLM）家族的命令生成和三个视觉语言模型（VLM）家族的场景理解进行了基准测试。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07509) | **Categories:** cs.RO, I.2.7; I.2.9; I.2.10

---

### [8] [Semantics-aware Predictive Inspection Path Planning](https://arxiv.org/abs/2506.06560)
*Mihir Dharmadhikari, Kostas Alexis*

Main category: cs.RO

TL;DR: 该论文提出了一种语义感知的预测规划方法，通过识别和利用环境中的重复语义模式来优化检查路径，从而显著减少检查时间。


<details>
  <summary>Details</summary>
Motivation: 针对工业环境中特定对象或结构（称为“语义”）的检查需求，例如船舶内部的压载水舱，通常呈现出结构化和重复的空间排列。

Method: 提出了一种语义感知的预测规划（SPP）范例，该范例利用语义场景图中的空间重复模式来预测环境的未见部分，并提出了两种定制的检查路径规划策略。

Result: 仿真和实验结果表明，与现有技术相比，在检查时间方面有显著改进，同时保持了相等或更好的语义表面覆盖率。

Conclusion: 在模拟和实际船舶压载舱的实验中，该方法在保持同等或更好语义表面覆盖率的同时，显著缩短了检查时间，优于现有技术。

Abstract: 本文提出了一种新颖的语义感知检查路径规划范例，称为“语义感知预测规划”（SPP）。需要检查特定对象或结构（称为“语义”）的工业环境，例如船舶内部的压载水舱，通常呈现出结构化和重复的语义空间排列。受此启发，我们首先贡献了一种算法，该算法识别语义场景图表示中语义的空间重复模式（精确或非精确），并使用这些模式预测环境中未见部分中图的演变。此外，还提出了两种针对压载水舱检查量身定制的检查路径规划策略，这些策略利用了这些预测。为了评估新型预测规划范例的性能，进行了仿真和实验评估。首先，我们进行了一项仿真研究，将该方法与相关的最新技术进行比较，并进一步展示了其处理不完善模式的能力的测试。其次，我们将我们的方法部署在运行在两艘真实船舶的压载舱内的耐碰撞无人机上。仿真和现场实验结果表明，与现有技术相比，在检查时间方面有显著改进，同时保持了相等或更好的语义表面覆盖率。一组描述该方法不同部分和现场部署的视频可在 https://tinyurl.com/spp-videos 上找到。这项工作的代码可在 https://github.com/ntnu-arl/predictive_planning_ros 上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06560) | **Categories:** cs.RO

---

### [9] [RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](https://arxiv.org/abs/2506.06677)
*Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, Si Liu*

Main category: cs.RO

TL;DR: RoboCerebra是一个新的基准，用于评估机器人长时程操作中的高级推理能力。


<details>
  <summary>Details</summary>
Motivation: 现有工作主要集中于反应式System 1策略，未能充分利用VLMs在语义推理和长时程规划方面的优势。由于当前基准测试的时间尺度和结构复杂性有限，这些以审慎的、目标导向的思维为特征的System 2能力仍有待探索。为了弥补这一差距，我们引入了RoboCerebra，这是一个用于评估长时程机器人操作中高级推理的基准。

Method: 该方法包括：（1）一个大规模的模拟数据集，具有扩展的任务范围和家庭环境中不同的子任务序列；（2）一个分层框架，将高级VLM规划器与低级视觉-语言-动作（VLA）控制器相结合；（3）一个通过结构化的System 1-System 2交互，针对规划、反思和记忆的评估协议。

Result: 与之前的基准测试相比，RoboCerebra具有明显更长的动作序列和更密集的注释。

Conclusion: 通过RoboCerebra基准测试，对最先进的视觉语言模型（VLMs）作为System 2模块进行了基准测试，并分析了它们在关键认知维度上的表现，从而推动了更有能力和更通用的机器人规划器的开发。

Abstract: 视觉语言模型（VLMs）的最新进展使得指令控制的机器人系统具有了更好的泛化能力。然而，目前大多数工作都集中在反应式的System 1策略上，没有充分利用VLMs在语义推理和长程规划方面的优势。由于当前基准测试的时间尺度和结构复杂性有限，这些以审慎的、目标导向的思维为特征的System 2能力仍有待探索。为了弥补这一差距，我们引入了RoboCerebra，这是一个用于评估长时程机器人操作中高级推理的基准。RoboCerebra包括：（1）一个大规模的模拟数据集，具有扩展的任务范围和家庭环境中不同的子任务序列；（2）一个分层框架，将高级VLM规划器与低级视觉-语言-动作（VLA）控制器相结合；（3）一个通过结构化的System 1-System 2交互，针对规划、反思和记忆的评估协议。该数据集通过自上而下的流程构建，GPT生成任务指令并将其分解为子任务序列。人类操作员在模拟环境中执行子任务，从而产生具有动态对象变化的高质量轨迹。与之前的基准测试相比，RoboCerebra具有明显更长的动作序列和更密集的注释。我们进一步将最先进的VLMs作为System 2模块进行基准测试，并分析它们在关键认知维度上的表现，从而推动了更有能力和更通用的机器人规划器的发展。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06677) | **Categories:** cs.RO, cs.CV

---

### [10] [RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](https://arxiv.org/abs/2506.06683)
*Shiying Duan, Pei Ren, Nanxiang Jiang, Zhengping Che, Jian Tang, Yifan Sun, Zhaoxin Fan, Wenjun Wu*

Main category: cs.RO

TL;DR: RoboPARA是一个基于大型语言模型的双臂机器人任务并行规划框架，它通过依赖图和图重遍历优化任务并行性。


<details>
  <summary>Details</summary>
Motivation: 现有方法在任务规划中取得了可喜的成果，但它们通常无法充分优化任务并行性，从而限制了双臂协作的潜力。

Method: RoboPARA 采用两阶段流程：(1) 基于依赖图的规划候选生成，构建有向无环图 (DAG) 以建模任务依赖关系并消除冗余；(2) 基于图重遍历的双臂并行规划，优化 DAG 遍历以最大化并行性，同时保持任务连贯性。

Result: 在 X-DAPT 数据集上的大量实验表明，RoboPARA 显着优于现有方法，尤其是在复杂的任务组合中。

Conclusion: RoboPARA在复杂任务组合中表现出更高的效率和可靠性。

Abstract: 双臂机器人在提高复杂多任务场景中的效率和灵活性方面发挥着关键作用。虽然现有的方法在任务规划中取得了可喜的成果，但它们通常无法充分优化任务并行性，从而限制了双臂协作的潜力。为了解决这个问题，我们提出了 RoboPARA，这是一种新颖的由大型语言模型 (LLM) 驱动的双臂任务并行规划框架。RoboPARA 采用两阶段流程：(1) 基于依赖图的规划候选生成，构建有向无环图 (DAG) 以建模任务依赖关系并消除冗余；(2) 基于图重遍历的双臂并行规划，优化 DAG 遍历以最大化并行性，同时保持任务连贯性。此外，我们还介绍了跨场景双臂并行任务数据集 (X-DAPT 数据集)，这是第一个专门用于评估跨不同场景和难度级别的双臂任务并行性的数据集。在 X-DAPT 数据集上的大量实验表明，RoboPARA 显着优于现有方法，尤其是在复杂的任务组合中，实现了更高的效率和可靠性。代码和数据集将在接受后发布。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06683) | **Categories:** cs.RO, cs.AI

---

### [11] [Hierarchical Intention Tracking with Switching Trees for Real-Time Adaptation to Dynamic Human Intentions during Collaboration](https://arxiv.org/abs/2506.07004)
*Zhe Huang, Ye-Ji Mun, Fatemeh Cheraghi Pouria, Katherine Driggs-Campbell*

Main category: cs.RO

TL;DR: 提出了一种分层意图跟踪（HIT）算法，用于协作机器人实时跟踪动态和分层的人类意图。


<details>
  <summary>Details</summary>
Motivation: 协作任务中，人类行为受随时间演变的多层意图指导，协作机器人需要实时准确地跟踪这些动态的人类意图，以适应这些变化。

Method: 提出了一种分层意图跟踪（HIT）算法，通过贝叶斯滤波、向上测量传播和向下后验传播，在所有层级上概率性地跟踪人类意图。

Result: 基于HIT的协作机器人系统优于现有的协作机器人解决方案，在效率、身体负荷和用户舒适度之间取得了平衡，同时确保了安全和任务完成。

Conclusion: 用户研究表明，基于HIT的协作机器人系统在效率、身体负荷和用户舒适度之间取得了平衡，同时确保了安全和任务完成。实验后调查进一步表明，HIT系统通过有效理解跨多个层次的人类意图，增强了用户信任，并最大限度地减少了对用户任务流程的干扰。

Abstract: 在协作任务中，人类行为受到多个随时间演变的意图层次的指导，例如任务序列偏好和交互策略。为了适应这些不断变化的偏好并及时纠正任何不准确的估计，协作机器人必须实时准确地跟踪这些动态的人类意图。我们提出了一种分层意图跟踪（HIT）算法，用于协作机器人，以有效且实时地跟踪动态和分层的人类意图。HIT将人类意图表示为具有任意深度的意图树，并通过贝叶斯滤波、向上测量传播和向下后验传播跨所有层级概率性地跟踪人类意图。我们开发了一种基于HIT的机器人系统，该系统在协作组装任务的交互任务树和验证任务树之间动态切换，使机器人能够有效地协调三个层次的人类意图：任务层（子任务目标位置）、交互层（与机器人交互的模式）和验证层（确认或纠正意图识别）。我们的用户研究表明，基于HIT的协作机器人系统超越了现有的协作机器人解决方案，在效率、身体负荷和用户舒适度之间取得了平衡，同时确保了安全和任务完成。实验后调查进一步表明，HIT系统通过有效理解跨多个层次的人类意图，增强了用户信任，并最大限度地减少了对用户任务流程的干扰。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07004) | **Categories:** cs.RO

---

### [12] [Prime the search: Using large language models for guiding geometric task and motion planning by warm-starting tree search](https://arxiv.org/abs/2506.07062)
*Dongryung Lee, Sejune Joo, Kimin Lee, Beomjoon Kim*

Main category: cs.RO

TL;DR: 提出了一种利用大型语言模型指导几何任务和运动规划的方法，并通过实验验证了其有效性。


<details>
  <summary>Details</summary>
Motivation: 将一组物体搬迁到指定区域的问题可以被视为几何任务和运动规划（G-TAMP）问题。传统的G-TAMP方法依赖于领域无关的启发式方法或从规划经验中学习来指导搜索，这通常需要大量的计算资源或数据。人类通常使用常识来直观地决定在G-TAMP问题中操纵哪些对象。因此，我们建议利用从互联网规模数据中获取常识知识的大型语言模型（LLM）来指导G-TAMP问题中的任务规划。

Method: 利用大型语言模型（LLM）来指导G-TAMP问题中的任务规划。设计了一种基于谓词的提示，该提示编码了从运动规划算法中导出的几何信息。然后查询LLM以生成任务计划，该计划随后用于搜索可行的连续参数集。将蒙特卡洛树搜索（MCTS）扩展到混合动作空间，并使用LLM来指导搜索。使用LLM来预热MCTS。

Result: 在六个不同的G-TAMP问题上，该方法优于以往的LLM规划器和纯搜索算法。

Conclusion: 该方法在六个不同的G-TAMP问题上优于以往的LLM规划器和纯搜索算法。

Abstract: 本文提出了一种利用大型语言模型（LLM）指导几何任务和运动规划（G-TAMP）问题中任务规划的方法。该方法设计了一种基于谓词的提示，编码了从运动规划算法中提取的几何信息，并用以查询LLM生成任务计划。该计划用于搜索可行的连续参数集。为了提高效率，该方法扩展了蒙特卡洛树搜索（MCTS）到混合动作空间，并使用LLM的输出来预热MCTS，避免了在每个节点都调用LLM所带来的高计算成本。实验结果表明，在六个不同的G-TAMP问题上，该方法优于以往的LLM规划器和纯搜索算法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07062) | **Categories:** cs.RO, cs.AI

---

### [13] [Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset](https://arxiv.org/abs/2506.07150)
*Xintao Yan, Erdao Liang, Jiawei Wang, Haojie Zhu, Henry X. Liu*

Main category: cs.RO

TL;DR: 该论文提出了一种自动化的交通信号数据修复方法，通过车辆轨迹和交通知识，显著提高了自动驾驶数据集的质量。


<details>
  <summary>Details</summary>
Motivation: Autonomous vehicle datasets often suffer from missing or inaccurate traffic signal data, which compromises their reliability and negatively impacts model performance.

Method: The paper introduces a fully automated approach that uses vehicle trajectory data and transportation domain knowledge to impute and rectify traffic signal information in the Waymo Open Motion Dataset (WOMD).

Result: The proposed method successfully imputed 71.7% of missing or unknown traffic signal states in the WOMD and reduced the estimated red-light running rate from 15.7% to 2.9%.

Conclusion: The proposed method significantly enhances the quality of autonomous vehicle datasets by accurately imputing missing traffic signal data and rectifying inaccuracies, leading to a substantial reduction in red-light running rates.

Abstract: 自动驾驶车辆（AV）数据集对于人工智能（AI）、自动驾驶和交通工程等研究领域具有重要意义。然而，这些数据集经常遇到与交通信号状态相关的问题，例如数据缺失或不准确。这些问题会损害数据集的可靠性，并对使用它们开发的模型的性能产生不利影响。本研究提出了一种全自动方法，旨在通过利用现有的车辆轨迹数据和交通领域的知识，有效地推算和纠正Waymo开放运动数据集（WOMD）中的交通信号信息。该方法具有鲁棒性和灵活性，能够处理现实场景中不同的交叉口几何形状和交通信号配置。已经在整个WOMD上进行了全面的验证，重点关注超过360,000个与交通信号相关的场景，总共有530,000个真实驾驶场景。在原始数据集中，71.7%的交通信号状态缺失或未知，所有这些状态都已通过我们提出的方法成功推算。此外，在没有真实信号状态的情况下，该方法的准确性是根据车辆轨迹中的闯红灯率来评估的。结果表明，该方法将估计的闯红灯率从原始数据中的15.7%降低到2.9%，从而证明了其在纠正数据不准确性方面的有效性。本文显著提高了AV数据集的质量，为更广泛的AI和AV研究社区做出了贡献，并使各种下游应用受益。代码和改进的交通信号数据在https://github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement上开源。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07150) | **Categories:** cs.RO

---

### [14] [UruBots Autonomous Cars Challenge Pro Team Description Paper for FIRA 2025](https://arxiv.org/abs/2506.07348)
*Pablo Moraes, Mónica Rodríguez, Sebastian Barcelona, Angel Da Silva, Santiago Fernandez, Hiago Sodre, Igor Nunes, Bruna Guterres, Ricardo Grando*

Main category: cs.RO

TL;DR: UruBots团队开发了一种基于深度学习的自动驾驶汽车，能够在特定赛道上实现自主导航和避障。


<details>
  <summary>Details</summary>
Motivation: 为参加2025年FIRA自动驾驶汽车挑战赛（Pro），开发一种能够自主导航通过不同赛道的紧凑型电动汽车。

Method: 使用深度学习模型处理相机图像并控制车辆运动，具体而言，使用超过一万张图像的数据集训练卷积神经网络（CNN）来有效驾驶车辆，通过两个输出（转向和油门）实现。

Result: 该自动驾驶汽车能够在30秒内完成赛道，速度约为每秒0.4米，同时避开障碍物。

Conclusion: 该论文展示了UruBots团队开发的自动驾驶汽车，该车能够在30秒内完成赛道，速度约为每秒0.4米，同时避开障碍物。

Abstract: 本文介绍了UruBots团队为参加2025年FIRA自动驾驶汽车挑战赛（Pro）而开发的自动驾驶汽车。该项目涉及构建一种紧凑型电动汽车，其尺寸与遥控车大致相同，能够自主导航通过不同的赛道。该设计融合了机械和电子组件以及机器学习算法，使车辆能够根据来自摄像头的视觉输入做出实时导航决策。我们使用深度学习模型来处理相机图像并控制车辆运动。使用超过一万张图像的数据集，我们训练了一个卷积神经网络（CNN）来有效地驾驶车辆，通过两个输出（转向和油门）实现。该车在30秒内完成了赛道，速度约为每秒0.4米，同时避开了障碍物。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07348) | **Categories:** cs.RO, cs.SY, eess.IV, eess.SY

---

### [15] [Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs](https://arxiv.org/abs/2506.07454)
*Jared Strader, Aaron Ray, Jacob Arkin, Mason B. Peterson, Yun Chang, Nathan Hughes, Christopher Bradley, Yi Xuan Jia, Carlos Nieto-Granda, Rajat Talak, Chuchu Fan, Luca Carlone, Jonathan P. How, Nicholas Roy*

Main category: cs.RO

TL;DR: 本文提出了一种基于3D场景图的多机器人系统，能够理解自然语言指令并在复杂环境中执行任务。


<details>
  <summary>Details</summary>
Motivation: 解决多机器人系统在复杂环境中执行自然语言指令的问题。

Method: 该文提出了一种利用3D场景图集成的多机器人系统，该系统结合了地图构建、定位以及任务和运动规划（TAMP）。

Result: 实验评估表明，该系统在真实世界的任务中表现良好。

Conclusion: 该系统能够在大型户外环境中执行复杂的自然语言指令。

Abstract: 本文介绍了一种多机器人系统，该系统集成了地图构建、定位以及任务和运动规划（TAMP），并通过3D场景图实现，从而能够执行用自然语言表达的复杂指令。我们的系统构建了一个共享的3D场景图，其中包含一个基于开放集对象构建的地图，该地图被用于多机器人3D场景图融合。这种表示支持实时、视角不变的重定位（通过基于对象的地图）和规划（通过3D场景图），从而允许机器人团队推理其周围环境并执行复杂任务。此外，我们还介绍了一种规划方法，该方法通过利用来自共享3D场景图和机器人能力的上下文，使用大型语言模型（LLM）将操作员意图转换为规划域定义语言（PDDL）目标。我们提供了该系统在大型户外环境中执行真实世界任务的性能实验评估。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07454) | **Categories:** cs.RO, cs.AI

---

### [16] [Fractional Collisions: A Framework for Risk Estimation of Counterfactual Conflicts using Autonomous Driving Behavior Simulations](https://arxiv.org/abs/2506.07540)
*Sreeja Roy-Singh, Sarvesh Kolekar, Daniel P. Bonny, Kyle Foss*

Main category: cs.RO

TL;DR: 本文提出了一种基于反事实模拟的碰撞风险评估方法，用于评估自动驾驶系统在减少碰撞方面的有效性。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在提供一种评估自动驾驶系统(ADS)碰撞风险的方法，利用来自ADS或自然驾驶数据库的传感器数据构建反事实模拟场景。

Method: 该方法通过检测和分类冲突类型、识别主体角色、识别响应者的反应点，并将人类行为预期建模为概率反事实轨迹来评估双主体的冲突。

Result: 该方法在合成模拟环境中预测的碰撞概率与实际碰撞概率相差在1%以内。在替换自然响应者后，ADS软件使自然碰撞减少了4倍，碰撞风险降低了约62%。在ADS测试车辆上收集的25万英里传感器数据也验证了该框架的效用。

Conclusion: 该方法通过合成模拟环境验证了其有效性，并评估了ADS软件在减少碰撞风险方面的能力。

Abstract: 本文提出了一种评估碰撞风险的方法，该方法基于自动驾驶系统（ADS）或自然驾驶数据库中的传感器数据构建反事实模拟场景。通过检测和分类冲突类型、识别主体角色（发起者或响应者）、识别响应者的反应点，并将人类行为预期建模为概率反事实轨迹来评估双主体的冲突。这些状态被用于计算碰撞时的速度差，结合碰撞模型，可以估计损失的严重程度，即概率性的伤害或财产损失，称之为部分碰撞。概率模型还可以扩展到包括与模拟、特征和主体相关的其他不确定性。我们使用来自VTTI的SHRP2数据库和Nexar行车记录仪数据的300多个碰撞和近碰撞场景的重建轨迹，在合成模拟环境中验证了该方法的有效性。我们的方法预测的部分碰撞与实际碰撞相差在1%以内。然后，我们通过用ADS模拟器替换这些合成重建中的自然响应者，并将结果与人类响应结果进行比较，来评估任意ADS软件版本的代理发起的碰撞风险。我们的ADS使自然碰撞减少了4倍，部分碰撞风险降低了约62%。该框架的效用还在ADS测试车辆上收集的25万英里的专有开放式传感器数据上得到了证明，这些数据使用任意ADS软件版本重新模拟。ADS发起的冲突导致了0.4次导致受伤和1.7次导致财产损失的部分碰撞，并且ADS在96%的代理发起的冲突中改善了碰撞风险。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07540) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [17] [Blending Participatory Design and Artificial Awareness for Trustworthy Autonomous Vehicles](https://arxiv.org/abs/2506.07633)
*Ana Tanevska, Ananthapathmanabhan Ratheesh Kumar, Arabinda Ghosh, Ernesto Casablanca, Ginevra Castellano, Sadegh Soudjani*

Main category: cs.RO

TL;DR: 本文通过构建人类驾驶员的马尔可夫链模型，研究了自动驾驶汽车的透明度对人类驾驶员行为的影响。


<details>
  <summary>Details</summary>
Motivation: 当前的机器人代理需要以适当的情境感知（SA）、风险意识、协调和决策来处理不确定的现实世界环境。为了使这些代理与人类用户交互，需要了解如何对交互场景中的人类进行建模，以及如何在代理和人类之间建立信任和透明度。

Method: 创建了一个人类驾驶员的数据驱动模型，并将其集成到SA架构中。通过以用户为中心的大规模人机交互研究，收集创建模型所需的数据，研究AV的透明度与用户行为之间的相互作用。

Result: 研究表明，根据AV的透明度、场景环境和用户的人口统计数据，模型转换存在显着差异。

Conclusion: 根据AV的透明度、场景环境和用户的人口统计数据，我们可以获得模型转换的显着差异。

Abstract: 当前的机器人代理，如自动驾驶汽车（AV）和无人机，需要以适当的情境感知（SA）、风险意识、协调和决策来处理不确定的现实世界环境。SymAware项目致力于通过设计多代理系统中的人工智能感知架构来解决这个问题，从而实现自动驾驶汽车和无人机的安全协作。然而，这些代理还需要与人类用户（驾驶员、行人、无人机操作员）互动，这反过来需要了解如何对互动场景中的人类进行建模，以及如何在代理和人类之间建立信任和透明度。在这项工作中，我们的目标是创建一个人类驾驶员的数据驱动模型，并将其集成到我们的SA架构中，并将我们的研究建立在值得信赖的人机交互原则之上。为了收集创建模型所需的数据，我们进行了一项以用户为中心的大规模人机交互研究，其中我们调查了AV的透明度与用户行为之间的相互作用。本文的贡献是双重的：首先，我们详细说明了我们的人机交互研究及其发现，其次，我们展示了从研究数据中计算出的人类驾驶员的马尔可夫链模型。我们的结果表明，根据AV的透明度、场景环境和用户的人口统计数据，我们可以在模型转换中获得显着差异。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07633) | **Categories:** cs.RO

---

### [18] [BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](https://arxiv.org/abs/2506.07961)
*Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, Tieniu Tan*

Main category: cs.RO

TL;DR: BridgeVLA通过3D到2D的投影和热图预测，显著提高了3D机器人操作的学习效率和泛化能力。


<details>
  <summary>Details</summary>
Motivation: 现有的视觉-语言-动作模型(VLA)方法未能充分利用3D数据的空间结构，导致样本效率低下。

Method: BridgeVLA通过将3D输入投影到多个2D图像，并利用2D热图进行动作预测，统一了输入和输出空间。

Result: BridgeVLA在RLBench、COLOSSEUM和GemBench等模拟基准测试中均优于现有技术，并在真实机器人实验中平均性能提高32%，仅用少量样本即可实现高成功率。

Conclusion: BridgeVLA在模拟和真实机器人实验中均表现出色，尤其在样本效率和泛化能力方面。

Abstract: 本文介绍了一种名为BridgeVLA的新型3D VLA模型，该模型(1)将3D输入投影到多个2D图像，确保与VLM主干的输入对齐，并且(2)利用2D热图进行动作预测，从而在一致的2D图像空间内统一输入和输出空间。此外，我们提出了一种可扩展的预训练方法，该方法使VLM主干具有在下游策略学习之前预测2D热图的能力。大量实验表明，该方法能够高效且有效地学习3D操作。BridgeVLA在三个模拟基准测试中均优于最先进的基线方法。在RLBench中，它将平均成功率从81.4％提高到88.2％。在COLOSSEUM中，它在具有挑战性的泛化设置中表现出明显更好的性能，从而将平均成功率从56.7％提高到64.0％。在GemBench中，它在平均成功率方面超过了所有比较基线方法。在真实机器人实验中，BridgeVLA的平均性能优于最先进的基线方法32％。它在多种分布外设置（包括视觉干扰和看不见的指令）中均能稳定地推广。值得注意的是，它仅需每个任务3个轨迹即可在10多个任务上实现96.8％的成功率，从而突出了其非凡的样本效率。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.07961) | **Categories:** cs.RO, cs.AI

---


## 统计机器学习 (Machine Learning Statistics) [stat.ML]
### [1] [On the Fundamental Impossibility of Hallucination Control in Large Language Models](https://arxiv.org/abs/2506.06382)
*Michał P. Karpowicz*

Main category: stat.ML

TL;DR: 该论文证明了大型语言模型无法同时满足真实性、信息守恒、知识相关性和最优性这四个基本属性。


<details>
  <summary>Details</summary>
Motivation: 解释了为什么不可能创建不产生幻觉的大型语言模型，以及我们应该寻找什么样的权衡。

Method: 通过将LLM推理建模为神经组件竞争以促进响应的思想拍卖，使用Green-Laffont定理证明了这种不可能性。

Result: 该数学框架为理解推理过程的本质提供了严格的基础，对模型架构、训练目标和评估方法具有重要意义。

Conclusion: 论文提出了一个形式上的不可能定理，证明没有推理机制能够同时满足四个基本属性：真实生成、语义信息守恒、相关知识揭示和知识约束最优性。

Abstract: 本文解释了为什么不可能创建不产生幻觉的大型语言模型，以及我们应该寻找什么样的权衡。它提出了一个形式上的不可能定理，证明没有推理机制能够同时满足四个基本属性：真实（非幻觉）生成、语义信息守恒、相关知识揭示和知识约束最优性。通过将LLM推理建模为神经组件竞争以促进响应的思想拍卖，我们使用Green-Laffont定理证明了这种不可能性。该数学框架为理解推理过程的本质提供了严格的基础，对模型架构、训练目标和评估方法具有重要意义。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.06382) | **Categories:** stat.ML, cs.AI, cs.CL, cs.GT, cs.LG

---
