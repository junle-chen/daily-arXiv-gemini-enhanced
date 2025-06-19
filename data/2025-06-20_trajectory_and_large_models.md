# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-20

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [cs.CR (2)](#cs-cr)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [cs.IT (1)](#cs-it)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (4)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence](https://arxiv.org/abs/2506.15672)
*Yao Zhang, Chenyang Lin, Shijie Tang, Haokun Chen, Shijie Zhou, Yunpu Ma, Volker Tresp*

Main category: cs.AI

TL;DR: SwarmAgentic是一个全自动的智能体系统生成框架，它从头开始构建智能体系统，并通过语言驱动的探索，将智能体功能和协作作为相互依赖的组件进行联合优化。


<details>
  <summary>Details</summary>
Motivation: Yet, existing agentic system generation frameworks lack full autonomy, missing from-scratch agent generation, self-optimizing agent functionality, and collaboration, limiting adaptability and scalability.

Method: We propose SwarmAgentic, a framework for fully automated agentic system generation that constructs agentic systems from scratch and jointly optimizes agent functionality and collaboration as interdependent components through language-driven exploration. To enable efficient search over system-level structures, SwarmAgentic maintains a population of candidate systems and evolves them via feedback-guided updates, drawing inspiration from Particle Swarm Optimization (PSO).

Result: Given only a task description and an objective function, SwarmAgentic outperforms all baselines, achieving a +261.8% relative improvement over ADAS on the TravelPlanner benchmark, highlighting the effectiveness of full automation in structurally unconstrained tasks.

Conclusion: This framework marks a significant step toward scalable and autonomous agentic system design, bridging swarm intelligence with fully automated system multi-agent generation.

Abstract: 大型语言模型的快速发展推动了智能体系统在决策、协调和任务执行方面的进步。然而，现有的智能体系统生成框架缺乏完全的自主性，缺少从头开始的智能体生成、自优化智能体功能和协作，从而限制了适应性和可扩展性。我们提出了SwarmAgentic，这是一个全自动的智能体系统生成框架，它从头开始构建智能体系统，并通过语言驱动的探索，将智能体功能和协作作为相互依赖的组件进行联合优化。为了实现对系统级结构的高效搜索，SwarmAgentic维护了一个候选系统群体，并通过反馈引导的更新来进化它们，其灵感来自粒子群优化（PSO）。我们在六个真实的、开放式的和探索性的任务中评估了我们的方法，这些任务涉及高层次的规划、系统级的协调和创造性的推理。仅给定任务描述和目标函数，SwarmAgentic优于所有基线，在TravelPlanner基准测试中实现了比ADAS高+261.8%的相对改进，突出了完全自动化在结构上无约束的任务中的有效性。该框架标志着在可扩展和自主的智能体系统设计方面迈出了重要一步，将群体智能与全自动系统多智能体生成联系起来。我们的代码已在https://yaoz720.github.io/SwarmAgentic/上公开发布。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15672) | **Categories:** cs.AI, cs.MA

---


## cs.CR [cs.CR]
### [1] [Advanced Prediction of Hypersonic Missile Trajectories with CNN-LSTM-GRU Architectures](https://arxiv.org/abs/2506.15043)
*Amir Hossein Baradaran*

Main category: cs.CR

TL;DR: 本文提出了一种混合深度学习方法，用于高精度预测高超音速导弹的复杂轨迹，从而显著提升防御能力。


<details>
  <summary>Details</summary>
Motivation: 高超音速导弹以其极高的速度和机动性构成了严峻的挑战，因此准确的轨迹预测对于有效的对抗措施至关重要。

Method: 提出了一种混合深度学习方法，集成了卷积神经网络（CNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）。

Result: 该方法成功地高精度预测了高超音速导弹的复杂轨迹。

Conclusion: 先进的机器学习技术增强了防御系统的预测能力。

Abstract: 国防工业的进步对于确保国家安全至关重要，能够为应对新兴威胁提供强大的保护。其中，高超音速导弹以其极高的速度和机动性构成了严峻的挑战，因此准确的轨迹预测对于有效的对抗措施至关重要。本文采用一种新颖的混合深度学习方法，集成了卷积神经网络（CNN）、长短期记忆网络（LSTM）和门控循环单元（GRU），旨在解决这一挑战。通过利用这些架构的优势，该方法成功地高精度预测了高超音速导弹的复杂轨迹，为防御策略和导弹拦截技术做出了重大贡献。这项研究证明了先进的机器学习技术在增强防御系统预测能力方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15043) | **Categories:** cs.CR, cs.AI

---

### [2] [Winter Soldier: Backdooring Language Models at Pre-Training with Indirect Data Poisoning](https://arxiv.org/abs/2506.14913)
*Wassim Bouaziz, Mathurin Videau, Nicolas Usunier, El-Mahdi El-Mhamdi*

Main category: cs.CR

TL;DR: 该论文提出了一种间接数据中毒方法，通过少量中毒token即可使语言模型学习并检测秘密，且不影响性能，从而保护数据集并追踪其使用情况。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）的预训练依赖于来自不同且难以管理的来源的海量文本数据集。虽然已经探索了成员推理攻击和隐藏canaries来追踪数据使用情况，但这些方法依赖于对训练数据的记忆，而LM提供商试图限制这种记忆。

Method: 使用基于梯度的优化prompt调整，使模型学习任意秘密序列。

Result: 验证了该方法在从头开始预训练的语言模型上的有效性，表明少于0.005%的中毒token足以隐蔽地使LM学习秘密，并以极高的置信度（$p < 10^{-55}$）检测到它，且具有理论上可证明的方案。

Conclusion: 在没有性能下降的情况下，可以通过对少量中毒token进行梯度优化prompt调整，隐蔽地使语言模型学习秘密并以极高的置信度检测它。

Abstract: 大型语言模型（LLM）的预训练依赖于来自不同且难以管理的来源的海量文本数据集。虽然已经探索了成员推理攻击和隐藏canaries来追踪数据使用情况，但这些方法依赖于对训练数据的记忆，而LM提供商试图限制这种记忆。在这项工作中，我们证明了间接数据中毒（目标行为在训练数据中不存在）不仅是可行的，而且还可以有效地保护数据集并追踪其使用情况。使用基于梯度的优化prompt调整，我们使模型学习任意秘密序列：对训练语料库中不存在的秘密提示的秘密响应。我们验证了该方法在从头开始预训练的语言模型上的有效性，表明少于0.005%的中毒token足以隐蔽地使LM学习秘密，并以极高的置信度（$p < 10^{-55}$）检测到它，且具有理论上可证明的方案。 重要的是，这发生在没有性能下降（在LM基准上）的情况下，并且尽管秘密从未出现在训练集中。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14913) | **Categories:** cs.CR, cs.LG, stat.ML

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review](https://arxiv.org/abs/2506.14831)
*Céline Finet, Stephane Da Silva Martins, Jean-Bernard Hayet, Ioannis Karamouzas, Javad Amirian, Sylvie Le Hégarat-Mascle, Julien Pettré, Emanuel Aldea*

Main category: cs.CV

TL;DR: 本研究综述了 2020-2024 年间基于深度学习的多智能体轨迹预测进展，并突出了该领域的挑战和未来方向。


<details>
  <summary>Details</summary>
Motivation: 随着数据驱动方法在人类轨迹预测 (HTP) 中兴起，更好地理解多智能体交互成为可能，这对自主导航和人群建模等领域具有重要意义。

Method: 根据架构设计、输入表示和预测策略对现有方法进行分类，特别关注使用 ETH/UCY 基准评估的模型。

Result: 回顾了基于深度学习的多智能体轨迹预测的最新进展，重点关注 2020 年至 2024 年间发表的研究。

Conclusion: 强调了多智能体 HTP 领域的关键挑战和未来研究方向。

Abstract: 随着强大的数据驱动方法在人类轨迹预测（HTP）中涌现，更深入地理解多智能体交互触手可及，这对自主导航和人群建模等领域具有重要意义。本综述回顾了基于深度学习的多智能体轨迹预测的一些最新进展，重点关注 2020 年至 2024 年间发表的研究。我们根据现有方法的架构设计、输入表示和整体预测策略对其进行分类，特别强调了使用 ETH/UCY 基准评估的模型。此外，我们还强调了多智能体 HTP 领域的关键挑战和未来研究方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14831) | **Categories:** cs.CV, cs.LG, cs.RO

---

### [2] [MapFM: Foundation Model-Driven HD Mapping with Multi-Task Contextual Learning](https://arxiv.org/abs/2506.15313)
*Leonid Ivanov, Vasily Yuryev, Dmitry Yudin*

Main category: cs.CV

TL;DR: MapFM 通过结合基础模型和多任务学习，显著提高了在线矢量化 HD 地图生成的准确性和质量。


<details>
  <summary>Details</summary>
Motivation: 高清 (HD) 地图和鸟瞰图 (BEV) 中的语义地图对于自动驾驶中准确定位、规划和决策至关重要。

Method: 提出了一种名为 MapFM 的增强型端到端模型，用于在线生成矢量化 HD 地图。该模型利用强大的基础模型对相机图像进行编码，显著提高了特征表示质量，并集成了辅助预测头，用于在 BEV 表示中进行语义分割。

Result: 通过集成辅助预测头和使用多任务学习，MapFM 能够更全面地表示场景，并最终提高预测矢量化 HD 地图的准确性和质量。

Conclusion: 通过多任务学习方法，MapFM 能够提供更丰富的上下文信息，从而提高预测矢量化 HD 地图的准确性和质量。

Abstract: 在自动驾驶中，高清 (HD) 地图和鸟瞰图 (BEV) 中的语义地图对于准确定位、规划和决策至关重要。本文介绍了一种名为 MapFM 的增强型端到端模型，用于在线生成矢量化 HD 地图。我们展示了通过结合强大的基础模型来编码相机图像，可以显著提高特征表示质量。为了进一步丰富模型对环境的理解并提高预测质量，我们集成了辅助预测头，用于在 BEV 表示中进行语义分割。这种多任务学习方法提供了更丰富的上下文监督，从而实现更全面的场景表示，并最终提高预测矢量化 HD 地图的准确性和质量。源代码可在 https://github.com/LIvanoff/MapFM 获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15313) | **Categories:** cs.CV, cs.AI

---


## cs.IT [cs.IT]
### [1] [LLM Agent for Hyper-Parameter Optimization](https://arxiv.org/abs/2506.15167)
*Wanzhe Wang, Jianqiu Peng, Menghao Hu, Weihuang Zhong, Tong Zhang, Shuai Wang, Yixin Zhang, Mingjie Shao, Wanli Ni*

Main category: cs.IT

TL;DR: 本文提出了一种基于大型语言模型（LLM）的智能体，用于自动调整超参数，从而显著提高了通信算法的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的基于启发式的超参数调整方法自动化程度低，性能不佳。

Method: 设计了一个基于大型语言模型（LLM）的智能体，用于自动超参数调整，应用了迭代框架和模型上下文协议（MCP）。

Result: 实验结果表明，通过LLM agent生成的超参数实现的最小总速率明显高于人工启发式和随机生成方法。

Conclusion: LLM agent 可以有效地找到高性能的超参数。

Abstract: 超参数对于通信算法的性能至关重要。然而，目前用于无线电地图无人机（UAV）轨迹和通信的带有交叉和变异的温启动粒子群优化（WS-PSO-CM）算法的超参数调整方法主要基于启发式，自动化程度低，性能不佳。在本文中，我们设计了一个基于大型语言模型（LLM）的智能体，用于自动超参数调整，其中应用了迭代框架和模型上下文协议（MCP）。特别地，首先通过配置文件设置LLM智能体，该文件指定任务、背景和输出格式。然后，LLM智能体由提示需求驱动，并迭代地调用WS-PSO-CM算法进行探索。最后，LLM智能体自主终止循环并返回一组超参数。我们的实验结果表明，通过LLM智能体生成的超参数实现的最小总速率明显高于人工启发式和随机生成方法。这表明具有PSO知识和WS-PSO-CM算法背景的LLM智能体在寻找高性能超参数方面非常有用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15167) | **Categories:** cs.IT, cs.AI, math.IT

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models](https://arxiv.org/abs/2506.15065)
*Trishna Chakraborty, Udita Ghosh, Xiaopan Zhang, Fahim Faisal Niloy, Yue Dong, Jiachen Li, Amit K. Roy-Chowdhury, Chengyu Song*

Main category: cs.LG

TL;DR: 该论文首次系统研究了LLM具身智能体在场景任务不一致情况下的幻觉问题，揭示了模型在处理不切实际任务时的局限性，并提供了改进建议。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型越来越多地被用作具身智能体的认知核心，但由于未能将用户指令与观察到的物理环境相结合而产生的幻觉，可能导致导航错误。

Method: 构建了一个幻觉探测集，可以在场景任务不一致的情况下，将幻觉率提高到比基本提示高 40 倍。

Result: 在两个模拟环境中评估了 12 个模型，发现模型表现出推理能力，但无法解决场景任务不一致的问题。

Conclusion: 模型在处理场景任务不一致时存在根本性限制，无法解决不切实际的任务。

Abstract: 大型语言模型（LLM）越来越多地被用作具身智能体的认知核心。然而，由于未能将用户指令与观察到的物理环境相结合而产生的幻觉，可能导致导航错误，例如搜索不存在的冰箱。在本文中，我们首次系统地研究了基于 LLM 的具身智能体在场景任务不一致的情况下执行长时任务时的幻觉问题。我们的目标是了解幻觉发生的程度，哪些类型的不一致会触发幻觉，以及当前模型的反应。为了实现这些目标，我们构建了一个幻觉探测集，该探测集建立在现有基准之上，能够将幻觉率提高到比基本提示高 40 倍。我们评估了两个模拟环境中的 12 个模型，发现虽然模型表现出推理能力，但它们无法解决场景任务不一致的问题——突出了处理不可行任务的根本局限性。我们还针对每种场景提供了理想模型行为的可操作见解，为开发更强大、更可靠的规划策略提供指导。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15065) | **Categories:** cs.LG, cs.RO

---

### [2] [Neural Canonical Polyadic Factorization for Traffic Analysis](https://arxiv.org/abs/2506.15079)
*Yikai Hou, Peng Tang*

Main category: cs.LG

TL;DR: NCPF模型结合CP分解和神经网络，有效解决高维交通数据插补问题，为下一代交通系统提供支持。


<details>
  <summary>Details</summary>
Motivation: 现代智能交通系统依赖于精确的时空交通分析来优化城市交通和基础设施弹性，但传感器故障和异构传感间隙导致普遍存在的数据缺失，从根本上阻碍了可靠的交通建模。

Method: 提出了一种神经规范Polyadic分解(NCPF)模型，该模型将低秩张量代数与深度表示学习相结合，用于鲁棒的交通数据插补。

Result: 在六个城市交通数据集上的广泛评估表明，NCPF优于六个最先进的基线。

Conclusion: NCPF模型通过结合CP分解的可解释性和神经网络的非线性表达能力，为高维交通数据插补提供了一种有效且灵活的方法。

Abstract: 现代智能交通系统依赖于精确的时空交通分析来优化城市交通和基础设施的弹性。然而，由传感器故障和异构传感间隙导致的普遍数据缺失从根本上阻碍了可靠的交通建模。本文提出了一种神经规范Polyadic分解(NCPF)模型，该模型将低秩张量代数与深度表示学习相结合，用于鲁棒的交通数据插补。该模型通过可学习的嵌入投影，创新地将CP分解嵌入到神经架构中，其中稀疏交通张量被编码成跨路段、时间间隔和移动性指标的密集潜在因子。一种分层特征融合机制采用Hadamard积来显式地建模多线性交互，而堆叠的多层感知器层非线性地细化这些表示，以捕获复杂的时空耦合。在六个城市交通数据集上的广泛评估表明，NCPF优于六个最先进的基线。通过统一CP分解的可解释因子分析与神经网络的非线性表达能力，NCPF为高维交通数据插补提供了一种有效而灵活的方法，为下一代交通数字孪生和自适应交通控制系统提供关键支持。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15079) | **Categories:** cs.LG, stat.ML

---

### [3] [Learning Task-Agnostic Skill Bases to Uncover Motor Primitives in Animal Behaviors](https://arxiv.org/abs/2506.15190)
*Jiyi Wang, Jingyang Ke, Bo Dai, Anqi Wu*

Main category: cs.LG

TL;DR: 该论文提出了一种基于技能的模仿学习框架（SKIL），用于理解动物行为，通过动态组合基本运动原语来生成复杂行为。


<details>
  <summary>Details</summary>
Motivation: 现有的行为分割方法通过在限制性生成假设下施加离散音节，过度简化了这个过程。

Method: 我们引入了基于技能的模仿学习（SKIL）用于行为理解，这是一种基于强化学习的模仿框架，它（1）通过利用转换概率的表征学习来推断可解释的技能集，即行为的潜在基函数，并且（2）将策略参数化为这些技能的动态混合。

Result: 在各种任务中，它可以识别可重用的技能组件，学习不断发展的组合策略，并生成超出传统离散模型能力的真实轨迹。

Conclusion: 通过动态组合基本运动原语，该方法为复杂动物行为的出现提供了一个简洁、有原则的解释。

Abstract: 动物灵活地重组一组有限的核心运动原语，以满足不同的任务需求，但现有的行为分割方法通过在限制性生成假设下施加离散音节，过度简化了这个过程。为了反映动物行为的生成过程，我们引入了基于技能的模仿学习（SKIL）用于行为理解，这是一种基于强化学习的模仿框架，它（1）通过利用转换概率的表征学习来推断可解释的技能集，即行为的潜在基函数，并且（2）将策略参数化为这些技能的动态混合。我们在一个简单的网格世界、一个离散的迷宫和自由移动动物的无约束视频上验证了我们的方法。在各种任务中，它可以识别可重用的技能组件，学习不断发展的组合策略，并生成超出传统离散模型能力的真实轨迹。通过利用具有组合表示的生成行为建模，我们的方法为复杂动物行为如何从基本运动原语的动态组合中出现提供了一个简洁、有原则的解释。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15190) | **Categories:** cs.LG, q-bio.NC

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Robust Instant Policy: Leveraging Student's t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation](https://arxiv.org/abs/2506.15157)
*Hanbit Oh, Andrea M. Salcedo-Vázquez, Ixchel G. Ramirez-Alpizar, Yukiyasu Domae*

Main category: cs.RO

TL;DR: 提出RIP算法，利用Student's t-回归模型，增强上下文模仿学习在机器人领域的鲁棒性，以抵抗LLM的幻觉问题。


<details>
  <summary>Details</summary>
Motivation: 基于LLM的瞬时策略的幻觉问题会降低其在机器人领域的可靠性，例如LLM有时会生成偏离给定演示的差轨迹。

Method: 提出了一种新的鲁棒的上下文模仿学习算法，称为鲁棒瞬时策略（RIP），它利用Student's t-回归模型来抵抗瞬时策略的幻觉轨迹，以实现可靠的轨迹生成。

Result: RIP在模拟和真实世界的环境中进行的实验表明，RIP明显优于最先进的IL方法，任务成功率至少提高了26％，尤其是在日常任务的低数据场景中。

Conclusion: RIP在模拟和真实世界的实验中显著优于最先进的IL方法，尤其是在日常任务的低数据场景中，任务成功率至少提高了26%。

Abstract: 模仿学习（IL）旨在通过观察一些人类演示，使机器人能够自主执行任务。最近，IL的一种变体，称为上下文IL，利用现成的，大型语言模型（LLM）作为即时策略，理解来自一些给定演示的上下文以执行新任务，而不是使用大规模演示来显式更新网络模型。然而，由于基于LLM的即时策略的幻觉问题，例如LLM有时会生成偏离给定演示的差轨迹，因此其在机器人领域的可靠性受到损害。为了缓解这个问题，我们提出了一种新的鲁棒的上下文模仿学习算法，称为鲁棒瞬时策略（RIP），它利用Student's t-回归模型来抵抗瞬时策略的幻觉轨迹，以实现可靠的轨迹生成。具体来说，RIP生成多个候选机器人轨迹，以从LLM完成给定的任务，并使用Student's t-分布对其进行聚合，这有利于忽略异常值（即幻觉）；因此，生成了针对幻觉的鲁棒轨迹。在模拟和真实世界环境中进行的实验表明，RIP明显优于最先进的IL方法，任务成功率至少提高了26％，尤其是在日常任务的低数据场景中。视频结果可在https://sites.google.com/view/robustinstantpolicy上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15157) | **Categories:** cs.RO, cs.CV

---

### [2] [Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers](https://arxiv.org/abs/2506.14855)
*Tommaso Belvedere, Michael Ziegltrum, Giulio Turrisi, Valerio Modugno*

Main category: cs.RO

TL;DR: F-MPPI通过增加局部线性反馈增益来增强标准MPPI，从而实现鲁棒、高频率的机器人控制。


<details>
  <summary>Details</summary>
Motivation: 模型预测路径积分控制是一种强大的基于采样的方法，由于其在处理非线性动力学和非凸成本方面的灵活性，适用于复杂的机器人任务。然而，它在实时、高频机器人控制场景中的适用性受到计算需求的限制。

Method: 通过计算从灵敏度分析中导出的局部线性反馈增益来增强标准MPPI，灵感来自基于Riccati的反馈，用于基于梯度的MPC。

Result: 在四足机器人执行不平坦地形上的动态运动和四旋翼飞行器执行具有板载计算的激进机动的两个机器人平台上的仿真和真实世界实验证明了F-MPPI的有效性。

Conclusion: 结合局部反馈显著提高了控制性能和稳定性，实现了适用于复杂机器人系统的高频率鲁棒操作。

Abstract: 模型预测路径积分控制（MPPI）是一种强大的基于采样的方法，适用于复杂的机器人任务，因为它在处理非线性动力学和非凸成本方面具有灵活性。然而，其在实时、高频机器人控制场景中的适用性受到计算需求的限制。本文介绍了一种新颖的框架Feedback-MPPI（F-MPPI），该框架通过计算从灵敏度分析中导出的局部线性反馈增益来增强标准MPPI，灵感来自基于Riccati的反馈，用于基于梯度的MPC。这些增益允许在当前状态下进行快速闭环校正，而无需在每个时间步进行完全重新优化。我们通过在两个机器人平台上的仿真和真实世界实验证明了F-MPPI的有效性：一个四足机器人执行不平坦地形上的动态运动，一个四旋翼飞行器执行具有板载计算的激进机动。结果表明，结合局部反馈显著提高了控制性能和稳定性，实现了适用于复杂机器人系统的高频率鲁棒操作。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14855) | **Categories:** cs.RO, cs.AI

---

### [3] [Towards Perception-based Collision Avoidance for UAVs when Guiding the Visually Impaired](https://arxiv.org/abs/2506.14857)
*Suman Raj, Swapnil Padhi, Ruchi Bhoot, Prince Modi, Yogesh Simmhan*

Main category: cs.RO

TL;DR: 本文提出了一种基于无人机的视觉辅助导航系统，旨在帮助视障人士在城市环境中安全导航。


<details>
  <summary>Details</summary>
Motivation: 本文探讨了使用无人机帮助视障人士（VIP）在户外城市环境中导航。

Method: 我们使用几何公式表示问题，并提出了一个基于多DNN的框架，用于无人机和VIP的避障。

Result: 在大学校园环境中的无人机-人系统评估验证了我们的算法在三种场景中的可行性。

Conclusion: 在大学校园环境中的无人机-人系统评估验证了我们的算法在三种场景中的可行性：当VIP在人行道上行走、靠近停放的车辆以及在拥挤的街道上。

Abstract: 无人机结合机载传感器、机器学习和计算机视觉算法的自主导航正在影响农业、物流和灾害管理等多个领域。在本文中，我们研究了使用无人机帮助视障人士（VIP）在户外城市环境中导航。具体来说，我们提出了一个基于感知的路径规划系统，用于在VIP附近进行局部规划，并结合基于GPS和地图的全局规划器进行粗略规划。我们使用几何公式表示问题，并提出了一个基于多DNN的框架，用于无人机和VIP的避障。在大学校园环境中的无人机-人系统评估验证了我们的算法在三种场景中的可行性：当VIP在人行道上行走、靠近停放的车辆以及在拥挤的街道上。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14857) | **Categories:** cs.RO, cs.CV

---

### [4] [DyNaVLM: Zero-Shot Vision-Language Navigation System with Dynamic Viewpoints and Self-Refining Graph Memory](https://arxiv.org/abs/2506.15096)
*Zihe Ji, Huangxuan Lin, Yue Gao*

Main category: cs.RO

TL;DR: DyNaVLM 是一个无需训练的端到端视觉语言导航框架，它使用动态动作空间和协作图记忆来实现高性能。


<details>
  <summary>Details</summary>
Motivation: 现有方法受限于固定的角度或距离间隔，该文旨在解决这个问题。

Method: 该文提出了 DyNaVLM，一个使用视觉语言模型 (VLM) 的端到端视觉语言导航框架，它允许智能体通过视觉语言推理自由选择导航目标。

Result: DyNaVLM 在 GOAT 和 ObjectNav 基准测试中表现出高性能。真实世界的测试进一步验证了其鲁棒性和泛化性。

Conclusion: 该系统通过动态动作空间、协作图记忆和无训练部署，为可扩展的具身机器人建立了一个新范例，弥合了离散 VLN 任务和连续现实世界导航之间的差距。

Abstract: 我们提出了 DyNaVLM，一个使用视觉语言模型 (VLM) 的端到端视觉语言导航框架。与之前受限于固定角度或距离间隔的方法不同，我们的系统使智能体能够通过视觉语言推理自由选择导航目标。其核心在于一个自完善的图记忆，它 1) 将对象位置存储为可执行的拓扑关系，2) 通过分布式图更新实现跨机器人记忆共享，3) 通过检索增强来增强 VLM 的决策能力。DyNaVLM 无需特定于任务的训练或微调即可运行，在 GOAT 和 ObjectNav 基准测试中表现出高性能。真实世界的测试进一步验证了其鲁棒性和泛化性。该系统的三项创新：动态动作空间公式、协作图记忆和无训练部署，为可扩展的具身机器人建立了一个新范例，弥合了离散 VLN 任务和连续现实世界导航之间的差距。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15096) | **Categories:** cs.RO

---


## eess.SY [eess.SY]
### [1] [Model Predictive Path-Following Control for a Quadrotor](https://arxiv.org/abs/2506.15447)
*David Leprich, Mario Rosenfelder, Mario Hermle, Jingshan Chen, Peter Eberhard*

Main category: eess.SY

TL;DR: 本文提出了一种基于模型预测控制的路径跟踪方法，并将其成功应用于四旋翼飞行器，同时提出了一种走廊路径跟踪方法以允许路径偏差。


<details>
  <summary>Details</summary>
Motivation: 现有的路径跟踪解决方案缺乏显式处理状态和输入约束的能力，通常采用保守的两阶段方法，或者仅适用于线性系统。为了解决这些挑战，本文研究了基于模型预测控制的路径跟踪框架。

Method: 提出了一种基于模型预测控制（MPC）的路径跟踪框架，并将其应用于Crazyflie四旋翼飞行器。该框架包含一个底层的姿态控制器，以满足四旋翼控制的实时性需求。

Result: 通过实际实验证明了该方法的有效性，并提出了一种走廊路径跟踪方法。

Conclusion: 通过实际实验验证了所提出的基于MPC的路径跟踪方法在四旋翼飞行器上的有效性，并提出了一种走廊路径跟踪方法，允许在路径跟踪过于严格的情况下进行偏差。

Abstract: 无人机辅助流程的自动化是一个复杂的任务。许多解决方案依赖于轨迹生成和跟踪，相比之下，路径跟踪控制是一种特别有前途的方法，它为无人机和其他车辆的自动化任务提供了一种直观和自然的方法。虽然已经提出了不同的路径跟踪问题的解决方案，但它们中的大多数缺乏显式处理状态和输入约束的能力，以保守的两阶段方法制定，或者仅适用于线性系统。为了解决这些挑战，本文建立在基于模型预测控制的路径跟踪框架之上，并将其应用扩展到Crazyflie四旋翼飞行器，并在硬件实验中进行了研究。模型预测路径跟踪控制公式中包含一个包含底层姿态控制器的级联控制结构，以满足四旋翼控制的具有挑战性的实时需求。通过实际实验证明了该方法的有效性，据作者所知，这代表了这种基于MPC的路径跟踪方法在四旋翼飞行器上的新颖应用。此外，作为原始方法的扩展，为了允许在路径的精确跟踪可能过于严格的情况下偏离路径，提出了一种走廊路径跟踪方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15447) | **Categories:** eess.SY, cs.RO, cs.SY, 93-XX

---
