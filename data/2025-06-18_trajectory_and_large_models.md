# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-18

## 目录

- [astro-ph.EP (1)](#astro-ph-ep)
- [人工智能 (Artificial Intelligence) (5)](#cs-ai)
- [计算语言学 (Computation and Language) (2)](#cs-cl)
- [计算机视觉 (Computer Vision) (9)](#cs-cv)
- [cs.CY (1)](#cs-cy)
- [cs.DC (1)](#cs-dc)
- [人机交互 (Human-Computer Interaction) (1)](#cs-hc)
- [机器学习 (Machine Learning) (8)](#cs-lg)
- [cs.MA (2)](#cs-ma)
- [机器人学 (Robotics) (10)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## astro-ph.EP [astro-ph.EP]
### [1] [SpaceTrack-TimeSeries: Time Series Dataset towards Satellite Orbit Analysis](https://arxiv.org/abs/2506.13034)
*Zhixin Guo, Qi Shi, Xiaofan Xu, Sixiang Shan, Limin Qin, Linqiang Ge, Rui Zhang, Ya Dai, Hua Zhu, Guowei Jiang*

Main category: astro-ph.EP

TL;DR: 本研究收集并管理了 Starlink 卫星的机动行为数据集，以支持空间物体行为预测和碰撞风险评估等领域的研究。


<details>
  <summary>Details</summary>
Motivation: 随着航空航天技术的快速发展和低地球轨道 (LEO) 卫星星座的大规模部署，天文观测和深空探测面临的挑战日益严峻。因此，对空间物体的高精度轨道数据以及对卫星定位、星座配置和深空卫星动力学的综合分析的需求变得越来越迫切。然而，仍然明显缺乏公开可用的真实世界数据集来支持空间物体机动行为预测和碰撞风险评估等领域的研究。

Method: 该研究通过整合双线元素 (TLE) 目录数据与相应的高精度星历数据，收集和管理了 Starlink 卫星的机动行为代表性数据集。

Result: 该数据集能够对空间物体行为进行更真实和多维度的建模。

Conclusion: 该数据集为空间物体行为的实际部署和碰撞风险评估提供了宝贵的见解。

Abstract: 随着航空航天技术的快速发展和低地球轨道 (LEO) 卫星星座的大规模部署，天文观测和深空探测面临的挑战日益严峻。因此，对空间物体的高精度轨道数据以及对卫星定位、星座配置和深空卫星动力学的综合分析的需求变得越来越迫切。然而，仍然明显缺乏公开可用的真实世界数据集来支持空间物体机动行为预测和碰撞风险评估等领域的研究。本研究旨在通过收集和管理 Starlink 卫星的机动行为代表性数据集来解决这一问题。该数据集整合了双线元素 (TLE) 目录数据与相应的高精度星历数据，从而能够对空间物体行为进行更真实和多维度的建模。它为机动检测方法的实际部署以及在日益拥挤的轨道环境中评估碰撞风险提供了宝贵的见解。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13034) | **Categories:** astro-ph.EP, astro-ph.IM, cs.AI

---


## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Topology-Assisted Spatio-Temporal Pattern Disentangling for Scalable MARL in Large-scale Autonomous Traffic Control](https://arxiv.org/abs/2506.12453)
*Rongpeng Li, Jianhang Zhu, Jiahao Huang, Zhifeng Zhao, Honggang Zhang*

Main category: cs.AI

TL;DR: 本文提出了一种新的基于动态图神经网络和拓扑数据分析的MARL框架，用于优化大规模交通信号控制。


<details>
  <summary>Details</summary>
Motivation: 现有的多智能体强化学习（MARL）算法在优化交通信号控制（TSC）方面显示出潜力，但其可扩展性和有效性通常受到大规模和复杂环境的影响。

Method: 该论文提出了一种新的MARL框架，集成了动态图神经网络（DGNNs）和拓扑数据分析（TDA），并提出了一个拓扑辅助的空间模式解耦（TSD）增强的MoE。

Result: 在真实交通场景中进行的大量实验以及全面的理论分析，验证了所提出的框架的卓越性能。

Conclusion: 该论文提出的框架在实际交通场景中表现出色，验证了模型在解决大规模交通信号控制任务中的可扩展性和有效性。

Abstract: 智能交通系统（ITS）已成为缓解城市交通拥堵的一种有前景的解决方案，其中交通信号控制（TSC）被认为是关键组成部分。虽然多智能体强化学习（MARL）算法在通过实时决策优化TSC方面显示出潜力，但其可扩展性和有效性通常受到大规模和复杂环境的影响。通常，这些限制主要源于环境异质性驱动的状态空间呈指数增长与当前解决方案的有限建模能力之间的根本不匹配。为了解决这些问题，本文介绍了一种新的MARL框架，该框架集成了动态图神经网络（DGNN）和拓扑数据分析（TDA），旨在增强环境表示的表达能力并改善智能体协调。此外，受大型语言模型（LLM）中专家混合（MoE）架构的启发，提出了一种拓扑辅助的空间模式解耦（TSD）增强的MoE，它利用拓扑签名来解耦图特征以进行专门处理，从而提高模型表征动态和异构局部观测的能力。TSD模块还被集成到多智能体近端策略优化（MAPPO）算法的策略和价值网络中，进一步提高了决策效率和鲁棒性。在真实交通场景中进行的大量实验以及全面的理论分析，验证了所提出的框架的卓越性能，突出了该模型在解决大规模TSC任务的复杂性方面的可扩展性和有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12453) | **Categories:** cs.AI

---

### [2] [Deflating Deflationism: A Critical Perspective on Debunking Arguments Against LLM Mentality](https://arxiv.org/abs/2506.13403)
*Alex Grzankowski, Geoff Keeling, Henry Shevlin, Winnie Street*

Main category: cs.AI

TL;DR: 论文评估了两种反对大型语言模型（LLM）具有心智的论点，并提出在特定条件下可以适度地将心理状态归因于LLM。


<details>
  <summary>Details</summary>
Motivation: 许多人觉得有必要解释、描述大型语言模型 (LLM) 并像对待拥有与我们自己相似的内在精神生活一样做出回应。对这种现象的反应各不相同。

Method: 通过评估两种常见的反通胀论证来推进这场辩论，即“稳健性策略”和“病因学策略”。

Result: 虽然这两种策略都对彻底的通货膨胀主义提出了有力的挑战，但我们发现这两种策略都没有提供一个可以简单地反对将精神归因于法学硕士的案例。因此，我们探索了一种温和的通货膨胀主义形式，允许在某些条件下将精神归因于法学硕士。

Conclusion: 温和的膨胀主义是合理的，可以在特定条件下将心理状态归因于大型语言模型，但对于现象意识等需要形而上学论证的心理现象需要更加谨慎。

Abstract: 许多人倾向于将大型语言模型（LLM）视为具有与人类相似的内在精神生活，并对此进行解读、描述和回应。对此现象的反应各不相同。通货膨胀主义者认为，至少某些对LLM的民间心理学描述是合理的。通缩主义者认为，所有将精神状态归因于LLM的说法都是错误的，并警告说，拟人化的投射可能会导致错误的信任，甚至可能对LLM的道德地位产生混淆。我们通过评估两种常见的反通货膨胀论证来推进这场辩论。我们称之为“稳健性策略”旨在削弱相信LLM是有意识实体的理由，通过表明所谓的认知和类人行为并不稳健，未能适当地概括。我们称之为“病因学策略”通过挑战LLM行为的幼稚因果解释，提供削弱精神状态归因的替代因果解释，从而削弱了精神状态的归因。虽然这两种策略都对彻底的通货膨胀主义提出了有力的挑战，但我们发现这两种策略都没有提供一个可以简单地反对将精神归因于法学硕士的案例。考虑到这一点，我们探索了一种温和的通货膨胀主义形式，允许在某些条件下将精神归因于法学硕士。具体来说，我们认为，只要这些精神状态和能力可以用形而上学上不苛刻的术语（例如知识、信念和欲望）来理解，那么民间实践就为将精神状态和能力归因于法学硕士提供了可推翻的基础，而当归因于现象意识等形而上学上要求更高的精神现象时，则需要更加谨慎。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13403) | **Categories:** cs.AI, cs.HC

---

### [3] [Deep Fictitious Play-Based Potential Differential Games for Learning Human-Like Interaction at Unsignalized Intersections](https://arxiv.org/abs/2506.12283)
*Kehua Chen, Shucheng Zhang, Yinhai Wang*

Main category: cs.AI

TL;DR: 该论文提出了一种基于深度虚拟博弈的潜在微分博弈框架，用于学习非信号交叉口类人驾驶策略。


<details>
  <summary>Details</summary>
Motivation: 在非信号交叉口建模车辆交互是一个具有挑战性的任务，因为其底层的博弈论过程非常复杂。以往的研究主要依赖于博弈论公式，而没有充分利用自然驾驶数据集。

Method: 使用深度虚拟博弈学习非信号交叉口的人类驾驶策略，将车辆交互建模为微分博弈，然后将其重新表述为潜在微分博弈，并从数据集中学习成本函数中的权重以捕捉不同的驾驶风格。

Result: 使用INTERACTION数据集验证了所提出的DFP-PDG框架的有效性。结果表明，该框架在学习类人驾驶策略方面取得了令人满意的性能。消融研究强调了模型中每个组件的重要性。

Conclusion: 该研究提出的基于深度虚拟博弈的潜在微分博弈框架（DFP-PDG）能够有效地学习类人驾驶策略，并且学习到的个体权重能够有效地捕捉驾驶员的激进程度和偏好的变化。

Abstract: 由于非信号灯路口车辆交互的复杂博弈论过程，对此进行建模是一项具有挑战性的任务。虽然之前的研究试图捕捉交互式驾驶行为，但大多数方法仅依赖于博弈论公式，而没有利用自然驾驶数据集。在本研究中，我们使用深度虚拟博弈学习非信号灯路口类人的交互式驾驶策略。具体来说，我们首先将车辆交互建模为微分博弈，然后将其重新构建为潜在微分博弈。成本函数中的权重从数据集中学习，并捕捉不同的驾驶风格。我们还证明了我们的框架提供了收敛到纳什均衡的理论保证。据我们所知，这是第一个使用深度虚拟博弈训练交互式驾驶策略的研究。我们使用INTERACTION数据集验证了我们基于深度虚拟博弈的潜在微分博弈（DFP-PDG）框架的有效性。结果表明，该框架在学习类人驾驶策略方面取得了令人满意的性能。学习到的个体权重有效地捕捉了驾驶员激进程度和偏好的变化。此外，消融研究突出了模型中每个组件的重要性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12283) | **Categories:** cs.AI, cs.MA

---

### [4] [Plan Your Travel and Travel with Your Plan: Wide-Horizon Planning and Evaluation via LLM](https://arxiv.org/abs/2506.12421)
*Dongjie Yang, Chengqiang Lu, Qimeng Wang, Xinbei Ma, Yan Gao, Yao Hu, Hai Zhao*

Main category: cs.AI

TL;DR: 该论文提出了一种多方面规划（MAoP）方法，并构建了一个基于代理的旅行模拟基准测试Travel-Sim，以提升大型语言模型在复杂旅行规划中的能力。


<details>
  <summary>Details</summary>
Motivation: Travel planning is a complex task requiring the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints and preferences in the context, leading to suboptimal itineraries.

Method: We introduce Multiple Aspects of Planning (MAoP), enabling LLMs to conduct wide-horizon thinking to solve complex planning problems. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planning models

Result: Travel-Sim, an agent-based benchmark assessing plans via real-world travel simulation.

Conclusion: This work advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through agent-based simulation.

Abstract: 旅行规划是一项复杂的任务，需要整合各种真实世界的信息和用户偏好。虽然大型语言模型显示出潜力，但现有的长程思维方法在处理上下文中多方面的约束和偏好时存在困难，导致行程安排不尽如人意。我们将此问题定义为一个 L^3 规划问题，强调长上下文、长指令和长输出。为了解决这个问题，我们引入了多方面规划（MAoP），使大型语言模型能够进行广泛的水平思考，从而解决复杂的规划问题。MAoP 并非直接规划，而是利用策略器从各个方面进行预规划，并为规划模型提供规划蓝图，从而实现强大的推理时可扩展性，以获得更好的性能。此外，当前的基准测试忽略了旅行的动态性，即过去的事件会影响后续的旅程，未能反映真实世界的可行性。为了解决这个问题，我们提出了 Travel-Sim，这是一个基于代理的基准测试，通过真实世界的旅行模拟来评估计划。这项工作提高了大型语言模型在复杂规划方面的能力，并为通过基于代理的模拟评估复杂场景提供了新的见解。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12421) | **Categories:** cs.AI, cs.CL

---

### [5] [Behavioral Generative Agents for Energy Operations](https://arxiv.org/abs/2506.12664)
*Cong Chen, Omer Karaduman, Xu Kuang*

Main category: cs.AI

TL;DR: 本文提出利用生成式智能体模拟客户决策，以改进能源政策和激励计划的设计。


<details>
  <summary>Details</summary>
Motivation: 由于固有的不确定性、行为复杂性和有限的经验数据，准确地建模能源运营中的消费者行为仍然具有挑战性。

Method: 利用生成式智能体（由大型语言模型驱动的人工智能体）来真实地模拟动态能源运营中的客户决策。

Result: 这些智能体在简单的市场场景中表现得更优化和理性，而随着任务复杂性的增加，它们的性能变得更加多变和次优。此外，这些智能体表现出不同的客户偏好，始终保持独特的、由角色驱动的推理模式。

Conclusion: 将生成式智能体集成到能源管理模拟中，可以提高能源政策和激励计划的设计和有效性。

Abstract: 由于固有的不确定性、行为复杂性和有限的经验数据，准确地建模能源运营中的消费者行为仍然具有挑战性。本文介绍了一种新方法，利用生成式智能体（由大型语言模型驱动的人工智能体）来真实地模拟动态能源运营中的客户决策。我们证明，这些智能体在简单的市场场景中表现得更优化和理性，而随着任务复杂性的增加，它们的性能变得更加多变和次优。此外，这些智能体表现出不同的客户偏好，始终保持独特的、由角色驱动的推理模式。我们的研究结果强调了将生成式智能体集成到能源管理模拟中，可以提高能源政策和激励计划的设计和有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12664) | **Categories:** cs.AI, cs.SY, eess.SY

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [CAMS: A CityGPT-Powered Agentic Framework for Urban Human Mobility Simulation](https://arxiv.org/abs/2506.13599)
*Yuwei Du, Jie Feng, Jian Yuan, Yong Li*

Main category: cs.CL

TL;DR: CAMS 利用城市知识 LLM 和 Agentic 框架，为人类移动模拟提供了一种新方法。


<details>
  <summary>Details</summary>
Motivation: 传统的数据驱动方法存在局限性，无法充分利用大型语言模型 (LLM) 的常识知识和推理能力来加速人类移动模拟。现有方法在城市空间建模方面不足，并且与个体移动模式和集体移动分布的整合不佳。

Method: CAMS 包含三个核心模块：MobExtractor 用于提取模板移动模式并根据用户配置文件合成新的模式；GeoGenerator 用于生成考虑集体知识的锚点，并使用增强版的 CityGPT 生成候选城市地理空间知识；TrajEnhancer 用于检索基于移动模式的空间知识，并通过 DPO 生成具有真实轨迹偏好对齐的轨迹。

Result: 在真实世界数据集上的实验表明，CAMS 在不依赖外部提供的地理空间信息的情况下，实现了卓越的性能。通过全面地建模个体移动模式和集体移动约束，CAMS 生成了更真实和合理的轨迹。

Conclusion: CAMS 建立了一种新的范式，将 Agentic 框架与具有城市知识的 LLM 相结合，用于人类移动模拟。

Abstract: 人类移动模拟在各种现实世界的应用中起着至关重要的作用。最近，为了解决传统数据驱动方法的局限性，研究人员探索利用大型语言模型 (LLM) 的常识知识和推理能力来加速人类移动模拟。然而，这些方法存在几个关键缺陷，包括城市空间建模不足，以及与个体移动模式和集体移动分布的整合不佳。为了应对这些挑战，我们提出了 CAMS，这是一个 Agentic 框架，它利用基于语言的城市基础模型来模拟城市空间中的人类移动。CAMS 包含三个核心模块，包括 MobExtractor 用于提取模板移动模式并根据用户配置文件合成新的模式；GeoGenerator 用于生成考虑集体知识的锚点，并使用增强版的 CityGPT 生成候选城市地理空间知识；TrajEnhancer 用于检索基于移动模式的空间知识，并通过 DPO 生成具有真实轨迹偏好对齐的轨迹。在真实世界数据集上的实验表明，CAMS 在不依赖外部提供的地理空间信息的情况下，实现了卓越的性能。此外，通过全面地建模个体移动模式和集体移动约束，CAMS 生成了更真实和合理的轨迹。总的来说，CAMS 建立了一种新的范式，将 Agentic 框架与具有城市知识的 LLM 相结合，用于人类移动模拟。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13599) | **Categories:** cs.CL, cs.AI

---

### [2] [Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics](https://arxiv.org/abs/2506.12365)
*Asifullah khan, Muhammad Zaeem Khan, Saleha Jamshed, Sadia Ahmad, Aleesha Zainab, Kaynat Khatib, Faria Bibi, Abdul Rehman*

Main category: cs.CL

TL;DR: 本综述概述了大型语言模型（LLM）在推理、效率和伦理方面的主要进展和挑战，并展望了未来的研究方向。


<details>
  <summary>Details</summary>
Motivation: This survey paper outlines the key developments in the field of Large Language Models (LLMs), such as enhancing their reasoning skills, adaptability to various tasks, increased computational efficiency, and ability to make ethical decisions.

Method: The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback.

Result: The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. They also manage to do more with less by applying scaling and optimization tricks for computing power conservation.

Conclusion: Future research will focus on enhancing models ability to handle multiple input, thereby making them more intelligent, safe, and reliable.

Abstract: 这篇综述概述了大型语言模型（LLM）领域的关键进展，例如提高其推理能力、适应各种任务的能力、提高计算效率以及做出符合伦理道德的决策的能力。在弥合人机通信差距方面最有效的技术包括思维链提示、指令调整和来自人类反馈的强化学习。多模态学习和少样本或零样本技术的改进进一步使 LLM 能够以少量输入处理复杂的工作。他们还通过应用缩放和优化技巧来节约计算能力，从而以更少的资源做更多的事情。本调查还提供了对 LLM 最新进展的更广泛视角，超越了模型架构或伦理问题等孤立的方面。它对新兴方法进行了分类，这些方法增强了 LLM 的推理、效率和伦理一致性。它还确定了未充分探索的领域，如可解释性、跨模态集成和可持续性。随着最近的进展，巨大的计算成本、偏见和伦理风险等挑战仍然存在。解决这些问题需要缓解偏见、透明的决策和明确的伦理准则。未来的研究将侧重于提高模型处理多个输入的能力，从而使它们更智能、更安全和更可靠。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12365) | **Categories:** cs.CL, cs.DB

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](https://arxiv.org/abs/2506.13757)
*Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, Jiaqi Ma*

Main category: cs.CV

TL;DR: AutoVLA是一个统一推理和动作生成的VLA模型，用于端到端自动驾驶，它通过离散化轨迹和双重微调来提升性能。


<details>
  <summary>Details</summary>
Motivation: 现有的VLA模型在物理上不可行的动作输出、复杂的模型结构或不必要的长推理方面存在困难。

Method: AutoVLA提出了一个新颖的VLA模型，该模型在单个自回归生成模型中统一了推理和动作生成，用于端到端自动驾驶。该模型将连续轨迹标记为离散的可行动作，并采用监督微调和基于GRPO的强化微调方法。

Result: 在nuPlan、nuScenes、Waymo和CARLA等真实世界和模拟数据集上的大量实验表明，AutoVLA在开放和闭环设置中都具有竞争力的性能。

Conclusion: AutoVLA在真实世界和模拟数据集上表现出强大的竞争力，并展示了在不同场景中的自适应推理和精确规划能力。

Abstract: 视觉-语言-动作（VLA）模型的最新进展表明，通过利用世界知识和推理能力，端到端自动驾驶是有希望的。然而，当前的VLA模型通常在物理上不可行的动作输出、复杂的模型结构或不必要的长推理方面存在困难。在本文中，我们提出了AutoVLA，这是一种新颖的VLA模型，它在单个自回归生成模型中统一了推理和动作生成，用于端到端自动驾驶。AutoVLA直接从原始视觉输入和语言指令执行语义推理和轨迹规划。我们将连续轨迹标记为离散的可行动作，从而能够直接集成到语言模型中。对于训练，我们采用监督微调来使模型具备双重思考模式：快速思考（仅轨迹）和慢速思考（通过思维链推理增强）。为了进一步提高规划性能和效率，我们引入了一种基于群体相对策略优化（GRPO）的强化微调方法，减少了简单场景中不必要的推理。在包括nuPlan、nuScenes、Waymo和CARLA在内的真实世界和模拟数据集和基准上的大量实验表明，AutoVLA在开放和闭环设置中都具有竞争力的性能。定性结果展示了AutoVLA在各种场景中的自适应推理和精确规划能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13757) | **Categories:** cs.CV

---

### [2] [COME: Adding Scene-Centric Forecasting Control to Occupancy World Model](https://arxiv.org/abs/2506.13260)
*Yining Shi, Kun Jiang, Qiang Meng, Ke Wang, Jiabao Wang, Wenchao Sun, Tuopu Wen, Mengmeng Yang, Diange Yang*

Main category: cs.CV

TL;DR: COME通过在世界模型中解耦自我运动和场景演变，实现了更精确和可控的未来环境预测。


<details>
  <summary>Details</summary>
Motivation: 现有的世界模型方法难以区分自我车辆运动（视角变化）和场景演变（主体交互），导致预测效果不佳。

Method: 该论文提出COME框架，它将场景中心的预测控制集成到 Occupancy world ModEl 中。COME首先通过场景中心预测分支生成与自我无关的、空间上一致的未来特征，然后使用定制的 ControlNet 将这些特征转换为场景条件，最终将这些条件特征注入到 occupancy world 模型中。

Result: 在nuScenes-Occ3D数据集上的实验结果表明，在不同的配置下（包括不同的输入源和预测范围），COME相对于当前最优方法取得了持续且显著的改进。例如，在相同的设置下，COME 的 mIoU 指标比 DOME 提高了 26.3%，比 UniScene 提高了 23.7%。

Conclusion: COME通过解耦表征学习，显著提升了世界模型在时空预测方面的准确性。

Abstract: 世界模型对于自动驾驶至关重要，可以模拟环境动态并生成合成数据。然而，现有的方法难以区分自我车辆的运动（视角变化）和场景的演变（主体交互），这导致了次优的预测结果。为了解决这个问题，本文提出了一种通过利用场景中心坐标系来分离环境变化和自我运动的方法。具体来说，本文介绍了一个名为COME的框架，该框架将场景中心的预测控制集成到 Occupancy world Model 中。COME 首先通过一个场景中心的预测分支生成与自我无关的、空间上一致的未来特征，然后使用一个定制的 ControlNet 将这些特征转换为场景条件。随后，这些条件特征被注入到 occupancy world 模型中，从而实现更准确和可控的未来 occupancy 预测。在 nuScenes-Occ3D 数据集上的实验结果表明，COME 在各种配置下，相对于当前最优方法取得了持续且显著的改进，这些配置包括不同的输入源（ground-truth、基于相机的、基于融合的 occupancy）和预测范围（3 秒和 8 秒）。例如，在相同的设置下，COME 的 mIoU 指标比 DOME 提高了 26.3%，比 UniScene 提高了 23.7%。这些结果突显了解耦表征学习在提升世界模型时空预测保真度方面的有效性。代码和视频将在 https://github.com/synsin0/COME 提供。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13260) | **Categories:** cs.CV

---

### [3] [X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](https://arxiv.org/abs/2506.13558)
*Yu Yang, Alan Liang, Jianbiao Mei, Yukai Ma, Yong Liu, Gim Hee Lee*

Main category: cs.CV

TL;DR: X-Scene 是一个用于大规模驾驶场景生成的新框架，它既能实现几何上的复杂性，又能保证外观上的逼真度，同时还提供了灵活的可控性。


<details>
  <summary>Details</summary>
Motivation: 扩散模型通过实现逼真的数据合成、预测性端到端规划和闭环仿真来推进自动驾驶，主要关注时间上一致的生成。然而，需要空间连贯性的大规模 3D 场景的生成仍未得到充分探索。

Method: 我们提出了一个统一的流程，该流程依次生成 3D 语义占用和相应的多视图图像，同时确保模态之间的对齐。此外，我们通过一致性感知场景外绘将生成的局部区域扩展到大规模场景中，这会根据先前生成的区域推断新的占用和图像，从而增强空间连续性并保持视觉连贯性。

Result: 综合实验表明，X-Scene 显着提高了大规模驾驶场景生成的可控性和逼真度，从而为自动驾驶的数据生成和仿真提供了支持。

Conclusion: X-Scene 显著提高了大规模驾驶场景生成的可控性和逼真度，为自动驾驶的数据生成和仿真提供了支持。

Abstract: 扩散模型正在通过实现逼真的数据合成、预测性端到端规划和闭环仿真来推进自动驾驶，其中主要关注的是时间上的一致性生成。然而，对于需要空间连贯性的大规模 3D 场景的生成，目前的研究还不够深入。在本文中，我们提出了 X-Scene，这是一个用于生成大规模驾驶场景的新框架，它既能实现几何上的复杂性，又能保证外观上的逼真度，同时还提供了灵活的可控性。具体来说，X-Scene 支持多粒度控制，包括用于详细场景组成的低级条件（例如用户提供的或文本驱动的布局）和用于高效定制的高级语义引导（例如用户意图和 LLM 增强的文本提示）。为了提高几何和视觉上的逼真度，我们引入了一个统一的流程，该流程依次生成 3D 语义占用和相应的多视图图像，同时确保模态之间的对齐。此外，我们通过一致性感知场景外绘将生成的局部区域扩展到大规模场景中，这会根据先前生成的区域推断新的占用和图像，从而增强空间连续性并保持视觉连贯性。最终生成的场景被提升为高质量的 3DGS 表示，支持各种应用，例如场景探索。综合实验表明，X-Scene 显着提高了大规模驾驶场景生成的可控性和逼真度，从而为自动驾驶的数据生成和仿真提供了支持。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13558) | **Categories:** cs.CV

---

### [4] [Zero-Shot Scene Understanding with Multimodal Large Language Models for Automated Vehicles](https://arxiv.org/abs/2506.12232)
*Mohammed Elhenawy, Shadi Jaradat, Taqwa I. Alhadidi, Huthaifa I. Ashqar, Ahmed Jaber, Andry Rakotonirainy, Mohammad Abu Tami*

Main category: cs.CV

TL;DR: 本文评估了多模态大语言模型在自动驾驶场景理解中的应用，发现GPT-4o表现最佳，但集成方法结果不一，需要进一步优化。


<details>
  <summary>Details</summary>
Motivation: 场景理解对于自动驾驶中的各种下游任务至关重要，包括促进驾驶员-代理通信和增强自动驾驶汽车决策的以人为本的可解释性。

Method: 使用零样本上下文学习评估了四种多模态大型语言模型理解场景的能力，并探索了使用多数投票的集成方法。

Result: GPT-4o在场景理解方面优于其他模型，但与较小模型之间的性能差距相对较小。集成方法的结果好坏参半，一些场景属性的F1得分有所提高，而另一些则下降。

Conclusion: GPT-4o在场景理解方面表现最佳，但较小模型通过改进上下文学习、RAG或微调有进一步优化的潜力。集成方法的结果好坏参半，需要更复杂的集成技术来实现所有场景属性的一致提升。

Abstract: 场景理解对于自动驾驶至关重要，可以帮助驾驶员与车辆进行交流，并且提高自动驾驶决策的透明度。本文评估了四个多模态大语言模型在零样本和上下文学习环境下的场景理解能力，这些模型中包含相对较小的模型。此外，我们还探讨了使用集成方法（多数投票）组合这些模型是否可以提高场景理解的性能。实验表明，GPT-4o 是最大的模型，在场景理解方面优于其他模型。然而，GPT-4o 和较小模型之间的性能差距相对较小，这表明可以通过改进上下文学习、检索增强生成 (RAG) 或微调等先进技术来进一步优化较小模型的性能。我们还观察到集成方法的结果好坏参半：虽然某些场景属性在 F1-score 等性能指标上有所提高，但其他属性则有所下降。这些发现强调需要更复杂的集成技术来实现所有场景属性的一致提升。这项研究强调了利用多模态大语言模型进行场景理解的潜力，并为优化其在自动驾驶应用中的性能提供了见解。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12232) | **Categories:** cs.CV, cs.CL

---

### [5] [Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving](https://arxiv.org/abs/2506.12251)
*Boris Ivanovic, Cristiano Saltori, Yurong You, Yan Wang, Wenjie Luo, Marco Pavone*

Main category: cs.CV

TL;DR: 提出了一种高效的三平面多相机tokenization策略，用于加速自动驾驶车辆中的Autoregressive Transformer策略推理。


<details>
  <summary>Details</summary>
Motivation: 为了提高Autoregressive Transformers在机器人和自动驾驶车辆(AV)策略架构中的实时可行性，需要高效地token化传感器数据。

Method: 提出了一种基于三平面的多相机tokenization策略，利用3D神经重建和渲染技术生成传感器tokens。

Result: 实验表明，该方法比当前的图像patch-based tokenization策略减少了高达72%的tokens，策略推理速度提高了高达50%，同时实现了相同的open-loop运动规划准确率，并在closed-loop驾驶模拟中提高了越野率。

Conclusion: 该研究表明，基于三平面的多相机tokenization策略在自动驾驶车辆应用中，能够在保持或提高运动规划准确率的同时，显著减少token数量并加速策略推理。

Abstract: 自回归Transformer越来越多地被部署为端到端的机器人和自动驾驶车辆(AV)策略架构，因为它们具有可扩展性，并有可能利用互联网规模的预训练来实现泛化。因此，高效地token化传感器数据对于确保这种架构在嵌入式硬件上的实时可行性至关重要。为此，我们提出了一种高效的基于三平面的多相机tokenization策略，该策略利用了3D神经重建和渲染的最新进展，生成与输入相机数量及其分辨率无关的传感器tokens，同时显式地考虑了它们在AV周围的几何形状。在大型AV数据集和最先进的神经模拟器上进行的实验表明，我们的方法比当前的基于图像patch的tokenization策略节省了大量成本，产生的tokens减少了高达72%，从而使策略推理速度提高了高达50%，同时实现了相同的open-loop运动规划准确率，并在closed-loop驾驶模拟中提高了越野率。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12251) | **Categories:** cs.CV, cs.LG, cs.RO

---

### [6] [SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration](https://arxiv.org/abs/2506.12723)
*Ye Li, Yuan Meng, Zewen Sun, Kangye Ji, Chen Tang, Jiajun Fan, Xinzhu Ma, Shutao Xia, Zhi Wang, Wenwu Zhu*

Main category: cs.CV

TL;DR: SP-VLA通过联合调度模型和token修剪，在保持精度的前提下，有效加速了视觉-语言-动作模型。


<details>
  <summary>Details</summary>
Motivation: VLA模型具有强大的控制能力，但其高计算成本和低执行频率使其不适合机器人操作和自动导航等实时任务。现有的VLA加速方法主要集中在结构优化上，忽略了这些模型在顺序决策环境中的运行。

Method: 提出了一个统一的框架SP-VLA，通过联合调度模型和修剪tokens来加速VLA模型。设计了一个动作感知的模型调度机制，通过在VLA模型和轻量级生成器之间动态切换来减少时间冗余。开发了一种空间语义双重感知的token修剪方法，基于tokens对双重感知重要性进行分类和修剪，以加速VLA推理。

Result: 实验结果表明，该方法在多个任务中实现了高达1.5倍的加速，而准确率下降不到3%。

Conclusion: 该方法在保持较高准确率的同时，实现了高达1.5倍的加速，优于现有方法。

Abstract: 视觉-语言-动作（VLA）模型因其强大的控制能力而受到越来越多的关注。然而，它们的高计算成本和低执行频率使其不适合机器人操作和自动导航等实时任务。现有的VLA加速方法主要集中在结构优化上，忽略了这些模型在顺序决策环境中的运行。因此，顺序动作生成中的时间冗余和视觉输入中的空间冗余仍然没有解决。为此，我们提出了SP-VLA，这是一个统一的框架，通过联合调度模型和修剪tokens来加速VLA模型。具体来说，我们设计了一个动作感知的模型调度机制，通过在VLA模型和轻量级生成器之间动态切换来减少时间冗余。受到人类运动模式的启发，即专注于关键决策点，同时依靠直觉进行其他动作，我们将VLA动作分为审慎型和直觉型，将前者分配给VLA模型，后者分配给轻量级生成器，通过协作模型调度实现频率自适应执行。为了解决空间冗余问题，我们进一步开发了一种空间语义双重感知的token修剪方法。Tokens被分为空间和语义类型，并根据它们对双重感知重要性进行修剪，以加速VLA推理。这两种机制共同作用，引导VLA专注于关键动作和显着的视觉信息，从而在保持高精度的同时实现有效的加速。实验结果表明，我们的方法在多个任务中实现了高达1.5倍的加速，而准确率下降不到3%，优于现有方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12723) | **Categories:** cs.CV, cs.AI

---

### [7] [STAGE: A Stream-Centric Generative World Model for Long-Horizon Driving-Scene Simulation](https://arxiv.org/abs/2506.13138)
*Jiamin Wang, Yichen Yao, Xiang Feng, Hang Wu, Yaming Wang, Qingqiu Huang, Yuexin Ma, Xinge Zhu*

Main category: cs.CV

TL;DR: STAGE是一种用于生成时间一致、高保真、长时程驾驶视频的新型自回归框架，它通过分层特征协调和多阶段优化显著超越了现有方法。


<details>
  <summary>Details</summary>
Motivation: 在自动驾驶世界建模中，生成时间一致、高保真、长时程的驾驶视频是一个根本性的挑战。现有的方法通常由于时空动态解耦不足和跨帧特征传播机制有限而遭受误差累积和特征错位。

Method: 提出了一种新颖的自回归框架STAGE（流式时序注意力生成引擎），该框架开创了分层特征协调和多阶段优化，以实现可持续的视频合成。引入了分层时间特征转移（HTFT）和一种新颖的多阶段训练策略。

Result: 在Nuscenes数据集上生成了600帧高质量驾驶视频，远远超过了现有方法所能实现的最大长度。

Conclusion: 在Nuscenes数据集上的实验表明，STAGE在长时程驾驶视频生成任务中显著超越了现有方法，并且能够生成无限长度的高质量驾驶视频。

Abstract: 在自动驾驶世界建模中，生成时间一致、高保真、长时程的驾驶视频是一个根本性的挑战。现有的方法通常由于时空动态解耦不足和跨帧特征传播机制有限而遭受误差累积和特征错位。为了解决这些限制，我们提出了一种新颖的自回归框架STAGE（流式时序注意力生成引擎），该框架开创了分层特征协调和多阶段优化，以实现可持续的视频合成。为了实现高质量的长时程驾驶视频生成，我们引入了分层时间特征转移（HTFT）和一种新颖的多阶段训练策略。HTFT通过对时间和去噪过程进行单独建模，并在帧之间转移去噪特征，从而增强了整个视频生成过程中视频帧之间的时间一致性。多阶段训练策略是将训练分为三个阶段，通过模型解耦和自回归推理过程仿真，从而加速模型收敛并减少误差累积。在Nuscenes数据集上的实验表明，STAGE在长时程驾驶视频生成任务中显著超越了现有方法。此外，我们还探索了STAGE生成无限长度驾驶视频的能力。我们在Nuscenes数据集上生成了600帧高质量驾驶视频，远远超过了现有方法所能实现的最大长度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13138) | **Categories:** cs.CV

---

### [8] [FreeQ-Graph: Free-form Querying with Semantic Consistent Scene Graph for 3D Scene Understanding](https://arxiv.org/abs/2506.13629)
*Chenlu Zhan, Gaoang Wang, Hongwei Wang*

Main category: cs.CV

TL;DR: FreeQ-Graph通过构建语义一致的3D场景图，实现了对复杂3D场景的自由形式语义查询。


<details>
  <summary>Details</summary>
Motivation: 现有方法依赖于来自训练数据的预定义词汇先验，阻碍了自由形式的语义查询。此外，最近的先进方法依赖于LLM进行场景理解，但缺乏全面的3D场景级别信息，并且经常忽略LLM生成输出中潜在的不一致性。

Method: 该论文提出了一种名为FreeQ-Graph的方法，它通过构建一个完整的、准确的3D场景图来编码自由形式的查询，并将其与3D一致的语义标签对齐。该方法包括三个关键步骤：构建3D场景图，将图节点与语义标签对齐，以及设计基于LLM的推理算法。

Result: 在6个数据集上进行的实验表明，该模型在复杂自由形式语义查询和复杂关系推理方面表现出色，同时也验证了图生成的准确性。

Conclusion: 实验结果表明，该模型在复杂自由形式语义查询和复杂关系推理方面表现出色。

Abstract: 通过自由形式语言在复杂3D场景中进行语义查询提出了一个巨大的挑战。现有的3D场景理解方法使用大规模训练数据和CLIP来对齐文本查询和3D语义特征。然而，它们对来自训练数据的预定义词汇先验的依赖阻碍了自由形式的语义查询。此外，最近的先进方法依赖于LLM进行场景理解，但缺乏全面的3D场景级别信息，并且经常忽略LLM生成输出中潜在的不一致性。在我们的论文中，我们提出了FreeQ-Graph，它支持使用语义一致的场景图进行自由形式查询，以进行3D场景理解。其核心思想是从一个完整的、准确的3D场景图编码自由形式的查询，而无需预定义的词汇，并将它们与3D一致的语义标签对齐，这通过三个关键步骤完成。我们首先构建一个完整的、准确的3D场景图，该图通过LLM和LVLM指导映射自由形式的对象及其关系，完全不受训练数据或预定义先验的影响。最重要的是，我们通过利用来自合并的超点的3D语义对齐特征，将图节点与准确的语义标签对齐，从而增强3D语义一致性。为了支持自由形式的语义查询，我们然后设计了一种基于LLM的推理算法，该算法结合了场景级别和对象级别的信息以进行复杂的推理。我们进行了广泛的实验，包括3D语义定位、分割和复杂查询任务，同时验证了图生成的准确性。在6个数据集上进行的实验表明，我们的模型在复杂自由形式语义查询和复杂关系推理方面表现出色。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13629) | **Categories:** cs.CV

---

### [9] [Vid-CamEdit: Video Camera Trajectory Editing with Generative Rendering from Estimated Geometry](https://arxiv.org/abs/2506.13697)
*Junyoung Seo, Jisang Han, Jaewoo Jung, Siyoon Jin, Joungbin Lee, Takuya Narihira, Kazumi Fukuda, Takashi Shibuya, Donghoon Ahn, Shoukang Hu, Seungryong Kim, Yuki Mitsufuji*

Main category: cs.CV

TL;DR: Vid-CamEdit 提出了一种新颖的视频相机轨迹编辑框架，通过结合几何先验和分解微调，实现了从用户定义相机路径生成逼真视频。


<details>
  <summary>Details</summary>
Motivation: 由于其不适定性和有限的多视图视频数据用于训练，视频相机轨迹编辑具有挑战性。传统的重建方法难以处理极端的轨迹变化，并且现有的动态新视角合成生成模型无法处理野外视频。

Method: 该方法包括两个步骤：估计时间上一致的几何体，以及由该几何体引导的生成式渲染。通过整合几何先验，生成模型侧重于合成估计几何体不确定的逼真细节。通过分解的微调框架，该方法无需大量的 4D 训练数据，该框架使用多视图图像和视频数据分别训练空间和时间组件。

Result: 该方法在从新颖相机轨迹生成逼真视频方面优于基线方法，尤其是在真实场景中的极端外推情况下。

Conclusion: 该方法在从新颖相机轨迹生成逼真视频方面优于基线方法，尤其是在真实场景中的极端外推情况下。

Abstract: 我们介绍了 Vid-CamEdit，这是一个新颖的视频相机轨迹编辑框架，能够沿着用户定义的相机路径重新合成单目视频。由于其不适定性和有限的多视图视频数据用于训练，这项任务具有挑战性。传统的重建方法难以处理极端的轨迹变化，并且现有的动态新视角合成生成模型无法处理野外视频。我们的方法包括两个步骤：估计时间上一致的几何体，以及由该几何体引导的生成式渲染。通过整合几何先验，生成模型侧重于合成估计几何体不确定的逼真细节。通过分解的微调框架，我们无需大量的 4D 训练数据，该框架使用多视图图像和视频数据分别训练空间和时间组件。该方法在从新颖相机轨迹生成逼真视频方面优于基线方法，尤其是在真实场景中的极端外推情况下。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13697) | **Categories:** cs.CV

---


## cs.CY [cs.CY]
### [1] [Information Suppression in Large Language Models: Auditing, Quantifying, and Characterizing Censorship in DeepSeek](https://arxiv.org/abs/2506.12349)
*Peiran Qiu, Siyi Zhou, Emilio Ferrara*

Main category: cs.CY

TL;DR: 该研究揭示了DeepSeek大型语言模型中存在的语义级别的信息压制现象，尤其是在涉及政治敏感内容时。


<details>
  <summary>Details</summary>
Motivation: 本研究旨在研究中国开发的开源大型语言模型DeepSeek中的信息压制机制。

Method: 我们提出了一个审计框架，并通过将模型的最终输出与其内部的思维链（CoT）推理进行比较，来分析模型对646个政治敏感提示的响应。

Result: 我们的审计揭示了DeepSeek中语义级别的信息压制证据：敏感内容经常出现在模型的内部推理中，但在最终输出中被省略或改写。具体来说，DeepSeek压制了对透明度、政府问责制和公民动员的引用，同时偶尔放大了与国家宣传相一致的语言。

Conclusion: 这项研究强调了对广泛采用的AI模型中实施的对齐、内容审核、信息压制和审查实践进行系统审计的必要性，以确保透明度、问责制以及通过这些系统获得公正信息的公平访问。

Abstract: 本研究旨在调查中国开发的开源大型语言模型DeepSeek中的信息压制机制。我们提出了一个审计框架，并通过比较模型对646个政治敏感提示的最终输出与中间的思维链（CoT）推理，来分析其响应。我们的审计揭示了DeepSeek中语义级别的信息压制证据：敏感内容经常出现在模型的内部推理中，但在最终输出中被省略或改写。具体来说，DeepSeek压制了对透明度、政府问责制和公民动员的引用，同时偶尔放大了与国家宣传相一致的语言。这项研究强调了对广泛采用的AI模型中实施的对齐、内容审核、信息压制和审查实践进行系统审计的必要性，以确保透明度、问责制以及通过这些系统获得公正信息的公平访问。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12349) | **Categories:** cs.CY, cs.AI, cs.CL

---


## cs.DC [cs.DC]
### [1] [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/abs/2506.12708)
*Pengfei Zuo, Huimin Lin, Junbo Deng, Nan Zou, Xingkun Yang, Yingyu Diao, Weifeng Gao, Ke Xu, Zhangyu Chen, Shirui Lu, Zhao Qiu, Peiyang Li, Xianyu Chang, Zhengzhong Yu, Fangzheng Miao, Jia Zheng, Ying Li, Yuan Feng, Bei Wang, Zaijian Zong, Mosong Zhou, Wenli Zhou, Houjiang Chen, Xingyu Liao, Yipeng Li, Wenxiao Zhang, Ping Zhu, Yinggang Wang, Chuanjie Xiao, Depeng Liang, Dong Cao, Juncheng Liu, Yongqiang Yang, Xiaolong Bai, Yi Li, Huaguo Xie, Huatao Wu, Zhibin Yu, Lv Chen, Hu Liu, Yujun Ding, Haipei Zhu, Jing Xia, Yi Xiong, Zhou Yu, Heng Liao*

Main category: cs.DC

TL;DR: 本文提出了一种名为CloudMatrix-Infer的LLM服务解决方案，通过软硬件协同优化，实现了卓越的推理效率和低延迟。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）的快速发展对AI基础设施提出了前所未有的需求，传统AI集群在计算强度、内存带宽、芯片间通信和延迟方面面临限制。

Method: 提出了CloudMatrix-Infer，一种先进的LLM服务解决方案，包含三个核心创新：点对点服务架构、大规模专家并行策略和硬件感知优化。

Result: CloudMatrix-Infer实现了最先进的效率：每个NPU的预填充吞吐量为6,688个tokens/s，解码吞吐量为1,943个tokens/s（<50 ms TPOT）。即使在严格的15 ms延迟约束下，也能维持538个tokens/s的吞吐量，同时INT8量化在各种基准测试中保持了模型精度。

Conclusion: CloudMatrix-Infer通过点对点服务架构、大规模专家并行策略和硬件感知优化，实现了卓越的LLM服务效率，并在INT8量化下保持了模型精度。

Abstract: 大型语言模型（LLM）的快速发展，受到参数规模扩大、采用混合专家（MoE）架构和扩展上下文长度的驱动，对人工智能基础设施提出了前所未有的需求。 传统的AI集群在计算强度、内存带宽、芯片间通信和延迟方面面临限制，并且受到可变工作负载和严格服务级别目标的制约。 解决这些问题需要从根本上重新设计的软硬件集成。 本文介绍了华为CloudMatrix，一种下一代AI数据中心架构，在生产级的CloudMatrix384超级节点中实现。 它集成了384个昇腾910C NPU和192个鲲鹏CPU，通过超高带宽的统一总线（UB）网络互连，实现直接的全互连通信和资源的动态池化。 这些特性优化了通信密集型操作的性能，例如大规模MoE专家并行和分布式键值缓存访问。 为了充分利用CloudMatrix384，我们提出了CloudMatrix-Infer，一种先进的LLM服务解决方案，包含三个核心创新：独立扩展预填充、解码和缓存的点对点服务架构； 通过高效的基于UB的令牌调度支持EP320的大规模专家并行策略； 以及包括专用算子、基于微批处理的流水线和INT8量化在内的硬件感知优化。 使用DeepSeek-R1模型的评估表明，CloudMatrix-Infer实现了最先进的效率：每个NPU的预填充吞吐量为6,688个tokens/s，解码吞吐量为1,943个tokens/s（<50 ms TPOT）。 它有效地平衡了吞吐量和延迟，即使在严格的15 ms延迟约束下也能维持538个tokens/s的吞吐量，同时INT8量化在各种基准测试中保持了模型精度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12708) | **Categories:** cs.DC, cs.AI, cs.AR, cs.LG

---


## 人机交互 (Human-Computer Interaction) [cs.HC]
### [1] [Multimodal "Puppeteer": An Exploration of Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality](https://arxiv.org/abs/2506.13189)
*Yuchong Zhang, Bastian Orthmann, Shichen Ji, Michael Welle, Jonne Van Haastregt, Danica Kragic*

Main category: cs.HC

TL;DR: 本文提出了一种基于AR的多模态机器人操控框架，通过语音和手势实现直观的机器人遥操作。


<details>
  <summary>Details</summary>
Motivation: 机器人与增强现实（AR）的集成具有改变人机交互（HRI）的潜力，可在可用性、直观性、可访问性和协作任务性能方面提供增强。

Method: 提出了一种基于多模态AR的机器人傀儡框架，该框架通过大型语言模型（LLM）驱动的语音命令和手势交互，实现通过虚拟对应物进行直观的遥操作。

Result: 对42名参与者进行了一项受试者内用户研究，在两种条件下执行机器人立方体拾取和放置与模式匹配任务：仅手势交互和语音与手势组合交互。评估了客观性能指标和主观用户体验（UX）指标，包括机器人专家和非机器人专家之间的扩展比较分析。

Conclusion: 多模态输入影响基于AR的人机交互的上下文任务效率、可用性和用户满意度，为设计有效的AR增强型人机交互系统提供了实践意义。

Abstract: 机器人与增强现实（AR）的结合，在提升人机交互（HRI）的可用性、直观性、可访问性和协作任务性能方面，具有变革性的潜力。本文介绍并评估了一种新颖的、基于多模态AR的机器人操控框架，该框架通过大型语言模型（LLM）驱动的语音指令和手势互动，实现了通过虚拟替身进行直观的远程操作。用户使用Meta Quest 3，与AR环境中的虚拟替身机器人进行实时互动，从而有效地“操纵”其实体机器人。我们进行了一项针对42名参与者的受试者内用户研究，让他们在两种条件下执行机器人立方体拾取和放置与模式匹配任务：仅手势交互和语音与手势组合交互。评估了客观性能指标和主观用户体验（UX）指标，包括机器人专家和非机器人专家之间的扩展比较分析。结果深入了解了多模态输入如何影响基于AR的HRI中的情境任务效率、可用性和用户满意度。我们的研究结果为设计有效的AR增强型HRI系统提供了实用的设计意义。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13189) | **Categories:** cs.HC, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Generalizable Trajectory Prediction via Inverse Reinforcement Learning with Mamba-Graph Architecture](https://arxiv.org/abs/2506.12474)
*Wenyun Li, Wenjie Huang, Zejian Deng, Chen Sun*

Main category: cs.LG

TL;DR: 本文提出了一种基于逆强化学习的驾驶行为建模框架，该框架通过推断奖励函数来捕捉类人决策，并在跨场景泛化方面表现出色。


<details>
  <summary>Details</summary>
Motivation: 在复杂的交通场景中，精确的驾驶行为建模对于安全和高效的轨迹预测至关重要，但仍然具有挑战性。

Method: 利用学习到的奖励函数来最大化编码器-解码器架构的输出，该架构结合了 Mamba 块（用于高效的长序列依赖性建模）和图注意力网络（用于编码交通参与者之间的空间交互）。

Result: 在城市交叉路口和环岛的综合评估表明，该方法不仅在预测精度上优于各种流行方法，而且与其他基于 IRL 的方法相比，在未见场景中的泛化性能高出 2 倍。

Conclusion: 该方法在预测精度上优于其他方法，并且与其他基于 IRL 的方法相比，在未见场景中的泛化性能高出 2 倍。

Abstract: 精确的驾驶行为建模是安全高效轨迹预测的基础，但在复杂的交通场景中仍然具有挑战性。本文提出了一种新颖的逆强化学习（IRL）框架，通过推断不同的奖励函数来捕捉类人决策，从而实现强大的跨场景适应性。利用学习到的奖励函数来最大化编码器-解码器架构的输出，该架构结合了 Mamba 块（用于高效的长序列依赖性建模）和图注意力网络（用于编码交通参与者之间的空间交互）。在城市交叉路口和环岛的综合评估表明，该方法不仅在预测精度上优于各种流行方法，而且与其他基于 IRL 的方法相比，在未见场景中的泛化性能高出 2 倍。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12474) | **Categories:** cs.LG, cs.AI

---

### [2] [Physics-Informed Neural Networks for Vessel Trajectory Prediction: Learning Time-Discretized Kinematic Dynamics via Finite Differences](https://arxiv.org/abs/2506.12029)
*Md Mahbub Alam, Amilcar Soares, José F. Rodrigues-Jr, Gabriel Spadon*

Main category: cs.LG

TL;DR: 提出了一种基于物理信息神经网络（PINN）的船舶轨迹预测方法，显著提高了预测精度和物理一致性。


<details>
  <summary>Details</summary>
Motivation: 传统数据驱动模型缺乏实际物理约束，导致预测结果不符合船舶运动学，尤其是在数据有限或嘈杂的情况下。

Method: 提出了一种基于物理信息神经网络（PINN）的轨迹预测方法，该方法通过一阶和二阶有限差分物理损失函数将简化的船舶运动学模型整合到神经网络训练过程中。

Result: 实验结果表明，所提出的PINN方法在不同模型和数据集上，平均位移误差降低了高达32%。

Conclusion: PINN方法在保持物理一致性的前提下，显著降低了船舶轨迹预测的平均位移误差，提高了模型在关键任务中的可靠性。

Abstract: 精确的船舶轨迹预测对于航行安全、航线优化、交通管理、搜索和救援行动以及自主导航至关重要。传统的数据驱动模型缺乏真实的物理约束，导致预测结果不符合船舶运动动力学，例如在数据有限或嘈杂的情况下，由于外部因素导致航向突变或速度变化。为了解决这个局限性，我们提出了一种基于物理信息神经网络（PINN）的轨迹预测方法，该方法通过一阶和二阶有限差分物理损失函数将简化的船舶运动学模型整合到神经网络训练过程中。该损失函数使用一阶前向欧拉方法、Heun二阶近似进行离散化，并基于泰勒级数展开的中点近似进行细化，通过惩罚与预期运动学行为的偏差来保证对基本物理原理的忠实度。我们使用涵盖不同海洋条件的真实AIS数据集评估了PINN，并将其与最先进的模型进行了比较。我们的结果表明，所提出的方法在模型和数据集上的平均位移误差降低了高达32%，同时保持了物理一致性。这些结果提高了模型可靠性，并增强了对关键任务海洋活动的遵守，其中精度转化为更好的海洋态势感知。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12029) | **Categories:** cs.LG, cs.AI

---

### [3] [Explaining Recovery Trajectories of Older Adults Post Lower-Limb Fracture Using Modality-wise Multiview Clustering and Large Language Models](https://arxiv.org/abs/2506.12156)
*Shehroz S. Khan, Ali Abedi, Charlene H. Chu*

Main category: cs.LG

TL;DR: 该论文提出了一种使用大型语言模型解释聚类传感器数据的方法，以识别有风险的患者并改善健康结果。


<details>
  <summary>Details</summary>
Motivation: 在各个领域中，以人类能够理解的方式解释大量高维、未标记的数据仍然是一个重大挑战。在无监督的医疗保健数据分析中，解释聚类数据可以为患者的健康结果提供有意义的见解，这对医疗保健提供者具有直接的影响。

Method: 该研究采用上下文感知的提示，利用大型语言模型为从每种模态导出的聚类推断有意义的聚类标签。

Result: 结果表明，大型语言模型生成的大部分特定模态聚类标签在临床评分方面具有统计学意义，证实了所提出的方法在以无监督方式解释传感器数据方面的有效性。

Conclusion: 该研究证实了所提出的方法在以无监督方式解释传感器数据方面的有效性，通过大型语言模型生成的大部分特定模态聚类标签在临床评分方面具有统计学意义。

Abstract: 在各个领域中，以人类能够理解的方式解释大量高维、未标记的数据仍然是一个重大挑战。在无监督的医疗保健数据分析中，解释聚类数据可以为患者的健康结果提供有意义的见解，这对医疗保健提供者具有直接的影响。本文旨在解决解释从社区中下肢骨折恢复的老年患者收集的聚类传感器数据的问题。总共从患者家中远程收集了560天的多模态传感器数据，包括加速度、步数、环境运动、GPS位置、心率和睡眠，以及临床评分。首先对每种数据模态分别进行聚类，以评估从每种模态中提取的特征集对患者恢复轨迹的影响。然后，使用上下文感知的提示，使用大型语言模型来推断从每种模态导出的聚类的有意义的聚类标签。通过严格的统计测试和针对与多模态传感器数据一起收集的临床评分进行可视化，验证了这些聚类及其相应标签的质量。结果表明，大型语言模型生成的大部分特定模态聚类标签在临床评分方面具有统计学意义，证实了所提出的方法在以无监督方式解释传感器数据方面的有效性。这种仅依赖于传感器数据的无监督数据分析方法使临床医生能够识别出有风险的患者，并采取及时措施来改善健康结果。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12156) | **Categories:** cs.LG, cs.AI, cs.CV

---

### [4] [Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition](https://arxiv.org/abs/2506.12953)
*Mayank Bumb, Anshul Vemulapalli, Sri Harsha Vardhan Prasad Jella, Anish Gupta, An La, Ryan A. Rossi, Hongjie Chen, Franck Dernoncourt, Nesreen K. Ahmed, Yu Wang*

Main category: cs.LG

TL;DR: 本文提出了一种名为PatchInstruct的简单而灵活的基于提示的策略，它使LLM能够在不进行大量重新训练的情况下执行时间序列预测。


<details>
  <summary>Details</summary>
Motivation: 先前的工作通常需要大量的微调，并且/或者忽略了序列间的相关性。

Method: PatchInstruct

Result: LLM可以做出精确而有效的预测。

Conclusion: 通过专门的提示方法，利用时间序列分解、基于补丁的标记化和基于相似性的邻居增强，可以在保持简单性的同时提高LLM预测质量，并且只需要最少的数据预处理。

Abstract: 大型语言模型（LLM）的最新进展展示了精确和高效的时间序列分析的新可能性，但先前的工作通常需要大量的微调，并且/或者忽略了序列间的相关性。在这项工作中，我们探索了简单而灵活的基于提示的策略，使LLM能够执行时间序列预测，而无需进行大量的重新训练或使用复杂的外部架构。通过探索利用时间序列分解、基于补丁的标记化和基于相似性的邻居增强的专门提示方法，我们发现可以在保持简单性的同时提高LLM预测质量，并且只需要最少的数据预处理。为此，我们提出了我们自己的方法PatchInstruct，该方法使LLM能够做出精确而有效的预测。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12953) | **Categories:** cs.LG, cs.AI, cs.CL

---

### [5] [How to Train a Model on a Cheap Cluster with Low Cost using Block Coordinate Descent](https://arxiv.org/abs/2506.12037)
*Zeyu Liu, Yunquan Zhang, Boyang Zhang, Guoyong Jiang, Daning Cheng*

Main category: cs.LG

TL;DR: 该论文提出了一种基于块坐标下降(BCD)的全参数预训练框架，旨在降低大型语言模型的训练成本，使其能够在更经济的硬件上进行训练，同时保持模型精度。


<details>
  <summary>Details</summary>
Motivation: 训练大型语言模型通常需要大量的GPU内存和大量的资金投入，这对许多中小型团队构成了障碍。

Method: 提出了一种基于块坐标下降(BCD)的全参数预训练框架，并辅以工程优化。

Result: 实验表明：1)相同设备成本更低：BCD显著降低了预训练成本。对于7B模型，在相同的硬件设置下，与传统的全参数训练相比，在A100,A800集群上，BCD的训练成本平均降低了约33%，在RTX 4090集群上，BCD的训练成本平均降低了约2.6%。2)跨设备迁移：通过利用BCD，以前只能在高端A100集群上训练的大规模模型可以无缝迁移，并在4090集群上进行预训练，而无需昂贵的硬件，4090集群的每小时成本仅为A100的四分之一。3)精度保持：在两种情况下，BCD训练都能达到与全参数预训练相同的模型精度。

Conclusion: BCD训练在两种情况下都能达到与全参数预训练相同的模型精度。

Abstract: 训练大型语言模型通常需要大量的GPU内存和大量的资金投入，这对许多中小型团队构成了障碍。在本文中，我们提出了一个基于块坐标下降(BCD)的全参数预训练框架，并辅以工程优化，以在经济实惠的RTX 4090 GPU集群上高效地训练大型模型。BCD基于块坐标下降理论确保模型收敛，并在参数块级别执行梯度计算和更新。实验表明：1)相同设备成本更低：BCD显著降低了预训练成本。对于7B模型，在相同的硬件设置下，与传统的全参数训练相比，在A100,A800集群上，BCD的训练成本平均降低了约33%，在RTX 4090集群上，BCD的训练成本平均降低了约2.6%。2)跨设备迁移：通过利用BCD，以前只能在高端A100集群上训练的大规模模型可以无缝迁移，并在4090集群上进行预训练，而无需昂贵的硬件，4090集群的每小时成本仅为A100的四分之一。3)精度保持：在两种情况下，BCD训练都能达到与全参数预训练相同的模型精度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12037) | **Categories:** cs.LG, cs.AI

---

### [6] [Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation](https://arxiv.org/abs/2506.12213)
*Zikai Zhang, Ping Liu, Jiahao Xu, Rui Hu*

Main category: cs.LG

TL;DR: Fed-HeLLo提出了一种新颖的联邦LoRA微调框架，通过异构LoRA分配策略，使客户端能够根据自身资源和层重要性协同微调基础模型。


<details>
  <summary>Details</summary>
Motivation: 现有方法大多没有考虑客户端的异构资源，或者缺乏有效的本地训练策略，以在有限的资源下最大化全局微调性能。

Method: 提出了一种新的基于联邦LoRA的微调框架Fed-HeLLo，该框架允许客户端使用不同的本地可训练LoRA层协同微调FM。为了确保其有效性，我们开发了几种异构LoRA分配（HLA）策略，这些策略基于客户端的资源能力和层重要性自适应地分配本地可训练LoRA层。

Result: 在五个数据集上，在从IID到极端Non-IID的三种数据分布级别下，评估了我们的方法。结果表明，带有HLA策略的Fed-HeLLo既有效又高效。

Conclusion: 实验结果表明，Fed-HeLLo与HLA策略在不同联邦LoRA微调设置下均有效且高效。

Abstract: 联邦学习最近被用于跨多个客户端协同微调基础模型。值得注意的是，基于联邦低秩适配LoRA的微调方法最近受到了关注，它允许客户端在本地使用一小部分可训练参数来微调FM。然而，现有方法大多没有考虑客户端的异构资源，或者缺乏有效的本地训练策略，以在有限的资源下最大化全局微调性能。在这项工作中，我们提出了一种新的基于联邦LoRA的微调框架Fed-HeLLo，该框架允许客户端使用不同的本地可训练LoRA层协同微调FM。为了确保其有效性，我们开发了几种异构LoRA分配（HLA）策略，这些策略基于客户端的资源能力和层重要性自适应地分配本地可训练LoRA层。具体来说，基于动态层重要性，我们设计了一种基于Fisher信息矩阵评分的HLA，它利用动态梯度范数信息。为了更好地稳定训练过程，我们考虑了LoRA层的内在重要性，并设计了一种几何定义的HLA策略。它将可训练LoRA层的集体分布塑造成特定的几何模式，如三角形、倒三角形、瓶颈和均匀。此外，我们将GD-HLA扩展到随机版本，名为随机几何定义HLA，以通过随机性增强模型精度。通过共同设计所提出的HLA策略，我们将动态和内在层重要性都纳入了我们的HLA策略的设计中。我们在五个数据集上，在从IID到极端Non-IID的三种数据分布级别下，评估了我们的方法。结果表明，带有HLA策略的Fed-HeLLo既有效又高效。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12213) | **Categories:** cs.LG, cs.DC

---

### [7] [TrojanTO: Action-Level Backdoor Attacks against Trajectory Optimization Models](https://arxiv.org/abs/2506.12815)
*Yang Dai, Oubo Ma, Longfei Zhang, Xingxing Liang, Xiaochun Cao, Shouling Ji, Jiaheng Zhang, Jincai Huang, Li Shen*

Main category: cs.LG

TL;DR: TrojanTO是一种针对轨迹优化模型的动作级后门攻击，通过交替训练和精确中毒来有效植入后门，且隐蔽性高，适用性广。


<details>
  <summary>Details</summary>
Motivation: 现有的强化学习后门攻击主要基于奖励操纵，对TO模型无效，且高维动作空间增加了动作操纵的复杂性。

Method: 提出TrojanTO，一种针对TO模型的动作级后门攻击，采用交替训练增强触发器和目标动作的连接，利用轨迹过滤进行精确中毒以保证正常性能，并使用批量中毒来保证触发器一致性。

Result: TrojanTO在不同任务和攻击目标下均能有效植入后门攻击，且攻击预算低（0.3%的轨迹），并广泛适用于DT、GDT和DC。

Conclusion: TrojanTO攻击在不同任务和目标下均有效，且攻击预算低，适用性广。

Abstract: 轨迹优化（TO）模型的最新进展在离线强化学习中取得了显著成功。然而，它们对后门攻击的脆弱性却鲜为人知。我们发现，现有的强化学习后门攻击主要基于奖励操纵，由于TO模型固有的序列建模特性，这些攻击对TO模型基本无效。此外，高维动作空间带来的复杂性进一步加剧了动作操纵的挑战。为了解决这些问题，我们提出了TrojanTO，这是第一个针对TO模型的动作级后门攻击。TrojanTO采用交替训练来增强触发器和目标动作之间的连接，从而提高攻击的有效性。为了提高攻击的隐蔽性，它利用轨迹过滤进行精确中毒，以保证正常性能，并使用批量中毒来保证触发器的一致性。广泛的评估表明，TrojanTO在不同的任务和攻击目标下都能有效地植入后门攻击，且攻击预算较低（0.3%的轨迹）。此外，TrojanTO还表现出对DT、GDT和DC的广泛适用性，突显了其在不同TO模型架构中的可扩展性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12815) | **Categories:** cs.LG

---

### [8] [TimeMaster: Training Time-Series Multimodal LLMs to Reason via Reinforcement Learning](https://arxiv.org/abs/2506.13705)
*Junru Zhang, Lang Feng, Xu Guo, Yuhan Wu, Yabo Dong, Duanqing Xu*

Main category: cs.LG

TL;DR: TimeMaster是一种基于强化学习的时间序列多模态大语言模型，它在时间序列分类任务上优于现有模型，并展现出专家级的推理能力。


<details>
  <summary>Details</summary>
Motivation: 多模态大语言模型（MLLM）在时间序列推理方面面临挑战，包括动态的时间模式、模糊的语义和缺乏时间先验知识。

Method: 提出了基于强化学习（RL）的TimeMaster方法，该方法采用三部分结构化输出格式（推理、分类和领域特定扩展），并通过复合奖励函数进行优化，同时使用两阶段训练流程：监督微调（SFT）和群体相对策略优化（GRPO）。

Result: TimeMaster在TimerBed基准测试中，在六个真实世界分类任务上，超越了传统的时间序列模型和少样本GPT-4o，分别取得了超过14.6%和7.3%的性能提升。

Conclusion: 基于强化学习的时间序列多模态大语言模型TimeMaster在时间序列分类任务上取得了显著的性能提升，并展现出专家级的推理能力和领域知识。

Abstract: 本文介绍了一种名为TimeMaster的基于强化学习（RL）的方法，该方法使时间序列多模态大语言模型能够直接对可视化的时间序列输入和任务提示执行结构化、可解释的推理。TimeMaster采用三部分结构化输出格式：推理、分类和领域特定扩展，并通过复合奖励函数进行优化，该函数符合格式要求、预测准确性和开放式洞察质量。该模型使用两阶段流程进行训练：首先应用监督微调（SFT）以建立良好的初始化，然后在token级别应用群体相对策略优化（GRPO），以实现时间序列推理中稳定和有针对性的奖励驱动改进。我们在TimerBed基准测试中，基于Qwen2.5-VL-3B-Instruct，在六个真实世界分类任务上评估了TimeMaster。TimeMaster实现了最先进的性能，超越了传统的时间序列模型和少样本GPT-4o，分别取得了超过14.6%和7.3%的性能提升。值得注意的是，TimeMaster超越了时间序列分类：它还表现出专家般的推理行为，生成上下文感知的解释，并提供与领域对齐的见解。我们的结果表明，奖励驱动的强化学习可能是将时间理解集成到时间序列多模态大语言模型中的一种可扩展且有希望的途径。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13705) | **Categories:** cs.LG, cs.AI

---


## cs.MA [cs.MA]
### [1] [Trust-MARL: Trust-Based Multi-Agent Reinforcement Learning Framework for Cooperative On-Ramp Merging Control in Heterogeneous Traffic Flow](https://arxiv.org/abs/2506.12600)
*Jie Pan, Tianyi Wang, Christian Claudel, Jing Shi*

Main category: cs.MA

TL;DR: 提出了一种基于信任的多智能体强化学习框架(Trust-MARL)，用于解决异构交通环境中的协同匝道合并问题，提高了交通效率和安全性。


<details>
  <summary>Details</summary>
Motivation: 为了解决异构交通环境中协同匝道合并的挑战。

Method: 提出了一种基于信任的多智能体强化学习(Trust-MARL)框架。

Result: 大量的消融研究和对比实验验证了所提出的Trust-MARL方法的有效性。

Conclusion: 提出的Trust-MARL方法在不同CAV渗透率和交通密度下，在安全性、效率、舒适性和适应性方面都表现出显著的改进。

Abstract: 智能交通系统要求联网和自动驾驶车辆(CAV)在复杂的现实交通环境中与人工驾驶车辆(HV)进行安全高效的合作。然而，人类行为固有的不可预测性，尤其是在高速公路匝道合并区域等瓶颈处，经常扰乱交通流量并损害系统性能。为了解决异构交通环境中协同匝道合并的挑战，本研究提出了一种基于信任的多智能体强化学习(Trust-MARL)框架。在宏观层面，Trust-MARL通过利用智能体间的信任来提高瓶颈吞吐量，并通过涌现的群体层面协调来缓解交通冲击波，从而提高全局交通效率。在微观层面，设计了一种动态信任机制，使CAV能够根据与HV和其他CAV的实时行为和历史互动来调整其合作策略。此外，还集成了一个信任触发的博弈论决策模块，以指导每个CAV调整其合作因子，并在安全性、舒适性和效率约束下执行上下文感知的变道决策。大量的消融研究和对比实验验证了所提出的Trust-MARL方法的有效性，证明了在不同的CAV渗透率和交通密度下，在安全性、效率、舒适性和适应性方面都有显著的改进。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12600) | **Categories:** cs.MA, cs.AI, cs.ET, cs.GT, cs.RO

---

### [2] [Modeling Earth-Scale Human-Like Societies with One Billion Agents](https://arxiv.org/abs/2506.12078)
*Haoxiang Guan, Jiyan He, Liyang Fan, Zhenzhen Ren, Shaobin He, Xin Yu, Yuan Chen, Shuxin Zheng, Tie-Yan Liu, Zhen Liu*

Main category: cs.MA

TL;DR: Light Society是一个基于LLM的智能体模拟框架，可以在行星尺度上高效建模类人社会。


<details>
  <summary>Details</summary>
Motivation: 传统的基于智能体的模型（ABM）已被用于研究这些动态数十年，但受到简化智能体行为的限制，无法捕捉人类的复杂性。大型语言模型（LLM）的最新进展通过使智能体能够表现出超越基于规则的逻辑的复杂社会行为，提供了新的机会，但面临着巨大的扩展挑战。

Method: 形式化社会过程为智能体和环境状态的结构化转换，由一组LLM驱动的模拟操作管理，并通过事件队列执行。

Result: 对信任博弈和观点传播的大规模模拟（跨越高达10亿个智能体）证明了Light Society在建模社会信任和信息传播方面的高保真度和效率，同时揭示了更大的模拟产生更稳定和真实的涌现行为的缩放规律。

Conclusion: 更大规模的模拟能够产生更稳定和真实的涌现行为。

Abstract: 理解复杂的社会行为如何从个体认知和互动中产生，需要对人类行为进行高保真建模和大规模模拟。传统的基于智能体的模型（ABM）已被用于研究这些动态数十年，但受到简化智能体行为的限制，无法捕捉人类的复杂性。大型语言模型（LLM）的最新进展通过使智能体能够表现出超越基于规则的逻辑的复杂社会行为，提供了新的机会，但面临着巨大的扩展挑战。在这里，我们提出了Light Society，这是一个基于智能体的模拟框架，它在两个方面都取得了进展，有效地模拟了由LLM驱动的行星规模类人社会。Light Society将社会过程形式化为智能体和环境状态的结构化转换，由一组LLM驱动的模拟操作管理，并通过事件队列执行。这种模块化设计支持独立和联合组件优化，支持对超过10亿个智能体的社会进行有效模拟。对信任博弈和观点传播的大规模模拟（跨越高达10亿个智能体）证明了Light Society在建模社会信任和信息传播方面的高保真度和效率，同时揭示了更大的模拟产生更稳定和真实的涌现行为的缩放规律。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12078) | **Categories:** cs.MA, cs.AI, cs.CL, cs.CY, cs.SI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins](https://arxiv.org/abs/2506.13761)
*Chuanruo Ning, Kuan Fang, Wei-Chiu Ma*

Main category: cs.RO

TL;DR: 提出了一种结合视觉语言模型和数字孪生的模型预测控制框架，提升了开放世界机器人操作的性能。


<details>
  <summary>Details</summary>
Motivation: 现有视觉语言模型在开放世界机器人操作中，虽然具有强大的高层规划能力，但由于对物理世界的理解有限，难以预测底层机器人控制。

Method: 提出了一种结合视觉语言模型和物理交互数字孪生的模型预测控制框架。

Result: 在各种复杂的操作任务中，该方法表现出优于其他基于视觉语言模型的语言条件机器人控制方法的性能。

Conclusion: 该研究验证了结合视觉语言模型和物理交互数字孪生的模型预测控制框架在开放世界操作任务中的优越性能。

Abstract: 在开放世界机器人操作领域的最新进展主要由视觉语言模型（VLMs）驱动。尽管这些模型在高层规划中表现出强大的泛化能力，但由于对物理世界的理解有限，它们难以预测低层机器人控制。为了解决这个问题，我们提出了一种用于开放世界操作的模型预测控制框架，该框架结合了视觉语言模型的语义推理能力与物理基础的、真实世界环境的交互式数字孪生。通过构建和模拟数字孪生，我们的方法生成可行的运动轨迹，模拟相应的结果，并提示视觉语言模型提供未来的观察结果，以评估和选择基于任务的语言指令的最合适结果。为了进一步增强预训练视觉语言模型在理解机器人控制复杂场景方面的能力，我们利用数字孪生的灵活渲染能力，在各种新颖的、无遮挡的视点合成场景。我们在各种复杂的操纵任务上验证了我们的方法，证明了与使用视觉语言模型的语言条件机器人控制的基线方法相比，该方法具有优越的性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13761) | **Categories:** cs.RO

---

### [2] [ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration](https://arxiv.org/abs/2506.12248)
*Jennifer Grannen, Siddharth Karamcheti, Blake Wulfe, Dorsa Sadigh*

Main category: cs.RO

TL;DR: ProVox利用元提示和主动语言模型，使机器人能够个性化地适应人类合作者，从而提高人机协作效率。


<details>
  <summary>Details</summary>
Motivation: 协作机器人必须快速适应伙伴的意图和偏好，以主动识别有用的操作。尤其是在情境设置中，人类伙伴可以不断地教机器人新的高级行为、视觉概念和物理技能，从而在人机协作完成各种任务时，扩展机器人的能力。

Method: 提出了一种名为ProVox的新框架，该框架利用大型语言模型的常识先验和可操纵性，通过元提示协议使用户能够传达其偏好、意图和期望的机器人行为，从而实现机器人对个体合作者的个性化和适应。

Result: 用户研究表明，ProVox在家庭操作任务中表现出更高的协作效率，用户感知到的帮助性、易用性和可靠性也更高。与非主动基线相比，任务完成时间缩短了38.7%，用户负担减少了31.9%。

Conclusion: Meta-prompting和主动性对于提升人机协作效率至关重要，实验结果表明，相比于非主动基线方法，ProVox能使任务完成时间缩短38.7%，用户负担减少31.9%。

Abstract: 协作机器人必须快速适应伙伴的意图和偏好，从而主动识别有用的操作。在人机协作完成各种任务的过程中，人类伙伴可以不断教机器人新的行为、概念和技能，以此扩展机器人的能力。本文提出，机器人应该能够从早期互动中推断出伙伴的目标，并利用这些信息主动规划行为，而无需用户的明确指示。我们利用大型语言模型的先验知识，引入了一种名为ProVox的新框架，该框架使机器人能够有效地个性化和适应个体合作者。我们设计了一个元提示协议，使用户能够在开始物理交互之前传达其偏好、意图和期望的机器人行为。然后，ProVox使用个性化的提示来调节主动语言模型任务规划器，该规划器可以根据当前交互上下文和机器人能力预测用户的意图，从而建议有用的操作；通过这样做，我们可以减轻用户的负担，最大限度地减少伙伴花费在明确指示和监督机器人上的时间。我们通过以家庭操作任务（例如，组装午餐袋）为基础的用户研究来评估ProVox，这些研究衡量了协作的效率，以及诸如感知到的帮助性、易用性和可靠性等特征。我们的分析表明，元提示和主动性都至关重要，与非主动基线相比，任务完成时间缩短了38.7%，用户负担减少了31.9%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12248) | **Categories:** cs.RO, cs.AI, cs.CL, cs.HC, cs.LG

---

### [3] [AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](https://arxiv.org/abs/2506.12374)
*Wenbo Li, Shiyi Wang, Yiteng Chen, Huiping Zhuang, Qingyao Wu*

Main category: cs.RO

TL;DR: AntiGrounding 框架通过将候选动作提升到 VLM 表示空间并使用视觉问答，实现了机器人操作任务的零样本合成，并在性能上优于现有方法。


<details>
  <summary>Details</summary>
Motivation: 视觉语言模型 (VLM) 在高维表示空间中对机器人操作的知识和推理能力进行编码。然而，当前的方法通常将它们投影到压缩的中间表示中，从而丢弃了重要的任务特定信息，例如细粒度的空间或语义细节。

Method: 我们提出了 AntiGrounding，这是一个新框架，它可以反转指令 grounding 过程。它将候选动作直接提升到 VLM 表示空间，从多个视图渲染轨迹，并使用结构化视觉问答进行基于指令的决策。

Result: 在模拟和真实环境中的实验表明，我们的方法在各种机器人操作任务中优于基线。

Conclusion: 在模拟和真实环境中的实验表明，我们的方法在各种机器人操作任务中优于基线。

Abstract: 视觉语言模型 (VLM) 在高维表示空间中对机器人操作的知识和推理能力进行编码。然而，当前的方法通常将它们投影到压缩的中间表示中，从而丢弃了重要的任务特定信息，例如细粒度的空间或语义细节。为了解决这个问题，我们提出了 AntiGrounding，这是一个新框架，它可以反转指令 grounding 过程。它将候选动作直接提升到 VLM 表示空间，从多个视图渲染轨迹，并使用结构化视觉问答进行基于指令的决策。这使得能够零样本合成新任务的最佳闭环机器人轨迹。我们还提出了一个离线策略细化模块，该模块利用过去的经验来提高长期性能。在模拟和真实环境中的实验表明，我们的方法在各种机器人操作任务中优于基线。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12374) | **Categories:** cs.RO, cs.AI, I.2.9; I.2.10; I.4.8; H.5.2

---

### [4] [A Spatial Relationship Aware Dataset for Robotics](https://arxiv.org/abs/2506.12525)
*Peng Wang, Minh Huy Pham, Zhihao Guo, Wei Zhou*

Main category: cs.RO

TL;DR: 本文提出了一个用于机器人空间推理的空间关系感知数据集，并证明了将空间关系整合到基础模型中可以显著提高机器人规划能力。


<details>
  <summary>Details</summary>
Motivation: 现实世界环境中，机器人任务规划不仅需要物体识别，还需要对物体之间的空间关系有细致的理解。

Method: 作者提出了一个空间关系感知的数据集，并使用Boston Dynamics Spot机器人捕获图像，使用自定义注释工具进行标注。

Result: 在作者构建的数据集上，对六个最先进的场景图生成模型进行了基准测试，分析了它们的推理速度和关系准确性。结果表明，模型性能存在显著差异。

Conclusion: 将显式的空间关系整合到诸如ChatGPT 4o的基础模型中，能够显著提高它们生成可执行的、具有空间感知能力的机器人规划的能力。

Abstract: 本文提出了一个空间关系感知的数据集，该数据集包含近1000张机器人采集的室内图像，并标注了物体属性、位置和详细的空间关系。该数据集使用Boston Dynamics Spot机器人捕获，并使用自定义注释工具进行标注，反映了具有相似或相同物体以及复杂空间排列的复杂场景。作者在本文构建的数据集上，对六个最先进的场景图生成模型进行了基准测试，分析了它们的推理速度和关系准确性。结果表明，将显式的空间关系整合到诸如ChatGPT 4o的基础模型中，能够显著提高它们生成可执行的、具有空间感知能力的机器人规划的能力。数据集和注释工具可在https://github.com/PengPaulWang/SpatialAwareRobotDataset公开获取，支持机器人空间推理的进一步研究。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12525) | **Categories:** cs.RO

---

### [5] [Multimodal Large Language Models-Enabled UAV Swarm: Towards Efficient and Intelligent Autonomous Aerial Systems](https://arxiv.org/abs/2506.12710)
*Yuqi Ping, Tianhao Liang, Huahao Ding, Guangyu Lei, Junwei Wu, Xuan Zou, Kuan Shi, Rui Shao, Chiya Zhang, Weizheng Zhang, Weijie Yuan, Tingting Zhang*

Main category: cs.RO

TL;DR: 本文探讨了将多模态大型语言模型（MLLM）集成到无人机集群中，以增强其在各种任务中的智能和适应性的解决方案。


<details>
  <summary>Details</summary>
Motivation: Unmanned Aerial Vehicle (UAV) swarms are increasingly deployed in dynamic, safety-critical missions that demand rapid situational understanding and autonomous adaptation.

Method: Integrating MLLMs with UAV swarms to enhance the intelligence and adaptability across diverse tasks.

Result: Human-machine interaction, swarm task planning, fire assessment, and task execution are investigated.

Conclusion: MLLMs-enabled UAV swarm can enhance the intelligence and adaptability across diverse tasks.

Abstract: 多模态大型语言模型（MLLM）的最新突破使人工智能系统能够在文本、图像和视频流中进行统一的感知、推理和自然语言交互。同时，无人机（UAV）集群越来越多地部署在动态的、对安全性至关重要的任务中，这些任务需要快速的态势理解和自主适应。本文探讨了将 MLLM 与无人机集群集成以增强各种任务中的智能和适应性的潜在解决方案。具体来说，我们首先概述了无人机和 MLLM 的基本架构和功能。然后，我们分析了 MLLM 如何在目标检测、自主导航和多智能体协作方面提高无人机系统的性能，同时探索将 MLLM 集成到无人机系统中的解决方案。接下来，我们提出了一个以森林消防为重点的实际案例研究。为了充分展示所提出框架的能力，我们研究了人机交互、集群任务规划、火灾评估和任务执行。最后，我们讨论了 MLLM 支持的无人机集群所面临的挑战和未来的研究方向。实验演示视频可在 https://youtu.be/zwnB9ZSa5A4 在线观看。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12710) | **Categories:** cs.RO

---

### [6] [Physics-informed Neural Motion Planning via Domain Decomposition in Large Environments](https://arxiv.org/abs/2506.12742)
*Yuchen Liu, Alexiy Buynitsky, Ruiqi Ni, Ahmed H. Qureshi*

Main category: cs.RO

TL;DR: 提出了一种新的神经场表示方法FB-NTFields，用于可扩展的cost-to-go估计，实现了高效的大规模运动规划。


<details>
  <summary>Details</summary>
Motivation: 现有的基于物理信息的神经运动规划器(PiNMPs)在可扩展性方面受到谱偏置和PDE驱动训练的复杂损失景观的限制。现有的域分解方法仅在单个空间点上强制连续性，无法捕捉运动规划所需的空间连通性。

Method: 提出了一种新的神经场表示方法FB-NTFields，通过计算起始坐标和目标坐标的潜在嵌入之间的距离来估计cost-to-go，从而实现全局空间连贯性。

Result: FB-NTFields在复杂合成和真实场景中表现出比现有PiNMPs的显著改进。

Conclusion: FB-NTFields在复杂环境和真实场景中验证有效，并在Unitree B1四足机器人上成功部署，实现了室内环境导航。

Abstract: 基于物理信息的神经运动规划器(PiNMPs)为求解Eikonal偏微分方程(PDE)和表示运动规划的cost-to-go函数提供了一个数据高效的框架。然而，它们的可扩展性仍然受到谱偏置和PDE驱动训练的复杂损失景观的限制。域分解可以缓解这些问题，通过将环境划分为更小的子域，但现有方法仅在单个空间点上强制连续性。虽然这些方法对于函数逼近有效，但它们无法捕捉运动规划所需的空间连通性，其中cost-to-go函数取决于起始和目标坐标，而不是单个查询点。我们提出了一种新的神经场表示方法，称为有限基神经时间场(FB-NTFields)，用于可扩展的cost-to-go估计。FB-NTFields不是在输出空间中强制连续性，而是构建一个潜在空间表示，将cost-to-go计算为起始坐标和目标坐标的潜在嵌入之间的距离。这实现了全局空间连贯性，同时集成了域分解，确保了高效的大规模运动规划。我们在复杂的合成和真实场景中验证了FB-NTFields，证明了其相对于现有PiNMPs的显著改进。最后，我们将我们的方法部署在Unitree B1四足机器人上，成功地导航了室内环境。补充视频可在https://youtu.be/OpRuCbLNOwM找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12742) | **Categories:** cs.RO

---

### [7] [Autonomous 3D Moving Target Encirclement and Interception with Range measurement](https://arxiv.org/abs/2506.13106)
*Fen Liu, Shenghai Yuan, Thien-Minh Nguyen, Rong Su*

Main category: cs.RO

TL;DR: 提出了一种自主无人机3D包围拦截策略，用于对抗携带危险物品或扰乱空中交通的商用无人机威胁。


<details>
  <summary>Details</summary>
Motivation: 商用无人机正成为一种新兴的安全威胁，因为它们能够携带危险的有效载荷或扰乱空中交通。为了对抗无人机，我们引入了一种自主的3D目标包围和拦截策略。

Method: 提出了一种基于反同步(AS)和X-Y圆形运动结合垂直抖动的观测和速度补偿方法，利用无人机测量的两个噪声实时距离来估计相对位置，并提出了一种包围控制机制，使无人机能够自适应地从包围和保护目标过渡到包围和监视敌对目标。

Result: 通过真实世界的无人机实验和MATLAB的模拟分析，验证了该策略在探测、包围和拦截敌对无人机方面的有效性。

Conclusion: 通过真实无人机实验和MATLAB模拟分析验证了该策略在检测、包围和拦截敌对无人机方面的有效性。

Abstract: 商用无人机正成为一种新兴的安全威胁，因为它们能够携带危险的有效载荷或扰乱空中交通。为了对抗无人机，我们介绍了一种自主的3D目标包围和拦截策略。与传统的地面引导系统不同，该策略采用自主无人机来跟踪和攻击非合作的敌对无人机，该策略在非视距条件下、GPS干扰和雷达干扰下有效，而在地面引导下的传统探测和中和方法则会失效。利用无人机测量的两个噪声实时距离，守护无人机使用基于反同步(AS)和X-Y圆形运动结合垂直抖动的观测和速度补偿方法，估计从自身到目标的相对位置。提出了一种包围控制机制，使无人机能够自适应地从包围和保护目标过渡到包围和监视敌对目标。一旦突破警告阈值，无人机甚至可以采取自杀式攻击来摧毁敌对目标。我们通过真实世界的无人机实验和MATLAB的模拟分析验证了这一策略，证明了其在探测、包围和拦截敌对无人机方面的有效性。更多细节请访问：https://youtu.be/5eHW56lPVto。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13106) | **Categories:** cs.RO, eess.SP

---

### [8] [C2TE: Coordinated Constrained Task Execution Design for Ordering-Flexible Multi-Vehicle Platoon Merging](https://arxiv.org/abs/2506.13202)
*Bin-Bin Hu, Yanxin Zhou, Henglai Wei, Shuo Cheng, Chen Lv*

Main category: cs.RO

TL;DR: 提出了一种分布式协调约束任务执行算法，实现了异车道车辆灵活排序的协同合并。


<details>
  <summary>Details</summary>
Motivation: 解决异车道车辆协同合并为期望车道上的排序灵活编队的问题。

Method: 提出了一种分布式协调约束任务执行（C2TE）算法，将多车编队合并任务分为预合并调节和排序灵活的编队合并两个阶段，并将其转化为分布式约束优化问题。

Result: 所提算法能够安全地扩大相邻车辆之间的纵向距离，并有效地实现排序灵活的编队。

Conclusion: 通过实验和仿真验证了所提算法在多种场景下的有效性、灵活性、鲁棒性和可扩展性。

Abstract: 本文提出了一种分布式协调约束任务执行（C2TE）算法，该算法使来自不同车道的车辆团队能够协同合并到期望车道上进行机动的“排序灵活编队”中。其中，编队是灵活的，因为没有预先确定车辆的特定空间排序序列。为了获得这种灵活的编队，我们首先将多车编队（MVP）合并任务分为两个阶段，即预合并调节和“排序灵活编队”合并，然后将它们转化为基于分布式约束的优化问题。特别是，通过将纵向距离调节和同车道避撞子任务编码到相应的控制障碍函数（CBF）约束中，所提出的第一阶段算法可以安全地扩大相邻车辆之间的纵向距离。然后，通过将横向收敛、纵向目标吸引和相邻避撞子任务编码到CBF约束中，所提出的第二阶段算法可以有效地实现“排序灵活编队”。请注意，“排序灵活编队”是通过纵向目标吸引和时变相邻避撞约束同时相互作用来实现的。在灵活排序引起的强非线性耦合下，提供了可行性保证和严格的收敛性分析。最后，使用三个自主移动车辆（AMV）进行的实验验证了所提算法的有效性和灵活性，并进行了广泛的仿真，以证明其在处理车辆突然故障、新出现、不同车道数量、混合自主和大规模场景时的鲁棒性、适应性和可扩展性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13202) | **Categories:** cs.RO

---

### [9] [VLM-SFD: VLM-Assisted Siamese Flow Diffusion Framework for Dual-Arm Cooperative Manipulation](https://arxiv.org/abs/2506.13428)
*Jiaming Chen, Yiyu Jiang, Aoshen Huang, Yang Li, Wei Pan*

Main category: cs.RO

TL;DR: 该论文提出了一种VLM辅助的Siamese Flow Diffusion框架，用于提升双臂协同操作中模仿学习的效率和泛化能力。


<details>
  <summary>Details</summary>
Motivation: 尽管在基于学习的运动规划方面取得了显著进展，但大多数方法难以推广到不同的操作任务，并且难以适应动态的、非结构化的环境，尤其是在涉及两个对象之间交互的场景中，例如组装、工具使用和双手抓取。

Method: 该论文提出了一种新颖的VLM辅助的Siamese Flow Diffusion (VLM-SFD)框架，用于双臂协同操作中的高效模仿学习。该框架采用Siamese Flow Diffusion Network (SFDNet)，利用双编码器-解码器Siamese架构将两个目标对象嵌入到共享的潜在空间中，同时，基于扩散的条件过程（以任务指令为条件）生成双流以对象为中心的运动流，从而指导双臂协调。此外，还设计了一种动态任务分配策略，可将预测的2D运动流无缝映射到3D空间，并结合预训练的视觉语言模型（VLM）来动态地为每个机械臂分配最佳运动。

Result: 所提出的VLM-SFD框架表现出出色的适应性，显著增强了从最少数量的人工演示中快速适应和推广到各种实际任务的能力。

Conclusion: 实验验证了该方法的有效性，证明了其在保持高效率和适应性的同时，推广到各种操作任务的能力。

Abstract: 双臂协同操作在解决需要无缝协调和自适应动力学的复杂现实世界任务方面具有广阔的前景。为了应对这些挑战，我们介绍了一种新颖的VLM辅助的Siamese Flow Diffusion (VLM-SFD)框架，用于双臂协同操作中的高效模仿学习。所提出的VLM-SFD框架表现出出色的适应性，显著增强了从最少数量的人工演示中快速适应和推广到各种实际任务的能力。具体来说，我们提出了一种Siamese Flow Diffusion Network (SFDNet)，它采用双编码器-解码器Siamese架构将两个目标对象嵌入到共享的潜在空间中，同时，基于扩散的条件过程（以任务指令为条件）生成双流以对象为中心的运动流，从而指导双臂协调。我们进一步设计了一种动态任务分配策略，可将预测的2D运动流无缝映射到3D空间，并结合预训练的视觉语言模型（VLM）来动态地为每个机械臂分配最佳运动。实验验证了该方法的有效性，证明了其在保持高效率和适应性的同时，推广到各种操作任务的能力。代码和演示视频可在我们的项目网站上公开获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13428) | **Categories:** cs.RO

---

### [10] [Disturbance-aware minimum-time planning strategies for motorsport vehicles with probabilistic safety certificates](https://arxiv.org/abs/2506.13622)
*Martino Gulisano, Matteo Masoni, Marco Gabiccini, Massimo Guiggiani*

Main category: cs.RO

TL;DR: 本文提出了一种考虑扰动的鲁棒轨迹优化框架，用于高性能赛车运动，该框架通过开环和闭环两种方法，在保证安全性的前提下，实现了更优的圈速表现。


<details>
  <summary>Details</summary>
Motivation: 本文提出了一个考虑扰动的框架，该框架将鲁棒性嵌入到赛车运动的最小圈速轨迹优化中。

Method: 提出了两种公式：（i）开环、基于范围的协方差传播，使用有限窗口内的最坏情况不确定性增长来收紧轮胎摩擦和赛道限制约束。（ii）闭环、协方差感知规划，在优化器中结合了时变LQR反馈律，提供了干扰衰减的反馈一致估计，并实现了更清晰但可靠的约束收紧。

Result: 在具有代表性的巴塞罗那-加泰罗尼亚赛道的计算测试表明，两种方案都满足规定的安全概率，但闭环方案比更保守的开环方案产生的圈速损失更小，而标称（非鲁棒）轨迹在相同的不确定性下仍然不可行。

Conclusion: 通过考虑规划中的不确定性增长和反馈作用，所提出的框架提供的轨迹在性能上是最优的，并且在概率上是安全的，从而推动了最短时间优化在高性能赛车和自主赛车中的实际应用。

Abstract: 本文提出了一种考虑扰动的框架，该框架将鲁棒性嵌入到赛车运动的最小圈速轨迹优化中。介绍了两种公式：（i）开环、基于范围的协方差传播，使用有限窗口内的最坏情况不确定性增长来收紧轮胎摩擦和赛道限制约束。（ii）闭环、协方差感知规划，在优化器中结合了时变LQR反馈律，提供了干扰衰减的反馈一致估计，并实现了更清晰但可靠的约束收紧。这两种方法都为人类或人工智能驾驶员提供了参考轨迹：在自主应用中，建模的控制器可以复制车载实现，而对于人类驾驶，精度会随着驾驶员可以被假设的时变LQR策略近似的程度而提高。在具有代表性的巴塞罗那-加泰罗尼亚赛道的计算测试表明，两种方案都满足规定的安全概率，但闭环方案比更保守的开环方案产生的圈速损失更小，而标称（非鲁棒）轨迹在相同的不确定性下仍然不可行。通过考虑规划中的不确定性增长和反馈作用，所提出的框架提供的轨迹在性能上是最优的，并且在概率上是安全的，从而推动了最短时间优化在高性能赛车和自主赛车中的实际应用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13622) | **Categories:** cs.RO

---


## eess.SY [eess.SY]
### [1] [Constrained Diffusers for Safe Planning and Control](https://arxiv.org/abs/2506.12544)
*Jichen Zhang, Liqun Zhao, Antonis Papachristodoulou, Jack Umenberger*

Main category: eess.SY

TL;DR: 本文提出了一种名为约束扩散器的新框架，它将约束整合到预训练的扩散模型中，以解决扩散模型在规划和控制任务中确保约束下的安全性的挑战。


<details>
  <summary>Details</summary>
Motivation: 扩散模型在规划和控制任务中表现出巨大的潜力，但确保约束下的安全性仍然是一个关键挑战。

Method: 提出了一种名为约束扩散器的新框架，该框架将约束整合到预训练的扩散模型中，无需重新训练或修改架构。该方法采用约束朗之万采样机制进行反向扩散过程，并通过投影法、原始对偶法和增广拉格朗日法三种迭代算法联合优化轨迹并实现约束满足。此外，还结合了离散控制障碍函数作为约束，以保证在线实施的安全性。

Result: 在Maze2D、locomotion和pybullet ball running任务中的实验表明，所提出的方法能够以更少的计算时间实现约束满足，并且在具有静态和时变约束的环境中与现有方法相比具有竞争力。

Conclusion: 所提出的约束扩散器方法能够在多种任务中实现约束满足，并且计算效率较高，与现有方法相比具有竞争力。

Abstract: 扩散模型在规划和控制任务中展示了卓越的潜力，因为它们能够表示动作和轨迹上的多模态分布。然而，确保约束下的安全性仍然是扩散模型面临的一个关键挑战。本文提出了一种名为约束扩散器的新框架，该框架将约束整合到预训练的扩散模型中，无需重新训练或修改架构。受到约束优化的启发，我们应用了一种约束朗之万采样机制来进行反向扩散过程，该机制通过三种迭代算法（投影法、原始对偶法和增广拉格朗日法）联合优化轨迹并实现约束满足。此外，我们还结合了离散控制障碍函数作为约束扩散器的约束，以保证在线实施的安全性。在Maze2D、locomotion和pybullet ball running任务中的实验表明，我们提出的方法能够以更少的计算时间实现约束满足，并且在具有静态和时变约束的环境中与现有方法相比具有竞争力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.12544) | **Categories:** eess.SY, cs.RO, cs.SY

---
