# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-19

## 目录

- [人工智能 (Artificial Intelligence) (6)](#cs-ai)
- [计算语言学 (Computation and Language) (1)](#cs-cl)
- [计算机视觉 (Computer Vision) (6)](#cs-cv)
- [cs.IR (1)](#cs-ir)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (12)](#cs-ro)
- [stat.AP (1)](#stat-ap)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [ADRD: LLM-Driven Autonomous Driving Based on Rule-based Decision Systems](https://arxiv.org/abs/2506.14299)
*Fanzhi Zeng, Siqi Wang, Chuzhao Zhu, Li Li*

Main category: cs.AI

TL;DR: ADRD框架利用大型语言模型生成可解释的规则，显著提升了自动驾驶决策系统的性能。


<details>
  <summary>Details</summary>
Motivation: 构建可解释的自动驾驶决策系统已成为学术研究的焦点。

Method: 利用大型语言模型生成可执行的、基于规则的决策系统，并引入ADRD框架，该框架集成了信息模块、代理模块和测试模块。

Result: ADRD在可解释性、响应速度和驾驶性能方面均优于传统的强化学习方法和最先进的基于LLM的方法。

Conclusion: ADRD框架在自动驾驶决策任务中表现出色，展示了透明、基于规则的决策系统在复杂驾驶场景中的巨大潜力。

Abstract: 如何构建一个可解释的自动驾驶决策系统已成为学术研究的焦点。在这项研究中，我们提出了一种新颖的方法，该方法利用大型语言模型（LLM）来生成可执行的、基于规则的决策系统，以应对这一挑战。具体来说，利用LLM强大的推理和编程能力，我们引入了ADRD（基于规则的决策系统的LLM驱动的自动驾驶）框架，该框架集成了三个核心模块：信息模块、代理模块和测试模块。该框架的运作方式是首先通过信息模块聚合上下文驾驶场景信息，然后利用代理模块生成基于规则的驾驶策略。这些策略通过与测试模块的持续交互进行迭代优化。广泛的实验评估表明，ADRD在自动驾驶决策任务中表现出卓越的性能。与传统的强化学习方法和最先进的基于LLM的方法相比，ADRD在可解释性、响应速度和驾驶性能方面显示出显著的优势。这些结果突出了该框架实现对复杂驾驶场景的全面和准确理解的能力，并强调了透明、基于规则的决策系统具有易于修改和广泛应用的广阔前景。据我们所知，这是第一个将大型语言模型与基于规则的系统集成用于自动驾驶决策的工作，我们的发现验证了其在现实世界中部署的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14299) | **Categories:** cs.AI

---

### [2] [From Points to Places: Towards Human Mobility-Driven Spatiotemporal Foundation Models via Understanding Places](https://arxiv.org/abs/2506.14570)
*Mohammad Hashemi, Andreas Zufle*

Main category: cs.AI

TL;DR: 本文提出了一种新的空间基础模型，通过整合地理位置语义与人类移动性来理解地点，从而实现更智能的空间决策。


<details>
  <summary>Details</summary>
Motivation: 为了支持跨不同地理位置和环境的可扩展和可转移的分析，需要一个用于时空数据的通用基础模型。现有的基础模型在处理移动数据的空间、时间和语义复杂性方面仍然有限。

Method: 提出了一种新的空间基础模型，该模型将地理位置语义与跨多个尺度的人类移动性相结合，从对离散兴趣点的建模转变为理解地点。

Result: 确定了适应性、可扩展性和多粒度推理方面的关键差距，并提出了以地点建模和实现高效学习为重点的研究方向。

Conclusion: 下一代地理空间智能需要可扩展的、具有上下文感知能力的模型，这些模型可以应用于个性化地点发现、物流优化和城市规划等。

Abstract: 捕捉人类移动性对于建模人们如何互动和穿梭于物理空间至关重要，反映了社会行为、资源获取和动态空间模式。为了支持跨不同地理位置和环境的可扩展和可转移的分析，需要一个用于时空数据的通用基础模型。虽然基础模型已经改变了语言和视觉，但它们在处理移动数据的空间、时间和语义复杂性方面仍然有限。这篇愿景论文提倡一种新的空间基础模型，该模型将地理位置语义与跨多个尺度的人类移动性相结合。我们愿景的核心是从对离散兴趣点的建模转变为理解地点：由人类行为和移动性塑造的动态的、上下文丰富的区域，可能包含许多兴趣点。我们确定了适应性、可扩展性和多粒度推理方面的关键差距，并提出了以地点建模和实现高效学习为重点的研究方向。我们的目标是指导下一代地理空间智能的可扩展的、具有上下文感知能力的模型的开发。这些模型解锁了强大的应用，从个性化地点发现和物流优化到城市规划，最终实现更智能、更快速的空间决策。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14570) | **Categories:** cs.AI

---

### [3] [Into the Unknown: Applying Inductive Spatial-Semantic Location Embeddings for Predicting Individuals' Mobility Beyond Visited Places](https://arxiv.org/abs/2506.14070)
*Xinglei Wang, Tao Cheng, Stephen Law, Zichao Zeng, Ilya Ilyankou, Junyuan Liu, Lu Yin, Weiming Huang, Natchapon Jongwiriyanurak*

Main category: cs.AI

TL;DR: CaLLiPer通过融合空间和语义信息进行对比学习，实现了更鲁棒的人类移动预测，尤其是在新地点预测方面。


<details>
  <summary>Details</summary>
Motivation: 传统方法依赖于从历史移动模式中学习的位置嵌入，限制了它们编码显式空间信息、整合丰富的城市语义环境以及适应先前未见位置的能力。

Method: 提出CaLLiPer，一种融合空间坐标和兴趣点语义特征的对比学习框架，用于位置嵌入。

Result: 在四个公共移动数据集上的大量实验表明，CaLLiPer在传统和归纳设置下始终优于强大的基线，尤其是在归纳场景中表现出色。

Conclusion: 多模态归纳位置嵌入能够提升人类移动预测系统的性能。

Abstract: 预测个人下一个位置是人类移动建模中的一项核心任务，对城市规划、交通、公共政策和个性化移动服务具有广泛的影响。传统方法主要依赖于从历史移动模式中学习的位置嵌入，这限制了它们编码显式空间信息、整合丰富的城市语义环境以及适应先前未见位置的能力。为了解决这些挑战，我们探索了CaLLiPer的应用——CaLLiPer是一种多模态表示学习框架，它通过对比学习融合兴趣点的空间坐标和语义特征——用于个人移动预测中的位置嵌入。CaLLiPer的嵌入在设计上是空间显式的、语义丰富的和归纳的，即使在新出现的位置的情况下也能实现强大的预测性能。通过在四个公共移动数据集上在传统和归纳设置下进行的大量实验，我们证明了CaLLiPer始终优于强大的基线，尤其是在归纳场景中表现出色。我们的研究结果强调了多模态归纳位置嵌入在提升人类移动预测系统能力方面的潜力。我们还发布了代码和数据(https://github.com/xlwang233/Into-the-Unknown)，以促进可重复性和未来的研究。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14070) | **Categories:** cs.AI

---

### [4] [LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?](https://arxiv.org/abs/2506.14496)
*Muhammad Atta Ur Rahman, Melanie Schranz*

Main category: cs.AI

TL;DR: 本文对比了传统群体智能和LLM驱动的群体，分析了LLM在群体系统中的机遇与挑战，并讨论了“群体”在AI中的新定义。


<details>
  <summary>Details</summary>
Motivation: 探讨在现代人工智能（AI）中，去中心化、可扩展性和涌现性是如何被重新定义的，对比传统群体算法与LLM驱动的群体。

Method: 本文通过实现和比较Boids和蚁群优化（ACO）算法，评估了延迟、资源使用和行为准确性，对比了传统群体算法与LLM驱动的群体。

Result: 实验评估了基于云和本地LLM在基于主体的群体中的适用性。

Conclusion: LLMs为群体系统提供了强大的推理和抽象能力，但也带来了计算和协调方面的新约束，对传统群体设计的概念提出了挑战。

Abstract: 本文对比了传统群体智能算法和大型语言模型（LLM）驱动的群体，探讨了在现代人工智能（AI）中，去中心化、可扩展性和涌现性是如何被重新定义的。我们通过实现和比较Boids和蚁群优化（ACO）算法，评估了延迟、资源使用和行为准确性。评估了基于云和本地LLM在基于主体的群体中的适用性。虽然LLM提供了强大的推理和抽象能力，但它们在计算和协调方面引入了新的约束，这挑战了传统的群体设计概念。这项研究强调了将LLM集成到群体系统中的机遇和限制，并讨论了现代AI研究中“群体”的演变定义。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14496) | **Categories:** cs.AI

---

### [5] [Toward Safety-First Human-Like Decision Making for Autonomous Vehicles in Time-Varying Traffic Flow](https://arxiv.org/abs/2506.14502)
*Xiao Wang, Junru Yu, Jun Huang, Qiong Wu, Ljubo Vacic, Changyin Sun*

Main category: cs.AI

TL;DR: 本文提出了一种安全第一的类人决策框架（SF-HLDM），用于提高自动驾驶汽车在复杂交通环境中的安全性、舒适性和社会兼容性。


<details>
  <summary>Details</summary>
Motivation: 尽管人工智能技术的最新进展已显示出提高运输效率和安全性的巨大潜力，但自动驾驶汽车（AV）在时变交通流中行驶仍然面临巨大挑战，尤其是在密集和交互式的情况下。同时，人类拥有自由意志，即使在完全相同的场景中，通常也不会做出相同的决定，导致数据驱动的方法存在迁移性差和搜索成本高的问题，从而降低了行为策略的效率和有效性。

Method: 该框架集成了分层渐进框架，该框架结合了用于其他道路使用者意图推断的时空注意（S-TA）机制、用于行为调节的社会合规性估计模块以及用于有效扩展搜索空间的深度进化强化学习（DERL）模型，从而避免陷入局部最优陷阱并降低过度拟合的风险，从而做出具有可解释性和灵活性的类人决策。

Result: SF-HLDM框架使自动驾驶AI智能体能够动态调整决策参数，以保持安全边际，同时遵守符合上下文的驾驶行为。

Conclusion: SF-HLDM框架使自动驾驶AI智能体能够动态调整决策参数，以保持安全边际，同时遵守符合上下文的驾驶行为。

Abstract: 尽管人工智能技术的最新进展已显示出提高运输效率和安全性的巨大潜力，但自动驾驶汽车（AV）在时变交通流中行驶仍然面临巨大挑战，尤其是在密集和交互式的情况下。同时，人类拥有自由意志，即使在完全相同的场景中，通常也不会做出相同的决定，导致数据驱动的方法存在迁移性差和搜索成本高的问题，从而降低了行为策略的效率和有效性。在这项研究中，我们提出了一种安全第一的类人决策框架（SF-HLDM），供自动驾驶汽车安全、舒适且在效率上具有社会兼容性地驾驶。该框架集成了分层渐进框架，该框架结合了用于其他道路使用者意图推断的时空注意（S-TA）机制、用于行为调节的社会合规性估计模块以及用于有效扩展搜索空间的深度进化强化学习（DERL）模型，从而避免陷入局部最优陷阱并降低过度拟合的风险，从而做出具有可解释性和灵活性的类人决策。SF-HLDM框架使自动驾驶AI智能体能够动态调整决策参数，以保持安全边际，同时遵守符合上下文的驾驶行为。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14502) | **Categories:** cs.AI

---

### [6] [AgentDistill: Training-Free Agent Distillation with Generalizable MCP Boxes](https://arxiv.org/abs/2506.14728)
*Jiahao Qiu, Xinzhe Juan, Yimin Wang, Ling Yang, Xuan Qi, Tongcheng Zhang, Jiacheng Guo, Yifu Lu, Zixin Yao, Hongru Wang, Shilong Liu, Xun Jiang, Liu Leqi, Mengdi Wang*

Main category: cs.AI

TL;DR: AgentDistill通过重用教师代理生成的任务解决模块（MCP），实现了高效且可扩展的代理知识转移，使小型语言模型也能达到大型语言模型的性能水平。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）的代理涉及规划、记忆和工具使用，但基于LLM的代理的提炼相对未被探索。现有的代理提炼方法通常重放完整的教师轨迹或模仿逐步的教师工具使用，但它们通常难以训练学生代理在新的环境中动态地规划和行动。

Method: 提出AgentDistill，一种新颖的免训练代理提炼框架，通过直接重用模型-上下文-协议（MCP），实现高效且可扩展的知识转移，这些MCP是由教师代理自主生成的结构化和可重用的任务解决模块。

Result: 在生物医学和数学基准测试中进行的实验表明，我们提炼的基于小型语言模型的学生代理可以达到与使用大型LLM（如OctoTools (GPT-4o)）的先进系统相当的性能。

Conclusion: 基于小型语言模型的学生代理通过复用提炼的MCP，在生物医学和数学基准测试中达到了与使用大型LLM（如OctoTools (GPT-4o)）的先进系统相当的性能，突显了该框架在构建可扩展且经济高效的智能代理方面的有效性。

Abstract: 知识蒸馏已成为一个成熟的领域，通过对齐大型语言模型（LLM）的输出或内部表示，将它们压缩成较小的模型。然而，基于LLM的代理的蒸馏，涉及到规划、记忆和工具的使用，仍然相对未被充分探索。现有的代理蒸馏方法通常重放完整的教师轨迹或模仿教师逐步的工具使用，但是它们经常难以训练学生代理在新的环境中动态地规划和行动。我们提出了AgentDistill，这是一个新颖的，免训练的代理蒸馏框架，它可以通过直接重用模型-上下文-协议（MCP）来实现高效和可扩展的知识转移。这些MCP是由教师代理自主生成的结构化和可重用的任务解决模块。重用这些提炼的MCP使得学生代理能够跨领域推广它们的能力，并以最少的监督或人工干预来解决新的问题。在生物医学和数学基准测试上的实验表明，我们提炼的，构建在小型语言模型之上的学生代理可以达到与使用大型LLM（例如OctoTools (GPT-4o)）的先进系统相当的性能，突出了我们的框架在构建可扩展且经济高效的智能代理方面的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14728) | **Categories:** cs.AI

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection](https://arxiv.org/abs/2506.13956)
*Shang-Chi Tsai, Seiya Kawano, Angel Garcia Contreras, Koichiro Yoshino, Yun-Nung Chen*

Main category: cs.CL

TL;DR: 本文提出了一种新的数据增强框架，通过模拟对话和生成图像来改进机器人辅助场景中的多模态模型。


<details>
  <summary>Details</summary>
Motivation: 在设计用于辅助日常人类活动的机器人时，利用周围环境的视觉线索来增强用户请求，从而提高意图理解至关重要。然而，收集包含视觉和语言元素的大规模数据集用于模型训练具有挑战性且耗时。

Method: 利用大型语言模型模拟潜在的对话和环境上下文，然后使用stable diffusion模型创建描述这些环境的图像。

Result: 基于从现实场景收集的数据集，实验结果表明该方法显著提高了机器人动作选择能力，达到了最先进的性能。

Conclusion: 实验结果表明，该方法显著提高了机器人动作选择能力，达到了最先进的性能。

Abstract: 在设计用于辅助日常人类活动的机器人时，利用周围环境的视觉线索来增强用户请求，从而提高意图理解至关重要。然而，收集包含视觉和语言元素的大规模数据集用于模型训练具有挑战性且耗时。为了解决这个问题，我们的论文介绍了一种新颖的框架，专注于机器人辅助场景中的数据增强，包括对话和相关的环境图像。这种方法包括利用一个复杂的大型语言模型来模拟潜在的对话和环境背景，然后使用一个稳定的扩散模型来创建描述这些环境的图像。额外生成的数据用于改进最新的多模态模型，使它们能够更准确地确定适当的动作，以响应用户与有限目标数据的交互。我们的实验结果，基于从现实场景收集的数据集，表明我们的方法显著提高了机器人的动作选择能力，达到了最先进的性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13956) | **Categories:** cs.CL, cs.AI, cs.RO

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [SceneAware: Scene-Constrained Pedestrian Trajectory Prediction with LLM-Guided Walkability](https://arxiv.org/abs/2506.14144)
*Juho Bai, Inwook Shim*

Main category: cs.CV

TL;DR: 该论文提出了一种名为SceneAware的行人轨迹预测框架，通过显式地结合场景理解，显著提高了预测的准确性和物理合理性。


<details>
  <summary>Details</summary>
Motivation: Existing pedestrian trajectory prediction methods often overlook the environmental context, which significantly influences human movement. This paper aims to address this limitation by explicitly incorporating scene understanding to improve prediction accuracy.

Method: The authors propose SceneAware, a framework that uses a Vision Transformer (ViT) for scene encoding and Multi-modal Large Language Models (MLLMs) to generate walkability masks, combined with a Transformer-based trajectory encoder and collision penalty mechanisms.

Result: Experiments on the ETH/UCY datasets show that SceneAware outperforms state-of-the-art methods by more than 50%, demonstrating consistent performance across various pedestrian movement types.

Conclusion: The study concludes that incorporating explicit scene information significantly improves the accuracy and physical plausibility of pedestrian trajectory predictions, demonstrating the effectiveness and reliability of the proposed scene-aware approach.

Abstract: 为了准确预测行人轨迹，这篇论文提出了一个名为SceneAware的新框架，它通过结合视觉Transformer（ViT）场景编码器处理环境上下文信息，并利用多模态大型语言模型（MLLM）生成二元可行走区域掩码，从而显式地融入了场景理解。该框架还结合了基于Transformer的轨迹编码器，并整合了碰撞惩罚机制，以确保预测的轨迹在物理上是合理的。在ETH/UCY基准数据集上的综合实验表明，SceneAware优于现有技术，性能提升超过50%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14144) | **Categories:** cs.CV, cs.AI

---

### [2] [Image Segmentation with Large Language Models: A Survey with Perspectives for Intelligent Transportation Systems](https://arxiv.org/abs/2506.14096)
*Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma*

Main category: cs.CV

TL;DR: 本文综述了大型语言模型（LLM）增强的图像分割技术在智能交通系统中的应用、挑战和未来方向。


<details>
  <summary>Details</summary>
Motivation: 在智能交通系统中，准确的场景理解对于安全和效率至关重要，而LLM与计算机视觉的结合为此提供了前所未有的能力。

Method: 本文对LLM增强的图像分割方法进行了分类，并重点介绍了其在智能交通系统中的应用。

Result: 强调了LLM增强图像分割在自动驾驶、交通监控和基础设施维护方面的增强作用。

Conclusion: LLM增强的图像分割技术在智能交通系统中具有巨大潜力，但需要解决实时性、可靠性和可解释性等关键挑战。

Abstract: 大型语言模型（LLM）与计算机视觉的集成正在深刻地改变图像分割等感知任务。对于智能交通系统（ITS）而言，准确的场景理解对于安全和效率至关重要，这种新范式提供了前所未有的能力。本文系统地回顾了LLM增强图像分割这一新兴领域，重点关注其在ITS中的应用、挑战和未来方向。我们根据其提示机制和核心架构对当前方法进行了分类，并重点介绍了这些创新如何增强自动驾驶、交通监控和基础设施维护的道路场景理解。最后，我们确定了关键挑战，包括实时性能和安全关键的可靠性，并概述了一个以可解释的、以人为中心的AI为中心的视角，作为这项技术在下一代交通系统中成功部署的先决条件。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14096) | **Categories:** cs.CV, cs.AI

---

### [3] [KDMOS:Knowledge Distillation for Motion Segmentation](https://arxiv.org/abs/2506.14130)
*Chunyu Cao, Jintao Cheng, Zeyu Chen, Linfan Zhan, Rui Fan, Zhijian He, Xiaoyu Tang*

Main category: cs.CV

TL;DR: 提出了一种基于logits知识蒸馏的运动目标分割框架，在保证实时性的同时提高了精度。


<details>
  <summary>Details</summary>
Motivation: 运动目标分割对于自动驾驶至关重要，但现有方法难以平衡精度和实时推理。

Method: 提出了一种基于logits的知识蒸馏框架，用于运动目标分割。

Result: 参数量减少了7.69%，显著减少了假阳性和假阴性。

Conclusion: 该方法在SemanticKITTI-MOS数据集的隐藏测试集上取得了78.8%的IoU，并在Apollo数据集上取得了有竞争力的结果。

Abstract: 运动目标分割（MOS）对于自动驾驶至关重要，因为它增强了定位、路径规划、地图构建、场景流估计和未来状态预测。虽然现有方法取得了良好的性能，但平衡精度和实时推理仍然是一个挑战。为了解决这个问题，我们提出了一种基于logits的知识蒸馏框架用于MOS，旨在提高精度，同时保持实时效率。具体来说，我们采用基于鸟瞰图（BEV）投影的模型作为学生，非投影模型作为教师。为了处理移动和非移动类别之间严重的失衡，我们将它们解耦并应用定制的蒸馏策略，使教师模型能够更好地学习关键的运动相关特征。这种方法显著减少了假阳性和假阴性。此外，我们还引入了动态上采样，优化了网络架构，并实现了7.69%的参数量减少，从而减轻了过拟合。我们的方法在SemanticKITTI-MOS数据集的隐藏测试集上取得了78.8%的IoU，并在Apollo数据集上取得了有竞争力的结果。KDMOS的实现可在https://github.com/SCNU-RISLAB/KDMOS上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14130) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [4] [Leader360V: The Large-scale, Real-world 360 Video Dataset for Multi-task Learning in Diverse Environment](https://arxiv.org/abs/2506.14271)
*Weiming Zhang, Dingwen Xiao, Aobotao Dai, Yexin Liu, Tianbo Pan, Shiqi Wen, Lei Chen, Lin Wang*

Main category: cs.CV

TL;DR: Leader360V是首个大规模带标签的真实世界360视频数据集，用于实例分割和跟踪。


<details>
  <summary>Details</summary>
Motivation: 由于固有的球面属性，如极地地区的严重失真和内容不连续性，使得360视频的标注成本高昂且复杂，因此缺乏大规模、带标签的真实世界数据集。

Method: 论文设计了一个自动标注流水线，巧妙地协调了预训练的2D分割器和大型语言模型来辅助标注，该流水线包含三个新颖的阶段。

Result: 广泛的用户研究和评估表明，该标注流水线的有效性。

Conclusion: Leader360V数据集显著提升了360视频分割和跟踪的模型性能，为更具扩展性的360场景理解铺平了道路。

Abstract: 360视频以360X180的超大视野捕获完整的周围场景。这使得360场景理解任务，例如分割和跟踪，对于自动驾驶、机器人等应用至关重要。随着最近基础模型的出现，但由于缺乏大规模、带标签的真实世界数据集，社区受到了阻碍。这是由于固有的球面属性，例如极地地区的严重失真和内容不连续性，使得注释成本高昂且复杂。本文介绍了Leader360V，这是第一个用于实例分割和跟踪的大规模、带标签的真实世界360视频数据集。我们的数据集具有很高的场景多样性，从室内和城市环境到自然和动态的室外场景。为了自动化标注，我们设计了一个自动标注流水线，巧妙地协调了预训练的2D分割器和大型语言模型来辅助标注。该流水线包含三个新颖的阶段。具体来说，在初始标注阶段，我们引入了一个语义和失真感知细化模块，该模块将来自多个2D分割器的对象掩码提议与LLM验证的语义标签相结合。然后将这些转换为掩码提示，以指导SAM2生成用于后续帧的失真感知掩码。在自动细化标注阶段，通过再次应用SDR或解决水平边界附近的非连续性来校正缺失或不完整的区域。人工修订阶段最终结合了LLM和人工标注员，以进一步完善和验证标注。广泛的用户研究和评估表明，我们的标注流水线是有效的。同时，实验证实Leader360V显着提高了360视频分割和跟踪的模型性能，为更具扩展性的360场景理解铺平了道路。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14271) | **Categories:** cs.CV

---

### [5] [Toward Rich Video Human-Motion2D Generation](https://arxiv.org/abs/2506.14428)
*Ruihao Xi, Xuekuan Wang, Yongcheng Li, Shuhua Li, Zichen Wang, Yiwei Wang, Feng Wei, Cairong Zhao*

Main category: cs.CV

TL;DR: 本文提出了RVHM2D模型，用于生成逼真且可控的单人和双人互动人体运动，并在Motion2D-Video-150K数据集上取得了领先性能。


<details>
  <summary>Details</summary>
Motivation: 由于数据稀缺和建模人际动态的复杂性，生成逼真且可控的人体运动，特别是涉及丰富的多角色互动的人体运动，仍然是一个巨大的挑战。

Method: 提出了一种新的基于扩散的丰富视频人体运动2D生成（RVHM2D）模型，该模型结合了增强的文本条件机制，利用双文本编码器（CLIP-L/B）或T5-XXL，具有全局和局部特征。设计了一个两阶段的训练策略：首先使用标准扩散目标训练模型，然后使用基于FID的奖励通过强化学习进行微调，以进一步提高运动真实感和文本对齐。

Result: RVHM2D在Motion2D-Video-150K基准测试中，在生成单人和双人互动场景方面均取得了领先的性能。

Conclusion: RVHM2D在Motion2D-Video-150K基准测试中，在生成单人和双人互动场景方面均取得了领先的性能。

Abstract: 由于数据稀缺和建模人际互动复杂性，生成逼真且可控的人体运动，特别是涉及丰富的多角色互动的人体运动，仍然是一个巨大的挑战。为了解决这些限制，我们首先引入了一个新的大型丰富视频人体运动2D数据集（Motion2D-Video-150K），该数据集包含15万个视频序列。Motion2D-Video-150K具有多样化的单人角色和双人互动动作的平衡分布，每个动作都配有详细的文本描述。在此数据集的基础上，我们提出了一种新的基于扩散的丰富视频人体运动2D生成（RVHM2D）模型。RVHM2D结合了增强的文本条件机制，利用双文本编码器（CLIP-L/B）或T5-XXL，具有全局和局部特征。我们设计了一个两阶段的训练策略：首先使用标准扩散目标训练模型，然后使用基于FID的奖励通过强化学习进行微调，以进一步提高运动真实感和文本对齐。大量的实验表明，RVHM2D在Motion2D-Video-150K基准测试中，在生成单人和双人互动场景方面均取得了领先的性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14428) | **Categories:** cs.CV

---

### [6] [SIRI-Bench: Challenging VLMs' Spatial Intelligence through Complex Reasoning Tasks](https://arxiv.org/abs/2506.14512)
*Zijian Song, Xiaoxin Lin, Qiuming Huang, Guangrun Wang, Liang Lin*

Main category: cs.CV

TL;DR: SIRI-Bench是一个新的基准测试，用于评估视觉语言模型在视频中的空间推理能力，结果表明现有模型在该任务上表现不佳。


<details>
  <summary>Details</summary>
Motivation: 缺乏对视觉语言模型在空间环境中复杂推理能力的系统评估。

Method: 提出了SIRI-Bench基准测试，用于评估视觉语言模型在视频空间推理任务中的空间智能；开发了一个自动场景创建引擎，利用多个专门的LLM代理从抽象数学问题生成逼真的3D场景。

Result: 实验结果表明，最先进的视觉语言模型在SIRI-Bench上表现不佳，突显了空间推理的挑战。

Conclusion: 最先进的视觉语言模型在SIRI-Bench上表现不佳，表明空间推理仍然是一个挑战。

Abstract: 大型语言模型（LLM）在复杂推理方面正经历着快速进步，在数学和编程中表现出卓越的泛化能力。相比之下，虽然空间智能对于视觉语言模型（VLM）在现实世界中的交互至关重要，但对其在空间环境中复杂推理能力的系统评估仍未得到充分探索。为了弥合这一差距，我们推出了SIRI-Bench，这是一个旨在通过基于视频的推理任务评估VLM空间智能的基准。SIRI-Bench包含近1K个视频-问题-答案三元组，其中每个问题都嵌入在逼真的3D场景中，并通过视频捕获。通过仔细设计问题和相应的3D场景，我们的基准确保了解决问题既需要空间理解以提取信息，又需要高级推理以得出解决方案，这使其成为评估VLM的一个具有挑战性的基准。为了促进大规模数据合成，我们开发了一个自动场景创建引擎。该引擎利用多个专门的LLM代理，可以从抽象数学问题生成逼真的3D场景，确保对原始描述的忠实性。实验结果表明，最先进的VLM在SIRI-Bench上表现不佳，突显了空间推理的挑战。我们希望我们的研究能引起研究人员对空间基础推理的关注，并推进VLM在视觉问题解决方面的应用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14512) | **Categories:** cs.CV

---


## cs.IR [cs.IR]
### [1] [XGraphRAG: Interactive Visual Analysis for Graph-based Retrieval-Augmented Generation](https://arxiv.org/abs/2506.13782)
*Ke Wang, Bo Pan, Yingchaojie Feng, Yuwei Wu, Jieyi Chen, Minfeng Zhu, Wei Chen*

Main category: cs.IR

TL;DR: 该论文提出了一种用于分析 GraphRAG 的可视化框架，以提高其可解释性和可访问性。


<details>
  <summary>Details</summary>
Motivation: 由于 GraphRAG 复杂的信息处理流程以及图构建和查询过程中涉及的大量 LLM 调用，开发者通常在分析 GraphRAG 在其数据集上的有效性时面临挑战，这限制了 GraphRAG 的可解释性和可访问性。

Method: 该研究提出了一种可视化分析框架，并开发了一个原型系统 XGraphRAG，该系统包含一组交互式可视化，以方便用户的分析过程。

Result: 该研究的评估表明，所提出的可视化分析框架是有效和可用的。

Conclusion: 该研究提出了一种可视化分析框架，可以帮助 RAG 开发者识别 GraphRAG 的关键召回并追踪这些召回在 GraphRAG 流程中的情况，从而促进失败案例的收集和改进机会的识别。评估表明该方法的有效性和可用性。

Abstract: 基于图的检索增强生成 (GraphRAG) 在利用外部知识库增强大型语言模型 (LLM) 的答案方面表现出强大的能力。与传统的 RAG 相比，它引入了一个图作为中间表示，以捕获语料库中更好的结构化关系知识，从而提高了生成结果的准确性和全面性。然而，由于 GraphRAG 复杂的信息处理流程以及图构建和查询过程中涉及的大量 LLM 调用，开发者通常在分析 GraphRAG 在其数据集上的有效性时面临挑战，这限制了 GraphRAG 的可解释性和可访问性。本研究提出了一种可视化分析框架，可以帮助 RAG 开发者识别 GraphRAG 的关键召回并追踪这些召回在 GraphRAG 流程中的情况。基于此框架，我们开发了 XGraphRAG，一个包含一组交互式可视化的原型系统，以方便用户的分析过程，促进失败案例的收集和改进机会的识别。我们的评估证明了该方法的有效性和可用性。我们的工作是开源的，可在 https://github.com/Gk0Wk/XGraphRAG 上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13782) | **Categories:** cs.IR, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [HiLight: A Hierarchical Reinforcement Learning Framework with Global Adversarial Guidance for Large-Scale Traffic Signal Control](https://arxiv.org/abs/2506.14391)
*Yaqiao Zhu, Hongkai Wen, Geyong Min, Man Luo*

Main category: cs.LG

TL;DR: HiLight：一种用于大规模交通信号控制的分层强化学习框架，具有全局对抗指导，可有效协调全局规划和局部执行。


<details>
  <summary>Details</summary>
Motivation: 现有的强化学习（RL）方法在扩展到大型网络同时保持全局协调方面面临挑战。集中式RL存在可扩展性问题，而分散式方法通常缺乏统一的目标，导致网络级效率有限。

Method: HiLight采用分层强化学习框架，包含一个高层Meta-Policy（使用Transformer-LSTM架构划分交通网络并生成子目标）和一个低层Sub-Policy（控制具有全局意识的单个交叉口）。引入对抗训练机制以提高全局规划和局部执行之间的一致性。

Result: HiLight在合成和真实世界的基准测试中都进行了评估，并在包含高峰过渡、恶劣天气和节假日高峰等各种交通条件的大型曼哈顿网络中进行了评估。实验结果表明，HiLight在大型场景中表现出显著的优势。

Conclusion: HiLight在大型交通网络中表现出显著优势，并在各种规模的标准基准测试中保持竞争力。

Abstract: 高效的交通信号控制（TSC）对于缓解城市拥堵至关重要，但现有的强化学习（RL）方法在扩展到大型网络同时保持全局协调方面面临挑战。集中式RL存在可扩展性问题，而分散式方法通常缺乏统一的目标，导致网络级效率有限。在本文中，我们提出了一种具有全局对抗指导的分层强化学习框架HiLight，用于大规模TSC。HiLight由一个高层Meta-Policy组成，该Meta-Policy使用Transformer-LSTM架构将交通网络划分为子区域并生成子目标；以及一个低层Sub-Policy，该Sub-Policy控制具有全局意识的单个交叉口。为了提高全局规划和局部执行之间的一致性，我们引入了一种对抗训练机制，其中Meta-Policy生成具有挑战性但信息丰富的子目标，而Sub-Policy学习超越这些目标，从而实现更有效的协调。我们在合成和真实世界的基准测试中评估了HiLight，此外还构建了一个包含高峰过渡、恶劣天气和节假日高峰等各种交通条件的大型曼哈顿网络。实验结果表明，HiLight在大型场景中表现出显著的优势，并在各种规模的标准基准测试中保持竞争力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14391) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics](https://arxiv.org/abs/2506.14009)
*Qianzhong Chen, Naixiang Gao, Suning Huang, JunEn Low, Timothy Chen, Jiankai Sun, Mac Schwager*

Main category: cs.RO

TL;DR: GRaD-Nav++是一个轻量级的机载VLA框架，它通过可微强化学习和混合专家模型，实现了对自然语言命令的实时响应和导航。


<details>
  <summary>Details</summary>
Motivation: 自主无人机需要在非结构化环境中解释和执行高级语言指令，但现有方法受到其对手工技能、广泛的参数调整或不适合机载使用的计算密集型模型的限制。

Method: 一种轻量级的视觉-语言-动作（VLA）框架，完全在机载运行，并实时遵循自然语言命令。该策略在逼真的3D高斯溅射（3DGS）模拟器中通过可微强化学习（DiffRL）进行训练，从而能够从视觉和语言输入中有效学习低级控制。其核心是混合专家（MoE）行动头，它可以自适应地路由计算，以提高泛化能力，同时减少遗忘。

Result: 在多任务泛化实验中，GRaD-Nav++在模拟训练任务上的成功率为83%，在未见任务上的成功率为75%。当部署在真实的硬件上时，它在训练任务上的成功率为67%，在未见任务上的成功率为50%。在多环境适应实验中，GRaD-Nav++在不同的模拟环境中的平均成功率为81%，在不同的现实世界环境中的平均成功率为67%。

Conclusion: 紧凑高效的模型能够在没有外部基础设施的情况下实现可靠的、语言引导的导航。

Abstract: 自主无人机需要在非结构化环境中解释和执行高级语言指令，但现有方法受到其对手工技能、广泛的参数调整或不适合机载使用的计算密集型模型的限制。我们介绍了GRaD-Nav++，这是一种轻量级的视觉-语言-动作（VLA）框架，它完全在机载运行，并实时遵循自然语言命令。我们的策略在逼真的3D高斯溅射（3DGS）模拟器中通过可微强化学习（DiffRL）进行训练，从而能够从视觉和语言输入中有效学习低级控制。其核心是混合专家（MoE）行动头，它可以自适应地路由计算，以提高泛化能力，同时减少遗忘。在多任务泛化实验中，GRaD-Nav++在模拟训练任务上的成功率为83%，在未见任务上的成功率为75%。当部署在真实的硬件上时，它在训练任务上的成功率为67%，在未见任务上的成功率为50%。在多环境适应实验中，GRaD-Nav++在不同的模拟环境中的平均成功率为81%，在不同的现实世界环境中的平均成功率为67%。这些结果为完全机载的视觉-语言-动作（VLA）飞行建立了一个新的基准，并表明紧凑高效的模型能够在没有外部基础设施的情况下实现可靠的、语言引导的导航。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14009) | **Categories:** cs.RO

---

### [2] [Narrate2Nav: Real-Time Visual Navigation with Implicit Language Reasoning in Human-Centric Environments](https://arxiv.org/abs/2506.14233)
*Amirreza Payandeh, Anuj Pokhrel, Daeun Song, Marcos Zampieri, Xuesu Xiao*

Main category: cs.RO

TL;DR: Narrate2Nav 通过自监督学习在视觉编码器中嵌入了隐式自然语言推理，从而提高了人机交互环境中机器人导航的性能。


<details>
  <summary>Details</summary>
Motivation: 大型视觉语言模型 (VLM) 在增强以人为中心的移动机器人导航方面显示出潜力，但它们的计算复杂性和对连续数值数据的有限敏感性阻碍了实时性能和精确的运动控制。

Method: 提出了一种基于 Barlow Twins 冗余减少损失的新型自监督学习框架 Narrate2Nav，以在视觉编码器中嵌入隐式自然语言推理、社会线索和人类意图。

Result: Narrate2Nav 在各种具有挑战性的场景中进行了广泛的评估，在离线未见数据集和真实世界实验中，其性能分别比次优基线提高了 52.94% 和 41.67%。

Conclusion: 在离线未见数据集和真实世界实验中，Narrate2Nav 的总体性能分别比次优基线提高了 52.94% 和 41.67%。

Abstract: 大型视觉语言模型 (VLM) 在增强以人为中心的移动机器人导航方面显示出潜力，它们可以理解上下文线索、人类意图和社会动态，同时表现出推理能力。然而，它们的计算复杂性和对连续数值数据的有限敏感性阻碍了实时性能和精确的运动控制。为此，我们提出了一种新的实时视觉-动作模型 Narrate2Nav，该模型利用一种基于 Barlow Twins 冗余减少损失的新型自监督学习框架，以在视觉编码器中嵌入隐式自然语言推理、社会线索和人类意图——从而在模型的潜在空间而不是 token 空间中实现推理。该模型在训练期间结合了 RGB 输入、运动命令和场景上下文的文本信号，以桥接从机器人观察到低级运动命令的连接，从而在部署期间实现短视距点目标导航。在离线未见数据集和真实世界实验中，对 Narrate2Nav 在各种具有挑战性的场景中进行了广泛的评估，结果表明，其总体性能分别比次优基线提高了 52.94% 和 41.67%。此外，对 Narrate2Nav 的视觉编码器注意力图与其他四个基线进行的定性比较分析表明，其对导航关键场景元素的注意力增强，突显了其在以人为中心的导航任务中的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14233) | **Categories:** cs.RO

---

### [3] [NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.14589)
*Ren Xin, Hongji Liu, Xiaodong Mei, Wenru Liu, Maosheng Ye, Zhili Chen, Jun Ma*

Main category: cs.RO

TL;DR: NetRoller通过新颖的适配器机制，实现了通用模型与专用驾驶模型的无缝集成，从而显著提升了自动驾驶性能。


<details>
  <summary>Details</summary>
Motivation: 将通用模型（GM）与自动驾驶任务中的专用模型（SM）集成，为缓解现有专用驾驶模型在数据多样性和模型容量方面的挑战提供了一种有前景的方法。然而，这种集成导致了异步系统的问题，这些问题源于GM和SM中固有的不同特性。

Method: NetRoller包含一套新颖的机制，促进通用模型和专用驾驶模型的无缝集成。它使用早期停止机制从LLM的推理过程中收集语义丰富且计算高效的表示，应用可学习的查询嵌入、无意义嵌入和位置层嵌入以促进稳健高效的跨模态翻译，并采用计算高效的查询移位和特征移位机制，通过少量epoch的微调来提高SM的性能。

Result: 在nuScenes数据集上进行的实验表明，通过NetRoller集成GM显著提高了规划任务中的人类相似性和安全性，并且在端到端自动驾驶的检测和建图任务中实现了显着的精度提升。

Conclusion: 通过NetRoller集成通用模型显著提高了自动驾驶规划任务中的人类相似性和安全性，并在检测和建图任务中实现了显著的精度提升。

Abstract: 将大型语言模型等通用模型(GM)与自动驾驶任务中的专用模型(SM)集成，为缓解现有专用驾驶模型在数据多样性和模型容量方面的挑战提供了一种有前景的方法。然而，这种集成导致了异步系统的问题，这些问题源于GM和SM中固有的不同特性。为了应对这一挑战，我们提出NetRoller，这是一种适配器，它包含一套新颖的机制，以促进通用模型和专用驾驶模型的无缝集成。具体来说，我们用于连接异步GM和SM的机制分为三个关键阶段。NetRoller首先使用早期停止机制从LLM的推理过程中收集语义丰富且计算高效的表示，从而在保持较低开销的同时保留对驾驶环境的关键见解。然后，它应用可学习的查询嵌入、无意义嵌入和位置层嵌入，以促进稳健高效的跨模态翻译。最后，它采用计算高效的查询移位和特征移位机制，通过少量epoch的微调来提高SM的性能。基于这三个阶段中形式化的机制，NetRoller使专用驾驶模型能够以其原生频率运行，同时保持对GM的情境感知。在nuScenes数据集上进行的实验表明，通过NetRoller集成GM显著提高了规划任务中的人类相似性和安全性，并且在端到端自动驾驶的检测和建图任务中实现了显着的精度提升。代码和模型可在https://github.com/Rex-sys-hk/NetRoller获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14589) | **Categories:** cs.RO

---

### [4] [Sequence Modeling for Time-Optimal Quadrotor Trajectory Optimization with Sampling-based Robustness Analysis](https://arxiv.org/abs/2506.13915)
*Katherine Mao, Hongzhan Yu, Ruipeng Zhang, Igor Spasojevic, M Ani Hsieh, Sicun Gao, Vijay Kumar*

Main category: cs.RO

TL;DR: 该论文提出了一种学习方法，通过模仿时间最优轨迹规划器来加速四旋翼飞行器的轨迹生成，并在实际硬件平台上验证了其可行性，能够推广到未见过的路径长度，显著提升了速度。


<details>
  <summary>Details</summary>
Motivation: 时间最优轨迹将四旋翼飞行器驱动到其动态极限，但计算此类轨迹涉及通过迭代非线性优化解决非凸问题，这使得它们对于实时应用而言成本过高。

Method: 该研究提出了一种基于学习的模型，该模型模仿基于模型的时间最优轨迹规划器来加速轨迹生成，并提出了一种数据增强方案，该方案将随机扰动应用于输入路径，以增强鲁棒性。

Result: 该方法实现了显着的加速，并在硬件四旋翼平台上验证了其实时可行性。实验表明，学习的模型可以推广到以前未见过的路径长度。

Conclusion: 该研究表明，学习模型可以有效地学习时间最优轨迹的潜在模式，并且能够推广到以前未见过的路径长度。

Abstract: 时间最优轨迹能够将四旋翼飞行器驱动到其动态极限，但计算此类轨迹需要通过迭代非线性优化方法解决非凸问题，这导致实时应用成本过高。本文研究了基于学习的模型，该模型模仿基于模型的时间最优轨迹规划器，以加速轨迹生成。给定一个无碰撞几何路径数据集，我们证明了建模架构可以有效地学习时间最优轨迹的潜在模式。我们引入了一个定量框架来分析学习模型的局部解析属性，并将它们与几何跟踪控制器的后向可达管联系起来。为了提高鲁棒性，我们提出了一种数据增强方案，该方案将随机扰动应用于输入路径。与经典规划器相比，我们的方法实现了显着的加速，并且我们在硬件四旋翼平台上验证了其实时可行性。实验表明，学习的模型可以推广到以前未见过的路径长度。该方法的代码可以在https://github.com/maokat12/lbTOPPQuad找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13915) | **Categories:** cs.RO

---

### [5] [DynaGuide: Steering Diffusion Polices with Active Dynamic Guidance](https://arxiv.org/abs/2506.13922)
*Maximilian Du, Shuran Song*

Main category: cs.RO

TL;DR: DynaGuide通过在扩散去噪过程中利用外部动力学模型的指导，实现了对扩散策略的有效引导。


<details>
  <summary>Details</summary>
Motivation: 在现实世界中部署大型复杂策略需要能够对其进行引导以适应具体情况。大多数常见的转向方法，如目标条件反射，需要训练机器人策略时考虑到测试时目标分布。为了克服这个限制。

Method: DynaGuide是一种扩散策略的转向方法，它在扩散去噪过程中使用来自外部动力学模型的指导。

Result: DynaGuide在铰接CALVIN任务上平均转向成功率为70%，并且在使用低质量目标进行转向时，性能优于目标条件反射5.4倍。此外，DynaGuide还成功地引导了现成的真实机器人策略，以表达对特定对象的偏好，甚至创造了新的行为。

Conclusion: DynaGuide在模拟和真实实验中表现出良好的性能，在铰接CALVIN任务上平均转向成功率为70%，并且在使用低质量目标进行转向时，性能优于目标条件反射5.4倍。此外，DynaGuide还成功地引导了现成的真实机器人策略，以表达对特定对象的偏好，甚至创造了新的行为。

Abstract: 在现实世界中部署大型复杂的策略需要能够根据情况对其进行引导。最常见的引导方法，如目标条件法，需要在训练机器人策略时考虑到测试时目标分布。为了克服这个限制，我们提出DynaGuide，一种扩散策略的引导方法，它在扩散去噪过程中使用来自外部动力学模型的指导。DynaGuide将动力学模型与基础策略分离，这使其具有多个优势，包括能够引导到多个目标，增强代表性不足的基础策略行为，并在低质量目标上保持鲁棒性。单独的引导信号还允许DynaGuide与现成的预训练扩散策略一起工作。我们在一系列模拟和真实实验中，针对其他引导方法展示了DynaGuide的性能和功能，结果表明，在一组铰接CALVIN任务上的平均引导成功率为70%，并且在使用低质量目标进行引导时，性能优于目标条件法5.4倍。我们还成功地引导了现成的真实机器人策略，以表达对特定对象的偏好，甚至创造了新的行为。视频和更多信息可以在项目网站上找到：https://dynaguide.github.io

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13922) | **Categories:** cs.RO

---

### [6] [Socially-aware Object Transportation by a Mobile Manipulator in Static Planar Environments with Obstacles](https://arxiv.org/abs/2506.13953)
*Caio C. G. Ribeiro, Leonardo R. D. Paes, Douglas G. Macharet*

Main category: cs.RO

TL;DR: 提出了一种基于 Risk-RRT* 框架的社交感知移动机械臂导航方法，该方法能够协调底座和机械臂的运动，在人群中安全有效地运输物体。


<details>
  <summary>Details</summary>
Motivation: 现有的方法主要为移动机器人开发，解决移动机械臂带来的独特挑战的研究存在显著差距。本文旨在解决在静态人群环境中，导航携带不可忽略负载的机器人移动机械臂，同时遵守社会规范的挑战。

Method: 提出了一种基于 Risk-RRT* 框架的方法，该方法能够协调移动底座和机械臂的运动。

Result: 在模拟环境中，该方法优于仅适用于移动设备的社交感知方法，突显了移动机械臂专用技术的必要性。

Conclusion: 该方法能够使机器人在导航、运输物体、避免碰撞和最小化社交不适方面表现出色。

Abstract: 在人类与机器人共存的环境中，具有社会意识的机器人导航至关重要，以确保安全和舒适。然而，现有的大多数方法主要是为移动机器人开发的，解决移动机械臂带来的独特挑战的研究存在显著差距。在本文中，我们解决了在静态人群环境中，导航携带不可忽略负载的机器人移动机械臂，同时遵守社会规范的挑战。我们的目标是开发一种能够使机器人同时操纵物体并在具有社会意识的情况下在位置之间导航的方法。我们提出了一种基于 Risk-RRT* 框架的方法，该方法能够协调移动底座和机械臂的运动。这种方法确保了无碰撞导航，同时遵守人类的社会偏好。我们在模拟环境中将我们的方法与应用于移动机械臂的具有社会意识的纯移动方法进行了比较。结果表明，有必要采用移动机械臂专用技术，我们的方法优于纯移动方法。我们的方法能够使机器人在导航、运输物体、避免碰撞和最大限度地减少社交不适方面表现出色。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.13953) | **Categories:** cs.RO

---

### [7] [A Hierarchical Test Platform for Vision Language Model (VLM)-Integrated Real-World Autonomous Driving](https://arxiv.org/abs/2506.14100)
*Yupeng Zhou, Can Cui, Juntong Peng, Zichong Yang, Juanwu Lu, Jitesh H Panchal, Bin Yao, Ziran Wang*

Main category: cs.RO

TL;DR: 该论文提出了一个用于评估集成视觉-语言模型的自动驾驶系统的分层真实世界测试平台，并在真实场景中验证了其有效性。


<details>
  <summary>Details</summary>
Motivation: 现有的基于模拟和数据集驱动的评估方法通常无法捕捉真实世界场景的完整复杂性，并且不能容易地适应具有灵活场景操作的可重复闭环测试。因此，将视觉-语言模型从广泛的网络规模数据调整到驾驶的安全关键环境提出了一个重大挑战，通常被称为领域转移。

Method: 该论文介绍了一个分层真实世界测试平台，专门用于评估集成视觉-语言模型的自动驾驶系统。该方法包括一个模块化、低延迟的车载中间件，一个清晰分离的感知-规划-控制架构，以及一套可配置的真实世界测试场景。

Result: 该论文通过一个涉及支持视觉-语言模型的自动驾驶车辆的案例研究，展示了所提出的平台的测试和评估能力。

Conclusion: 该论文展示了一个基于视觉-语言模型的自动驾驶车辆的案例研究，证明了该平台的有效性，并强调了该测试框架在不同条件下支持稳健实验的能力。

Abstract: 视觉-语言模型（VLM）通过在大量的图像-文本对上进行预训练，展示了在自动驾驶领域中实现多模态推理的潜力。然而，将这些模型从广泛的网络数据调整到安全攸关的驾驶环境，提出了一个重大挑战，通常被称为领域迁移。现有的基于模拟和数据集驱动的评估方法虽然有价值，但通常无法捕捉真实世界场景的完整复杂性，并且不能容易地适应具有灵活场景操作的可重复闭环测试。在本文中，我们介绍了一个分层真实世界测试平台，专门用于评估集成VLM的自动驾驶系统。我们的方法包括一个模块化、低延迟的车载中间件，可以无缝集成各种VLM，一个清晰分离的感知-规划-控制架构，可以容纳基于VLM和传统的模块，以及一套在封闭赛道上可配置的真实世界测试场景，从而促进可控但真实的评估。我们通过一个涉及VLM自动驾驶车辆的案例研究，展示了该平台在测试和评估方面的有效性，强调了我们的测试框架如何在各种条件下支持稳健的实验。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14100) | **Categories:** cs.RO, eess.SY

---

### [8] [AMPLIFY: Actionless Motion Priors for Robot Learning from Videos](https://arxiv.org/abs/2506.14198)
*Jeremy A. Collins, Loránd Cheng, Kunal Aneja, Albert Wilcox, Benjamin Joffe, Animesh Garg*

Main category: cs.RO

TL;DR: AMPLIFY利用大规模无动作视频数据学习机器人策略，通过将视觉动态编码为运动标记，实现了高效和可泛化的世界模型。


<details>
  <summary>Details</summary>
Motivation: 机器人动作标记数据稀缺且昂贵，限制了学习策略的泛化能力。相比之下，大量的无动作视频数据很容易获得，但将这些观察结果转化为有效的策略仍然是一个挑战。

Method: AMPLIFY通过将视觉动态编码成从关键点轨迹导出的紧凑、离散的运动标记，从而利用大规模视频数据。该方法分离了视觉运动预测和动作推理，将学习什么运动定义任务的挑战与机器人如何执行任务的挑战分离开来。

Result: 实验结果表明，所学习的动力学是准确且广泛有用的，与先前的方法相比，MSE提高了3.7倍，像素预测精度提高了2.5倍以上。在下游策略学习中，我们的动态预测在低数据情况下实现了1.2-2.2倍的改进，通过从无动作的人类视频中学习，平均改进了1.4倍，并且首次从零分布内动作数据推广到LIBERO任务。

Conclusion: AMPLIFY提出了一种新范式，利用异构数据源构建高效、可泛化的世界模型。

Abstract: 用于机器人的动作标记数据稀缺且昂贵，限制了学习策略的泛化能力。相比之下，大量的无动作视频数据很容易获得，但将这些观察结果转化为有效的策略仍然是一个挑战。我们介绍了AMPLIFY，这是一种新颖的框架，它通过将视觉动态编码成从关键点轨迹导出的紧凑、离散的运动标记，从而利用大规模视频数据。我们的模块化方法分离了视觉运动预测和动作推理，将学习什么运动定义任务的挑战与机器人如何执行任务的挑战分离开来。我们在大量的无动作视频上训练前向动力学模型，并在有限的动作标记示例上训练逆向动力学模型，从而实现独立扩展。广泛的评估表明，所学习的动力学是准确的，与先前的方法相比，MSE提高了3.7倍，像素预测精度提高了2.5倍以上，并且用途广泛。在下游策略学习中，我们的动态预测在低数据情况下实现了1.2-2.2倍的改进，通过从无动作的人类视频中学习，平均改进了1.4倍，并且首次从零分布内动作数据推广到LIBERO任务。除了机器人控制之外，我们发现AMPLIFY学习的动态是一种通用的潜在世界模型，可以提高视频预测质量。我们的结果提出了一种新范式，利用异构数据源构建高效、可泛化的世界模型。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14198) | **Categories:** cs.RO, cs.CV, cs.LG

---

### [9] [Socially Aware Robot Crowd Navigation via Online Uncertainty-Driven Risk Adaptation](https://arxiv.org/abs/2506.14305)
*Zhirui Sun, Xingrong Diao, Yao Wang, Bi-Ke Zhu, Jiankun Wang*

Main category: cs.RO

TL;DR: LR-MPC 是一种数据驱动的导航算法，通过学习风险来平衡机器人导航中的效率、安全性和社交感知。


<details>
  <summary>Details</summary>
Motivation: 在人机共享的拥挤环境中导航仍然具有挑战性，因为机器人需要高效移动，同时尊重人类的运动习惯。然而，许多现有方法强调安全或效率，而忽略了社交感知。

Method: LR-MPC 包含两个阶段：离线风险学习阶段，在该阶段使用来自基于启发式 MPC 的基线 (HR-MPC) 的风险数据训练概率集成神经网络 (PENN)；以及在线自适应推理阶段，在该阶段采样局部航路点并通过 Multi-RRT 规划器进行全局引导。PENN 评估每个候选航路点的风险，并使用认知和偶然不确定性过滤预测，以确保稳健的决策。选择最安全的航路点作为 MPC 输入以进行实时导航。

Result: 大量实验表明，LR-MPC 在成功率和社会感知方面优于基线方法，使机器人能够在复杂人群中以高适应性和低干扰性导航。

Conclusion: LR-MPC在成功率和社会感知方面优于基线方法，使机器人能够在复杂人群中以高适应性和低干扰性导航。

Abstract: 在人机共享的拥挤环境中导航仍然具有挑战性，因为机器人需要高效移动，同时尊重人类的运动习惯。然而，许多现有方法强调安全或效率，而忽略了社交感知。本文提出了一种学习风险模型预测控制（LR-MPC），这是一种数据驱动的导航算法，可以平衡效率、安全性和社交感知。LR-MPC 包含两个阶段：离线风险学习阶段，在该阶段使用来自基于启发式 MPC 的基线 (HR-MPC) 的风险数据训练概率集成神经网络 (PENN)；以及在线自适应推理阶段，在该阶段采样局部航路点并通过 Multi-RRT 规划器进行全局引导。PENN 评估每个候选航路点的风险，并使用认知和偶然不确定性过滤预测，以确保稳健的决策。选择最安全的航路点作为 MPC 输入以进行实时导航。大量实验表明，LR-MPC 在成功率和社会感知方面优于基线方法，使机器人能够在复杂人群中以高适应性和低干扰性导航。有关此工作的网站位于 https://sites.google.com/view/lr-mpc。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14305) | **Categories:** cs.RO

---

### [10] [Can Pretrained Vision-Language Embeddings Alone Guide Robot Navigation?](https://arxiv.org/abs/2506.14507)
*Nitesh Subedi, Adam Haroon, Shreyan Ganguly, Samuel T. K. Tetteh, Prajwal Koirala, Cody Fleming, Soumik Sarkar*

Main category: cs.RO

TL;DR: 该论文研究了仅使用预训练视觉-语言嵌入进行机器人导航的可行性，结果表明其在基本语言理解方面有效，但在长程规划上存在局限性。


<details>
  <summary>Details</summary>
Motivation: 探讨是否无需额外的微调或专用模块，仅使用预训练的嵌入就能成功引导导航。

Method: 直接在来自特权专家演示的冻结视觉-语言嵌入上训练行为克隆策略。

Result: 在语言指定的导航任务中，该方法达到了 74% 的成功率，但平均步数是专家的 3.2 倍。

Conclusion: 预训练的嵌入能够有效地支持基本的语言理解，但在长程规划和空间推理方面存在不足。

Abstract: 基础模型通过提供丰富的语义表示，无需特定任务的训练，从而彻底改变了机器人技术。虽然许多方法将预训练的视觉-语言模型（VLM）与专门的导航架构相结合，但一个根本问题仍然存在：是否仅凭这些预训练的嵌入就能成功引导导航，而无需额外的微调或专门的模块？我们提出了一个极简框架，通过直接在来自特权专家收集的演示的冻结视觉-语言嵌入上训练行为克隆策略来解耦这个问题。我们的方法在导航到语言指定目标时达到了 74% 的成功率，而了解状态的专家则达到了 100%，尽管平均需要多 3.2 倍的步数。这种性能差距表明，预训练的嵌入能够有效地支持基本的语言理解，但在长程规划和空间推理方面存在不足。通过提供这个经验基线，我们强调了使用基础模型作为具身任务的直接嵌入的能力和局限性，为面临系统复杂性和资源受限场景下的性能之间实际设计权衡的机器人研究人员提供了重要的见解。我们的代码可在 https://github.com/oadamharoon/text2nav 找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14507) | **Categories:** cs.RO

---

### [11] [Casper: Inferring Diverse Intents for Assistive Teleoperation with Vision Language Models](https://arxiv.org/abs/2506.14727)
*Huihan Liu, Rutav Shah, Shuijing Liu, Jack Pittenger, Mingyo Seo, Yuchen Cui, Yonatan Bisk, Roberto Martín-Martín, Yuke Zhu*

Main category: cs.RO

TL;DR: Casper是一个辅助遥操作系统，它利用视觉语言模型中的常识知识进行意图推断和技能执行，从而提升人机协作效率和用户满意度。


<details>
  <summary>Details</summary>
Motivation: 现实世界辅助遥操作的一个核心挑战是机器人需要从用户控制输入中推断出广泛的人类意图，并以正确的动作来辅助用户。现有的方法或者局限于简单的、预定义的场景，或者限制于训练时特定于任务的数据分布，限制了它们对现实世界辅助的支持。

Method: Casper包含一个开放世界感知模块，用于广义理解新颖物体和场景；一个由VLM驱动的意图推断机制，利用常识推理来解释遥操作用户输入的片段；以及一个技能库，扩展了先前辅助遥操作系统的范围，以支持多样化的、长期的移动操作任务。

Result: 大量的经验评估，包括人体研究和系统消融实验，表明Casper提高了任务性能，降低了人类认知负荷，并实现了比直接遥操作和辅助遥操作基线更高的用户满意度。

Conclusion: Casper通过利用预训练的视觉语言模型中的常识知识，提高了任务完成效率，降低了人类认知负荷，并实现了比直接遥操作和辅助遥操作基线更高的用户满意度。

Abstract: 辅助遥操作是一种人与机器人共享控制权的模式，它能够在各种非结构化环境中实现高效、直观的人机协作。现实世界中辅助遥操作面临的核心挑战是，机器人需要从用户的控制输入中推断出广泛的人类意图，并以正确的动作来辅助用户。现有的方法要么局限于简单的、预定义的场景，要么限制于训练时特定于任务的数据分布，这限制了它们对现实世界辅助的支持。我们介绍了一种名为Casper的辅助遥操作系统，该系统利用预训练的视觉语言模型（VLM）中嵌入的常识知识，进行实时意图推断和灵活的技能执行。Casper包含一个开放世界感知模块，用于广义理解新颖物体和场景；一个由VLM驱动的意图推断机制，利用常识推理来解释遥操作用户输入的片段；以及一个技能库，扩展了先前辅助遥操作系统的范围，以支持多样化的、长期的移动操作任务。大量的经验评估，包括人体研究和系统消融实验，表明Casper提高了任务性能，降低了人类认知负荷，并实现了比直接遥操作和辅助遥操作基线更高的用户满意度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14727) | **Categories:** cs.RO, cs.AI

---

### [12] [RobotSmith: Generative Robotic Tool Design for Acquisition of Complex Manipulation Skills](https://arxiv.org/abs/2506.14763)
*Chunru Lin, Haotian Yuan, Yian Wang, Xiaowen Qiu, Tsun-Hsuan Wang, Minghao Guo, Bohan Wang, Yashraj Narang, Dieter Fox, Chuang Gan*

Main category: cs.RO

TL;DR: RobotSmith 提出了一种自动化流程，利用视觉语言模型和物理模拟为机器人设计和使用工具，并在各种操作任务中表现出色。


<details>
  <summary>Details</summary>
Motivation: 赋予机器人工具设计能力对于使它们能够解决复杂的、原本难以处理的操作任务至关重要。简单地检索人类设计的工具可能并不理想，因为许多工具（例如，擀面杖）对于机器人操纵器来说难以操作。此外，现有的工具设计方法要么依赖于具有有限参数调整的预定义模板，要么应用未针对工具创建进行优化的通用 3D 生成方法。

Method: 提出了一种名为 RobotSmith 的自动化流程，该流程利用视觉语言模型 (VLM) 中嵌入的隐式物理知识以及物理模拟提供的更精确的物理知识来设计和使用机器人操作工具。该系统 (1) 使用协作 VLM 代理迭代地提出工具设计，(2) 生成用于工具使用的低级机器人轨迹，以及 (3) 共同优化工具几何形状和使用以提高任务性能。

Result: 在涉及刚性、可变形和流体对象的各种操作任务中评估了该方法。实验表明，该方法在任务成功率和整体性能方面始终优于强大的基线。值得注意的是，该方法实现了 50.0% 的平均成功率，大大超过了其他基线，例如 3D 生成 (21.4%) 和工具检索 (11.1%)。

Conclusion: 该方法在各种操作任务中始终优于强大的基线，在任务成功率和整体性能方面均优于其他方法。在现实环境中部署表明，生成的工具及其使用计划可以有效地转移到物理执行中，从而验证了该方法的实用性和泛化能力。

Abstract: 赋予机器人工具设计能力对于使它们能够解决复杂的、原本难以处理的操作任务至关重要。虽然最近的生成框架可以自动合成任务设置，例如 3D 场景和奖励函数，但它们尚未解决工具使用场景的挑战。简单地检索人类设计的工具可能并不理想，因为许多工具（例如，擀面杖）对于机器人操纵器来说难以操作。此外，现有的工具设计方法要么依赖于具有有限参数调整的预定义模板，要么应用未针对工具创建进行优化的通用 3D 生成方法。为了解决这些限制，我们提出了一种名为 RobotSmith 的自动化流程，该流程利用视觉语言模型 (VLM) 中嵌入的隐式物理知识以及物理模拟提供的更精确的物理知识来设计和使用机器人操作工具。我们的系统 (1) 使用协作 VLM 代理迭代地提出工具设计，(2) 生成用于工具使用的低级机器人轨迹，以及 (3) 共同优化工具几何形状和使用以提高任务性能。我们在涉及刚性、可变形和流体对象的各种操作任务中评估了我们的方法。实验表明，我们的方法在任务成功率和整体性能方面始终优于强大的基线。值得注意的是，我们的方法实现了 50.0% 的平均成功率，大大超过了其他基线，例如 3D 生成 (21.4%) 和工具检索 (11.1%)。最后，我们在现实环境中部署了我们的系统，证明了生成的工具及其使用计划可以有效地转移到物理执行中，从而验证了我们的方法的实用性和泛化能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14763) | **Categories:** cs.RO

---


## stat.AP [stat.AP]
### [1] [Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior](https://arxiv.org/abs/2506.14762)
*Chengyuan Zhang, Cathy Wu, Lijun Sun*

Main category: stat.AP

TL;DR: 该论文提出了一个基于因子隐马尔可夫模型（FHMM-IDM）的机制转换框架，用于建模人类驾驶行为，能够有效解耦驾驶员内在行为与外部交通环境。


<details>
  <summary>Details</summary>
Motivation: 传统的车辆跟随模型（如智能驾驶员模型IDM）受限于其简约和单机制结构，无法捕捉人类驾驶的多模态特性。在单一驾驶状态下，人类驾驶员会产生许多不同的驾驶行为，这迫使模型平均不同的行为，从而降低了模型的保真度，并使其参数难以解释。

Method: 提出了一个机制转换框架，该框架允许驾驶行为由不同的IDM参数集控制，每个参数集对应于一种可解释的行为模式。使用具有IDM动态的因子隐马尔可夫模型（FHMM-IDM）实例化该框架，该模型通过两个独立的潜在马尔可夫过程，将内在驾驶机制（例如，激进加速、稳态跟随）与外部交通场景（例如，自由流动、拥堵、走走停停）显式分离。

Result: 在HighD数据集上的实验表明，FHMM-IDM能够揭示人类驾驶中可解释的结构，有效地从上下文交通状况中解耦内部驾驶员行为，并揭示动态的机制转换模式。

Conclusion: FHMM-IDM模型能够有效地从上下文交通状况中解耦内部驾驶员行为，并揭示动态的机制转换模式。

Abstract: 精确且可解释的车辆跟随模型对于交通仿真和自动驾驶车辆的开发至关重要。然而，经典的车辆跟随模型（如智能驾驶员模型IDM）受限于其简约和单机制结构。它们无法捕捉人类驾驶的多模态特性，即在单一驾驶状态（例如，速度、相对速度和间隙）下，人类驾驶员会产生许多不同的驾驶行为。这迫使模型平均不同的行为，从而降低了模型的保真度，并使其参数难以解释。为了克服这个问题，我们引入了一个机制转换框架，该框架允许驾驶行为由不同的IDM参数集控制，每个参数集对应于一种可解释的行为模式。这种设计使模型能够在可解释的行为模式之间动态切换，而不是平均不同的驾驶环境。我们使用具有IDM动态的因子隐马尔可夫模型（FHMM-IDM）实例化该框架，该模型通过两个独立的潜在马尔可夫过程，将内在驾驶机制（例如，激进加速、稳态跟随）与外部交通场景（例如，自由流动、拥堵、走走停停）显式分离。通过马尔可夫链蒙特卡罗（MCMC）的贝叶斯推断用于联合估计特定于机制的参数、转换动态和潜在状态轨迹。在HighD数据集上的实验表明，FHMM-IDM能够揭示人类驾驶中可解释的结构，有效地从上下文交通状况中解耦内部驾驶员行为，并揭示动态的机制转换模式。该框架为在不确定性下建模依赖于上下文的驾驶行为提供了一个易于处理且有原则的解决方案，从而提高了交通仿真的保真度、安全分析的有效性以及更以人为本的ADAS的开发。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14762) | **Categories:** stat.AP, cs.LG, cs.RO

---
