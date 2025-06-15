# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-16

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算语言学 (Computation and Language) (1)](#cs-cl)
- [计算机视觉 (Computer Vision) (8)](#cs-cv)
- [cs.DC (1)](#cs-dc)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [cs.MM (1)](#cs-mm)
- [cs.NI (1)](#cs-ni)
- [机器人学 (Robotics) (7)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models](https://arxiv.org/abs/2506.10853)
*Yu Zhang, Yang Hu, De Wang*

Main category: cs.AI

TL;DR: 该论文提出了一种结合CoT推理和MCP的框架，以提升LLM在模拟人类时空行为方面的能力。


<details>
  <summary>Details</summary>
Motivation: 传统方法在模拟人类时空行为方面存在计算成本高、泛化能力有限和可扩展性差的问题。大型语言模型（LLM）虽然有潜力作为“世界模拟器”，但在时空推理方面面临挑战，包括空间认知有限、缺乏物理约束理解和群体同质化倾向。

Method: 该论文提出了一个结合思维链（CoT）推理和模型上下文协议（MCP）的框架，通过五阶段认知框架和六类MCP工具来实现对人类时空行为的模拟。

Result: 在上海陆家嘴地区的实验验证了该框架的有效性，生成结果与真实移动信令数据高度相似，不同基础模型的生成质量得分达到7.86至8.36。并行处理实验显示，生成时间从每样本1.30分钟减少到0.17分钟。

Conclusion: 该研究证明了结合CoT推理和MCP框架能够有效提升LLM在模拟人类时空行为方面的能力，为城市计算提供了一种可行的方法。

Abstract: 人类时空行为模拟对于城市规划研究至关重要，但传统的基于规则和统计的方法存在计算成本高、泛化能力有限和可扩展性差的问题。虽然大型语言模型（LLM）作为“世界模拟器”展现出潜力，但它们在时空推理方面面临挑战，包括空间认知有限、缺乏物理约束理解和群体同质化倾向。本文介绍了一个框架，该框架集成了思维链（CoT）推理与模型上下文协议（MCP），以增强LLM在模拟与验证数据模式相符的时空行为方面的能力。该方法结合了类人渐进推理（通过五阶段认知框架）与全面的数据处理（通过六个专业的MCP工具类别：时间管理、空间导航、环境感知、个人记忆、社会协作和经验评估）。在上海陆家嘴地区的实验验证了该框架在1000个生成样本中的有效性。结果表明，该框架与真实的移动信令数据具有高度相似性，在不同的基础模型上实现了7.86至8.36的生成质量评分。并行处理实验显示出效率的提升，当从2个进程扩展到12个进程时，每个样本的生成时间从1.30分钟减少到0.17分钟。这项工作有助于将CoT推理与MCP集成，以进行城市行为建模，从而推进LLM在城市计算中的应用，并为合成移动性数据的生成提供了一种实用的方法。该框架为智能城市规划、交通预测和参与式城市设计应用奠定了基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10853) | **Categories:** cs.AI, cs.CY

---

### [2] [Mirage-1: Augmenting and Updating GUI Agent with Hierarchical Multimodal Skills](https://arxiv.org/abs/2506.10387)
*Yuquan Xie, Zaijing Li, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Dongmei Jiang, Liqiang Nie*

Main category: cs.AI

TL;DR: Mirage-1通过分层多模态技能和技能增强的蒙特卡洛树搜索，显著提升了GUI智能体在长程任务中的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的多模态大型语言模型（MLLM）作为GUI智能体在在线环境中的长程任务中表现不佳，主要是由于知识不足以及离线和在线领域之间存在差距。

Method: 提出了一种分层多模态技能（HMS）模块和技能增强的蒙特卡洛树搜索（SA-MCTS）算法。

Result: Mirage-1在AndroidWorld、MobileMiniWob++、Mind2Web-Live和AndroidLH上分别比之前的智能体高出32%、19%、15%和79%。

Conclusion: Mirage-1在多个GUI任务中表现优于现有智能体，尤其是在长程任务中。

Abstract: 最近利用多模态大型语言模型（MLLM）作为GUI智能体的尝试已经取得了有希望的成果。然而，这些智能体仍然在在线环境中的长程任务中挣扎，主要是由于知识不足以及离线和在线领域之间固有的差距。在本文中，受到人类如何在开放环境中泛化知识的启发，我们提出了一个分层多模态技能（HMS）模块，以解决知识不足的问题。它逐步将轨迹抽象为执行技能、核心技能，最终抽象为元技能，为长程任务规划提供分层知识结构。为了弥合领域差距，我们提出了技能增强的蒙特卡洛树搜索（SA-MCTS）算法，该算法有效地利用了在离线环境中获得的技能，以减少在线树探索期间的动作搜索空间。在HMS的基础上，我们提出了Mirage-1，一个多模态、跨平台、即插即用的GUI智能体。为了验证Mirage-1在真实长程场景中的性能，我们构建了一个新的基准，AndroidLH。实验结果表明，Mirage-1在AndroidWorld、MobileMiniWob++、Mind2Web-Live和AndroidLH上分别比之前的智能体高出32%、19%、15%和79%。项目页面：https://cybertronagent.github.io/Mirage-1.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10387) | **Categories:** cs.AI

---

### [3] [GenPlanX. Generation of Plans and Execution](https://arxiv.org/abs/2506.10897)
*Daniel Borrajo, Giuseppe Canonaco, Tomás de la Rosa, Alfredo Garrachón, Sriram Gopalakrishnan, Simerjot Kaur, Marianela Morales, Sunandita Patra, Alberto Pozanco, Keshav Ramani, Charese Smiley, Pietro Totis, Manuela Veloso*

Main category: cs.AI

TL;DR: GenPlanX 集成了 LLM 和经典 AI 规划引擎，以实现对自然语言规划任务的理解和执行。


<details>
  <summary>Details</summary>
Motivation: 传统的 AI 规划技术缺乏理解自然语言描述的规划任务的能力。大型语言模型 (LLM) 的出现为人机交互带来了新的能力。

Method: GenPlanX 结合了 LLM 用于理解自然语言规划任务描述，并结合了经典的 AI 规划引擎以及执行和监控框架。

Result: GenPlanX 在协助用户完成办公室相关任务方面表现出有效性，并通过无缝的人工智能协作，突出了其简化工作流程和提高生产力的潜力。

Conclusion: GenPlanX 通过将 LLM 与经典 AI 规划引擎集成，实现了对基于自然语言描述的规划任务的处理，并通过执行和监控框架提高了人机协作效率。

Abstract: 传统的 AI 规划技术擅长生成复杂任务的行动序列，但缺乏理解自然语言描述的规划任务的能力。大型语言模型 (LLM) 的出现，为人机交互带来了新的能力。在规划任务的背景下，LLM 在理解人类意图方面表现出色。本文介绍了 GenPlanX，它集成了 LLM，用于理解基于自然语言的规划任务描述，并结合了经典的 AI 规划引擎以及执行和监控框架。我们通过 GenPlanX 在协助用户完成办公室相关任务中的有效性，突出了其通过无缝的人工智能协作来简化工作流程和提高生产力的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10897) | **Categories:** cs.AI

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](https://arxiv.org/abs/2506.10887)
*Yixiao Huang, Hanlin Zhu, Tianyu Guo, Jiantao Jiao, Somayeh Sojoudi, Michael I. Jordan, Stuart Russell, Song Mei*

Main category: cs.CL

TL;DR: 该论文揭示了大型语言模型中上下文推理驱动泛化和幻觉，并提供了理论解释。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型可以通过微调获得新知识，但这个过程表现出一种令人困惑的二元性：模型可以从新事实中显著地概括，但也容易产生错误的幻觉信息。然而，这种现象的原因仍然知之甚少。

Method: 作者将上下文推理形式化为一个合成的事实回忆任务，并通过实验证明，具有分解输出和值矩阵的单层单头注意力Transformer可以学习解决此任务。

Result: 实验结果表明，上下文推理确实驱动了泛化和幻觉，这取决于相关概念是否具有因果关系。理论分析表明，上下文推理能力可归因于梯度下降的隐式偏差，梯度下降倾向于最小化组合输出-值矩阵的核范数的解决方案。

Conclusion: 该研究为理解大型语言模型中的上下文推理现象提供了理论基础，并为分析和缓解知识注入中不良行为提供了一个新的视角。

Abstract: 大型语言模型（LLM）可以通过微调获得新知识，但这个过程表现出一种令人困惑的二元性：模型可以从新事实中显著地概括，但也容易产生错误的幻觉信息。然而，这种现象的原因仍然知之甚少。在这项工作中，我们认为这两种行为都源于一种称为上下文推理（OCR）的单一机制：通过关联概念来推断含义的能力，即使这些概念之间没有因果联系。我们对五个著名LLM的实验证实，上下文推理确实驱动了泛化和幻觉，这取决于相关概念是否具有因果关系。为了对这种现象建立严格的理论理解，我们将上下文推理形式化为一个合成的事实回忆任务。我们通过实验表明，具有分解输出和值矩阵的单层单头注意力Transformer可以学习解决此任务，而具有组合权重的模型则不能，这突出了矩阵分解的关键作用。我们的理论分析表明，上下文推理能力可归因于梯度下降的隐式偏差，梯度下降倾向于最小化组合输出-值矩阵的核范数的解决方案。这种数学结构解释了为什么模型学习以高样本效率关联事实和含义，无论相关性是因果关系还是仅仅是虚假的。最终，我们的工作为理解上下文推理现象提供了理论基础，并为分析和缓解知识注入中不良行为提供了一个新的视角。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10887) | **Categories:** cs.CL, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [RoCA: Robust Cross-Domain End-to-End Autonomous Driving](https://arxiv.org/abs/2506.10145)
*Rajeev Yasarla, Shizhong Han, Hsin-Pai Cheng, Litian Liu, Shweta Mahajan, Apratim Bhattacharyya, Yunxiao Shi, Risheek Garrepalli, Hong Cai, Fatih Porikli*

Main category: cs.CV

TL;DR: RoCA通过学习token表示和概率推理，实现了鲁棒的跨域端到端自动驾驶。


<details>
  <summary>Details</summary>
Motivation: 现有端到端自动驾驶方法在跨域部署时面临挑战，大型语言模型虽然具有开放世界知识，但不能保证跨域驾驶性能，且领域自适应期间的再训练成本高昂。

Method: RoCA通过高斯过程学习一组基础tokens及其轨迹，这些tokens编码了自我车辆和周围车辆的信息，从而对未来轨迹进行概率推断。

Result: 在各种跨域场景下进行的广泛评估表明，RoCA实现了强大的领域泛化和适应性能。

Conclusion: RoCA通过学习跨越不同驾驶场景的基础tokens及其轨迹，提高了E2E模型的泛化能力，并在新目标域上实现了鲁棒的适应。

Abstract: 端到端（E2E）自动驾驶最近作为一种新的范例出现，具有巨大的潜力。然而，很少有研究关注跨领域（例如，城市）部署的实际挑战。虽然一些工作已经结合大型语言模型（LLM）来利用其开放世界知识，但LLM不能保证跨领域驾驶性能，并且可能在领域适应期间产生过高的再训练成本。在本文中，我们提出RoCA，一种用于鲁棒的跨领域E2E自动驾驶的新颖框架。RoCA制定了E2E管道中编码自我车辆和周围车辆信息的tokens的联合概率分布。通过使用高斯过程（GP）进行实例化，RoCA学习一组具有相应轨迹的基础tokens，这些tokens跨越不同的驾驶场景。然后，给定任何驾驶场景，它能够概率性地推断未来轨迹。通过在源域训练中使用RoCA和基础E2E模型，我们提高了基础模型的泛化能力，而无需额外的推理计算。此外，RoCA能够在新的目标域上实现鲁棒的适应，显著优于直接微调。我们广泛地在各种跨领域场景中评估RoCA，并表明它实现了强大的领域泛化和适应性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10145) | **Categories:** cs.CV

---

### [2] [Motion-R1: Chain-of-Thought Reasoning and Reinforcement Learning for Human Motion Generation](https://arxiv.org/abs/2506.10353)
*Runqi Ouyang, Haoyun Li, Zhenyuan Zhang, Xiaofeng Wang, Zheng Zhu, Guan Huang, Xingang Wang*

Main category: cs.CV

TL;DR: Motion-R1通过集成Chain-of-Thought机制和Group Relative Policy Optimization，提升了文本到动作生成的可控性、一致性和多样性。


<details>
  <summary>Details</summary>
Motivation: 现有方法无法捕捉深层的语言结构和逻辑推理，导致生成的动作缺乏可控性、一致性和多样性。

Method: Motion-R1集成了Chain-of-Thought机制，将复杂的文本指令分解为逻辑结构化的动作路径，并采用Group Relative Policy Optimization进行训练。

Result: Motion-R1在需要细致的语义理解和长期时间连贯性的场景中，相比现有技术取得了有竞争力的或更优越的性能。

Conclusion: Motion-R1在多个基准数据集上表现出色，尤其是在需要细致的语义理解和长期时间连贯性的场景中。

Abstract: 大型语言模型在自然语言理解和推理方面的最新进展为文本到动作生成开辟了新的可能性。虽然现有的方法在语义对齐和动作合成方面取得了显著进展，但它们通常依赖于端到端的映射策略，而这些策略未能捕捉到深层的语言结构和逻辑推理。因此，生成的动作往往缺乏可控性、一致性和多样性。为了解决这些局限性，我们提出了Motion-R1，一个统一的动作-语言建模框架，它集成了Chain-of-Thought机制。通过将复杂的文本指令显式地分解为逻辑结构化的动作路径，Motion-R1为动作生成提供了高层次的语义指导，显著增强了模型解释和执行多步骤、长时程和组合丰富的命令的能力。为了训练我们的模型，我们采用了Group Relative Policy Optimization，这是一种为大型模型设计的强化学习算法，它利用动作质量反馈来联合优化推理链和动作合成。在多个基准数据集上进行的大量实验表明，Motion-R1相比于最先进的方法，取得了有竞争力的或更优越的性能，特别是在需要细致的语义理解和长期时间连贯性的场景中。代码、模型和数据将公开提供。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10353) | **Categories:** cs.CV

---

### [3] [Test-Time Adaptation for Generalizable Task Progress Estimation](https://arxiv.org/abs/2506.10085)
*Christos Ziakas, Alessandra Russo*

Main category: cs.CV

TL;DR: 本文提出了一种测试时自适应方法，通过优化自监督目标，使进度估计模型能够在线适应测试轨迹，从而提高性能。


<details>
  <summary>Details</summary>
Motivation: 我们提出了一种测试时自适应方法，该方法通过优化学习到的自监督目标，使进度估计模型能够在线适应测试轨迹的视觉和时间上下文。

Method: 我们提出了一种测试时自适应方法，该方法通过优化学习到的自监督目标，使进度估计模型能够在线适应测试轨迹的视觉和时间上下文。

Result: 我们的测试时自适应方法优于使用自回归视觉语言模型的最新上下文学习方法。

Conclusion: 该测试时自适应方法能够从单个训练环境推广到各种各样的分布外任务、环境和实施方式，优于使用自回归视觉语言模型的最新上下文学习方法。

Abstract: 我们提出了一种测试时自适应方法，该方法通过优化学习到的自监督目标，使进度估计模型能够在线适应测试轨迹的视觉和时间上下文。为此，我们引入了一种基于梯度的元学习策略，以在专家视觉轨迹及其自然语言任务描述上训练模型，从而使测试时自适应能够提高依赖于语义内容而非时间顺序的进度估计。我们的测试时自适应方法可以从单个训练环境推广到各种各样的分布外任务、环境和实施方式，优于使用自回归视觉语言模型的最新上下文学习方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10085) | **Categories:** cs.CV, cs.AI, I.2.6; I.2.9; I.2.10

---

### [4] [EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models](https://arxiv.org/abs/2506.10100)
*Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, Linfeng Zhang*

Main category: cs.CV

TL;DR: EfficientVLA是一个VLA推理加速框架，通过剪枝、优化视觉处理和缓存中间特征，显著提高了推理速度和效率。


<details>
  <summary>Details</summary>
Motivation: VLA模型在具身智能方面展示了变革的潜力，但由于广泛的固有和推理时冗余导致的高计算和内存需求而受到严重阻碍。现有的加速工作通常针对孤立的低效率，这种零散的解决方案通常不能全面地解决整个VLA管道中各种计算和内存瓶颈，从而限制了实际的可部署性。

Method: EfficientVLA通过协同整合三种有针对性的策略来系统地消除这些障碍：(1) 剪枝语言模块中功能上无关紧要的层，由层间冗余分析指导；(2) 通过任务感知策略优化视觉处理路径，该策略选择一个紧凑、多样化的视觉token集合，平衡任务关键性和信息覆盖；(3) 通过战略性地缓存和重用关键的中间特征，减轻迭代的基于扩散的动作头中的时间计算冗余。

Result: EfficientVLA在CogACT模型上实现了1.93倍的推理加速，并将FLOPs降低到28.9%，同时在SIMPLER基准测试中仅损失了0.6%的成功率。

Conclusion: 通过在CogACT模型上的实验，EfficientVLA在SIMPLER基准测试中仅损失0.6%的成功率的情况下，实现了1.93倍的推理加速，并将FLOPs降低到28.9%。

Abstract: 视觉-语言-动作（VLA）模型，特别是基于扩散的架构，在具身智能方面展示了变革的潜力，但由于广泛的固有和推理时冗余导致的高计算和内存需求而受到严重阻碍。现有的加速工作通常针对孤立的低效率，这种零散的解决方案通常不能全面地解决整个VLA管道中各种计算和内存瓶颈，从而限制了实际的可部署性。我们介绍了一种结构化的、无需训练的推理加速框架EfficientVLA，它通过协同利用多方面的冗余来系统地消除这些障碍。EfficientVLA协同整合了三种有针对性的策略：(1) 剪枝语言模块中功能上无关紧要的层，由层间冗余分析指导；(2) 通过任务感知策略优化视觉处理路径，该策略选择一个紧凑、多样化的视觉token集合，平衡任务关键性和信息覆盖；(3) 通过战略性地缓存和重用关键的中间特征，减轻迭代的基于扩散的动作头中的时间计算冗余。我们将我们的方法应用于标准VLA模型CogACT，在SIMPLER基准测试中仅损失0.6%的成功率的情况下，实现了1.93倍的推理加速，并将FLOPs降低到28.9%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10100) | **Categories:** cs.CV

---

### [5] [DySS: Dynamic Queries and State-Space Learning for Efficient 3D Object Detection from Multi-Camera Videos](https://arxiv.org/abs/2506.10242)
*Rajeev Yasarla, Shizhong Han, Hong Cai, Fatih Porikli*

Main category: cs.CV

TL;DR: DySS通过状态空间学习和动态查询，实现了高效且高性能的相机BEV 3D目标检测。


<details>
  <summary>Details</summary>
Motivation: 现有基于相机的BEV 3D目标检测方法依赖于密集的BEV特征，构建成本高昂，或者需要大量查询，当使用更多视频帧时，运行成本会变得很高。

Method: DySS采用状态空间模型（SSM）按时间步长顺序处理采样特征，并通过合并、移除和分割操作动态更新查询。

Result: DySS在nuScenes测试集上达到65.31 NDS和57.4 mAP，在验证集上达到56.2 NDS和46.2 mAP，以及33 FPS的实时推理速度。

Conclusion: DySS在nuScenes数据集上实现了最先进的3D目标检测性能，并在验证集上达到了实时推理速度。

Abstract: 基于相机的鸟瞰图（BEV）3D目标检测是自动驾驶中最重要的感知任务之一。早期的方法依赖于密集的BEV特征，这导致构建成本很高。最近的研究探索了基于稀疏查询的检测方法。然而，它们仍然需要大量的查询，并且当使用更多的视频帧时，运行成本会变得很高。在本文中，我们提出了一种新颖的方法DySS，它采用了状态空间学习和动态查询。更具体地说，DySS利用状态空间模型（SSM）按时间步长顺序处理采样的特征。为了鼓励模型更好地捕获潜在的运动和对应关系信息，我们引入了未来预测和掩码重建的辅助任务，以更好地训练SSM。然后，SSM的状态提供了场景的信息丰富但高效的概括。基于状态空间学习的特征，我们通过合并、移除和分割操作动态更新查询，这有助于在整个网络中维护一个有用、精简的检测查询集。我们提出的DySS实现了卓越的检测性能和高效的推理。具体来说，在nuScenes测试集上，DySS达到了65.31 NDS和57.4 mAP，优于最新的技术水平。在val split上，DySS达到了56.2 NDS和46.2 mAP，以及33 FPS的实时推理速度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10242) | **Categories:** cs.CV

---

### [6] [DanceChat: Large Language Model-Guided Music-to-Dance Generation](https://arxiv.org/abs/2506.10574)
*Qing Wang, Xiaohang Yang, Yilan Dong, Naveen Raj Govindaraj, Gregory Slabaugh, Shanxin Yuan*

Main category: cs.CV

TL;DR: DanceChat利用大型语言模型引导音乐到舞蹈的生成，通过文本指令显式地指导舞蹈动作，从而生成更多样化和更符合音乐风格的舞蹈。


<details>
  <summary>Details</summary>
Motivation: 音乐到舞蹈的生成旨在合成以音乐输入为条件的人类舞蹈动作。由于音乐和舞蹈动作之间的语义差距，以及音乐只能提供抽象的线索，例如旋律、节奏和情感，而没有明确指定物理动作，因此仍然存在巨大的挑战。此外，一个单一的音乐可以产生多种合理的舞蹈解释。这种一对多的映射需要额外的指导，因为仅靠音乐为生成各种舞蹈动作提供的信息有限。配对的音乐和舞蹈数据的稀缺性进一步加剧了这一挑战，这限制了模型学习各种舞蹈模式的能力。

Method: 我们提出了DanceChat，一个基于大型语言模型（LLM）引导的音乐到舞蹈生成方法。该方法包括三个模块：（1）一个基于LLM的伪指令生成模块，该模块根据音乐风格和结构生成文本舞蹈指导；（2）一个多模态特征提取和融合模块，该模块将音乐、节奏和文本指导整合到一个共享表示中；（3）一个基于扩散的运动合成模块，以及一个多模态对齐损失，确保生成的舞蹈与音乐和文本提示对齐。

Result: 在AIST++上的大量实验和人工评估表明，DanceChat在质量和数量上都优于目前最好的方法。

Conclusion: DanceChat在AIST++数据集上和人工评估中，都比目前最好的方法表现更好，在质量和数量上都超过了它们。

Abstract: 音乐到舞蹈的生成旨在合成以音乐输入为条件的人类舞蹈动作。尽管最近取得了进展，但由于音乐和舞蹈动作之间的语义差距，仍然存在巨大的挑战，因为音乐只提供抽象的线索，如旋律、节奏和情感，而没有明确指定物理动作。此外，一个单一的音乐可以产生多种合理的舞蹈解释。这种一对多的映射需要额外的指导，因为仅靠音乐为生成各种舞蹈动作提供的信息有限。配对的音乐和舞蹈数据的稀缺性进一步加剧了这一挑战，这限制了模型学习各种舞蹈模式的能力。在本文中，我们介绍了一种大型语言模型（LLM）引导的音乐到舞蹈生成方法DanceChat。我们使用LLM作为编舞，提供文本运动指令，为舞蹈生成提供明确的、高层次的指导。这种方法超越了仅从音乐中进行隐式学习，使模型能够生成更多样化且与音乐风格更好对齐的舞蹈。我们的方法包括三个组成部分：（1）一个基于LLM的伪指令生成模块，该模块根据音乐风格和结构生成文本舞蹈指导；（2）一个多模态特征提取和融合模块，该模块将音乐、节奏和文本指导整合到一个共享表示中；（3）一个基于扩散的运动合成模块，以及一个多模态对齐损失，确保生成的舞蹈与音乐和文本提示对齐。在AIST++上的大量实验和人工评估表明，DanceChat在质量和数量上都优于目前最好的方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10574) | **Categories:** cs.CV, cs.MM, cs.SD, eess.AS

---

### [7] [Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework](https://arxiv.org/abs/2506.10685)
*Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo*

Main category: cs.CV

TL;DR: 本文提出了一种新颖的无源对抗验证码（UAC）框架，通过文本提示生成高保真对抗样本，有效攻击现有验证码系统。


<details>
  <summary>Details</summary>
Motivation: 随着深度学习的快速发展，传统的 CAPTCHA 方案越来越容易受到深度神经网络 (DNN) 驱动的自动攻击。现有的对抗性攻击方法通常依赖于原始图像特征，导致扭曲，从而阻碍了人类的解释，并限制了在缺乏初始输入图像的场景中的适用性。

Method: 提出了一种新颖的框架，即无源对抗验证码 (UAC)，该框架生成由攻击者指定的文本提示引导的高保真对抗性示例。利用大型语言模型 (LLM)，UAC 增强了验证码的多样性，并支持有针对性和无针对性的攻击。对于有针对性的攻击，EDICT 方法优化了扩散模型中的双重潜在变量，以获得卓越的图像质量。在无针对性的攻击中，特别是对于黑盒场景，我们引入了双路径无源对抗验证码 (BP-UAC)，这是一种两步优化策略，采用多模态梯度和双路径优化来实现有效的错误分类。

Result: 实验表明，BP-UAC 在各种系统中实现了高攻击成功率，生成了对人类和 DNN 来说都无法区分的自然验证码。

Conclusion: BP-UAC 在各种系统中实现了高攻击成功率，生成了对人类和 DNN 来说都无法区分的自然验证码。

Abstract: 随着深度学习的快速发展，传统的验证码方案越来越容易受到深度神经网络（DNN）驱动的自动攻击。现有的对抗攻击方法通常依赖于原始图像特征，导致图像扭曲，阻碍人类识别，并限制了在缺少初始输入图像的场景中的适用性。为了解决这些挑战，我们提出了一种新颖的无源对抗验证码（UAC）框架，该框架生成由攻击者指定的文本提示引导的高保真对抗样本。通过利用大型语言模型（LLM），UAC 增强了验证码的多样性，并支持有针对性和无针对性的攻击。对于有针对性的攻击，EDICT方法优化扩散模型中的双重潜在变量，以获得卓越的图像质量。在无目标攻击中，特别是对于黑盒场景，我们引入了双路径无源对抗验证码（BP-UAC），这是一种两步优化策略，采用多模态梯度和双路径优化来实现有效的错误分类。实验表明，BP-UAC 在各种系统中实现了高攻击成功率，生成了对人类和 DNN 来说都无法区分的自然验证码。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10685) | **Categories:** cs.CV, cs.CR

---

### [8] [SlotPi: Physics-informed Object-centric Reasoning Models](https://arxiv.org/abs/2506.10778)
*Jian Li, Wan Han, Ning Lin, Yu-Liang Zhan, Ruizhi Chengze, Haining Wang, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, Hao Sun*

Main category: cs.CV

TL;DR: SlotPi是一种基于槽的物理信息物体中心推理模型，它集成了物理模块和时空预测模块，在动态预测方面表现出强大的适应性。


<details>
  <summary>Details</summary>
Motivation: 目前，以物体为中心的动态模拟方法忽略了两个关键方面：1）将物理知识整合到模型中；2）验证模型在不同场景中的适应性。

Method: SlotPi集成了基于哈密顿原理的物理模块和用于动态预测的时空预测模块。

Result: 实验突出了该模型在基准和流体数据集上的预测和视觉问答（VQA）等任务中的优势。此外，我们创建了一个包含对象交互、流体动力学和流体-对象交互的真实世界数据集，我们验证了我们模型的能力。

Conclusion: 该模型在所有数据集上表现出强大的性能，突显了其强大的适应性，为开发更高级的世界模型奠定了基础。

Abstract: 通过视觉观察理解和推理受物理定律支配的动力学，类似于人类在现实世界中的能力，提出了重大的挑战。目前，以物体为中心的动态模拟方法虽然模拟了人类的行为，取得了显著的进展，但忽略了两个关键方面：1）将物理知识整合到模型中。人类通过观察世界获得物理方面的洞察力，并将这些知识应用于准确地推理各种动态场景；2）验证模型在不同场景中的适应性。现实世界的动力学，特别是那些涉及流体和物体的动力学，需要模型不仅能捕捉物体间的相互作用，还能模拟流体流动特性。为了解决这些差距，我们引入了SlotPi，一个基于槽的物理信息物体中心推理模型。SlotPi集成了基于哈密顿原理的物理模块和用于动态预测的时空预测模块。我们的实验突出了该模型在基准和流体数据集上的预测和视觉问答（VQA）等任务中的优势。此外，我们创建了一个包含对象交互、流体动力学和流体-对象交互的真实世界数据集，我们验证了我们模型的能力。该模型在所有数据集上表现出强大的性能，突显了其强大的适应性，为开发更高级的世界模型奠定了基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10778) | **Categories:** cs.CV, cs.AI, cs.LG

---


## cs.DC [cs.DC]
### [1] [HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration](https://arxiv.org/abs/2506.10401)
*Jiaqi Lv, Xufeng He, Yanchen Liu, Xu Dai, Yang Hu, Shouyi Yin*

Main category: cs.DC

TL;DR: 该论文提出了一个新框架，利用AI编译器和图数据增强来提高LLM在CUDA转译中的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的方法在工作负载覆盖率和泛化性方面面临限制，并且通常会产生大量的开发成本。此外，现有的LLM在CUDA转译中的性能，特别是在高性能代码方面，仍然欠佳。主要原因是缺乏高质量的训练数据集。

Method: 利用AI编译器和自动优化技术，提出了一个用于生成高性能CUDA和相应平台代码对的新框架，并采用基于图的数据增强方法。

Result: 实验结果表明，该框架显著提高了CUDA转换的性能。

Conclusion: 实验结果表明，该框架显著提高了CUDA转换的性能，突显了LLM在解决CUDA生态系统中的兼容性挑战方面的潜力。

Abstract: 深度学习的快速发展推动了模型参数和计算需求的指数级增长。NVIDIA GPU及其基于CUDA的软件生态系统为并行计算提供了强大的支持，大大缓解了计算瓶颈。同时，由于用户编程习惯的培养和GPU的高性能，CUDA生态系统在并行软件领域占据了主导地位。这种主导地位要求其他硬件平台支持具有性能可移植性的基于CUDA的软件。然而，由于并行编程范式和硬件架构的差异，将CUDA代码转换为其他平台带来了巨大的挑战。现有的方法依赖于语言扩展、领域特定语言（DSL）或编译器，但在工作负载覆盖率和泛化性方面面临限制。此外，这些方法通常会产生大量的开发成本。最近，LLM在各种垂直领域，尤其是在代码相关任务中，表现出了非凡的潜力。然而，现有的LLM在CUDA转译中的性能，特别是在高性能代码方面，仍然欠佳。这种限制的主要原因在于缺乏高质量的训练数据集。为了应对这些挑战，我们提出了一个新颖的框架，该框架利用AI编译器和自动优化技术来生成高性能CUDA和相应的平台代码对。我们进一步利用基于图的数据增强方法增强了该框架，并推出了HPCTransEval，这是一个用于评估LLM在CUDA转译方面的性能的基准。我们以CUDA到CPU的转译作为案例，对领先的LLM进行了实验。结果表明，我们的框架显著提高了CUDA转译的性能，突显了LLM在解决CUDA生态系统中的兼容性挑战方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10401) | **Categories:** cs.DC, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices](https://arxiv.org/abs/2506.10443)
*Zhaode Wang, Jingbang Yang, Xinyu Qian, Shiwen Xing, Xiaotang Jiang, Chengfei Lv, Shengyu Zhang*

Main category: cs.LG

TL;DR: MNN-LLM是一个加速大型语言模型在移动设备上部署的框架，通过模型量化和优化计算策略，实现了显著的性能提升。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型规模庞大，推理时计算资源消耗高，成本高昂，因此边缘设备推理是一个有前景的解决方案。边缘推理的主要挑战包括内存使用和推理速度。

Method: MNN-LLM框架通过模型量化、DRAM-Flash混合存储、权重和输入重排、多核负载均衡、混合精度浮点运算和几何计算等策略来优化LLM在移动设备上的部署。

Result: MNN-LLM框架与当前主流的LLM专用框架相比，实现了高达8.6倍的速度提升。

Conclusion: MNN-LLM框架通过模型量化和DRAM-Flash混合存储有效降低了内存使用，并通过优化计算策略显著提高了移动设备上LLM的推理速度。

Abstract: 大型语言模型（LLM）在各种任务中表现出卓越的性能。然而，它们庞大的规模导致推理过程中大量的计算资源消耗，从而导致高成本。因此，边缘设备推理提供了一个有希望的解决方案。边缘推理的主要挑战包括内存使用和推理速度。本文介绍MNN-LLM，一个专门为加速大型语言模型在移动设备上的部署而设计的框架。MNN-LLM通过模型量化和DRAM-Flash混合存储来解决LLM的运行时特性，有效地减少了内存使用。它基于移动CPU指令集和GPU特性重新排列权重和输入，同时采用诸如多核负载平衡、混合精度浮点运算和几何计算等策略来提高性能。值得注意的是，与当前主流的LLM专用框架相比，MNN-LLM实现了高达8.6倍的速度提升。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10443) | **Categories:** cs.LG

---

### [2] [Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs](https://arxiv.org/abs/2506.10630)
*Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, Jintao Zhang*

Main category: cs.LG

TL;DR: Time-R1是一个用于时间序列预测的，通过强化学习微调LLM以提升多步推理能力的两阶段框架。


<details>
  <summary>Details</summary>
Motivation: 现有时间序列预测方法缺乏显式的推理过程，而新兴的慢思考LLM虽然具有潜力，但直接应用存在计算成本高、隐私风险和领域知识不足等问题。

Method: 提出了Time-R1，一个两阶段强化微调框架，包括有监督微调的预热适应阶段和利用强化学习提升模型泛化能力的阶段。同时，设计了细粒度的多目标奖励，并引入了GRIP以优化模型对有效推理路径的探索。

Result: 实验表明，Time-R1在多个数据集上显著提高了预测性能。

Conclusion: Time-R1通过两阶段强化微调框架，显著提升了LLM在时间序列预测上的多步推理能力，并在多个数据集上验证了其有效性。

Abstract: 为了提升时间序列预测（TSF）能力，已经提出了各种方法来提高预测精度，这些方法从统计技术发展到数据驱动的深度学习架构。尽管它们有效，但大多数现有方法仍然坚持快速思考模式——依赖于提取历史模式并将它们映射到未来值作为其核心建模理念，缺乏包含中间时间序列推理的显式思考过程。与此同时，新兴的慢思考LLM（例如，OpenAI-o1）已经展示了卓越的多步推理能力，为克服这些问题提供了另一种方法。然而，仅仅依靠prompt工程存在几个局限性——包括高计算成本、隐私风险和深入的领域特定时间序列推理能力有限。为了解决这些限制，一种更有希望的方法是训练LLM来发展慢思考能力并获得强大的时间序列推理技能。为此，我们提出了Time-R1，这是一个两阶段强化微调框架，旨在增强LLM在时间序列预测中的多步推理能力。具体来说，第一阶段进行有监督的微调以进行预热适应，而第二阶段采用强化学习来提高模型的泛化能力。特别地，我们专门为时间序列预测设计了一个细粒度的多目标奖励，然后引入GRIP（用于策略优化的基于组的相对重要性），它利用非均匀采样来进一步鼓励和优化模型对有效推理路径的探索。实验表明，Time-R1 显着提高了各种数据集的预测性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10630) | **Categories:** cs.LG, cs.AI

---

### [3] [NoLoCo: No-all-reduce Low Communication Training Method for Large Models](https://arxiv.org/abs/2506.10911)
*Jari Kolehmainen, Nikolay Blagoev, John Donaghy, Oğuzhan Ersoy, Christopher Nies*

Main category: cs.LG

TL;DR: NoLoCo是一种无需显式全局通信的新型优化方法，通过隐式同步模型权重，提高了大型语言模型在通信受限环境下的训练效率。


<details>
  <summary>Details</summary>
Motivation: 现有的大型语言模型训练方法依赖于高带宽互连的大规模加速器集群，扩展这些集群成本高昂且不切实际。即使是最新的低通信训练方法仍然需要同步模型参数，这在低带宽网络上可能会变得非常昂贵。

Method: 提出了一种名为NoLoCo的新型优化方法，该方法通过Nesterov动量优化器的变体隐式同步模型权重，无需显式全局通信。

Result: NoLoCo在通信开销方面显著低于完全分片的数据并行训练，甚至低于广泛使用的低通信训练方法DiLoCo。在数百个加速器通过互联网进行训练时，同步步骤本身估计比DiLoCo中使用的all-reduce快一个数量级。与DiLoCo相比，我们还观察到在各种模型尺寸和加速器数量下，收敛速度提高了4%。

Conclusion: NoLoCo在各种模型尺寸和加速器数量下都表现出优于DiLoCo的性能，尤其是在通信受限的环境中。

Abstract: 训练大型语言模型通常通过在包含数万个加速器的集群上使用优化方法来完成，这些加速器通过高带宽互连进行通信。扩展这些集群既昂贵又不切实际，限制了可以训练的模型的大小。最近的一些研究提出了通信强度较低的训练方法，避免了对高度连接的计算集群的需求。这些最先进的低通信训练方法仍然采用模型参数的同步步骤，当对所有模型副本执行时，在低带宽网络上可能会变得昂贵。在这项工作中，我们提出了一种新颖的优化方法NoLoCo，该方法在训练期间不会显式同步所有模型参数，因此不需要任何集体通信。NoLoCo通过Nesterov动量优化器的一种新颖变体隐式地同步模型权重，方法是部分地将模型权重与随机选择的另一个模型权重进行平均。我们为我们提出的优化器提供了理论收敛分析以及来自语言模型训练的经验结果。我们在1.25亿到68亿参数之间的各种加速器数量和模型尺寸上对NoLoCo进行了基准测试。我们的方法比完全分片的数据并行训练甚至广泛使用的低通信训练方法DiLoCo需要更少的通信开销。据估计，在数百个加速器通过互联网进行训练时，同步步骤本身比DiLoCo中使用的all-reduce快一个数量级。我们也没有任何全局阻塞通信，从而减少了加速器的空闲时间。与DiLoCo相比，我们还观察到在各种模型尺寸和加速器数量下，收敛速度提高了4%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10911) | **Categories:** cs.LG

---


## cs.MM [cs.MM]
### [1] [Multimodal Large Language Models: A Survey](https://arxiv.org/abs/2506.10016)
*Longzhen Han, Awes Mubarak, Almas Baimagambetov, Nikolaos Polatidis, Thar Baker*

Main category: cs.MM

TL;DR: 本综述概述了多模态大型语言模型的发展，重点介绍了其架构、技术和挑战，旨在实现更通用的多模态系统。


<details>
  <summary>Details</summary>
Motivation: 多模态大型语言模型（MLLM）已迅速发展，超越了文本生成，现在涵盖了包括图像、音乐、视频、人体运动和 3D 对象在内的各种输出模态，通过在统一架构下将语言与其他感官模态集成。

Method: 该研究对六种主要的生成模态进行了分类，并探讨了自监督学习（SSL）、混合专家（MoE）、人类反馈强化学习（RLHF）和思维链（CoT）提示等基础技术如何实现跨模态能力。

Result: 分析了关键模型、架构趋势和新兴的跨模态协同作用，同时强调了可转移的技术和未解决的挑战。诸如 transformers 和扩散模型之类的架构创新是这种融合的基础，从而实现了跨模态转移和模块化专业化。

Conclusion: 多模态大型语言模型正朝着更通用、自适应和可解释的方向发展，但在评估、模块化和结构化推理方面仍面临挑战。

Abstract: 多模态大型语言模型（MLLM）的演进迅速，已不再局限于文本生成，而是扩展到包括图像、音乐、视频、人体运动和3D对象等多种输出模态，通过在统一的架构下整合语言和其他感官模态来实现。本综述对六种主要的生成模态进行了分类，并探讨了自监督学习（SSL）、混合专家（MoE）、人类反馈强化学习（RLHF）和思维链（CoT）提示等基础技术如何实现跨模态能力。我们分析了关键模型、架构趋势和新兴的跨模态协同效应，同时强调了可转移的技术和尚未解决的挑战。诸如transformers和扩散模型之类的架构创新是这种融合的基础，从而实现了跨模态转移和模块化专业化。我们强调了新兴的协同模式，并指出了在评估、模块化和结构化推理方面存在的开放性挑战。本次综述为MLLM的发展提供了一个统一的视角，并确定了通往更通用、自适应和可解释的多模态系统的关键路径。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10016) | **Categories:** cs.MM, cs.AI, cs.CL

---


## cs.NI [cs.NI]
### [1] [AI5GTest: AI-Driven Specification-Aware Automated Testing and Validation of 5G O-RAN Components](https://arxiv.org/abs/2506.10111)
*Abiodun Ganiyu, Pranshav Gajjar, Vijay K Shah*

Main category: cs.NI

TL;DR: AI5GTest是一个AI驱动的O-RAN组件自动化测试框架，通过协同的LLM框架显著减少了测试时间并保持了验证准确性。


<details>
  <summary>Details</summary>
Motivation: 现有的O-RAN测试框架依赖于手动流程，容易出错，并且缺乏一致性和可扩展性。

Method: AI5GTest利用一个协同的大型语言模型（LLM）框架，包括Gen-LLM、Val-LLM和Debug-LLM，以自动化O-RAN组件的验证。

Result: AI5GTest在从O-RAN TIFG和WG5-IOT测试规范获得的一系列测试用例中进行了评估，结果表明与传统的手动方法相比，总体测试执行时间显著减少，同时保持了较高的验证准确性。

Conclusion: AI5GTest显著减少了测试执行时间，同时保持了较高的验证准确性。

Abstract: 开放无线接入网络（O-RAN）通过促进互操作性、供应商多样性和快速创新，改变了电信行业。然而，其分离的架构引入了复杂的测试挑战，尤其是在根据O-RAN联盟和3GPP规范验证多供应商组件方面。现有的框架，如开放测试和集成中心（OTIC）提供的框架，严重依赖于手动流程，这些流程是分散的，容易出现人为错误，导致不一致和可扩展性问题。为了解决这些限制，我们提出了AI5GTest——一个人工智能驱动的、规范感知的测试框架，旨在自动化O-RAN组件的验证。AI5GTest利用一个协同的大型语言模型（LLM）框架，包括Gen-LLM、Val-LLM和Debug-LLM。Gen-LLM根据3GPP和O-RAN规范自动生成测试用例的预期程序流程，而Val-LLM将信令消息与这些流程进行交叉引用，以验证合规性并检测偏差。如果出现异常，Debug-LLM会执行根本原因分析，从而深入了解故障原因。为了提高透明度和可信度，AI5GTest结合了一个人机协作机制，其中Gen-LLM在继续进行验证之前，向测试人员展示前k个相关的官方规范以供批准。AI5GTest使用从O-RAN TIFG和WG5-IOT测试规范获得的一系列测试用例进行了评估，结果表明与传统的手动方法相比，总体测试执行时间显著减少，同时保持了较高的验证准确性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10111) | **Categories:** cs.NI, cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding](https://arxiv.org/abs/2506.10756)
*Yuhang Zhang, Haosheng Yu, Jiaping Xiao, Mir Feroskhan*

Main category: cs.RO

TL;DR: VLFly 是一个用于无人机的视觉语言导航框架，它使用大型语言模型和视觉语言模型来实现开放词汇目标理解和通用导航能力。


<details>
  <summary>Details</summary>
Motivation: 视觉语言导航 (VLN) 是自主机器人领域一个长期存在的挑战，旨在使智能体能够在复杂环境中按照人类指令进行导航。该领域仍然存在两个关键瓶颈：推广到分布外环境以及依赖于固定的离散动作空间。

Method: VLFly 整合了三个模块：一个基于大型语言模型 (LLM) 的指令编码器，它将高级语言重新制定为结构化提示，一个由视觉语言模型 (VLM) 驱动的目标检索器，它通过视觉语言相似性将这些提示与目标图像匹配，以及一个航路点规划器，它为实时无人机控制生成可执行的轨迹。

Result: VLFly 在各种仿真环境中进行了评估，无需额外微调，并且始终优于所有基线。此外，在直接和间接指令下，室内和室外环境中的真实 VLN 任务表明，即使在存在抽象语言输入的情况下，VLFly 也能实现强大的开放词汇目标理解和通用导航能力。

Conclusion: VLFly 在各种仿真环境中进行了评估，无需额外微调，并且始终优于所有基线。此外，在直接和间接指令下，室内和室外环境中的真实 VLN 任务表明，即使在存在抽象语言输入的情况下，VLFly 也能实现强大的开放词汇目标理解和通用导航能力。

Abstract: 视觉语言导航 (VLN) 是自主机器人领域一个长期存在的挑战，旨在使智能体能够在复杂环境中按照人类指令进行导航。该领域仍然存在两个关键瓶颈：推广到分布外环境以及依赖于固定的离散动作空间。为了应对这些挑战，我们提出了一种专为无人机 (UAV) 量身定制的框架 Vision-Language Fly (VLFly)，用于执行语言引导的飞行。VLFly 无需定位或主动测距传感器，仅从机载单目摄像头捕获的自我中心观测输出连续速度命令。VLFly 整合了三个模块：一个基于大型语言模型 (LLM) 的指令编码器，它将高级语言重新制定为结构化提示，一个由视觉语言模型 (VLM) 驱动的目标检索器，它通过视觉语言相似性将这些提示与目标图像匹配，以及一个航路点规划器，它为实时无人机控制生成可执行的轨迹。VLFly 在各种仿真环境中进行了评估，无需额外微调，并且始终优于所有基线。此外，在直接和间接指令下，室内和室外环境中的真实 VLN 任务表明，即使在存在抽象语言输入的情况下，VLFly 也能实现强大的开放词汇目标理解和通用导航能力，即使在存在抽象语言输入的情况下。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10756) | **Categories:** cs.RO, cs.AI

---

### [2] [A Navigation Framework Utilizing Vision-Language Models](https://arxiv.org/abs/2506.10172)
*Yicheng Duan, Kaiyu tang*

Main category: cs.RO

TL;DR: 本文提出了一种模块化的视觉语言导航框架，通过结合冻结的视觉语言模型和轻量级规划逻辑，实现了灵活高效的导航。


<details>
  <summary>Details</summary>
Motivation: 视觉语言导航（VLN）在具身人工智能中提出了一个复杂的挑战，要求智能体解释自然语言指令并在视觉丰富的陌生环境中导航。大型视觉语言模型（LVLMs）的最新进展，如CLIP和Flamingo，显着提高了多模态理解，但也带来了与计算成本和实时部署相关的新挑战。

Method: 通过将冻结的视觉语言模型Qwen2.5-VL-7B-Instruct与轻量级规划逻辑相结合，我们旨在实现灵活、快速和适应性强的导航，而无需进行广泛的模型微调。我们的框架利用提示工程、结构化历史管理和双帧视觉输入策略来增强导航步骤中的决策连续性。

Result: 我们在VLN-CE设置中使用Matterport3D数据集和Habitat-Lab模拟环境在Room-to-Room基准上评估了我们的系统。我们的初步结果显示在推广到严格评估设置下的未见环境方面存在挑战。

Conclusion: 虽然在严格的评估设置下，我们的初步结果显示在推广到未见环境方面存在挑战，但我们的模块化方法为可扩展和高效的导航系统奠定了基础，突出了通过增强的环境先验和扩展的多模态输入集成来实现未来改进的有希望的方向。

Abstract: 视觉语言导航（VLN）在具身人工智能中提出了一个复杂的挑战，要求智能体解释自然语言指令并在视觉丰富的陌生环境中导航。大型视觉语言模型（LVLMs）的最新进展，如CLIP和Flamingo，显着提高了多模态理解，但也带来了与计算成本和实时部署相关的新挑战。在这个项目中，我们提出了一个模块化的、即插即用的导航框架，将视觉语言理解与动作规划分离。通过将冻结的视觉语言模型Qwen2.5-VL-7B-Instruct与轻量级规划逻辑相结合，我们旨在实现灵活、快速和适应性强的导航，而无需进行广泛的模型微调。我们的框架利用提示工程、结构化历史管理和双帧视觉输入策略来增强导航步骤中的决策连续性。我们在VLN-CE设置中使用Matterport3D数据集和Habitat-Lab模拟环境在Room-to-Room基准上评估了我们的系统。虽然我们的初步结果显示在推广到严格评估设置下的未见环境方面存在挑战，但我们的模块化方法为可扩展和高效的导航系统奠定了基础，突出了通过增强的环境先验和扩展的多模态输入集成来实现未来改进的有希望的方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10172) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [3] [Leveraging LLMs for Mission Planning in Precision Agriculture](https://arxiv.org/abs/2506.10093)
*Marcos Abel Zuzuárregui, Stefano Carpin*

Main category: cs.RO

TL;DR: 提出了一种利用大型语言模型赋能机器人自主完成复杂农业数据收集任务的端到端系统。


<details>
  <summary>Details</summary>
Motivation: 终端用户通常缺乏技术专长，难以使机器人适应执行不同的任务。

Method: 利用大型语言模型ChatGPT，用户可以使用自然语言指令将复杂的数据收集任务分配给自主机器人。

Result: 通过大量的实验，强调了LLM在这方面的优势和局限性，特别是在空间推理和解决复杂的路由挑战方面，并展示了所提出的实现如何克服这些挑战。

Conclusion: 大型语言模型在机器人任务中的应用有优势和局限性，该论文提出了一种实现方法来克服这些局限性。

Abstract: 机器人和人工智能在推进精准农业方面具有巨大的潜力。虽然机器人系统已经成功地应用于各种任务，但使它们适应执行不同的任务仍然具有挑战性，特别是因为终端用户通常缺乏技术专长。在本文中，我们提出了一个端到端的系统，该系统利用大型语言模型（LLM），特别是ChatGPT，使用户能够使用自然语言指令将复杂的数据收集任务分配给自主机器人。为了提高可重用性，任务计划使用现有的IEEE任务规范标准进行编码，并通过ROS2节点在机器人上执行，ROS2节点将高级任务描述与现有的ROS库连接起来。通过大量的实验，我们强调了LLM在这方面的优势和局限性，特别是在空间推理和解决复杂的路由挑战方面，并展示了我们提出的实现如何克服这些挑战。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10093) | **Categories:** cs.RO, cs.AI

---

### [4] [Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models](https://arxiv.org/abs/2506.10098)
*Christian Reichenbächer, Philipp Rank, Jochen Hipp, Oliver Bringmann*

Main category: cs.RO

TL;DR: 高斯混合Copula模型更适合驾驶场景数据，为自动驾驶安全验证提供了一个有前景的基础。


<details>
  <summary>Details</summary>
Motivation: 在基于场景的安全评估中，风险量化取决于具体参数组合的可能性，因此需要了解场景参数的联合概率分布。

Method: 将高斯混合Copula模型应用于驾驶场景的统计建模。

Result: 在1800万个场景实例上的评估表明，高斯混合Copula模型在似然性和Sinkhorn距离方面都更适合数据。

Conclusion: 高斯混合Copula模型为未来基于场景的验证框架提供了一个引人注目的基础。

Abstract: 本文首次将高斯混合Copula模型应用于驾驶场景的统计建模，以用于自动驾驶系统的安全验证。场景参数的联合概率分布知识对于基于场景的安全评估至关重要，因为风险量化取决于具体参数组合的可能性。高斯混合Copula模型结合了高斯混合模型的多模态表达能力和copula的灵活性，能够对边际分布和依赖性进行单独建模。我们使用来自联合国第157号条例中定义的场景的真实驾驶数据，将高斯混合Copula模型与先前提出的方法（高斯混合模型和高斯Copula模型）进行了基准测试。我们在1800万个场景实例上的评估表明，高斯混合Copula模型在似然性和Sinkhorn距离方面都更适合数据。这些结果表明，高斯混合Copula模型为未来基于场景的验证框架提供了一个引人注目的基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10098) | **Categories:** cs.RO, cs.LG

---

### [5] [One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture](https://arxiv.org/abs/2506.10106)
*Marcos Abel Zuzuárregui, Mustafa Melih Toslak, Stefano Carpin*

Main category: cs.RO

TL;DR: 该论文提出了一种自然语言机器人任务规划器，使非专业人员可以通过通用界面控制异构机器人，从而使精准农业中的机器人自动化更易于非技术用户访问。


<details>
  <summary>Details</summary>
Motivation: 人工智能正在改变精准农业，为农民提供新的工具来简化他们的日常操作。虽然这些技术进步有望提高效率，但它们通常会引入额外的复杂性和陡峭的学习曲线，这对必须在技术采用与现有工作量之间取得平衡的非技术用户来说尤其具有挑战性。

Method: 利用大型语言模型 (LLM) 和预定义的基元，将人类语言无缝转换为可由不同机器人平台执行的中间描述。

Result: 结果表明，该架构具有足够的通用性，可以支持不同的机器人，并且足够强大，可以执行复杂的任务请求。

Conclusion: 该研究表明，该架构具有足够的通用性，可以支持不同的机器人，并且足够强大，可以执行复杂的任务请求。这项工作代表着朝着使精准农业中的机器人自动化更易于非技术用户访问迈出的重要一步。

Abstract: 人工智能正在改变精准农业，为农民提供新的工具来简化他们的日常操作。虽然这些技术进步有望提高效率，但它们通常会引入额外的复杂性和陡峭的学习曲线，这对必须在技术采用与现有工作量之间取得平衡的非技术用户来说尤其具有挑战性。在本文中，我们提出了一种自然语言 (NL) 机器人任务规划器，使非专业人员可以通过通用界面控制异构机器人。通过利用大型语言模型 (LLM) 和预定义的基元，我们的架构可以将人类语言无缝转换为可由不同机器人平台执行的中间描述。借助该系统，用户无需编写任何代码即可制定复杂的农业任务。在本文中，我们通过涉及机器人操作和计算机视觉任务的新型实验，扩展了我们之前为轮式机器人任务规划量身定制的系统。我们的结果表明，该架构具有足够的通用性，可以支持不同的机器人，并且足够强大，可以执行复杂的任务请求。这项工作代表着朝着使精准农业中的机器人自动化更易于非技术用户访问迈出的重要一步。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10106) | **Categories:** cs.RO, cs.AI

---

### [6] [Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving](https://arxiv.org/abs/2506.10317)
*Akshar Tumu, Henrik I. Christensen, Marcell Vazquez-Chanlatte, Chikao Tsuchiya, Dhaval Bhanderi*

Main category: cs.RO

TL;DR: 本文提出了一种轻量级的方法，通过结合 OSM 地图和道路设计手册中的信息来改进车道拓扑预测。


<details>
  <summary>Details</summary>
Motivation: 车道拓扑预测是安全可靠的自动驾驶的关键组成部分。准确理解道路环境有助于完成这项任务。

Method: 通过结合来自 OSM 地图的结构化道路元数据和来自道路设计手册的车道宽度先验知识，以轻量级的方式增强了基于地图先验的在线车道拓扑预测模型 SMERF。

Result: 该方法在车道和交通元素检测及其关联方面均有改进。我们使用四种拓扑感知指标来全面评估模型性能。

Conclusion: 该方法在不同的拓扑结构和条件下都表现出良好的泛化和扩展能力。

Abstract: 车道拓扑预测是安全可靠的自动驾驶的关键组成部分。准确理解道路环境有助于完成这项任务。我们观察到，这些信息通常遵循自然语言中编码的约定，通过反映道路结构的设计规范和捕捉道路功能的道路名称来体现。我们通过结合来自 OSM 地图的结构化道路元数据和来自道路设计手册的车道宽度先验知识，以轻量级的方式增强了基于地图先验的在线车道拓扑预测模型 SMERF。我们在两个地理位置不同的复杂交叉路口场景中评估了我们的方法。我们的方法在车道和交通元素检测及其关联方面均有改进。我们使用四种拓扑感知指标来全面评估模型性能。这些结果证明了我们的方法能够推广和扩展到不同的拓扑结构和条件。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10317) | **Categories:** cs.RO, cs.AI

---

### [7] [GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation](https://arxiv.org/abs/2506.10966)
*Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang*

Main category: cs.RO

TL;DR: GenManip是一个现实的桌面模拟平台，通过LLM驱动的场景图合成多样化任务，用于评估和提升机器人策略的泛化能力。


<details>
  <summary>Details</summary>
Motivation: 现实世界中的机器人操作仍然具有挑战性，尤其是在鲁棒泛化方面。现有的模拟平台缺乏足够的支持来探索策略如何适应不同的指令和场景。因此，它们落后于人们对指令遵循基础模型（如LLM）日益增长的兴趣，这些模型的适应性至关重要，但在公平比较中仍未得到充分探索。

Method: 提出了GenManip，一个现实的桌面模拟平台，专为策略泛化研究定制。它具有一个自动管道，通过LLM驱动的面向任务的场景图来合成大规模、多样化的任务，使用10K注释的3D对象资产。

Result: 结果表明，虽然数据缩放有利于端到端方法，但通过基础模型增强的模块化系统在不同的场景中能更有效地推广。

Conclusion: 模块化系统在不同场景中能更有效地推广

Abstract: 现实世界中的机器人操作仍然具有挑战性，尤其是在鲁棒泛化方面。现有的模拟平台缺乏足够的支持来探索策略如何适应不同的指令和场景。因此，它们落后于人们对指令遵循基础模型（如LLM）日益增长的兴趣，这些模型的适应性至关重要，但在公平比较中仍未得到充分探索。为了弥合这一差距，我们推出了GenManip，这是一个现实的桌面模拟平台，专为策略泛化研究定制。它具有一个自动管道，通过LLM驱动的面向任务的场景图来合成大规模、多样化的任务，使用10K注释的3D对象资产。为了系统地评估泛化，我们提出了GenManip-Bench，这是一个通过人工循环校正改进的200个场景的基准。我们评估了两种策略类型：（1）集成基础模型的模块化操作系统的感知、推理和规划，以及（2）通过可扩展的数据收集训练的端到端策略。结果表明，虽然数据缩放有利于端到端方法，但通过基础模型增强的模块化系统在不同的场景中能更有效地推广。我们预计该平台将有助于促进对在现实条件下推进策略泛化的重要见解。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10966) | **Categories:** cs.RO

---
