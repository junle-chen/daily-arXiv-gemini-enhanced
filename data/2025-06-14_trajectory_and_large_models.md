# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-14

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (7)](#cs-cv)
- [cs.DC (1)](#cs-dc)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [cs.MM (1)](#cs-mm)
- [cs.NI (1)](#cs-ni)
- [机器人学 (Robotics) (7)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models](https://arxiv.org/abs/2506.10853)
*Yu Zhang, Yang Hu, De Wang*

Main category: cs.AI

TL;DR: 该论文提出了一种结合CoT推理和MCP的框架，以提升LLM在城市时空行为模拟方面的能力。


<details>
  <summary>Details</summary>
Motivation: 传统的人类时空行为模拟方法计算成本高、泛化能力有限且可扩展性差。大型语言模型（LLM）在时空推理方面面临空间认知有限、缺乏物理约束理解和群体同质化倾向等挑战。

Method: 该论文提出了一种结合思维链（CoT）推理和模型上下文协议（MCP）的框架，通过五阶段认知框架进行类人渐进推理，并结合六个专业MCP工具类别进行综合数据处理，以增强LLM在模拟时空行为方面的能力。

Result: 在上海陆家嘴地区的实验验证了该框架的有效性，生成结果与真实移动信令数据高度相似，不同基础模型的生成质量得分达到7.86至8.36。并行处理实验显示，从2个进程扩展到12个进程时，每个样本的生成时间从1.30分钟减少到0.17分钟，效率得到显著提高。

Conclusion: 该研究证明了结合CoT推理和MCP的框架能够有效提升LLM在模拟时空行为方面的能力，并为城市计算提供了一种实用的合成移动数据生成方法。

Abstract: 人类时空行为模拟对于城市规划研究至关重要，但传统的基于规则和统计的方法存在计算成本高、泛化能力有限和可扩展性差的问题。虽然大型语言模型（LLM）作为“世界模拟器”显示出潜力，但它们在时空推理方面面临挑战，包括空间认知有限、缺乏物理约束理解和群体同质化倾向。本文介绍了一个框架，该框架集成了思维链（CoT）推理与模型上下文协议（MCP），以增强LLM在模拟与验证数据模式相符的时空行为方面的能力。该方法论结合了通过五阶段认知框架进行的人类式渐进推理，以及通过六个专业MCP工具类别进行的综合数据处理：时间管理、空间导航、环境感知、个人记忆、社会协作和经验评估。在上海陆家嘴地区进行的实验验证了该框架在1000个生成样本中的有效性。结果表明，与真实移动信令数据具有高度相似性，不同基础模型的生成质量得分达到7.86至8.36。并行处理实验显示了效率的提高，当从2个进程扩展到12个进程时，每个样本的生成时间从1.30分钟减少到0.17分钟。这项工作有助于将CoT推理与MCP集成用于城市行为建模，推进LLM在城市计算中的应用，并为合成移动数据生成提供了一种实用的方法。该框架为智慧城市规划、交通预测和参与式城市设计应用奠定了基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10853) | **Categories:** cs.AI, cs.CY

---

### [2] [Mirage-1: Augmenting and Updating GUI Agent with Hierarchical Multimodal Skills](https://arxiv.org/abs/2506.10387)
*Yuquan Xie, Zaijing Li, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Dongmei Jiang, Liqiang Nie*

Main category: cs.AI

TL;DR: Mirage-1通过分层多模态技能和技能增强的蒙特卡洛树搜索，显著提升了GUI代理在长时程任务中的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的多模态大型语言模型（MLLM）GUI代理在在线环境中的长时程任务中表现不佳，主要是由于知识不足以及离线和在线领域之间的固有差距。

Method: 提出分层多模态技能（HMS）模块和技能增强的蒙特卡洛树搜索（SA-MCTS）算法。

Result: Mirage-1在AndroidWorld、MobileMiniWob++、Mind2Web-Live和AndroidLH上的性能分别优于之前的代理32%、19%、15%和79%。

Conclusion: Mirage-1在多个GUI任务中表现优于现有模型，尤其是在AndroidLH基准测试中。

Abstract: 最近利用多模态大型语言模型（MLLM）作为GUI代理的尝试已经取得了有希望的结果。然而，这些代理仍然在在线环境中的长时程任务中挣扎，这主要是由于知识不足以及离线和在线领域之间的固有差距。在本文中，受到人类如何在开放环境中概括知识的启发，我们提出了一个分层多模态技能（HMS）模块，以解决知识不足的问题。它逐步将轨迹抽象为执行技能、核心技能，最终抽象为元技能，从而为长时程任务规划提供分层知识结构。为了弥合领域差距，我们提出了技能增强的蒙特卡洛树搜索（SA-MCTS）算法，该算法有效地利用了在离线环境中获得的技能，以减少在线树探索期间的动作搜索空间。在HMS的基础上，我们提出了Mirage-1，一个多模态、跨平台、即插即用的GUI代理。为了验证Mirage-1在真实长时程场景中的性能，我们构建了一个新的基准测试AndroidLH。实验结果表明，Mirage-1在AndroidWorld、MobileMiniWob++、Mind2Web-Live和AndroidLH上的性能分别优于之前的代理32%、19%、15%和79%。项目页面：https://cybertronagent.github.io/Mirage-1.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10387) | **Categories:** cs.AI

---

### [3] [GenPlanX. Generation of Plans and Execution](https://arxiv.org/abs/2506.10897)
*Daniel Borrajo, Giuseppe Canonaco, Tomás de la Rosa, Alfredo Garrachón, Sriram Gopalakrishnan, Simerjot Kaur, Marianela Morales, Sunandita Patra, Alberto Pozanco, Keshav Ramani, Charese Smiley, Pietro Totis, Manuela Veloso*

Main category: cs.AI

TL;DR: GenPlanX集成了LLM和经典AI规划引擎，通过自然语言实现了人机协作，从而简化了工作流程并提高了生产力。


<details>
  <summary>Details</summary>
Motivation: 传统的AI规划技术可以为复杂的任务生成动作序列，但是它们缺乏理解自然语言描述的规划任务的能力。大型语言模型(LLM)的出现为人机交互带来了新的能力。

Method: GenPlanX集成了用于规划任务的自然语言描述的LLM、经典AI规划引擎以及执行和监控框架。

Result: GenPlanX在协助用户处理办公室相关任务方面的有效性。

Conclusion: GenPlanX展示了通过无缝人机协作简化工作流程和提高生产力的潜力。

Abstract: 传统的AI规划技术为复杂任务生成动作序列，但无法理解自然语言描述的规划任务。大型语言模型(LLM)的出现，为人机交互带来了新的能力。在规划任务的背景下，LLM在理解人类意图方面表现出色。本文介绍了GenPlanX，它集成了LLM（用于规划任务的自然语言描述）、经典AI规划引擎以及执行和监控框架。我们通过GenPlanX在协助用户处理办公室相关任务方面的有效性，突出了它通过无缝人机协作简化工作流程和提高生产力的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10897) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [RoCA: Robust Cross-Domain End-to-End Autonomous Driving](https://arxiv.org/abs/2506.10145)
*Rajeev Yasarla, Shizhong Han, Hsin-Pai Cheng, Litian Liu, Shweta Mahajan, Apratim Bhattacharyya, Yunxiao Shi, Risheek Garrepalli, Hong Cai, Fatih Porikli*

Main category: cs.CV

TL;DR: RoCA通过学习tokens联合概率分布和使用高斯过程进行实例化，提升了端到端自动驾驶模型的跨域泛化和适应能力。


<details>
  <summary>Details</summary>
Motivation: 现有端到端自动驾驶模型在跨域部署时面临挑战，大型语言模型虽能利用开放世界知识，但不能保证跨域驾驶性能，且领域自适应的再训练成本高昂。

Method: RoCA框架通过学习ego和周围车辆信息的tokens联合概率分布，并使用高斯过程进行实例化，学习一组基础tokens及其对应轨迹。

Result: 在各种跨域场景下对RoCA进行了广泛的评估，结果表明RoCA实现了强大的领域泛化和适应性能，显著优于直接微调。

Conclusion: RoCA通过学习跨越不同驾驶场景的基础tokens及其对应轨迹，并结合高斯过程进行概率推理，从而提升了E2E自动驾驶模型的泛化性和适应性。

Abstract: 端到端(E2E)自动驾驶最近作为一种新的范例出现，具有巨大的潜力。然而，很少有研究关注跨领域(如城市)部署的实际挑战。虽然一些工作已经结合大型语言模型(llm)来利用他们的开放世界知识，但llm不能保证跨领域的驾驶性能，并且可能在领域适应过程中产生过高的再训练成本。在本文中，我们提出了RoCA，一个新的鲁棒的跨域E2E自动驾驶框架。RoCA在E2E管道中制定了对自我和周围车辆信息进行编码的tokens的联合概率分布。通过使用高斯过程(GP)进行实例化，RoCA学习了一组具有相应轨迹的基础tokens，这些轨迹跨越了不同的驾驶场景。然后，给定任何驾驶场景，它能够概率地推断未来的轨迹。通过在源域训练中使用RoCA和基础E2E模型，我们提高了基础模型的泛化能力，而不需要额外的推理计算。此外，RoCA能够在新的目标领域实现鲁棒的自适应，显著优于直接微调。我们广泛地评估了RoCA在各种跨域场景，并表明它实现了强大的领域泛化和适应性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10145) | **Categories:** cs.CV

---

### [2] [Motion-R1: Chain-of-Thought Reasoning and Reinforcement Learning for Human Motion Generation](https://arxiv.org/abs/2506.10353)
*Runqi Ouyang, Haoyun Li, Zhenyuan Zhang, Xiaofeng Wang, Zheng Zhu, Guan Huang, Xingang Wang*

Main category: cs.CV

TL;DR: Motion-R1 通过引入 Chain-of-Thought 机制，增强了文本到运动生成中对复杂指令的理解和执行能力。


<details>
  <summary>Details</summary>
Motivation: 现有的文本到运动生成方法通常依赖于端到端映射策略，无法捕获深层的语言结构和逻辑推理，导致生成的运动缺乏可控性、一致性和多样性。

Method: Motion-R1 提出了一个统一的运动-语言建模框架，该框架集成了 Chain-of-Thought 机制，通过将复杂的文本指令分解为逻辑结构化的动作路径，为运动生成提供高级语义指导。

Result: Motion-R1 在需要细致的语义理解和长期时间连贯性的场景中，实现了与最先进方法相比具有竞争力的或更优越的性能。

Conclusion: Motion-R1 在多个基准数据集上表现出与最先进方法相当或更优越的性能，尤其是在需要细致的语义理解和长期时间连贯性的场景中。

Abstract: 大型语言模型在自然语言理解和推理方面的最新进展，为文本到运动生成开辟了新的可能性。虽然现有的方法在语义对齐和运动合成方面取得了显著进展，但它们通常依赖于端到端映射策略，无法捕获深层的语言结构和逻辑推理。因此，生成的运动往往缺乏可控性、一致性和多样性。为了解决这些局限性，我们提出了 Motion-R1，这是一个统一的运动-语言建模框架，它集成了 Chain-of-Thought 机制。通过将复杂的文本指令分解为逻辑结构化的动作路径，Motion-R1 为运动生成提供高级语义指导，显著增强了模型解释和执行多步骤、长时程和组合丰富的命令的能力。为了训练我们的模型，我们采用了 Group Relative Policy Optimization，这是一种专为大型模型设计的强化学习算法，它利用运动质量反馈来联合优化推理链和运动合成。在多个基准数据集上进行的大量实验表明，Motion-R1 实现了与最先进方法相比具有竞争力的或更优越的性能，尤其是在需要细致的语义理解和长期时间连贯性的场景中。代码、模型和数据将公开提供。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10353) | **Categories:** cs.CV

---

### [3] [DanceChat: Large Language Model-Guided Music-to-Dance Generation](https://arxiv.org/abs/2506.10574)
*Qing Wang, Xiaohang Yang, Yilan Dong, Naveen Raj Govindaraj, Gregory Slabaugh, Shanxin Yuan*

Main category: cs.CV

TL;DR: DanceChat 是一种利用大型语言模型（LLM）生成文本舞蹈指导，从而实现更多样化和音乐风格对齐的音乐到舞蹈生成方法。


<details>
  <summary>Details</summary>
Motivation: 音乐到舞蹈的生成任务面临音乐和舞蹈动作之间的语义鸿沟挑战，以及音乐到多种舞蹈诠释的一对多映射问题。此外，配对的音乐和舞蹈数据稀缺，限制了模型学习多样化舞蹈模式的能力。

Method: DanceChat 包含三个模块：一个基于LLM的伪指令生成模块，用于生成文本舞蹈指导；一个多模态特征提取和融合模块，将音乐、节奏和文本指导整合到一个共享表示中；以及一个基于扩散的运动合成模块，结合多模态对齐损失，确保生成的舞蹈与音乐和文本提示对齐。

Result: 在 AIST++ 数据集上进行了大量实验，并通过人工评估表明，DanceChat 在质量和数量上均优于现有技术水平的方法。

Conclusion: DanceChat在AIST++数据集上和人工评估中，都超越了现有最佳方法，无论是在质量上还是数量上。

Abstract: 音乐到舞蹈的生成旨在合成以音乐输入为条件的人类舞蹈动作。尽管最近取得了进展，但由于音乐和舞蹈动作之间的语义差距仍然存在重大挑战，因为音乐仅提供抽象的线索，例如旋律、律动和情感，而没有明确指定物理动作。此外，一首音乐可以产生多种合理的舞蹈诠释。这种一对多的映射需要额外的指导，因为仅音乐就为生成各种舞蹈动作提供的信息有限。配对的音乐和舞蹈数据的稀缺性进一步加剧了这一挑战，这限制了模型学习各种舞蹈模式的能力。在本文中，我们介绍了一种大型语言模型 (LLM) 指导的音乐到舞蹈生成方法 DanceChat。我们使用 LLM 作为编舞，提供文本运动指令，为舞蹈生成提供明确的、高级的指导。这种方法超越了仅从音乐中进行的隐式学习，使模型能够生成更多样化且与音乐风格更好对齐的舞蹈。我们的方法包括三个组成部分：(1) 一个基于 LLM 的伪指令生成模块，该模块基于音乐风格和结构生成文本舞蹈指导，(2) 一个多模态特征提取和融合模块，该模块将音乐、节奏和文本指导集成到共享表示中，以及 (3) 一个基于扩散的运动合成模块，以及一个多模态对齐损失，该损失确保生成的舞蹈与音乐和文本提示对齐。在 AIST++ 上进行的大量实验和人工评估表明，DanceChat 在质量和数量上均优于最先进的方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10574) | **Categories:** cs.CV, cs.MM, cs.SD, eess.AS

---

### [4] [Test-Time Adaptation for Generalizable Task Progress Estimation](https://arxiv.org/abs/2506.10085)
*Christos Ziakas, Alessandra Russo*

Main category: cs.CV

TL;DR: 该论文提出了一种测试时自适应方法，通过优化自监督目标，使进度估计模型能够更好地适应测试轨迹的视觉和时间上下文。


<details>
  <summary>Details</summary>
Motivation: 我们提出了一种测试时自适应方法，该方法使进度估计模型能够在线适应测试轨迹的视觉和时间上下文。

Method: 我们提出了一种测试时自适应方法，该方法通过优化学习到的自监督目标，使进度估计模型能够在线适应测试轨迹的视觉和时间上下文。为此，我们引入了一种基于梯度的元学习策略，以在专家视觉轨迹及其自然语言任务描述上训练模型，从而使测试时自适应能够提高依赖于语义内容而非时间顺序的进度估计。

Result: 我们的测试时自适应方法可以从单一训练环境推广到不同的分布外任务、环境和实现，优于使用自回归视觉语言模型的最新上下文学习方法。

Conclusion: 我们的测试时自适应方法可以从单一训练环境推广到不同的分布外任务、环境和实现，优于使用自回归视觉语言模型的最新上下文学习方法。

Abstract: 我们提出了一种测试时自适应方法，通过优化学习到的自监督目标，使进度估计模型能够在线适应测试轨迹的视觉和时间上下文。为此，我们引入了一种基于梯度的元学习策略，以在专家视觉轨迹及其自然语言任务描述上训练模型，从而使测试时自适应能够提高依赖于语义内容而非时间顺序的进度估计。我们的测试时自适应方法可以从单一训练环境推广到不同的分布外任务、环境和实现，优于使用自回归视觉语言模型的最新上下文学习方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10085) | **Categories:** cs.CV, cs.AI, I.2.6; I.2.9; I.2.10

---

### [5] [EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models](https://arxiv.org/abs/2506.10100)
*Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, Linfeng Zhang*

Main category: cs.CV

TL;DR: EfficientVLA通过协同优化语言、视觉和动作模块，显著加速了VLA模型的推理过程，同时保持了性能。


<details>
  <summary>Details</summary>
Motivation: 现有的加速方法通常针对孤立的低效率，这种零碎的解决方案通常不能全面地解决整个VLA管道中各种计算和内存瓶颈，从而限制了实际部署。

Method: EfficientVLA通过协同整合三种有针对性的策略来系统地消除这些障碍：(1) 剪枝语言模块中功能上不重要的层，以层间冗余分析为指导；(2) 通过任务感知策略优化视觉处理路径，选择紧凑、多样化的视觉token集合，平衡任务关键性和信息覆盖；(3) 通过战略性地缓存和重用关键中间特征，减轻迭代的基于扩散的动作头中的时间计算冗余。

Result: EfficientVLA在CogACT模型上实现了1.93倍的推理加速，并将FLOPs降低至28.9%，在SIMPLER基准测试中仅损失了0.6%的成功率。

Conclusion: 通过在CogACT模型上的实验，EfficientVLA在SIMPLER基准测试中仅损失0.6%的成功率的情况下，实现了1.93倍的推理加速，并将FLOPs降低至28.9%。

Abstract: 视觉-语言-动作（VLA）模型，特别是基于扩散的架构，展示了具身智能的变革潜力，但由于大量固有的和推理时的冗余，导致计算和内存需求非常高，从而严重阻碍了其发展。虽然现有的加速工作通常针对孤立的低效率，但这种零碎的解决方案通常不能全面地解决整个VLA管道中各种计算和内存瓶颈，从而限制了实际部署。我们介绍了一种结构化的、无需训练的推理加速框架EfficientVLA，它通过协同利用多方面的冗余，系统地消除了这些障碍。EfficientVLA协同整合了三种有针对性的策略：(1) 剪枝语言模块中功能上不重要的层，以层间冗余分析为指导；(2) 通过任务感知策略优化视觉处理路径，选择紧凑、多样化的视觉token集合，平衡任务关键性和信息覆盖；(3) 通过战略性地缓存和重用关键中间特征，减轻迭代的基于扩散的动作头中的时间计算冗余。我们将我们的方法应用于标准的VLA模型CogACT，在SIMPLER基准测试中仅损失0.6%的成功率的情况下，实现了1.93倍的推理加速，并将FLOPs降低至28.9%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10100) | **Categories:** cs.CV

---

### [6] [DySS: Dynamic Queries and State-Space Learning for Efficient 3D Object Detection from Multi-Camera Videos](https://arxiv.org/abs/2506.10242)
*Rajeev Yasarla, Shizhong Han, Hong Cai, Fatih Porikli*

Main category: cs.CV

TL;DR: DySS通过状态空间学习和动态查询，实现了高效且高性能的基于相机的BEV 3D目标检测。


<details>
  <summary>Details</summary>
Motivation: 基于相机的鸟瞰图（BEV）3D目标检测是自动驾驶中最重要的感知任务之一。然而，现有方法依赖于密集的BEV特征，构建成本高昂，或者需要大量的查询，当使用更多的视频帧时，运行成本会很高。

Method: DySS采用状态空间模型（SSM）按时间步长顺序处理采样的特征，并通过合并、删除和拆分操作动态更新查询。

Result: 在nuScenes测试集上，DySS达到了65.31 NDS和57.4 mAP，优于最新的技术水平。在val分割上，DySS达到了56.2 NDS和46.2 mAP，以及33 FPS的实时推理速度。

Conclusion: DySS通过状态空间学习的特征和动态查询更新，实现了卓越的检测性能和高效的推理。

Abstract: 基于相机的鸟瞰图（BEV）3D目标检测是自动驾驶中最重要的感知任务之一。先前的方法依赖于密集的BEV特征，这导致构建成本很高。最近的研究探索了基于稀疏查询的检测。然而，它们仍然需要大量的查询，并且当使用更多的视频帧时，运行成本会变得很高。在本文中，我们提出了一种新的方法DySS，它采用了状态空间学习和动态查询。更具体地说，DySS利用状态空间模型（SSM）按时间步长顺序处理采样的特征。为了鼓励模型更好地捕捉潜在的运动和对应信息，我们引入了未来预测和掩蔽重建的辅助任务，以更好地训练SSM。然后，SSM的状态提供了信息丰富但高效的场景概括。基于状态空间学习的特征，我们通过合并、删除和拆分操作动态更新查询，这有助于在整个网络中维护一个有用、精简的检测查询集。我们提出的DySS实现了卓越的检测性能和高效的推理。具体来说，在nuScenes测试集上，DySS达到了65.31 NDS和57.4 mAP，优于最新的技术水平。在val分割上，DySS达到了56.2 NDS和46.2 mAP，以及33 FPS的实时推理速度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10242) | **Categories:** cs.CV

---

### [7] [SlotPi: Physics-informed Object-centric Reasoning Models](https://arxiv.org/abs/2506.10778)
*Jian Li, Wan Han, Ning Lin, Yu-Liang Zhan, Ruizhi Chengze, Haining Wang, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, Hao Sun*

Main category: cs.CV

TL;DR: SlotPi是一个基于槽的物理信息对象中心推理模型，它集成了物理模块和时空预测模块，并在各种数据集上表现出强大的适应性。


<details>
  <summary>Details</summary>
Motivation: 当前面向对象的动态模拟方法忽略了两个关键方面：1）将物理知识集成到模型中；2）验证模型在不同场景中的适应性。

Method: SlotPi集成了基于哈密顿原理的物理模块和用于动态预测的时空预测模块。

Result: 该模型在基准和流体数据集上的预测和视觉问答（VQA）等任务中表现出优势。此外，我们创建了一个包含对象交互、流体动力学和流体-对象交互的真实世界数据集，我们在此数据集上验证了我们模型的能力。

Conclusion: 该模型在所有数据集上表现出强大的性能，突显了其强大的适应性，为开发更高级的世界模型奠定了基础。

Abstract: 通过视觉观察理解和推理受物理定律支配的动态，类似于人类在现实世界中的能力，提出了巨大的挑战。目前，模仿人类行为的以对象为中心的动态模拟方法已经取得了显著的进展，但忽略了两个关键方面：1）将物理知识集成到模型中。人类通过观察世界获得物理知识，并将这些知识应用于准确地推理各种动态场景；2）验证模型在不同场景中的适应性。现实世界的动力学，特别是那些涉及流体和物体的动力学，需要不仅能捕捉物体相互作用，还能模拟流体流动特性的模型。为了解决这些差距，我们引入了SlotPi，一个基于槽的物理信息对象中心推理模型。SlotPi集成了基于哈密顿原理的物理模块和用于动态预测的时空预测模块。我们的实验突出了该模型在基准和流体数据集上的预测和视觉问答（VQA）等任务中的优势。此外，我们还创建了一个包含对象交互、流体动力学和流体-对象交互的真实世界数据集，我们在此数据集上验证了我们模型的能力。该模型在所有数据集上表现出强大的性能，突显了其强大的适应性，为开发更高级的世界模型奠定了基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10778) | **Categories:** cs.CV, cs.AI, cs.LG

---


## cs.DC [cs.DC]
### [1] [HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration](https://arxiv.org/abs/2506.10401)
*Jiaqi Lv, Xufeng He, Yanchen Liu, Xu Dai, Yang Hu, Shouyi Yin*

Main category: cs.DC

TL;DR: 该论文提出了一个新框架，利用AI编译器和自动优化技术生成高性能CUDA代码，并通过实验验证了其在CUDA转译方面的有效性。


<details>
  <summary>Details</summary>
Motivation: CUDA生态系统在并行软件领域占据主导地位，这要求其他硬件平台支持基于CUDA的软件，但由于并行编程范式和硬件架构的差异，将CUDA代码转换为其他平台面临着巨大的挑战。现有方法在工作负载覆盖和泛化方面存在局限性，并且通常会产生巨大的开发成本。现有LLM在CUDA转译中的性能，尤其是在高性能代码方面，仍然不是最优的，主要原因是缺乏高质量的训练数据集。

Method: 利用AI编译器和自动优化技术，提出了一个用于生成高性能CUDA和相应平台代码对的新框架，并使用基于图的数据增强方法对其进行了增强。

Result: 通过在领先的LLM上使用CUDA-to-CPU转译作为案例研究进行实验，证明了该框架显着提高了CUDA转译的性能。

Conclusion: 实验结果表明，该框架显著提高了CUDA转译的性能，突显了LLM在解决CUDA生态系统中的兼容性挑战方面的潜力。

Abstract: 深度学习的快速发展推动了模型参数和计算需求的指数级增长。NVIDIA GPU及其基于CUDA的软件生态系统为并行计算提供了强大的支持，显着缓解了计算瓶颈。同时，由于用户编程习惯的培养和GPU的高性能，CUDA生态系统在并行软件领域占据了主导地位。这种主导地位要求其他硬件平台支持具有性能可移植性的基于CUDA的软件。然而，由于并行编程范式和硬件架构的差异，将CUDA代码转换为其他平台面临着巨大的挑战。现有方法依赖于语言扩展、领域特定语言 (DSL) 或编译器，但在工作负载覆盖和通用性方面面临限制。此外，这些方法通常会产生巨大的开发成本。最近，LLM在各种垂直领域，尤其是在代码相关任务中，表现出了非凡的潜力。然而，现有LLM在CUDA转译中的性能，尤其是在高性能代码方面，仍然不是最优的。这种限制的主要原因在于缺乏高质量的训练数据集。为了应对这些挑战，我们提出了一个新颖的框架，用于生成高性能CUDA和相应的平台代码对，利用AI编译器和自动优化技术。我们进一步使用基于图的数据增强方法增强了该框架，并推出了HPCTransEval，这是一个用于评估LLM在CUDA转译方面的性能的基准。我们以CUDA-to-CPU转译作为案例研究，对领先的LLM进行了实验。结果表明，我们的框架显着提高了CUDA转译，突显了LLM在解决CUDA生态系统中的兼容性挑战方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10401) | **Categories:** cs.DC, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices](https://arxiv.org/abs/2506.10443)
*Zhaode Wang, Jingbang Yang, Xinyu Qian, Shiwen Xing, Xiaotang Jiang, Chengfei Lv, Shengyu Zhang*

Main category: cs.LG

TL;DR: MNN-LLM是一个用于加速移动设备上大型语言模型推理的框架，通过多种优化策略实现了显著的性能提升。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型在边缘设备上的部署面临内存占用和推理速度的挑战。

Method: 通过模型量化、DRAM-Flash混合存储、权重和输入重排以及多核负载均衡等策略优化LLM在移动设备上的推理。

Result: MNN-LLM相比主流LLM框架实现了高达8.6倍的速度提升。

Conclusion: MNN-LLM框架在移动设备上实现了显著的LLM推理加速。

Abstract: 大型语言模型（LLM）在各种任务中表现出卓越的性能。然而，它们庞大的规模导致推理过程中大量的计算资源消耗，从而导致高成本。因此，边缘设备推理提供了一个有希望的解决方案。边缘推理的主要挑战包括内存使用和推理速度。本文介绍MNN-LLM，一个专门为加速大型语言模型在移动设备上的部署而设计的框架。MNN-LLM通过模型量化和DRAM-Flash混合存储来解决LLM的运行时特性，有效地减少了内存使用。它基于移动CPU指令集和GPU特性重新排列权重和输入，同时采用多核负载平衡、混合精度浮点运算和几何计算等策略来提高性能。值得注意的是，与当前主流的LLM专用框架相比，MNN-LLM实现了高达8.6倍的速度提升。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10443) | **Categories:** cs.LG

---

### [2] [Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs](https://arxiv.org/abs/2506.10630)
*Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, Jintao Zhang*

Main category: cs.LG

TL;DR: Time-R1 是一个两阶段强化微调框架，旨在增强 LLM 对时间序列预测的多步推理能力。


<details>
  <summary>Details</summary>
Motivation: 为了推进时间序列预测 (TSF)，已经提出了各种方法来提高预测准确性，从统计技术发展到数据驱动的深度学习架构。尽管它们有效，但大多数现有方法仍然坚持快速思考模式——依赖于提取历史模式并将它们映射到未来值作为其核心建模理念，缺乏包含中间时间序列推理的显式思考过程。与此同时，新兴的慢思考法学硕士（例如，OpenAI-o1）已经显示出卓越的多步推理能力，为克服这些问题提供了另一种方法。然而，仅靠提示工程就存在一些局限性——包括高计算成本、隐私风险以及深入的领域特定时间序列推理能力有限。

Method: 我们提出了 Time-R1，这是一个两阶段强化微调框架，旨在增强法学硕士对时间序列预测的多步推理能力。具体来说，第一阶段进行监督微调以进行预热适应，而第二阶段采用强化学习来提高模型的泛化能力。特别是，我们专门为时间序列预测设计了一个细粒度的多目标奖励，然后引入 GRIP（基于组的策略优化相对重要性），它利用非均匀抽样来进一步鼓励和优化模型对有效推理路径的探索。

Result: Time-R1 显著提高了各种数据集上的预测性能。

Conclusion: 实验表明，Time-R1 显著提高了各种数据集上的预测性能。

Abstract: 为了推进时间序列预测 (TSF)，已经提出了各种方法来提高预测准确性，从统计技术发展到数据驱动的深度学习架构。尽管它们有效，但大多数现有方法仍然坚持快速思考模式——依赖于提取历史模式并将它们映射到未来值作为其核心建模理念，缺乏包含中间时间序列推理的显式思考过程。与此同时，新兴的慢思考法学硕士（例如，OpenAI-o1）已经显示出卓越的多步推理能力，为克服这些问题提供了另一种方法。然而，仅靠提示工程就存在一些局限性——包括高计算成本、隐私风险以及深入的领域特定时间序列推理能力有限。为了解决这些限制，一种更有希望的方法是训练法学硕士开发慢思考能力并获得强大的时间序列推理技能。为此，我们提出了 Time-R1，这是一个两阶段强化微调框架，旨在增强法学硕士对时间序列预测的多步推理能力。具体来说，第一阶段进行监督微调以进行预热适应，而第二阶段采用强化学习来提高模型的泛化能力。特别是，我们专门为时间序列预测设计了一个细粒度的多目标奖励，然后引入 GRIP（基于组的策略优化相对重要性），它利用非均匀抽样来进一步鼓励和优化模型对有效推理路径的探索。实验表明，Time-R1 显著提高了各种数据集上的预测性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10630) | **Categories:** cs.LG, cs.AI

---

### [3] [NoLoCo: No-all-reduce Low Communication Training Method for Large Models](https://arxiv.org/abs/2506.10911)
*Jari Kolehmainen, Nikolay Blagoev, John Donaghy, Oğuzhan Ersoy, Christopher Nies*

Main category: cs.LG

TL;DR: NoLoCo提出了一种无需显式同步模型参数的优化方法，降低了通信开销，提高了训练效率。


<details>
  <summary>Details</summary>
Motivation: 大规模语言模型的训练通常需要在包含数万个加速器的集群上进行，这种集群的扩展成本高昂且不切实际，限制了可以训练的模型大小。现有的低通信训练方法仍然需要同步模型参数，在低带宽网络上成本很高。

Method: 提出了一种名为NoLoCo的新型优化方法，该方法通过Nesterov动量优化器的变体隐式同步模型权重，无需显式同步所有模型参数，从而避免了集体通信。

Result: 在125M到6.8B参数的不同模型大小和加速器数量下，NoLoCo的通信开销明显低于完全分片数据并行训练，甚至低于广泛使用的低通信训练方法DiLoCo。同步步骤本身比DiLoCo中使用的all-reduce快一个数量级，且加速器空闲时间更少。与DiLoCo相比，NoLoCo在各种模型大小和加速器数量下，收敛速度提高了4%。

Conclusion: NoLoCo在不同模型大小和加速器数量下，比DiLoCo收敛速度快4%，且通信开销更低，加速器空闲时间更少。

Abstract: 训练大型语言模型通常通过在包含数万个加速器的集群上进行优化方法来完成，这些加速器通过高带宽互连进行通信。扩展这些集群成本很高，并且可能变得不切实际，从而限制了可以训练的模型的大小。最近的一些研究提出了通信强度较低的训练方法，避免了对高度连接的计算集群的需求。这些最先进的低通信训练方法仍然采用模型参数的同步步骤，当在所有模型副本上执行时，在低带宽网络上可能会变得昂贵。在这项工作中，我们提出了一种新颖的优化方法NoLoCo，该方法在训练期间不会显式同步所有模型参数，因此不需要任何集体通信。NoLoCo通过Nesterov动量优化器的一种新变体隐式同步模型权重，该变体通过与随机选择的另一个模型权重进行部分平均来实现。我们为我们提出的优化器提供了理论收敛分析以及来自语言模型训练的经验结果。我们在1.25亿到68亿参数的各种加速器数量和模型大小上对NoLoCo进行了基准测试。我们的方法比完全分片数据并行训练甚至广泛使用的低通信训练方法DiLoCo需要显着更少的通信开销。据估计，在互联网上训练的数百个加速器中，同步步骤本身比DiLoCo中使用的all-reduce快一个数量级。我们也没有任何全局阻塞通信，从而减少了加速器的空闲时间。与DiLoCo相比，我们还观察到在各种模型大小和加速器数量下，收敛速度提高了高达4%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10911) | **Categories:** cs.LG

---


## cs.MM [cs.MM]
### [1] [Multimodal Large Language Models: A Survey](https://arxiv.org/abs/2506.10016)
*Longzhen Han, Awes Mubarak, Almas Baimagambetov, Nikolaos Polatidis, Thar Baker*

Main category: cs.MM

TL;DR: 本综述概述了多模态大型语言模型的发展，并探讨了实现通用多模态系统的关键路径。


<details>
  <summary>Details</summary>
Motivation: 多模态大型语言模型（MLLM）已迅速发展，超越了文本生成，现在涵盖了包括图像、音乐、视频、人体运动和 3D 对象在内的各种输出模态。

Method: 该综述通过整合语言和其他感官模态，实现了跨模态能力的涌现，并在评估、模块化和结构化推理方面仍面临挑战。

Result: 分析了关键模型、架构趋势和新兴的跨模态协同作用，同时强调了可转移的技术和未解决的挑战。Transformer和扩散模型等架构创新是这种融合的基础，实现了跨模态迁移和模块化专业化。

Conclusion: 多模态大型语言模型通过整合语言和其他感官模态，实现了跨模态能力的涌现，并在评估、模块化和结构化推理方面仍面临挑战。

Abstract: 多模态大型语言模型（MLLM）已经超越了文本生成，扩展到包括图像、音乐、视频、人体运动和3D对象等多种输出模态。本综述对六种主要的生成模态进行了分类，并研究了自监督学习（SSL）、混合专家（MoE）、人类反馈强化学习（RLHF）和思维链（CoT）提示等基础技术如何实现跨模态能力。我们分析了关键模型、架构趋势和新兴的跨模态协同效应，同时强调了可转移的技术和尚未解决的挑战。Transformer和扩散模型等架构创新是这种融合的基础，实现了跨模态迁移和模块化专业化。我们强调了新兴的协同模式，并指出了在评估、模块化和结构化推理方面的开放性挑战。本综述为MLLM的发展提供了一个统一的视角，并确定了通向更通用、自适应和可解释的多模态系统的关键路径。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10016) | **Categories:** cs.MM, cs.AI, cs.CL

---


## cs.NI [cs.NI]
### [1] [AI5GTest: AI-Driven Specification-Aware Automated Testing and Validation of 5G O-RAN Components](https://arxiv.org/abs/2506.10111)
*Abiodun Ganiyu, Pranshav Gajjar, Vijay K Shah*

Main category: cs.NI

TL;DR: AI5GTest是一个基于AI的O-RAN组件自动化验证框架，通过协同LLM框架显著减少了测试时间并保持了高准确性。


<details>
  <summary>Details</summary>
Motivation: 现有的O-RAN测试框架依赖于手动流程，容易出错，并且缺乏一致性和可扩展性。

Method: AI5GTest利用一个协同的大型语言模型（LLM）框架，包括Gen-LLM、Val-LLM和Debug-LLM，以自动化O-RAN组件的验证。

Result: AI5GTest在O-RAN TIFG和WG5-IOT测试规范的测试用例中进行了评估，结果表明，与传统的手动方法相比，整体测试执行时间显著减少，同时保持了较高的验证准确性。

Conclusion: AI5GTest显著减少了测试执行时间，同时保持了较高的验证准确性。

Abstract: 开放无线接入网络（O-RAN）通过促进互操作性、供应商多样性和快速创新，改变了电信行业。然而，其分离的架构引入了复杂的测试挑战，特别是在根据O-RAN联盟和3GPP规范验证多供应商组件方面。现有的框架，如开放测试和集成中心（OTIC）提供的框架，严重依赖于手动流程，这些流程是分散的，容易出现人为错误，导致不一致和可扩展性问题。为了解决这些限制，我们提出了AI5GTest——一个人工智能驱动的、规范感知的测试框架，旨在自动化O-RAN组件的验证。AI5GTest利用一个协同的大型语言模型（LLM）框架，包括Gen-LLM、Val-LLM和Debug-LLM。Gen-LLM根据3GPP和O-RAN规范自动生成测试用例的预期程序流程，而Val-LLM将信令消息与这些流程进行交叉引用，以验证合规性并检测偏差。如果出现异常，Debug-LLM会执行根本原因分析，从而深入了解故障原因。为了提高透明度和可信度，AI5GTest结合了一个人工参与的机制，其中Gen-LLM向测试人员呈现前k个相关的官方规范，以供批准，然后再继续进行验证。AI5GTest使用从O-RAN TIFG和WG5-IOT测试规范获得的一系列测试用例进行评估，结果表明，与传统的手动方法相比，整体测试执行时间显著减少，同时保持了较高的验证准确性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10111) | **Categories:** cs.NI, cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding](https://arxiv.org/abs/2506.10756)
*Yuhang Zhang, Haosheng Yu, Jiaping Xiao, Mir Feroskhan*

Main category: cs.RO

TL;DR: VLFly是一种新的无人机视觉语言导航框架，它使用大型语言模型和视觉语言模型来实现强大的泛化能力和开放词汇目标理解。


<details>
  <summary>Details</summary>
Motivation: 视觉语言导航（VLN）是自主机器人领域一个长期存在的挑战，旨在使智能体能够在复杂环境中按照人类指令进行导航。该领域仍然存在两个关键瓶颈：推广到分布外环境和依赖于固定的离散动作空间。

Method: VLFly集成了三个模块：一个基于大型语言模型（LLM）的指令编码器，将高级语言重构为结构化提示，一个由视觉语言模型（VLM）驱动的目标检索器，通过视觉语言相似性将这些提示与目标图像匹配，以及一个航点规划器，为实时无人机控制生成可执行的轨迹。

Result: VLFly在各种模拟环境中都优于所有基线。此外，在直接和间接指令下，室内和室外环境中的真实世界VLN任务表明，即使在存在抽象语言输入的情况下，VLFly也能实现强大的开放词汇目标理解和广义导航能力。

Conclusion: VLFly在各种模拟环境中都优于所有基线，并在室内和室外环境中的真实世界VLN任务中实现了强大的开放词汇目标理解和广义导航能力。

Abstract: 视觉语言导航（VLN）是自主机器人领域的一个长期挑战，旨在使智能体能够在复杂环境中按照人类指令进行导航。该领域仍然存在两个关键瓶颈：推广到分布外环境和依赖于固定的离散动作空间。为了应对这些挑战，我们提出了一种专为无人机（UAV）量身定制的框架Vision-Language Fly（VLFly），以执行语言引导的飞行。VLFly不需要定位或主动测距传感器，仅从机载单目摄像头捕获的自我中心观测输出连续速度命令。VLFly集成了三个模块：一个基于大型语言模型（LLM）的指令编码器，将高级语言重构为结构化提示，一个由视觉语言模型（VLM）驱动的目标检索器，通过视觉语言相似性将这些提示与目标图像匹配，以及一个航点规划器，为实时无人机控制生成可执行的轨迹。VLFly在各种模拟环境中进行了评估，无需额外的微调，并且始终优于所有基线。此外，在直接和间接指令下，室内和室外环境中的真实世界VLN任务表明，即使在存在抽象语言输入的情况下，VLFly也能实现强大的开放词汇目标理解和广义导航能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10756) | **Categories:** cs.RO, cs.AI

---

### [2] [A Navigation Framework Utilizing Vision-Language Models](https://arxiv.org/abs/2506.10172)
*Yicheng Duan, Kaiyu tang*

Main category: cs.RO

TL;DR: 该论文提出了一种模块化的视觉-语言导航框架，通过集成大型语言模型和轻量级规划逻辑，旨在实现高效且适应性强的导航。


<details>
  <summary>Details</summary>
Motivation: 视觉-语言导航（VLN）在具身智能中提出了一个复杂的挑战，要求智能体解释自然语言指令并在视觉丰富的陌生环境中导航。大型视觉-语言模型（LVLM）的最新进展，如CLIP和Flamingo，显着提高了多模态理解，但也带来了与计算成本和实时部署相关的新挑战。

Method: 通过将冻结的视觉-语言模型Qwen2.5-VL-7B-Instruct与轻量级规划逻辑相结合，我们旨在实现灵活、快速和适应性强的导航，而无需进行大量的模型微调。我们的框架利用提示工程、结构化历史管理和双帧视觉输入策略来增强导航步骤中的决策连续性。

Result: 我们在VLN-CE设置中使用Matterport3D数据集和Habitat-Lab模拟环境，在Room-to-Room基准上评估了我们的系统。我们的初步结果显示出泛化到严格评估设置下未见环境的挑战。

Conclusion: 虽然在严格的评估环境下，我们的初步结果显示出泛化到未见环境的挑战，但我们的模块化方法为可扩展和高效的导航系统奠定了基础，突出了未来通过增强环境先验和扩展多模态输入集成来改进的有希望的方向。

Abstract: 视觉-语言导航（VLN）在具身智能领域提出了复杂的挑战，要求智能体理解自然语言指令并在视觉丰富的陌生环境中导航。大型视觉-语言模型（LVLM）如CLIP和Flamingo的最新进展虽然显著提升了多模态理解能力，但也带来了计算成本和实时部署方面的新挑战。本项目提出了一种模块化的、即插即用的导航框架，将视觉-语言理解与动作规划解耦。通过集成冻结的视觉-语言模型Qwen2.5-VL-7B-Instruct与轻量级的规划逻辑，旨在实现灵活、快速和适应性强的导航，而无需进行大量的模型微调。我们的框架利用提示工程、结构化历史管理和双帧视觉输入策略来增强导航步骤中的决策连续性。我们在Matterport3D数据集和Habitat-Lab模拟环境中的Room-to-Room基准上，在VLN-CE设置下评估了我们的系统。虽然初步结果显示在严格的评估设置下，泛化到未见环境存在挑战，但我们的模块化方法为可扩展和高效的导航系统奠定了基础，并突出了未来通过增强环境先验知识和扩展多模态输入集成来改进的有希望的方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10172) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [3] [Leveraging LLMs for Mission Planning in Precision Agriculture](https://arxiv.org/abs/2506.10093)
*Marcos Abel Zuzuárregui, Stefano Carpin*

Main category: cs.RO

TL;DR: 本文提出了一种利用大型语言模型和ROS2的端到端系统，使用户能够通过自然语言指令将复杂的数据收集任务分配给自主机器人。


<details>
  <summary>Details</summary>
Motivation: 机器人和人工智能在精准农业中具有巨大潜力，但终端用户缺乏技术专长，难以使机器人适应执行多样化任务。

Method: 利用大型语言模型（ChatGPT）将自然语言指令转换为机器人任务，并使用IEEE任务规范标准编码任务计划，通过ROS2节点在机器人上执行。

Result: 实验结果突出了大型语言模型在机器人任务分配中的优势和局限性，特别是在空间推理和解决复杂路径规划挑战方面，并展示了所提出的实现如何克服这些局限性。

Conclusion: 大型语言模型在机器人任务分配中表现出潜力，但空间推理和复杂路径规划方面存在局限性，可以通过改进实现克服。

Abstract: 机器人和人工智能在推进精准农业方面具有巨大潜力。虽然机器人系统已成功部署于各种任务，但使其适应执行多样化任务仍然具有挑战性，特别是由于终端用户通常缺乏技术专长。在本文中，我们提出了一个端到端系统，该系统利用大型语言模型（LLM），特别是ChatGPT，使用户能够使用自然语言指令将复杂的数据收集任务分配给自主机器人。为了提高可重用性，任务计划使用现有的IEEE任务规范标准进行编码，并通过ROS2节点在机器人上执行，这些节点将高级任务描述与现有的ROS库连接起来。通过大量的实验，我们强调了LLM在这种环境中的优势和局限性，特别是在空间推理和解决复杂路由挑战方面，并展示了我们提出的实现如何克服这些局限性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10093) | **Categories:** cs.RO, cs.AI

---

### [4] [One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture](https://arxiv.org/abs/2506.10106)
*Marcos Abel Zuzuárregui, Mustafa Melih Toslak, Stefano Carpin*

Main category: cs.RO

TL;DR: 本文提出了一个自然语言机器人任务规划器，使非专业人员可以通过通用界面控制异构机器人。


<details>
  <summary>Details</summary>
Motivation: 人工智能正在改变精准农业，为农民提供了简化日常运营的新工具。虽然这些技术进步有望提高效率，但它们通常会引入额外的复杂性和陡峭的学习曲线，这对必须平衡技术采用与现有工作负载的非技术用户来说尤其具有挑战性。

Method: 利用大型语言模型 (LLM) 和预定义的基元，将人类语言无缝转换为可由不同机器人平台执行的中间描述。

Result: 结果表明，该架构具有足够的通用性，可以支持各种机器人，并且足够强大，可以执行复杂的任务请求。

Conclusion: 该研究表明，该架构具有足够的通用性，可以支持各种机器人，并且足够强大，可以执行复杂的任务请求。这项工作代表着朝着使精准农业中的机器人自动化更易于非技术用户访问的方向迈出的重要一步。

Abstract: 人工智能正在改变精准农业，为农民提供新的工具来简化他们的日常操作。虽然这些技术进步有望提高效率，但它们通常会引入额外的复杂性和陡峭的学习曲线，这对必须平衡技术采用与现有工作负载的非技术用户来说尤其具有挑战性。在本文中，我们提出了一个自然语言 (NL) 机器人任务规划器，使非专业人员可以通过通用界面控制异构机器人。通过利用大型语言模型 (LLM) 和预定义的基元，我们的架构可以将人类语言无缝转换为可由不同机器人平台执行的中间描述。使用此系统，用户无需编写任何代码即可制定复杂的农业任务。在本文中，我们通过涉及机器人操作和计算机视觉任务的新一类实验，扩展了我们之前为轮式机器人任务规划量身定制的系统。我们的结果表明，该架构具有足够的通用性，可以支持各种机器人，并且足够强大，可以执行复杂的任务请求。这项工作代表着朝着使精准农业中的机器人自动化更易于非技术用户访问的方向迈出的重要一步。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10106) | **Categories:** cs.RO, cs.AI

---

### [5] [A Unified Framework for Probabilistic Dynamic-, Trajectory- and Vision-based Virtual Fixtures](https://arxiv.org/abs/2506.10239)
*Maximilian Mühlbauer, Freek Stulp, Sylvain Calinon, Alin Albu-Schäffer, João Silvério*

Main category: cs.RO

TL;DR: 本文提出了一个统一的概率虚拟夹具框架，可以在手动、半自动和完全自主模式之间无缝切换，并实验验证了其在不同机器人上的有效性。


<details>
  <summary>Details</summary>
Motivation: 概率虚拟夹具（VF）能够基于学习或感知的不确定性，为任务的每个阶段自适应地选择最合适的触觉反馈。虽然保持人在环路中仍然至关重要，例如，为了确保高精度，某些任务阶段的部分自动化对于提高生产力至关重要。

Method: 我们提出了一个统一的概率虚拟夹具框架，该框架可以在手动夹具、半自动夹具（由人处理精确任务）和完全自主之间无缝切换。我们引入了一种新的基于概率动态系统的VF，用于粗略引导，使机器人能够自主完成某些任务阶段，同时保持人工操作员的参与。对于需要精确引导的任务，我们扩展了基于概率位置的轨迹夹具，使其具有自动化功能，从而可以实现无缝的人机交互以及几何感知和最佳阻抗增益。对于需要非常精确引导的手动任务，我们还扩展了具有相同几何感知和阻抗行为的视觉伺服夹具。

Result: 在不同的机器人上进行了实验验证，展示了多种操作模式和夹具编程的简易性。

Conclusion: 该方法在不同机器人上进行了实验验证，展示了多种操作模式和易于编程的夹具。

Abstract: 概率虚拟夹具(VF)能够基于学习或感知的不确定性，为任务的每个阶段自适应地选择最合适的触觉反馈。虽然保持人在环路中仍然至关重要，例如，为了确保高精度，某些任务阶段的部分自动化对于提高生产力至关重要。我们提出了一个统一的概率虚拟夹具框架，该框架可以在手动夹具、半自动夹具（由人处理精确任务）和完全自主之间无缝切换。我们引入了一种新的基于概率动态系统的VF，用于粗略引导，使机器人能够自主完成某些任务阶段，同时保持人工操作员的参与。对于需要精确引导的任务，我们扩展了基于概率位置的轨迹夹具，使其具有自动化功能，从而可以实现无缝的人机交互以及几何感知和最佳阻抗增益。对于需要非常精确引导的手动任务，我们还扩展了具有相同几何感知和阻抗行为的视觉伺服夹具。我们通过在不同的机器人上进行实验验证了我们的方法，展示了多种操作模式和夹具编程的简易性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10239) | **Categories:** cs.RO

---

### [6] [Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving](https://arxiv.org/abs/2506.10317)
*Akshar Tumu, Henrik I. Christensen, Marcell Vazquez-Chanlatte, Chikao Tsuchiya, Dhaval Bhanderi*

Main category: cs.RO

TL;DR: 通过融合道路元数据和车道宽度先验知识，该研究改进了SMERF模型的车道拓扑预测能力。


<details>
  <summary>Details</summary>
Motivation: 车道拓扑预测是安全可靠的自动导航的关键组成部分。准确理解道路环境有助于完成这项任务。

Method: 通过结合来自OSM地图的结构化道路元数据和来自道路设计手册的车道宽度先验知识，并将其与道路中心线编码相结合，以轻量级的方式增强了SMERF模型。

Result: 该方法在车道和交通元素检测及其关联方面均有改进。使用四个拓扑感知指标全面评估模型性能。

Conclusion: 该方法在不同的拓扑结构和条件下都表现出良好的泛化和扩展能力。

Abstract: 车道拓扑预测是安全可靠的自动导航的关键组成部分。准确理解道路环境有助于完成这项任务。我们观察到，这些信息通常遵循自然语言中编码的约定，通过反映道路结构的设计规范和捕获道路功能的道路名称。我们以轻量级的方式将这些信息增强到SMERF（一种基于地图先验的在线车道拓扑预测模型）中，方法是将来自OSM地图的结构化道路元数据和来自道路设计手册的车道宽度先验知识与道路中心线编码相结合。我们在两种地理位置不同的复杂交叉路口场景中评估了我们的方法。我们的方法在车道和交通元素检测及其关联方面均有改进。我们使用四个拓扑感知指标报告结果，以全面评估模型性能。这些结果证明了我们的方法能够推广和扩展到不同的拓扑结构和条件。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10317) | **Categories:** cs.RO, cs.AI

---

### [7] [GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation](https://arxiv.org/abs/2506.10966)
*Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang*

Main category: cs.RO

TL;DR: GenManip是一个新的仿真平台，旨在通过使用LLM驱动的场景生成和基准测试，促进机器人策略泛化研究。


<details>
  <summary>Details</summary>
Motivation: Existing simulation platforms lack sufficient support for exploring how policies adapt to varied instructions and scenarios, lagging behind the growing interest in instruction-following foundation models.

Method: A realistic tabletop simulation platform (GenManip) with an automatic pipeline via LLM-driven task-oriented scene graph to synthesize large-scale, diverse tasks using 10K annotated 3D object assets, and a benchmark of 200 scenarios (GenManip-Bench).

Result: Data scaling benefits end-to-end methods, modular systems enhanced with foundation models generalize more effectively across diverse scenarios.

Conclusion: Modular manipulation systems enhanced with foundation models generalize more effectively across diverse scenarios compared to end-to-end policies.

Abstract: 现实世界中的机器人操作仍然具有挑战性，特别是在鲁棒泛化方面。现有的仿真平台缺乏足够的支持来探索策略如何适应不同的指令和场景。因此，它们落后于人们对指令跟随基础模型（如LLM）日益增长的兴趣，这些模型的适应性至关重要，但在公平的比较中仍未得到充分探索。为了弥合这一差距，我们推出了GenManip，这是一个逼真的桌面仿真平台，专为策略泛化研究而定制。它具有一个自动管道，通过LLM驱动的面向任务的场景图来合成大规模、多样化的任务，使用10K个带注释的3D对象资产。为了系统地评估泛化能力，我们提出了GenManip-Bench，这是一个包含200个场景的基准，通过人工循环校正进行改进。我们评估了两种策略类型：（1）集成了基础模型以进行感知、推理和规划的模块化操作系；（2）通过可扩展的数据收集训练的端到端策略。结果表明，虽然数据缩放有利于端到端方法，但通过基础模型增强的模块化系统在各种场景中更有效地泛化。我们预计该平台将有助于深入了解在实际条件下推进策略泛化。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.10966) | **Categories:** cs.RO

---
