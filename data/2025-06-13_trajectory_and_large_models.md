# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-13

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [cs.CR (1)](#cs-cr)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [cs.DC (1)](#cs-dc)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (6)](#cs-ro)
- [统计机器学习 (Machine Learning Statistics) (1)](#stat-ml)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
*Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas*

Main category: cs.AI

TL;DR: 该论文提出了一种结合互联网视频和少量机器人交互数据的自监督学习方法，用于构建能够在物理世界中进行规划的世界模型。


<details>
  <summary>Details</summary>
Motivation: 现代人工智能面临的主要挑战是学习理解世界，并在很大程度上通过观察来学习行动。本文探讨了一种自监督方法，该方法结合了互联网规模的视频数据和少量的交互数据（机器人轨迹），以开发能够理解、预测和规划物理世界的模型。

Method: 该研究首先在包含超过100万小时互联网视频的视频和图像数据集上预训练一个无动作的联合嵌入预测架构V-JEPA 2，然后在Droid数据集上使用不到62小时的无标签机器人视频对一个潜在的动作条件世界模型V-JEPA 2-AC进行后训练。

Result: V-JEPA 2 在运动理解方面取得了强大的性能（在 Something-Something v2 上达到 77.3 的 top-1 准确率），并在人类动作预测方面取得了最先进的性能（在 Epic-Kitchens-100 上达到 39.7 的 recall-at-5），超过了以前特定于任务的模型。此外，在将 V-JEPA 2 与大型语言模型对齐后，我们在 80 亿参数规模的多个视频问答任务上展示了最先进的性能（例如，在 PerceptionTest 上为 84.0，在 TempCompass 上为 76.9）。最后，我们展示了如何通过使用来自 Droid 数据集的不到 62 小时的无标签机器人视频对潜在的动作条件世界模型 V-JEPA 2-AC 进行后训练，将自监督学习应用于机器人规划任务。我们在两个不同的实验室中零样本部署 V-JEPA 2-AC 在 Franka 机械臂上，并能够使用图像目标进行规划来拾取和放置物体。值得注意的是，这是在没有从这些环境中的机器人收集任何数据，并且没有任何特定于任务的训练或奖励的情况下实现的。

Conclusion: 通过结合网络规模的视频数据和少量机器人交互数据，该研究展示了自监督学习可以产生一个能够在物理世界中进行规划的世界模型。

Abstract: 现代人工智能面临的一个主要挑战是学习理解世界，并通过大量观察来学习行动。本文探索了一种自监督方法，该方法结合了互联网规模的视频数据和少量的交互数据（机器人轨迹），以开发能够理解、预测和规划物理世界的模型。我们首先在包含超过 100 万小时互联网视频的视频和图像数据集上预训练一个无动作的联合嵌入预测架构 V-JEPA 2。V-JEPA 2 在运动理解方面取得了强大的性能（在 Something-Something v2 上达到 77.3 的 top-1 准确率），并在人类动作预测方面取得了最先进的性能（在 Epic-Kitchens-100 上达到 39.7 的 recall-at-5），超过了以前特定于任务的模型。此外，在将 V-JEPA 2 与大型语言模型对齐后，我们在 80 亿参数规模的多个视频问答任务上展示了最先进的性能（例如，在 PerceptionTest 上为 84.0，在 TempCompass 上为 76.9）。最后，我们展示了如何通过使用来自 Droid 数据集的不到 62 小时的无标签机器人视频对潜在的动作条件世界模型 V-JEPA 2-AC 进行后训练，将自监督学习应用于机器人规划任务。我们在两个不同的实验室中零样本部署 V-JEPA 2-AC 在 Franka 机械臂上，并能够使用图像目标进行规划来拾取和放置物体。值得注意的是，这是在没有从这些环境中的机器人收集任何数据，并且没有任何特定于任务的训练或奖励的情况下实现的。这项工作展示了如何通过网络规模数据和少量机器人交互数据的自监督学习，产生一个能够在物理世界中进行规划的世界模型。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09985) | **Categories:** cs.AI, cs.CV, cs.LG, cs.RO

---

### [2] [Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning](https://arxiv.org/abs/2506.09498)
*Jaesik Yoon, Hyeonseo Cho, Yoshua Bengio, Sungjin Ahn*

Main category: cs.AI

TL;DR: Fast-MCTD 通过并行化和稀疏化显著提高了蒙特卡洛树扩散（MCTD）的速度和可扩展性，同时保持或提高了规划性能。


<details>
  <summary>Details</summary>
Motivation: 扩散模型在轨迹规划中表现出色，但其非序列性限制了其在长时程推理任务中的有效性。MCTD 通过结合扩散模型和树搜索，在复杂规划问题上取得了 SOTA 性能，但计算开销较大。

Method: 提出了 Fast-MCTD，它集成了两种技术：并行 MCTD，通过延迟树更新和冗余感知选择实现并行 rollout；稀疏 MCTD，通过轨迹粗化减少 rollout 长度。

Result: Fast-MCTD 比标准 MCTD 提速高达 100 倍，同时保持或提高了规划性能。在某些任务上，其推理速度甚至超过了 Diffuser，尽管 Diffuser 不需要搜索并且解的质量较差。

Conclusion: Fast-MCTD 是一种实用的、可扩展的基于扩散的推理时推理解决方案，它通过并行化和稀疏化显著提高了 MCTD 的速度和可扩展性，同时保持或提高了规划性能。

Abstract: 扩散模型最近成为轨迹规划的强大方法。然而，它们固有的非序列性质限制了它们在测试时长期推理任务中的有效性。最近提出的蒙特卡洛树扩散（MCTD）通过将扩散与基于树的搜索相结合，提供了一个有希望的解决方案，在复杂的规划问题上实现了最先进的性能。尽管它有优势，但我们的分析表明，由于树搜索的顺序性质和迭代去噪的成本，MCTD 产生了大量的计算开销。为了解决这个问题，我们提出了 Fast-MCTD，这是一种更有效的变体，它保留了 MCTD 的优势，同时显著提高了其速度和可扩展性。Fast-MCTD 集成了两种技术：并行 MCTD，它通过延迟树更新和冗余感知选择实现并行 rollout；稀疏 MCTD，它通过轨迹粗化减少 rollout 长度。实验表明，Fast-MCTD 比标准 MCTD 提速高达 100 倍，同时保持或提高了规划性能。值得注意的是，在某些任务上，它的推理速度甚至超过了 Diffuser，尽管 Diffuser 不需要搜索并且产生较弱的解决方案。这些结果将 Fast-MCTD 定位为一种实用的、可扩展的基于扩散的推理时推理解决方案。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09498) | **Categories:** cs.AI

---


## cs.CR [cs.CR]
### [1] [What is the Cost of Differential Privacy for Deep Learning-Based Trajectory Generation?](https://arxiv.org/abs/2506.09312)
*Erik Buchholz, Natasha Fernandes, David D. Nguyen, Alsharif Abuadbba, Surya Nepal, Salil S. Kanhere*

Main category: cs.CR

TL;DR: DP轨迹生成具有挑战性，形式化保证需大数据集和受限用例。


<details>
  <summary>Details</summary>
Motivation: 虽然位置轨迹提供了有价值的见解，但它们也揭示了敏感的个人信息。差分隐私（DP）提供了正式的保护，但实现有利的效用-隐私权衡仍然具有挑战性。目前的工作探索了基于深度学习的生成模型来生成合成轨迹。但是，当前的模型缺乏正式的隐私保证，并且在生成过程中依赖于从真实数据得出的条件信息。

Method: 我们为条件生成提出了一种新的DP机制，该机制提供了形式化保证，并评估了其对效用的影响。

Result: 我们的结果表明，DP-SGD会显着影响性能，但是如果数据集足够大，则仍然保留一些效用。所提出的DP机制提高了训练稳定性，尤其是当与DP-SGD结合使用时，对于不稳定的模型（例如GAN）以及在较小的数据集上。扩散模型在没有保证的情况下可产生最佳效用，但是使用DP-SGD，GAN的性能最佳，这表明在针对形式保证时，最佳的非私有模型不一定是最佳的。

Conclusion: DP轨迹生成仍然具有挑战性，目前的形式化保证只能通过大型数据集和在受约束的用例中实现。

Abstract: 虽然位置轨迹提供了有价值的见解，但它们也揭示了敏感的个人信息。差分隐私（DP）提供了正式的保护，但实现有利的效用-隐私权衡仍然具有挑战性。最近的工作探索了基于深度学习的生成模型来生成合成轨迹。但是，当前的模型缺乏正式的隐私保证，并且在生成过程中依赖于从真实数据得出的条件信息。这项工作研究了在此类模型中强制执行DP的效用成本，从而解决了跨两个数据集和11个效用指标的三个研究问题。(1) 我们评估了DP-SGD（深度学习的标准DP训练方法）如何影响最新生成模型的效用。(2) 由于DP-SGD仅限于无条件模型，因此我们为条件生成提出了一种新颖的DP机制，该机制提供了形式保证，并评估了其对效用的影响。(3) 我们分析了模型类型（扩散、VAE和GAN）如何影响效用-隐私权衡。我们的结果表明，DP-SGD会显着影响性能，但是如果数据集足够大，则仍然保留一些效用。所提出的DP机制提高了训练稳定性，尤其是当与DP-SGD结合使用时，对于不稳定的模型（例如GAN）以及在较小的数据集上。扩散模型在没有保证的情况下可产生最佳效用，但是使用DP-SGD，GAN的性能最佳，这表明在针对形式保证时，最佳的非私有模型不一定是最佳的。总而言之，DP轨迹生成仍然是一项具有挑战性的任务，并且目前的形式保证只能通过大型数据集和在受约束的用例中实现。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09312) | **Categories:** cs.CR, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [OctoNav: Towards Generalist Embodied Navigation](https://arxiv.org/abs/2506.09839)
*Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, Si Liu*

Main category: cs.CV

TL;DR: 本文提出了 OctoNav-Bench 和 OctoNav-R1，旨在构建能够遵循复杂指令的通用具身导航代理，并通过“行动前思考”提高模型推理能力。


<details>
  <summary>Details</summary>
Motivation: 以往的导航研究被划分为不同的任务/能力，例如ObjNav、ImgNav和VLN，它们在任务目标和模态上有所不同，使得数据集和方法都是单独设计的。本文旨在构建能够遵循包含多模态和多能力任意组合的自由形式指令的通用导航代理。

Method: 提出了一个基于MLLM并适应于VLA类型模型的OctoNav-R1，以及一个包含动作/TBA-SFT、Nav-GPRO和在线RL三个阶段的混合训练范式（HTP）。

Result: OctoNav-R1 相比现有方法表现出卓越的性能。

Conclusion: OctoNav-R1 在性能上优于现有方法，展示了其在具身导航领域实现“行动前思考”的潜力。

Abstract: 具身导航是具身人工智能的重要基石。然而，以往的导航研究被划分为不同的任务/能力，例如ObjNav、ImgNav和VLN，它们在任务目标和模态上有所不同，使得数据集和方法都是单独设计的。在这项工作中，我们朝着通用导航代理迈进了一步，它可以遵循包含多模态和多能力任意组合的自由形式指令。为了实现这一目标，我们提出了一个大规模的基准测试和相应的方法，分别称为OctoNav-Bench和OctoNav-R1。具体来说，OctoNav-Bench具有连续环境，并通过设计的标注流程构建。我们精心制作了指令-轨迹对，其中指令以自由形式呈现，具有任意的模态和能力。此外，我们在OctoNav-Bench中构建了一个“行动前思考”（TBA-CoT）数据集，以提供行动背后的思考过程。对于OctoNav-R1，我们以MLLM为基础，并将其适配为一个VLA类型的模型，该模型可以仅基于2D视觉观察产生低级动作。此外，我们设计了一个包含三个阶段的混合训练范式（HTP），即动作/TBA-SFT、Nav-GPRO和在线RL阶段。每个阶段都包含专门设计的学习策略和奖励。重要的是，对于TBA-SFT和Nav-GRPO的设计，我们受到了OpenAI-o1和DeepSeek-R1的启发，它们通过“行动前思考”展示了令人印象深刻的推理能力。因此，我们的目标是研究如何在具身导航领域实现“行动前思考”，以提高模型对通用性的推理能力。具体来说，我们提出TBA-SFT来利用TBA-CoT数据集来微调模型，作为一个冷启动阶段，然后利用Nav-GPRO来提高其思考能力。最后，OctoNav-R1 相比现有方法表现出卓越的性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09839) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [ReSim: Reliable World Simulation for Autonomous Driving](https://arxiv.org/abs/2506.09981)
*Jiazhi Yang, Kashyap Chitta, Shenyuan Gao, Long Chen, Yuqian Shao, Xiaosong Jia, Hongyang Li, Andreas Geiger, Xiangyu Yue, Li Chen*

Main category: cs.CV

TL;DR: ReSim 通过结合真实和模拟数据，构建了一个可控的驾驶世界模型，能够可靠地模拟各种驾驶行为，并使用 Video2Reward 模块估计奖励。


<details>
  <summary>Details</summary>
Motivation: 现有的驾驶世界模型难以模拟危险或非专业行为，限制了它们在策略评估等任务中的应用。本文旨在通过丰富真实世界人类演示数据，使其包含来自驾驶模拟器的各种非专业数据，来解决这一挑战。

Method: 通过使用扩散 Transformer 架构的视频生成器，并设计多种策略来有效整合调节信号，从而构建可控的世界模型。

Result: ReSim 在视觉保真度上提高了 44%，对专业和非专业行为的可控性提高了 50% 以上，并在 NAVSIM 上的规划和策略选择性能分别提高了 2% 和 25%。

Conclusion: ReSim 实现了对各种开放世界驾驶场景下各种行为（包括危险的非专业行为）的可靠模拟，并通过 Video2Reward 模块估计奖励。

Abstract: 如何可靠地模拟各种自我驾驶行为下的未来驾驶场景？最近的驾驶世界模型仅在真实世界的驾驶数据上开发，这些数据主要由安全的专家轨迹组成，难以模拟危险或非专业行为，因为这些行为在数据中很少见。这一限制限制了它们在策略评估等任务中的应用。在这项工作中，我们通过使用从驾驶模拟器（例如，CARLA）收集的各种非专业数据来丰富真实世界的人类演示，并构建一个在此异构语料库上训练的可控世界模型，从而应对这一挑战。从具有扩散 Transformer 架构的视频生成器开始，我们设计了几种策略来有效地整合调节信号，并提高预测的可控性和保真度。由此产生的模型 ReSim 能够在各种动作下可靠地模拟各种开放世界驾驶场景，包括危险的非专业动作。为了缩小高保真模拟与需要奖励信号来判断不同动作的应用之间的差距，我们引入了一个 Video2Reward 模块，该模块可以从 ReSim 模拟的未来中估计奖励。我们的 ReSim 范例实现了高达 44% 的视觉保真度，将专家和非专家行动的可控性提高了 50% 以上，并分别提高了 NAVSIM 上 2% 和 25% 的规划和策略选择性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09981) | **Categories:** cs.CV, cs.RO

---

### [3] [AD^2-Bench: A Hierarchical CoT Benchmark for MLLM in Autonomous Driving under Adverse Conditions](https://arxiv.org/abs/2506.09557)
*Zhaoyang Wei, Chenhui Qiang, Bowen Jiang, Xumeng Han, Xuehui Yu, Zhenjun Han*

Main category: cs.CV

TL;DR: AD^2-Bench是一个专为恶劣天气和复杂场景下的自动驾驶设计的链式思考基准测试，包含5.4k个手动注释的CoT实例，旨在评估和提升MLLM的推理能力。


<details>
  <summary>Details</summary>
Motivation: 现有的基准测试在很大程度上忽略了在这些特定和具有挑战性的场景中对CoT过程进行严格评估的需求。

Method: 提出了AD^2-Bench，一个专门为恶劣天气和复杂场景下的自动驾驶设计的Chain-of-Thought基准测试。

Result: AD^2-Bench包含超过5.4k个高质量、手动注释的CoT实例，每个中间推理步骤都被视为一个具有明确ground truth的原子单元。

Conclusion: 在AD^2-Bench上的评估结果表明，现有MLLM在恶劣天气和复杂场景下的自动驾驶推理能力不足，准确率低于60%，需要进一步提升。

Abstract: 链式思考（CoT）推理已经成为一种增强多模态大型模型（MLLM）的结构化、多步骤决策能力的强大方法，这对于在恶劣天气条件和复杂交通环境下的自动驾驶尤为重要。然而，现有的基准测试在很大程度上忽略了在这些特定和具有挑战性的场景中对CoT过程进行严格评估的需求。为了解决这个关键差距，我们推出了AD^2-Bench，这是第一个专门为恶劣天气和复杂场景下的自动驾驶设计的链式思考基准测试。AD^2-Bench的精心构建是为了满足三个关键标准：跨越不同恶劣环境的全面数据覆盖、支持多步骤推理的细粒度注释，以及为评估CoT性能量身定制的专用评估框架。AD^2-Bench的核心贡献是其超过5.4k个高质量、手动注释的CoT实例的广泛集合。这些注释中的每个中间推理步骤都被视为一个具有明确ground truth的原子单元，从而能够对MLLM在文本级别、点级别和区域级别视觉提示下的推理过程进行前所未有的细粒度分析。我们对最先进的MLLM在AD^2-Bench上的全面评估显示，准确率低于60%，这突显了基准测试的难度以及推进稳健、可解释的端到端自动驾驶系统的必要性。因此，AD^2-Bench提供了一个标准化的评估平台，通过改进MLLM在自动驾驶中的推理来推动研究，使其成为一种宝贵的资源。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09557) | **Categories:** cs.CV, cs.AI

---

### [4] [ECAM: A Contrastive Learning Approach to Avoid Environmental Collision in Trajectory Forecasting](https://arxiv.org/abs/2506.09626)
*Giacomo Rosin, Muhammad Rameez Ur Rahman, Sebastiano Vascon*

Main category: cs.CV

TL;DR: 本文提出了一个名为ECAM的环境避碰模块，通过对比学习增强轨迹预测模型在复杂环境下的避碰能力。


<details>
  <summary>Details</summary>
Motivation: 现有的轨迹预测方法通常忽略环境的影响，导致与障碍物发生碰撞。

Method: 提出了ECAM（环境避碰模块），一个基于对比学习的模块，以增强环境避碰能力。

Result: 在ETH/UCY数据集上的评估表明，该方法在定量和定性上都展示了其避碰能力，与最先进的方法集成后，碰撞率显著降低（-40/50%）。

Conclusion: 通过与ECAM模块集成，最先进的方法可以显著降低碰撞率（-40/50%）。

Abstract: 人类轨迹预测在自动驾驶、机器人和监控等应用中至关重要。精确的预测需要模型考虑各种因素，包括社会互动、多模态预测、行人意图和环境背景。虽然现有方法考虑了这些因素，但它们通常忽略了环境的影响，导致与障碍物发生碰撞。本文介绍了一种名为ECAM（环境避碰模块）的基于对比学习的模块，以增强环境避碰能力。所提出的模块可以集成到现有的轨迹预测模型中，提高其生成无碰撞预测的能力。我们在ETH/UCY数据集上评估了我们的方法，并通过定量和定性实验证明了其避碰能力。实验表明，与所提出的模块集成后，最先进的方法可以显著降低（-40/50%）碰撞率。代码可在https://github.com/CVML-CFU/ECAM获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09626) | **Categories:** cs.CV

---


## cs.DC [cs.DC]
### [1] [EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model](https://arxiv.org/abs/2506.09061)
*Alyssa Pinnock, Shakya Jayakody, Kawsher A Roxy, Md Rubel Ahmed*

Main category: cs.DC

TL;DR: EdgeProfiler是一个用于评估轻量级LLM在边缘设备上性能的快速Profiling框架，通过量化和建模实现精度、能效和计算可行性的平衡。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型(LLM)虽然在自然语言理解和生成方面表现出色，但其高昂的计算、内存和功耗需求通常限制了它们在云环境中的应用。EdgeProfiler旨在解决在资源受限的边缘环境中评估LLM性能的挑战。

Method: EdgeProfiler框架通过采用激进的量化技术和严格的内存约束，对轻量级LLM（如TinyLLaMA、Gemma3.1B等）在边缘设备上的性能进行评估，并使用分析建模来估计延迟、FLOPs和能耗。

Result: 实验结果表明，4比特量化可以将模型内存使用量减少约60-70%，同时保持精度在全精度基线的2-5%以内。推理速度比FP16基线提高了2-3倍。功耗建模估计INT4配置的能耗降低了35-50%。

Conclusion: 在边缘设备上部署轻量级LLM需要仔细权衡精度、能效和计算可行性，而EdgeProfiler为此提供了一个有效的评估框架。

Abstract: 本文介绍了一种快速Profiling框架EdgeProfiler，用于评估边缘系统上的轻量级大型语言模型（LLM）。虽然LLM在自然语言理解和生成方面提供了卓越的性能，但它们的高计算、内存和功耗需求通常将它们限制在云环境中。EdgeProfiler通过提供一种系统的方法来评估资源受限的边缘环境中的LLM性能，从而应对这些挑战。该框架使用激进的量化技术和严格的内存约束来分析小型LLM（包括TinyLLaMA、Gemma3.1B、Llama3.2-1B和DeepSeek-r1-1.5B）。分析建模用于估计延迟、FLOPs和能耗。Profiling显示，4位量化将模型内存使用量减少了约60-70%，同时保持了在全精度基线2-5%的精度。与各种边缘设备上的FP16基线相比，推理速度提高了2-3倍。功耗建模估计INT4配置的能耗降低了35-50%，从而可以在Raspberry Pi 4/5和Jetson Orin Nano Super等硬件上进行实际部署。我们的研究结果强调了针对边缘环境中轻量级LLM进行高效Profiling的重要性，从而平衡了精度、能效和计算可行性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09061) | **Categories:** cs.DC, cs.AI, cs.PF

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [On-the-Fly Adaptive Distillation of Transformer to Dual-State Linear Attention](https://arxiv.org/abs/2506.09316)
*Yeonju Ro, Zhenyu Zhang, Souvik Kundu, Zhangyang Wang, Aditya Akella*

Main category: cs.LG

TL;DR: 该论文提出了一种双状态线性注意力（DSLA）和在线自适应蒸馏框架（Serve），以加速大型语言模型的推理过程，同时保持性能。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）擅长通过自注意力捕捉全局token依赖关系，但在冗长的输入上面临巨大的计算和内存成本。虽然亚二次方法（例如，线性注意力）可以降低这些成本，但由于过度强调最近的token，它们通常会降低准确性。

Method: 提出了双状态线性注意力（DSLA），它维护两个专门的隐藏状态：一个用于保存历史上下文，一个用于跟踪最近信息。同时，提出了Serve，一个在线自适应蒸馏框架，该框架在推理时根据基于敏感度的层排序，逐步用DSLA层替换Transformer层。

Result: 在常识推理、长文本QA和文本摘要任务上，Serve的推理速度比Llama2-7B快2.3倍，比混合Zamba-7B快3.0倍，同时在下游任务中保持了相当的性能。消融研究表明，DSLA的双重状态捕获了全局和局部依赖关系，解决了先前线性注意力中历史token表示不足的问题。

Conclusion: Serve框架通过在线自适应蒸馏，用DSLA层逐步替换Transformer层，在常识推理、长文本QA和文本摘要任务上，实现了比Llama2-7B快2.3倍、比Zamba-7B快3.0倍的推理速度，同时保持了相当的性能。

Abstract: 大型语言模型(llm)擅长通过自注意力捕捉全局token依赖关系，但面临着在冗长输入上计算和内存成本过高的问题。虽然二次方法(例如，线性注意)可以降低这些成本，但由于过度强调最近的token，它们通常会降低准确性。在这项工作中，我们首先提出了双状态线性注意(dsla)，这是一种新颖的设计，它维护两个专门的隐藏状态——一个用于保存历史上下文，另一个用于跟踪recency——从而减轻了线性注意架构的典型短程偏差。为了进一步平衡动态工作负载条件下的效率和准确性，我们引入了Serve，一个在线自适应蒸馏框架，它在推理时根据基于敏感度的层排序，逐步用DSLA层替换Transformer层。Serve使用链式微调策略，以确保每个新转换的DSLA层与先前替换的层保持一致，从而保持整体质量。在常识推理、长文本问答和文本摘要方面的大量评估表明，Serve的推理速度比Llama2-7B快2.3倍，比混合Zamba-7B快3.0倍，同时在下游任务中保持了相当的性能。我们的消融研究表明，DSLA的双重状态捕获了全局和局部依赖关系，解决了先前线性注意中历史token表示不足的问题。代码可在https://github.com/utnslab/DSLA-Serve上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09316) | **Categories:** cs.LG

---

### [2] ["What are my options?": Explaining RL Agents with Diverse Near-Optimal Alternatives (Extended)](https://arxiv.org/abs/2506.09901)
*Noel Brindise, Vijeth Hebbar, Riya Shah, Cedric Langbort*

Main category: cs.LG

TL;DR: DNA通过寻找不同的近优策略来解释强化学习代理的行为，从而为轨迹规划提供多种选择。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在为轨迹规划代理寻找一组合理的“选项”，优化策略以在欧几里得空间中产生性质上不同的轨迹。

Method: DNA使用局部修改的Q学习问题中的奖励塑造来解决具有保证的epsilon-最优性的不同策略。

Result: 实验结果表明，DNA成功返回了性质上不同的策略，这些策略构成了模拟中意义上不同的“选项”。

Conclusion: DNA方法成功返回了在模拟中构成有意义的不同“选项”的性质上不同的策略，为RL中的探索和自适应规划开辟了新的可能性。

Abstract: 本文对一种新的可解释强化学习方法进行了扩展讨论，该方法称为多样性近优替代方案（DNA），该方法首次在L4DC 2025上提出。DNA旨在为轨迹规划代理寻找一组合理的“选项”，优化策略以在欧几里得空间中产生性质上不同的轨迹。本着可解释性的精神，这些不同的策略用于“解释”代理的选项，即人类用户可以从中选择的可用轨迹形状。特别地，DNA适用于基于马尔可夫决策过程的价值函数策略，其中代理仅限于连续轨迹。本文描述了DNA，它使用局部修改的Q学习问题中的奖励塑造来解决具有保证的epsilon-最优性的不同策略。我们表明，它成功返回了在模拟中构成有意义的不同“选项”的性质上不同的策略，包括与质量多样性随机优化领域中的相关方法的简要比较。除了解释性动机之外，这项工作为强化学习中的探索和自适应规划开辟了新的可能性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09901) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation](https://arxiv.org/abs/2506.09485)
*Yuxin Liu, Zhenghao Peng, Xuanhao Cui, Bolei Zhou*

Main category: cs.RO

TL;DR: Adv-BMT 框架通过逆向预测交通运动来增强自动驾驶场景，有效降低碰撞率。


<details>
  <summary>Details</summary>
Motivation: 基于场景的测试对于验证自动驾驶 (AD) 系统的性能至关重要。然而，这种测试受到现实世界中现有数据集中稀缺的长尾、安全关键场景的限制。

Method: 提出了 Adv-BMT 框架，该框架使用双向运动 Transformer (BMT) 模型执行逆向交通运动预测，以增强真实世界的场景与多样化和现实的对抗性交互。

Result: 实验结果验证了 Adv-BMT 生成的碰撞场景的质量。

Conclusion: 通过在增强数据集上进行训练，与之前的工作相比，碰撞率降低了 20%。

Abstract: 基于场景的测试对于验证自动驾驶 (AD) 系统的性能至关重要。然而，这种测试受到现实世界中现有数据集中稀缺的长尾、安全关键场景的限制。为了解决数据问题，我们提出了 Adv-BMT 框架，该框架通过多样化和现实的对抗性交互来增强真实世界的场景。Adv-BMT 的核心组件是一个双向运动 Transformer (BMT) 模型，用于执行逆向交通运动预测，它以场景中最后一个时间步的智能体信息作为输入，并以时间顺序的逆向重建交通，直到初始时间步。Adv-BMT 框架是一个两阶段的流水线：它首先进行对抗性初始化，然后进行逆向运动预测。与之前的工作不同，我们不需要任何碰撞数据进行预训练，并且能够生成真实和多样化的碰撞交互。我们的实验结果验证了 Adv-BMT 生成的碰撞场景的质量：在我们的增强数据集上进行训练将比之前的工作降低 20% 的碰撞率。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09485) | **Categories:** cs.RO, cs.AI, cs.GR

---

### [2] [Integrating Quantized LLMs into Robotics Systems as Edge AI to Leverage their Natural Language Processing Capabilities](https://arxiv.org/abs/2506.09581)
*Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, David Sobrín-Hidalgo, Ángel Manuel Guerrero-Higueras, Vicente MatellÁn-Olivera*

Main category: cs.RO

TL;DR: llama_ros是一个将量化大型语言模型集成到ROS 2机器人系统中的工具，旨在提高机器人的决策和交互能力。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）在过去一年中取得了巨大的进步，导致这些模型在多个领域中增加，以应对自然语言任务。将这些模型集成到机器人技术中也有助于改善人机交互、导航、规划和决策等多个方面。

Method: 该论文介绍了一种名为llama_ros的工具，该工具旨在将量化的LLM集成到使用ROS 2的机器人系统中。llama_ros利用高度优化的运行时引擎llama.cpp，能够在资源受限的环境中高效执行量化的LLM，作为机器人系统中的边缘人工智能。

Result: 该论文提供了一些关于使用llama_ros进行机器人规划和可解释性的用例。

Conclusion: llama_ros通过利用量化LLM，使机器人能够利用自然语言理解和生成来增强决策和交互能力，并可与提示工程、知识图谱、本体或其他工具结合使用，以提高自主机器人的能力。

Abstract: 大型语言模型（LLM）在过去一年中取得了巨大的进步，越来越多的模型被应用到各个领域以解决自然语言任务。将这些模型集成到机器人技术中也有助于改善人机交互、导航、规划和决策等多个方面。因此，本文介绍了一种名为llama_ros的工具，该工具旨在将量化的LLM集成到使用ROS 2的机器人系统中。llama_ros利用高度优化的运行时引擎llama.cpp，能够在资源受限的环境中高效执行量化的LLM，作为资源受限的机器人系统中的边缘人工智能，从而解决计算效率和内存限制的挑战。通过部署量化的LLM，llama_ros使机器人能够利用自然语言理解和生成来增强决策和交互能力，并可与提示工程、知识图谱、本体或其他工具结合使用，以提高自主机器人的能力。此外，本文还提供了一些关于使用llama_ros进行机器人规划和可解释性的用例。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09581) | **Categories:** cs.RO

---

### [3] [Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.09800)
*Haochen Liu, Tianyu Li, Haohan Yang, Li Chen, Caojun Wang, Ke Guo, Haochen Tian, Hongchen Li, Hongyang Li, Chen Lv*

Main category: cs.RO

TL;DR: R2SE 是一种新的学习流程，它通过强化学习不断改进困难驾驶场景，同时保持通用驾驶策略，从而提高端到端自动驾驶系统的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的基于模仿学习 (IL) 的模型难以泛化到困难案例，并且在部署后缺乏纠正反馈循环。强化学习 (RL) 虽然可以解决困难案例，但常常过度拟合到特定的驾驶案例，导致通用知识的灾难性遗忘和样本效率低下。

Method: 提出了一种名为Reinforced Refinement with Self-aware Expansion (R2SE) 的新型学习流程，该流程不断改进困难领域，同时保持通用驾驶策略，用于模型无关的端到端驾驶系统。R2SE 具有三个关键组件：1) 通过硬案例分配进行通用预训练；2) 残差强化专家微调；3) 自感知适配器扩展。

Result: 在闭环仿真和真实世界数据集中的实验结果表明，R2SE 在泛化性、安全性和长时程策略鲁棒性方面优于最先进的 E2E 系统。

Conclusion: 实验结果表明，R2SE在泛化性、安全性和长时程策略鲁棒性方面优于现有E2E系统，突出了强化改进对于可扩展自动驾驶的有效性。

Abstract: 端到端自动驾驶已经成为一种很有前景的范例，它使用基于学习的模块化集成将传感器输入直接映射到规划动作。然而，现有的基于模仿学习 (IL) 的模型难以泛化到困难案例，并且在部署后缺乏纠正反馈循环。虽然强化学习 (RL) 提供了一种潜在的解决方案来处理具有最优性的困难案例，但它常常受到过度拟合特定驾驶案例的阻碍，导致通用知识的灾难性遗忘和样本效率低下。为了克服这些挑战，我们提出了一种名为Reinforced Refinement with Self-aware Expansion (R2SE) 的新型学习流程，该流程不断改进困难领域，同时保持通用驾驶策略，用于模型无关的端到端驾驶系统。通过强化微调和促进持续改进的策略扩展，R2SE 具有三个关键组件：1) 通过硬案例分配进行通用预训练，训练通用模仿学习 (IL) 驾驶系统，同时动态识别容易出错的案例以进行有针对性的改进；2) 残差强化专家微调，使用强化学习 (RL) 优化残差校正，以提高困难案例领域的性能，同时保留全局驾驶知识；3) 自感知适配器扩展，将专家策略动态集成回通用模型，从而增强持续性能改进。在闭环仿真和真实世界数据集中的实验结果表明，与最先进的 E2E 系统相比，在泛化性、安全性和长时程策略鲁棒性方面有所改进，突出了强化改进对于可扩展自动驾驶的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09800) | **Categories:** cs.RO

---

### [4] [Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints](https://arxiv.org/abs/2506.09859)
*Huajian Liu, Yixuan Feng, Wei Dong, Kunpeng Fan, Chao Wang, Yongzhuo Gao*

Main category: cs.RO

TL;DR: 本文提出了一种基于强化学习和图神经网络的分层机器人导航框架，用于在动态环境中实现高效的局部规划。


<details>
  <summary>Details</summary>
Motivation: 解决在具有异构约束的动态环境中机器人导航的问题。

Method: 利用通过强化学习（RL）训练的图神经网络来有效估计机器人的cost-to-go，并将其制定为局部目标推荐。

Result: 仿真和真实世界的实验表明，该方法在计算效率和训练可扩展性方面具有显著优势。

Conclusion: 该方法在复杂的动态环境中有效地解决了局部规划问题，并实现了最先进的性能。

Abstract: 本文提出了一种新颖的分层框架，用于在具有异构约束的动态环境中进行机器人导航。我们的方法利用通过强化学习（RL）训练的图神经网络来有效估计机器人的cost-to-go，并将其制定为局部目标推荐。然后，采用考虑运动学约束的时空路径搜索模块来生成参考轨迹，以促进解决用于显式约束执行的非凸优化问题。更重要的是，我们引入了一种增量动作屏蔽机制和一种特权学习策略，从而能够对所提出的规划器进行端到端训练。仿真和真实世界的实验表明，该方法在复杂的动态环境中有效地解决了局部规划问题，并实现了最先进的性能。与现有的学习-优化混合方法相比，我们的方法消除了对高保真仿真环境的依赖，在计算效率和训练可扩展性方面具有显著优势。该代码将在论文被接受后开源。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09859) | **Categories:** cs.RO

---

### [5] [From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models](https://arxiv.org/abs/2506.09930)
*Irving Fang, Juexiao Zhang, Shengbang Tong, Chen Feng*

Main category: cs.RO

TL;DR: VLA模型拥有良好的意图，但行动执行不佳，并且在动作数据上进行微调会削弱VLM原有的通用推理能力。


<details>
  <summary>Details</summary>
Motivation: 现有的VLA评估仍然不足；传统的模仿学习基准由于缺乏语言指令而不适用。新兴的VLA基准虽然结合了语言，但评估任务有限，并且没有研究VLM预训练对下游机器人策略的泛化能力有多大贡献。同时，许多研究依赖于不同机构独立设计的真实机器人装置，这为重现性和可访问性带来了障碍。

Method: 提出了一个包含50个模拟任务的统一探测套件，涵盖语言指令、视觉和对象等10个子类别，并在此基础上系统地评估了几种最先进的VLA架构。

Result: 结果表明，VLM主干网络赋予VLA强大的感知理解和高层次规划能力，但不能可靠地转化为精确的运动执行；当面对分布外的观察时，策略通常表现出连贯的意图，但在动作执行方面会失败。此外，在动作数据上进行微调会削弱VLM原有的通用推理能力。

Conclusion: VLM主干网络赋予VLA强大的感知理解和高层次规划能力，但不能可靠地转化为精确的运动执行；在动作数据上进行微调会削弱VLM原有的通用推理能力。

Abstract: 视觉-语言-动作（VLA）模型相比传统的机器人模仿学习，具有利用大型视觉-语言模型（VLM）的广泛泛化能力，从而产生通用的机器人策略的潜力。然而，目前对VLA的评估仍然不足。传统的模仿学习基准由于缺乏语言指令而不适用。新兴的VLA基准虽然结合了语言，但评估任务有限，并且没有研究VLM预训练对下游机器人策略的泛化能力有多大贡献。同时，许多研究依赖于不同机构独立设计的真实机器人装置，这为重现性和可访问性带来了障碍。为了解决这个问题，我们提出了一个统一的探测套件，包含50个模拟任务，涵盖语言指令、视觉和对象等10个子类别。我们在此基础上系统地评估了几种最先进的VLA架构，以了解它们的泛化能力。我们的结果表明，虽然VLM主干网络赋予VLA强大的感知理解和高层次规划能力，但不能可靠地转化为精确的运动执行：当面对分布外的观察时，策略通常表现出连贯的意图，但在动作执行方面会失败。此外，在动作数据上进行微调会削弱VLM原有的通用推理能力。我们发布了我们的任务套件和评估代码，作为未来VLA的标准基准，并推动缩小感知到行动差距的研究。更多信息，包括源代码，请访问https://ai4ce.github.io/INT-ACT/

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09930) | **Categories:** cs.RO, cs.CV

---

### [6] [Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation](https://arxiv.org/abs/2506.09990)
*Wenbo Zhang, Tianrun Hu, Yanyuan Qiao, Hanbo Zhang, Yuchu Qin, Yang Li, Jiajun Liu, Tao Kong, Lingqiao Liu, Xiao Ma*

Main category: cs.RO

TL;DR: CoA通过反向推理生成动作轨迹，在操作任务中实现了强大的性能。


<details>
  <summary>Details</summary>
Motivation: 与预测下一步动作的传统方法不同，CoA通过动作级的思维链(CoT)过程，利用特定于任务的目标进行显式的后向推理，生成整个轨迹。

Method: 提出了一种基于轨迹自回归建模的新型视觉运动策略范式Chain-of-Action (CoA)。

Result: CoA具有强大的空间泛化能力，同时保留了视觉运动策略的灵活性和简单性。

Conclusion: CoA在60个RLBench任务和8个真实世界操作任务中实现了最先进的性能。

Abstract: 我们提出了一种名为Chain-of-Action (CoA) 的新型视觉运动策略范式，它建立在轨迹自回归建模的基础上。与传统的前向预测下一步动作的方法不同，CoA 通过动作级别的思维链 (CoT) 过程，利用特定任务的目标进行显式的后向推理，从而生成完整的轨迹。这一过程统一在单个自回归结构中：(1) 第一个 token 对应于一个稳定的关键帧动作，它编码了特定于任务的目标；(2) 后续的动作 token 以自回归的方式生成，并以初始关键帧和先前预测的动作为条件。这种反向动作推理强制执行了一种全局到局部的结构，允许每个局部动作受到最终目标的严格约束。为了进一步实现动作推理结构，CoA 结合了四个互补设计：连续动作 token 表示；用于可变长度轨迹生成的动态停止；反向时间集成；以及多 token 预测，以平衡动作块建模与全局结构。因此，CoA 具有强大的空间泛化能力，同时保留了视觉运动策略的灵活性和简单性。在实验上，我们观察到 CoA 在 60 个 RLBench 任务和 8 个真实世界操作任务中实现了最先进的性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09990) | **Categories:** cs.RO, cs.CV, cs.LG

---


## 统计机器学习 (Machine Learning Statistics) [stat.ML]
### [1] [Attention-Bayesian Hybrid Approach to Modular Multiple Particle Tracking](https://arxiv.org/abs/2506.09441)
*Piyush Mishra, Philippe Roudot*

Main category: stat.ML

TL;DR: 提出了一种混合跟踪框架，结合了自注意力机制和贝叶斯滤波，以提高多粒子跟踪在复杂场景下的精度和鲁棒性。


<details>
  <summary>Details</summary>
Motivation: 由于轨迹假设的组合爆炸，在嘈杂和混乱的场景中跟踪多个粒子仍然具有挑战性，轨迹假设的组合爆炸随着粒子和帧的数量呈超指数级增长。

Method: 通过使用transformer编码器推断跨帧检测之间的软关联，解决标签预测问题，从而执行轨迹到检测的关联。

Result: 该方法在跟踪精度和对虚假检测的鲁棒性方面有所提高。

Conclusion: 该方法提高了跟踪精度和对虚假检测的鲁棒性，为高杂波多粒子跟踪场景提供了一个解决方案。

Abstract: 在嘈杂和混乱的场景中跟踪多个粒子仍然具有挑战性，这是由于轨迹假设的组合爆炸，其随着粒子和帧的数量呈超指数级增长。Transformer架构在抵抗这种高组合负载方面表现出显著的改进。然而，在轨迹假设集减少的场景中，其性能仍然低于传统的贝叶斯滤波方法。这表明，虽然transformer擅长缩小可能的关联，但它们可能无法在局部稀疏场景中达到贝叶斯方法的最优性。因此，我们引入了一种混合跟踪框架，该框架结合了自注意力学习粒子行为潜在表示的能力以及贝叶斯滤波的可靠性和可解释性。我们通过解决标签预测问题来执行轨迹到检测的关联，使用transformer编码器来推断跨帧检测之间的软关联。这减少了假设集，从而能够在贝叶斯滤波框架中实现高效的多粒子跟踪。我们的方法证明了改进的跟踪精度和对虚假检测的鲁棒性，为高杂波多粒子跟踪场景提供了一个解决方案。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09441) | **Categories:** stat.ML, cs.LG

---
