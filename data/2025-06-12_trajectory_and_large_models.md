# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-12

## 目录

- [人工智能 (Artificial Intelligence) (5)](#cs-ai)
- [计算语言学 (Computation and Language) (1)](#cs-cl)
- [计算机视觉 (Computer Vision) (11)](#cs-cv)
- [机器学习 (Machine Learning) (6)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness](https://arxiv.org/abs/2506.08532)
*Yanwei Gong, Xiaolin Chang*

Main category: cs.AI

TL;DR: 提出了一种结合DRL和LLM的无人机轨迹规划框架，以提高在低空经济约束下的规划效率和效果。


<details>
  <summary>Details</summary>
Motivation: 低空经济的快速增长推动了无人机（UAV）的广泛应用。这种日益增长的部署为复杂城市环境中的无人机轨迹规划带来了新的挑战。然而，现有的研究往往忽略了关键因素，如城市空域限制和经济效率，这在低空经济背景下至关重要。

Method: 提出了一种结合DRL与大型语言模型（LLM）推理的新型无人机轨迹规划框架，以实现安全、合规和经济可行的路径规划。

Result: 实验结果表明，该方法在数据收集率、避碰、成功着陆、法规遵从性和能源效率等多项指标上均优于现有基线。

Conclusion: 实验结果表明，该方法在数据收集率、避碰、成功着陆、法规遵从性和能源效率等多项指标上均优于现有基线，验证了该方法在解决低空经济网络约束下的无人机轨迹规划关键挑战方面的有效性。

Abstract: 低空经济的快速发展推动了无人机（UAV）的广泛应用。日益增长的无人机部署给复杂城市环境中的无人机轨迹规划带来了新的挑战。然而，现有的研究往往忽略了城市空域限制和经济效率等关键因素，而这些因素在低空经济背景下至关重要。深度强化学习（DRL）被认为是解决这些问题的一个有希望的方案，但其在实际应用中受到学习效率低的限制。为了克服这一限制，我们提出了一种结合DRL与大型语言模型（LLM）推理的新型无人机轨迹规划框架，以实现安全、合规和经济可行的路径规划。实验结果表明，我们的方法在数据收集率、避碰、成功着陆、法规遵从性和能源效率等多项指标上均优于现有基线。这些结果验证了我们的方法在解决低空经济网络约束下的无人机轨迹规划关键挑战方面的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08532) | **Categories:** cs.AI

---

### [2] [IntTrajSim: Trajectory Prediction for Simulating Multi-Vehicle driving at Signalized Intersections](https://arxiv.org/abs/2506.08957)
*Yash Ranjan, Rahul Sengupta, Anand Rangarajan, Sanjay Ranka*

Main category: cs.AI

TL;DR: 本文提出了一种新的交通仿真评估方法，并设计了一个更优的轨迹预测模型，能够更好地模拟路口交通行为。


<details>
  <summary>Details</summary>
Motivation: 交通仿真器在研究道路基础设施的运营效率方面被广泛使用，但其基于规则的方法限制了它们模仿真实驾驶行为的能力。交通路口是道路基础设施的关键组成部分，无论是在安全风险（近 28% 的致命碰撞和 58% 的非致命碰撞发生在路口）还是道路走廊的运营效率方面。这提出了一个重要的问题：我们能否创建一个数据驱动的仿真器，可以模仿交通路口驾驶行为的宏观和微观统计数据？

Method: 提出了一种多头自注意力机制的轨迹预测模型，该模型融合了信号信息。

Result: 提出的模型在交通工程相关指标上优于以往模型。

Conclusion: 本文提出了一种基于多头自注意力机制的轨迹预测模型，该模型在交通工程相关指标上优于以往模型。

Abstract: 交通仿真器被广泛用于研究道路基础设施的运营效率，但它们基于规则的方法限制了它们模仿真实驾驶行为的能力。交通路口是道路基础设施的关键组成部分，无论是在安全风险（近 28% 的致命碰撞和 58% 的非致命碰撞发生在路口）还是道路走廊的运营效率方面。这就提出了一个重要的问题：我们能否创建一个数据驱动的仿真器，可以模仿交通路口驾驶行为的宏观和微观统计数据？基于深度生成建模的轨迹预测模型为模拟路口车辆的复杂动态提供了一个良好的起点。但它们没有在“实时”微观仿真场景中进行测试，也没有在交通工程相关的指标上进行评估。在这项研究中，我们提出了交通工程相关的指标来评估生成轨迹预测模型，并提供了一个 simulation-in-the-loop 管道来实现这一点。我们还提供了一个基于多头自注意力机制的轨迹预测模型，该模型融合了信号信息，并在评估指标上优于我们之前的模型。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08957) | **Categories:** cs.AI, cs.LG

---

### [3] [Cognitive Weave: Synthesizing Abstracted Knowledge with a Spatio-Temporal Resonance Graph](https://arxiv.org/abs/2506.08098)
*Akash Vishwakarma, Hojin Lee, Mohith Suresh, Priyam Shankar Sharma, Rahul Vishwakarma, Sparsh Gupta, Yuvraj Anupam Chauhan*

Main category: cs.AI

TL;DR: 认知编织：一种基于多层时空共振图的新型记忆框架，显著提升了LLM在复杂任务中的表现。


<details>
  <summary>Details</summary>
Motivation: 当前记忆系统在结构灵活性、时间感知以及从原始交互数据中综合更高层次见解的能力方面存在根本性局限性。

Method: 提出了一种新颖的记忆框架“认知编织”，它围绕一个多层时空共振图 (STRG) 构建，该图将信息管理为语义丰富的洞察粒子 (IP)，并通过专用的语义预言接口 (SOI) 动态地用共振键、指示符和情境印记丰富这些粒子。

Result: 在长程规划任务中，认知编织的任务完成率平均提高 34%，查询延迟减少 42%。

Conclusion: 认知编织在长程规划、演进式问答和多会话对话连贯性方面显著优于现有方法，任务完成率平均提高 34%，查询延迟减少 42%。

Abstract: 基于大型语言模型 (LLM) 的智能代理的出现，需要超越单纯数据存储的记忆架构，以实现持续学习、细致推理和动态适应。当前的记忆系统在结构灵活性、时间感知以及从原始交互数据中综合更高层次见解的能力方面通常面临根本性局限性。本文介绍了一种新颖的记忆框架“认知编织”，它围绕一个多层时空共振图 (STRG) 构建。该图将信息管理为语义丰富的洞察粒子 (IP)，这些粒子通过专用的语义预言接口 (SOI) 动态地用共振键、指示符和情境印记丰富。这些 IP 通过类型化的关系链相互连接，形成一个不断演变的知识体系。认知编织的一个关键组成部分是认知改进过程，这是一种自主机制，包括综合洞察聚合 (IA)，这是一种从已识别的相关 IP 集群中提取的精简的、更高层次的知识结构。我们展示了全面的实验结果，证明了认知编织在长程规划任务、演进式问答场景和多会话对话连贯性方面相对于现有方法的显着增强。与最先进的基线相比，该系统在任务完成率方面平均提高了 34%，查询延迟平均减少了 42%。此外，本文还探讨了这种先进记忆系统中固有的伦理考量，讨论了其对 LLM 中长期记忆的影响，并概述了有希望的未来研究方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08098) | **Categories:** cs.AI

---

### [4] [Evaluating Generative Vehicle Trajectory Models for Traffic Intersection Dynamics](https://arxiv.org/abs/2506.08963)
*Yash Ranjan, Rahul Sengupta, Anand Rangarajan, Sanjay Ranka*

Main category: cs.AI

TL;DR: 本文提出了一种交通分析工具，用于评估交通预测模型在微观模拟中产生的违反交通规则的行为。


<details>
  <summary>Details</summary>
Motivation: 交通路口是城市道路网络的重要组成部分，但它们也是轨迹冲突和容易发生事故的区域。交通信号灯控制的交叉口交通动态深度生成模型可以极大地帮助交通部门更好地理解效率和安全问题。目前，模型主要在计算指标上进行评估，主要关注轨迹重建误差。他们没有在“实时”微观模拟场景中进行在线评估。此外，这些指标没有充分考虑交通工程的具体问题，例如闯红灯、不允许停车等。

Method: 提出了一种综合分析工具，用于训练、运行和评估模型，并使用能够更好洞察模型性能的指标。

Result: 在微观模拟器中，在未知的交通条件下，在线评估预测模型的性能。结果表明，尽管使用了理想化的轨迹作为输入并实现了较低的轨迹重建误差，但生成的轨迹显示出违反交通规则的行为。引入了新的指标来评估这种不良行为，并展示了结果。

Conclusion: 尽管使用了理想化的轨迹作为输入并实现了较低的轨迹重建误差，但生成的轨迹显示出违反交通规则的行为。

Abstract: 交通路口对于城市道路网络至关重要，因为它们调节着人员和货物的流动。然而，它们也是轨迹冲突的区域，容易发生事故。信号交叉口交通动态的深度生成模型可以极大地帮助交通部门更好地理解效率和安全方面。目前，模型在计算指标上进行评估，主要关注轨迹重建误差。它们没有在“实时”微观模拟场景中进行在线评估。此外，这些指标没有充分考虑交通工程的具体问题，例如闯红灯、不允许停车等。在这项工作中，我们提供了一个综合分析工具来训练、运行和评估模型，并使用能够更好洞察模型性能的指标，从交通工程的角度来看。我们在一个大型数据集上训练了一个最先进的多车辆轨迹预测模型，该数据集是通过运行真实城市交叉口的校准场景收集的。然后，在微观模拟器中，在未知的交通条件下，在线评估预测模型的性能。我们表明，尽管使用了理想化的轨迹作为输入，并实现了较低的轨迹重建误差，但生成的轨迹显示出违反交通规则的行为。我们引入了新的指标来评估这种不良行为，并展示了我们的结果。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08963) | **Categories:** cs.AI

---

### [5] [VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](https://arxiv.org/abs/2506.09049)
*Li Kang, Xiufeng Song, Heng Zhou, Yiran Qin, Jie Yang, Xiaohong Liu, Philip Torr, Lei Bai, Zhenfei Yin*

Main category: cs.AI

TL;DR: VIKI-Bench和VIKI-R为推进具身人工智能系统中多智能体、视觉驱动的协作提供了一个统一的测试平台和方法。


<details>
  <summary>Details</summary>
Motivation: 在动态环境中协调多个具身智能体仍然是人工智能中的一个核心挑战，需要感知驱动的推理和可扩展的协作策略。现有的基于VLM的方法在支持多样化的具身类型方面仍然有限。

Method: 提出了一种名为VIKI-R的两阶段框架，该框架使用思维链注释演示对预训练的视觉语言模型（VLM）进行微调，然后在多级奖励信号下进行强化学习。

Result: VIKI-R在所有任务级别上都显著优于基线方法，并且强化学习能够实现异构智能体之间组合协作模式的出现。

Conclusion: VIKI-R在所有任务级别上都显著优于基线方法，并且强化学习能够实现异构智能体之间组合协作模式的出现。

Abstract: 在动态环境中协调多个具身智能体仍然是人工智能领域的核心挑战，这需要感知驱动的推理和可扩展的协作策略。虽然最近的工作已经利用大型语言模型（LLM）进行多智能体规划，但也有一些工作开始探索视觉语言模型（VLM）进行视觉推理。然而，这些基于VLM的方法在支持多样化的具身类型方面仍然有限。在这项工作中，我们介绍了VIKI-Bench，这是第一个为具身多智能体协作量身定制的分层基准，具有三个结构化级别：智能体激活、任务规划和轨迹感知。VIKI-Bench包括不同的机器人具身、多视角视觉观察和结构化监督信号，以评估基于视觉输入的推理。为了证明VIKI-Bench的效用，我们提出了一种名为VIKI-R的两阶段框架，该框架使用思维链注释演示对预训练的视觉语言模型（VLM）进行微调，然后在多级奖励信号下进行强化学习。我们的大量实验表明，VIKI-R在所有任务级别上都显著优于基线方法。此外，我们表明，强化学习能够实现异构智能体之间组合协作模式的出现。总之，VIKI-Bench和VIKI-R为推进具身人工智能系统中多智能体、视觉驱动的协作提供了一个统一的测试平台和方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09049) | **Categories:** cs.AI, cs.CV, cs.RO

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [Low-resource domain adaptation while minimizing energy and hardware resource consumption](https://arxiv.org/abs/2506.08433)
*Hernán Maina, Nicolás Wolovick, Luciana Benotti*

Main category: cs.CL

TL;DR: 通过调整数值精度和数据并行策略，可以在资源受限的环境中经济高效地进行领域自适应。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型的训练成本高昂，领域自适应是使模型适应不同文化和价值观环境的一种有前景的策略，但其计算成本仍然是一个显著的障碍，特别是对于缺乏大型基础设施的研究团队。

Method: 评估了不同数值精度和数据并行策略对训练速度和模型准确率的影响。

Result: 研究结果表明，调整数值精度和数据并行策略可以有效地进行领域自适应，同时保持模型准确率。

Conclusion: 通过调整数值精度和数据并行策略，可以在资源受限的环境中有效地进行领域自适应，同时保持模型准确率。

Abstract: 训练大型语言模型（LLM）在能源、硬件和标注数据方面的成本很高，这通常导致一种根植于主要文化和价值观的定位（Santy et al., 2023）。领域自适应已经成为一种有前景的策略，可以更好地使模型与不同的文化和价值观环境对齐（Hershcovich et al., 2022），但其计算成本仍然是一个显著的障碍，特别是对于缺乏大型基础设施的研究团队。在本文中，我们评估了使用不同的数值精度和数据并行化策略如何影响训练速度（作为能源和硬件消耗的代表）和模型准确率，目的是促进低资源环境中的领域自适应。我们的发现与能源效率、可访问性或有限的硬件可用性是关键考虑因素的任何设置相关。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08433) | **Categories:** cs.CL, cs.DC, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.08052)
*Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang*

Main category: cs.CV

TL;DR: ReCogDrive通过整合视觉语言模型与扩散规划器，并在三阶段训练下，显著提升了自动驾驶在复杂场景中的性能。


<details>
  <summary>Details</summary>
Motivation: 端到端自动驾驶在长尾场景中性能显著下降；视觉语言模型与实际驾驶数据存在领域差异；离散语言空间与连续动作空间存在维度不匹配；模仿学习倾向于捕捉次优行为。

Method: ReCogDrive采用三阶段训练范式，包括：使用驾驶问答数据集训练视觉语言模型，使用扩散模型进行模仿学习，以及使用强化学习微调扩散规划器。

Result: 在NAVSIM基准测试中，ReCogDrive的PDMS达到89.6，超过了之前的最佳视觉方案5.6 PDMS。

Conclusion: 通过在NAVSIM基准测试上的评估，ReCogDrive达到了89.6的PDMS，超越了之前的视觉SOTA 5.6 PDMS，确立了新的技术水平。

Abstract: 尽管端到端自动驾驶取得了显著进展，但在罕见和长尾场景中，其性能会显著下降。最近的方法试图利用视觉-语言模型（VLM）丰富的世界知识来应对这一挑战，但这些方法存在几个局限性：（1）VLM的预训练数据与真实驾驶数据之间存在显著的领域差距；（2）离散语言空间与连续动作空间之间存在维度不匹配；（3）模仿学习倾向于捕捉数据集中存在的平均行为，这可能并非最优甚至危险。在本文中，我们提出了ReCogDrive，一种集成了VLM与扩散规划器的自动驾驶系统，该系统采用三阶段训练范式。第一阶段，我们使用大规模驾驶问答数据集来训练VLM，从而缓解通用内容与真实驾驶场景之间的领域差异。第二阶段，我们采用基于扩散的规划器来执行模仿学习，将来自潜在语言空间的表示映射到连续驾驶动作。最后，我们使用强化学习和NAVSIM非反应式模拟器来微调扩散规划器，使模型能够生成更安全、更像人类的驾驶轨迹。我们在面向规划的NAVSIM基准上评估了我们的方法，实现了89.6的PDMS，并创造了超越之前仅视觉SOTA 5.6 PDMS的最新技术水平。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08052) | **Categories:** cs.CV, cs.RO

---

### [2] [TrajFlow: Multi-modal Motion Prediction via Flow Matching](https://arxiv.org/abs/2506.08541)
*Qi Yan, Brian Zhang, Yutong Zhang, Daniel Yang, Joshua White, Di Chen, Jiachao Liu, Langechuan Liu, Binnan Zhuang, Shaoshuai Shi, Renjie Liao*

Main category: cs.CV

TL;DR: TrajFlow通过单次预测多个轨迹、引入排序损失和自条件训练，实现了高效准确的自动驾驶运动预测。


<details>
  <summary>Details</summary>
Motivation: 在动态真实世界的条件下，高效准确的运动预测对于确保自动驾驶的安全性和明智的决策至关重要，尤其是在需要多模式预测的情况下。

Method: TrajFlow预测多个合理的未来轨迹，并在Plackett-Luce分布的基础上提出了排序损失，并设计了一种自条件训练技术。

Result: TrajFlow在Waymo Open Motion Dataset (WOMD) 大型数据集上进行了广泛的实验，证明了其在各种关键指标上都达到了最先进的性能。

Conclusion: TrajFlow在Waymo Open Motion Dataset上实现了最先进的性能，证明了其在自动驾驶安全应用中的有效性。

Abstract: 在自动驾驶中，高效准确的运动预测至关重要，尤其是在需要多模态预测的动态真实场景下。我们提出了一种新的基于流匹配的运动预测框架TrajFlow，它解决了现有生成轨迹预测方法的可扩展性和效率挑战。与传统的生成方法不同，TrajFlow通过单次预测多个合理的未来轨迹，显著降低了计算开销，同时保持了预测之间的一致性。此外，我们提出了一种基于Plackett-Luce分布的排序损失，以提高预测轨迹的不确定性估计。此外，我们设计了一种自条件训练技术，该技术在第二次前向传播期间重复使用模型自身的预测来构建噪声输入，从而提高泛化能力并加速推理。在Waymo Open Motion Dataset (WOMD) 大型数据集上进行的广泛实验表明，TrajFlow在各种关键指标上都达到了最先进的性能，突显了其在安全关键型自动驾驶应用中的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08541) | **Categories:** cs.CV, cs.AI

---

### [3] [Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models](https://arxiv.org/abs/2506.09042)
*Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, Seung Wook Kim, Jun Gao, Laura Leal-Taixe, Mike Chen, Sanja Fidler, Huan Ling*

Main category: cs.CV

TL;DR: Cosmos-Drive-Dreams 是一个合成数据生成流程，旨在生成具有挑战性的驾驶场景，以改善自动驾驶系统的训练和测试。


<details>
  <summary>Details</summary>
Motivation: 为自动驾驶汽车 (AV) 等安全关键物理 AI 系统收集和标注真实数据既耗时又昂贵，尤其难以捕获稀有边缘案例，而这些案例在 AV 系统的训练和测试中起着关键作用。

Method: Cosmos-Drive-Dreams 提出了一个合成数据生成 (SDG) 流程，该流程使用 NVIDIA Cosmos 世界基础模型定制的 Cosmos-Drive 模型套件，用于生成可控、高保真、多视角和时空一致的驾驶视频。

Result: 实验表明，Cosmos-Drive-Dreams 生成的数据有助于缓解长尾分布问题，并增强 3D 车道检测、3D 目标检测和驾驶策略学习等下游任务的泛化能力。

Conclusion: 通过合成数据，Cosmos-Drive-Dreams 缓解了长尾分布问题，并增强了 3D 车道检测、3D 目标检测和驾驶策略学习等下游任务的泛化能力。

Abstract: 为自动驾驶汽车 (AV) 等安全关键物理 AI 系统收集和标注真实世界的数据既耗时又昂贵。捕获罕见的边缘情况尤其具有挑战性，这些情况在自动驾驶系统的训练和测试中起着关键作用。为了解决这个挑战，我们推出了 Cosmos-Drive-Dreams——一个合成数据生成 (SDG) 流程，旨在生成具有挑战性的场景，以促进感知和驾驶策略训练等下游任务。Cosmos-Drive 为该流程提供支持，它是一套专门从 NVIDIA Cosmos 世界基础模型为驾驶领域定制的模型，能够生成可控、高保真、多视角和时空一致的驾驶视频。我们通过应用 Cosmos-Drive-Dreams 来扩展具有高保真和挑战性场景的驾驶数据集的数量和多样性，展示了这些模型的效用。实验表明，我们生成的数据有助于缓解长尾分布问题，并增强 3D 车道检测、3D 目标检测和驾驶策略学习等下游任务的泛化能力。我们通过 NVIDIA 的 Cosmos 平台开源了我们的流程工具包、数据集和模型权重。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09042) | **Categories:** cs.CV

---

### [4] [MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning](https://arxiv.org/abs/2506.08694)
*Mohammadreza Salehi, Shashanka Venkataramanan, Ioana Simion, Efstratios Gavves, Cees G. M. Snoek, Yuki M Asano*

Main category: cs.CV

TL;DR: 该论文提出了一种运动引导的自监督学习框架，通过聚类密集点轨迹来学习时空一致的视频表征。


<details>
  <summary>Details</summary>
Motivation: 现有的密集自监督学习方法依赖于静态增强，这在对象变形、遮挡和相机运动下会失效，导致随时间推移的不一致的特征学习，从而限制了其在视频领域的应用。

Method: 该论文提出了一种运动引导的自监督学习框架，该框架通过聚类密集点轨迹来学习时空一致的表征。利用现成的点跟踪器提取长程运动轨迹，并通过基于动量编码器的最优传输机制优化特征聚类。为了确保时间一致性，该方法沿跟踪点传播聚类分配，从而在视角变化时保持跨视角的特征一致性。

Result: 在六个图像和视频数据集以及四个评估基准上，该方法将state-of-the-art结果提高了1%到6%。

Conclusion: 通过利用运动信息作为隐式监督信号，该方法学习到的表征在帧间具有更好的泛化能力，提高了在动态场景和遮挡场景中的鲁棒性，并在多个图像和视频数据集上取得了state-of-the-art的结果。

Abstract: 稠密的自监督学习在学习像素和patch级别的表征方面显示出了巨大的潜力，但是由于运动动态的复杂性，将其扩展到视频仍然具有挑战性。现有的方法依赖于静态增强，这在对象变形、遮挡和相机运动下会失效，导致随时间推移的不一致的特征学习。我们提出了一种运动引导的自监督学习框架，该框架通过聚类密集点轨迹来学习时空一致的表征。通过利用现成的点跟踪器，我们提取长程运动轨迹，并通过基于动量编码器的最优传输机制优化特征聚类。为了确保时间一致性，我们沿跟踪点传播聚类分配，从而在视角变化时保持跨视角的特征一致性。通过整合运动作为隐式监督信号，我们的方法学习到的表征在帧间具有更好的泛化能力，提高了在动态场景和遮挡场景中的鲁棒性。通过从强大的图像预训练模型初始化并利用视频数据进行训练，我们在六个图像和视频数据集以及四个评估基准上将最先进水平提高了1%到6%。该实现可在我们的GitHub存储库中公开获得：https://github.com/SMSD75/MoSiC/tree/main

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08694) | **Categories:** cs.CV

---

### [5] [Surgeon Style Fingerprinting and Privacy Risk Quantification via Discrete Diffusion Models in a Vision-Language-Action Framework](https://arxiv.org/abs/2506.08185)
*Huixin Zhan, Jason H. Moore*

Main category: cs.CV

TL;DR: 该论文提出了一种利用离散扩散框架和视觉-语言-动作管道，在机器人手术中对外科医生特定指纹进行建模的新方法，并在个性化与隐私保护之间进行了权衡。


<details>
  <summary>Details</summary>
Motivation: 由于训练、经验和运动行为的差异，外科医生表现出不同的手术风格 - 但当前的人工智能系统通常忽略这种个性化信号。

Method: 我们提出了一种新颖的方法，使用与视觉-语言-动作（VLA）管道集成的离散扩散框架，在机器人手术中对细粒度的、外科医生特定的指纹进行建模。我们的方法将手势预测公式化为一个结构化的序列去噪任务，该任务以多模态输入为条件，包括内窥镜视频、手术意图语言以及外科医生身份和技能的隐私感知嵌入。

Result: 我们在JIGSAWS数据集上评估了我们的方法，并证明它可以准确地重建手势序列，同时学习每个外科医生独特的有意义的运动指纹。

Conclusion: 个性化嵌入虽然提高了性能，但也增加了身份泄露的风险，揭示了在手术建模中平衡个性化与隐私风险的重要性。

Abstract: 由于外科医生的训练、经验和运动行为存在差异，他们的手术风格也各不相同，但目前的人工智能系统往往忽略了这种个性化的信号。我们提出了一种新颖的方法，利用与视觉-语言-动作（VLA）管道集成的离散扩散框架，在机器人手术中对外科医生特有的精细指纹进行建模。我们的方法将手势预测构建为一个结构化的序列去噪任务，该任务以多模态输入为条件，包括内窥镜视频、手术意图语言以及外科医生身份和技能的隐私感知嵌入。个性化的外科医生指纹通过使用第三方语言模型的自然语言提示进行编码，从而使模型能够在不暴露明确身份的情况下保留个体行为风格。我们在JIGSAWS数据集上评估了我们的方法，结果表明，该方法能够准确地重建手势序列，同时学习每个外科医生独特的有意义的运动指纹。为了量化个性化的隐私影响，我们进行了成员推理攻击，发现更具表现力的嵌入可以提高任务性能，但同时也增加了身份泄露的可能性。这些发现表明，虽然个性化嵌入提高了性能，但也增加了身份泄露的风险，揭示了在手术建模中平衡个性化与隐私风险的重要性。代码可在以下网址获取：https://github.com/huixin-zhan-ai/Surgeon_style_fingerprinting。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08185) | **Categories:** cs.CV, cs.AI

---

### [6] [MLVTG: Mamba-Based Feature Alignment and LLM-Driven Purification for Multi-Modal Video Temporal Grounding](https://arxiv.org/abs/2506.08512)
*Zhiyi Zhu, Xiaoyu Wu, Zihao Liu, Linlin Yang*

Main category: cs.CV

TL;DR: MLVTG通过MambaAligner和LLMRefiner实现了更精确的视频时序定位，并在多个数据集上取得了SOTA性能。


<details>
  <summary>Details</summary>
Motivation: 现有的基于Transformer的方法通常存在冗余注意力和次优的多模态对齐问题。

Method: MLVTG集成了MambaAligner和LLMRefiner两个关键模块。MambaAligner使用堆叠的Vision Mamba块作为主干，代替Transformer来建模时间依赖关系并提取鲁棒的视频表示以进行多模态对齐。LLMRefiner利用预训练的大型语言模型（LLM）的特定冻结层来隐式传递语义先验，从而增强多模态对齐，而无需微调。

Result: 在QVHighlights、Charades-STA和TVSum上的大量实验表明，MLVTG实现了最先进的性能，并且显著优于现有的基线。

Conclusion: MLVTG通过MambaAligner和LLMRefiner实现了最先进的视频时序定位性能，显著优于现有基线。

Abstract: 视频时序定位（VTG）旨在定位与自然语言查询相对应的视频片段，是视频理解中一项基础但具有挑战性的任务。现有的基于Transformer的方法通常存在冗余注意力和次优的多模态对齐问题。为了解决这些限制，我们提出了一种新的框架MLVTG，它集成了两个关键模块：MambaAligner和LLMRefiner。MambaAligner使用堆叠的Vision Mamba块作为主干，代替Transformer来建模时间依赖关系并提取鲁棒的视频表示以进行多模态对齐。LLMRefiner利用预训练的大型语言模型（LLM）的特定冻结层来隐式传递语义先验，从而增强多模态对齐，而无需微调。这种双重对齐策略，即通过结构化状态空间动力学进行时间建模，以及通过文本先验进行语义提纯，能够实现更精确的定位。在QVHighlights、Charades-STA和TVSum上的大量实验表明，MLVTG实现了最先进的性能，并且显著优于现有基线。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08512) | **Categories:** cs.CV, cs.AI

---

### [7] [From Pixels to Graphs: using Scene and Knowledge Graphs for HD-EPIC VQA Challenge](https://arxiv.org/abs/2506.08553)
*Agnese Taluzzi, Davide Gesualdi, Riccardo Santambrogio, Chiara Plizzari, Francesca Palermo, Simone Mentasti, Matteo Matteucci*

Main category: cs.CV

TL;DR: SceneNet和KnowledgeNet结合多模态信息与常识知识，在HD-EPIC VQA挑战赛中表现出色。


<details>
  <summary>Details</summary>
Motivation: 该报告旨在解决HD-EPIC VQA挑战赛中的复杂第一人称视角视觉问答（VQA）任务。

Method: SceneNet利用多模态大型语言模型（MLLM）生成场景图，以捕捉细粒度的对象交互、空间关系和时间定位事件；KnowledgeNet结合ConceptNet的外部常识知识，引入实体之间的高层语义连接。

Result: SceneNet和KnowledgeNet在HD-EPIC基准测试的七个类别中表现出不同的优势，并且它们在框架中的结合在挑战赛中实现了44.21%的总体准确率。

Conclusion: 提出的SceneNet和KnowledgeNet方法在HD-EPIC VQA挑战赛中表现出有效性，结合两者在复杂的第一人称视角VQA任务中取得了44.21%的准确率。

Abstract: 本报告介绍了SceneNet和KnowledgeNet，这是我们为HD-EPIC VQA 2025挑战赛开发的两种方法。SceneNet利用多模态大型语言模型（MLLM）生成的场景图来捕捉细粒度的对象交互、空间关系和时间定位事件。同时，KnowledgeNet结合ConceptNet的外部常识知识，引入实体之间的高层语义连接，从而实现超越直接观察到的视觉证据的推理。每种方法在HD-EPIC基准测试的七个类别中都表现出独特的优势，并且它们在我们的框架中的结合在挑战赛中实现了44.21%的总体准确率，突显了其在复杂的第一人称视角VQA任务中的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08553) | **Categories:** cs.CV

---

### [8] [Generating Vision-Language Navigation Instructions Incorporated Fine-Grained Alignment Annotations](https://arxiv.org/abs/2506.08566)
*Yibo Cui, Liang Xie, Yu Zhao, Jiawei Sun, Erwei Yin*

Main category: cs.CV

TL;DR: FCA-NIG框架自动生成具有细粒度跨模态注释的导航指令，显著提升视觉-语言导航代理的性能。


<details>
  <summary>Details</summary>
Motivation: 视觉-语言导航（VLN）使智能代理能够通过整合视觉感知和自然语言指令来导航环境，但由于缺乏细粒度的跨模态对齐注释而面临重大挑战。现有的数据集主要侧重于全局指令-轨迹匹配，而忽略了对于准确的导航动作决策至关重要的子指令级别和实体级别的对齐。

Method: 我们提出了FCA-NIG，这是一个生成框架，可以自动构建具有双层细粒度跨模态注释的导航指令。在该框架中，首先将增强的轨迹分成子轨迹，然后通过基于GLIP的地标检测、精心设计的指令构建、基于OFA-Speaker的R2R类指令生成和CLIP驱动的实体选择进行处理，从而生成带有实体-地标注释的子指令-轨迹对。最后，将这些子对聚合以形成完整的指令-轨迹对。

Result: 该框架生成了FCA-R2R数据集，这是第一个具有精确的子指令-子轨迹和实体-地标对齐的大规模增强数据集。

Conclusion: 实验表明，使用FCA-R2R训练可以显著提高多个最先进的VLN代理的性能，包括SF、EnvDrop、RecBERT和HAMT。结合子指令-轨迹对齐增强了代理的状态感知和决策准确性，而实体-地标对齐进一步提高了导航性能和泛化能力。这些结果突出了FCA-NIG在生成高质量、可扩展的训练数据方面的有效性，无需手动注释，从而推进了复杂导航任务中的细粒度跨模态学习。

Abstract: 视觉-语言导航（VLN）使智能代理能够通过整合视觉感知和自然语言指令来导航环境，但由于缺乏细粒度的跨模态对齐注释而面临重大挑战。现有的数据集主要侧重于全局指令-轨迹匹配，而忽略了对于准确的导航动作决策至关重要的子指令级别和实体级别的对齐。为了解决这个限制，我们提出了FCA-NIG，这是一个生成框架，可以自动构建具有双层细粒度跨模态注释的导航指令。在该框架中，首先将增强的轨迹分成子轨迹，然后通过基于GLIP的地标检测、精心设计的指令构建、基于OFA-Speaker的R2R类指令生成和CLIP驱动的实体选择进行处理，从而生成带有实体-地标注释的子指令-轨迹对。最后，将这些子对聚合以形成完整的指令-轨迹对。该框架生成了FCA-R2R数据集，这是第一个具有精确的子指令-子轨迹和实体-地标对齐的大规模增强数据集。大量的实验表明，使用FCA-R2R训练可以显著提高多个最先进的VLN代理的性能，包括SF、EnvDrop、RecBERT和HAMT。结合子指令-轨迹对齐增强了代理的状态感知和决策准确性，而实体-地标对齐进一步提高了导航性能和泛化能力。这些结果突出了FCA-NIG在生成高质量、可扩展的训练数据方面的有效性，无需手动注释，从而推进了复杂导航任务中的细粒度跨模态学习。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08566) | **Categories:** cs.CV

---

### [9] [TraGraph-GS: Trajectory Graph-based Gaussian Splatting for Arbitrary Large-Scale Scene Rendering](https://arxiv.org/abs/2506.08704)
*Xiaohan Zhang, Sitong Wang, Yushen Yan, Yi Yang, Mingda Xu, Qi Liu*

Main category: cs.CV

TL;DR: TraGraph-GS利用轨迹图实现了大规模场景的高精度渲染，显著提升了新视角合成的质量。


<details>
  <summary>Details</summary>
Motivation: Existing novel view synthesis methods for large-scale scenes struggle with arbitrary camera trajectories and Gaussian overlap issues when merging regions, leading to poor generalization and distorted texture details.

Method: The authors propose a graph-based spatial partitioning method with a regularization constraint for texture and distant object rendering, along with a progressive rendering strategy to reduce Gaussian overlap artifacts.

Result: Experimental results show that TraGraph-GS achieves an average improvement of 1.86 dB in PSNR on aerial datasets and 1.62 dB on ground datasets compared to state-of-the-art approaches.

Conclusion: The paper introduces TraGraph-GS, a novel approach using trajectory graphs for high-precision rendering of large-scale scenes, demonstrating significant improvements in PSNR compared to existing methods on both aerial and ground datasets.

Abstract: 针对大规模场景的高质量新视角合成是 3D 计算机视觉中一个具有挑战性的难题。现有方法通常将大型场景划分为多个区域，使用高斯溅射为每个区域重建 3D 表示，并最终合并它们以进行新视角渲染。它们可以准确地渲染特定场景，但由于两个原因，它们不能有效地泛化：（1）刚性空间分割技术难以处理任意相机轨迹，以及（2）区域的合并导致高斯重叠，从而扭曲纹理细节。为了解决这些挑战，我们提出了 TraGraph-GS，它利用轨迹图来实现任意大规模场景的高精度渲染。我们提出了一种基于图的大规模场景空间划分方法，该方法结合了正则化约束，以增强纹理和远处物体的渲染，以及一种渐进式渲染策略，以减轻由高斯重叠引起的伪影。实验结果表明，它在四个航空数据集和四个地面数据集上都表现出卓越的性能，并突出了其卓越的效率：与最先进的方法相比，我们的方法在航空数据集上的 PSNR 平均提高了 1.86 dB，在地面数据集上的 PSNR 平均提高了 1.62 dB。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08704) | **Categories:** cs.CV

---

### [10] [Geometric deep learning for local growth prediction on abdominal aortic aneurysm surfaces](https://arxiv.org/abs/2506.08729)
*Dieuwertje Alblas, Patryk Rygiel, Julian Suk, Kaj O. Kappe, Marieke Hofman, Christoph Brune, Kak Khee Yeung, Jelmer M. Wolterink*

Main category: cs.CV

TL;DR: 本文提出了一种基于 SE(3) 对称变换模型的 AAA 生长预测方法，该方法直接在血管模型表面上进行预测，并取得了良好的预测效果，有望改善个性化监测策略。


<details>
  <summary>Details</summary>
Motivation: 腹主动脉瘤 (AAA) 是腹主动脉的进行性局灶性扩张。AAA 可能会破裂，存活率仅为 20%。目前的临床指南建议，当男性最大 AAA 直径超过 55 毫米或女性超过 50 毫米时，进行选择性手术修复。不符合这些标准的患者会定期接受监测，监测间隔基于最大 AAA 直径。然而，该直径没有考虑到 3D AAA 形状与其生长之间的复杂关系，使得标准化间隔可能不合适。个性化的 AAA 生长预测可以改善监测策略。

Method: 我们提出使用 SE(3) 对称变换模型来直接预测血管模型表面的 AAA 生长，该表面富含局部多物理特征。与参数化 AAA 形状的其他工作相比，这种表示保留了血管表面的解剖结构和几何保真度。

Result: 经过训练，我们的模型可以预测下一次扫描时刻的 AAA 生长，中值直径误差为 1.18 毫米。我们进一步证明了我们的模型在识别患者是否将在两年内有资格接受选择性修复方面的效用（acc = 0.93）。最后，我们评估了我们的模型在外部验证集上的泛化能力，该验证集由来自不同医院的 7 名 AAA 患者的 25 次 CTA 扫描组成。

Conclusion: 局部方向性 AAA 从血管表面的生长预测是可行的，并且可能有助于个性化监测策略。

Abstract: 腹主动脉瘤 (AAA) 是腹主动脉的进行性局部扩张。AAA 可能会破裂，存活率仅为 20%。目前的临床指南建议，当男性最大 AAA 直径超过 55 毫米或女性超过 50 毫米时，进行选择性手术修复。不符合这些标准的患者会定期接受监测，监测间隔基于最大 AAA 直径。然而，该直径没有考虑到 3D AAA 形状与其生长之间的复杂关系，使得标准化间隔可能不合适。个性化的 AAA 生长预测可以改善监测策略。我们提出使用 SE(3) 对称变换模型来直接预测血管模型表面的 AAA 生长，该表面富含局部多物理特征。与参数化 AAA 形状的其他工作相比，这种表示保留了血管表面的解剖结构和几何保真度。我们使用 24 名 AAA 患者的 113 次计算机断层扫描血管造影 (CTA) 扫描的纵向数据集，以不规则采样间隔训练我们的模型。经过训练，我们的模型可以预测下一次扫描时刻的 AAA 生长，中值直径误差为 1.18 毫米。我们进一步证明了我们的模型在识别患者是否将在两年内有资格接受选择性修复方面的效用（acc = 0.93）。最后，我们评估了我们的模型在外部验证集上的泛化能力，该验证集由来自不同医院的 7 名 AAA 患者的 25 次 CTA 扫描组成。我们的结果表明，局部方向性 AAA 从血管表面的生长预测是可行的，并且可能有助于个性化监测策略。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08729) | **Categories:** cs.CV, cs.AI

---

### [11] [Video-CoT: A Comprehensive Dataset for Spatiotemporal Understanding of Videos Based on Chain-of-Thought](https://arxiv.org/abs/2506.08817)
*Shuyi Zhang, Xiaoshuai Hao, Yingbo Tang, Lingfeng Zhang, Pengwei Wang, Zhongyuan Wang, Hongxuan Ma, Shanghang Zhang*

Main category: cs.CV

TL;DR: Video-CoT数据集通过提供大规模时空问答对和CoT标注，旨在提升视频理解中模型的时空推理能力。


<details>
  <summary>Details</summary>
Motivation: Large-scale vision-language models (VLMs) often struggle to capture the nuanced, spatiotemporal details essential for thorough video analysis.

Method: The authors introduce Video-CoT, a dataset with 192,000 spatiotemporal question-answer pairs and 23,000 CoT-annotated samples, along with a benchmark for evaluation.

Result: Experiments show that current VLMs face significant challenges in achieving satisfactory performance on the Video-CoT dataset.

Conclusion: Current VLMs struggle with spatiotemporal understanding, highlighting the need for improved models.

Abstract: 为了提升视频分析中时空理解的能力，本研究引入了Video-CoT数据集，该数据集包含192,000个精细的时空问答对和23,000个高质量的CoT标注样本。此外，还提供了一个全面的基准，用于评估这些任务，每个任务包含750张图像和定制的评估指标。实验表明，当前的VLMs在实现令人满意的性能方面面临重大挑战，突显了有效时空理解的困难。Video-CoT数据集和基准为多媒体理解开辟了新的研究途径，并支持未来需要高级视频分析功能的智能系统创新。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08817) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Scaling Laws of Motion Forecasting and Planning -- A Technical Report](https://arxiv.org/abs/2506.08228)
*Mustafa Baniodeh, Kratarth Goel, Scott Ettinger, Carlos Fuertes, Ari Seff, Tim Shen, Cole Gulino, Chenjie Yang, Ghassen Jerfel, Dokook Choe, Rui Wang, Vinutha Kallem, Sergio Casas, Rami Al-Rfou, Benjamin Sapp, Dragomir Anguelov*

Main category: cs.LG

TL;DR: 研究表明，优化自动驾驶中运动预测和规划模型的训练及推理缩放是提升性能的关键，且使用通用驾驶数据训练能提升自我智能体性能。


<details>
  <summary>Details</summary>
Motivation: 研究自动驾驶领域中联合运动预测和规划任务的模型性能扩展规律。

Method: 使用一个包含50万小时驾驶数据集的encoder-decoder自回归transformer模型。

Result: 模型性能随着总计算预算的增加而呈幂律函数关系提高，并且模型训练损失和模型评估指标之间存在很强的相关性。闭环指标也随着缩放而提高。随着训练计算预算的增长，最佳缩放需要以数据集大小的1.5倍速度增加模型大小。较小模型的输出进行采样和聚类使其与较大的模型相比具有竞争力，直到超过交叉点，较大的模型变得更有效率。

Conclusion: 优化运动预测和规划模型的训练和推理时间缩放特性是提高其性能以解决各种驾驶场景的关键手段。此外，使用其他智能体的通用日志驾驶数据进行训练可以提高自我智能体的性能。

Abstract: 我们研究了一系列encoder-decoder自回归transformer模型在自动驾驶领域的联合运动预测和规划任务上的经验缩放规律。使用一个包含50万小时驾驶数据集，我们证明了，与语言建模类似，模型性能随着总计算预算的增加而呈幂律函数关系提高，并且我们观察到模型训练损失和模型评估指标之间存在很强的相关性。最有趣的是，闭环指标也随着缩放而提高，这对于开放环路指标在模型开发和爬坡中的适用性具有重要意义。我们还研究了transformer参数数量和训练数据大小对于训练计算最佳模型的最佳缩放比例。我们发现，随着训练计算预算的增长，最佳缩放需要以数据集大小的1.5倍速度增加模型大小。我们还研究了推理时计算缩放，我们观察到对较小模型的输出进行采样和聚类使其与较大的模型相比具有竞争力，直到超过交叉点，较大的模型变得更有效率。总的来说，我们的实验结果表明，优化运动预测和规划模型的训练和推理时间缩放特性是提高其性能以解决各种驾驶场景的关键手段。最后，我们简要地研究了使用其他智能体的通用日志驾驶数据进行训练以提高自我智能体的性能的效用，这是解决用于大容量模型训练的机器人数据稀缺性的一个重要研究领域。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08228) | **Categories:** cs.LG, cs.AI, cs.RO

---

### [2] [FlowBERT: Prompt-tuned BERT for variable flow field prediction](https://arxiv.org/abs/2506.08021)
*Weihao Zou, Weibing Feng, Pin Wu*

Main category: cs.LG

TL;DR: 该论文提出了一种基于大型语言模型知识迁移的通用流场预测框架，能够以更低的计算成本和更好的泛化能力进行快速流体动力学预测。


<details>
  <summary>Details</summary>
Motivation: 该研究旨在解决传统计算流体动力学(CFD)方法的高计算成本和现有深度学习模型的有限的跨条件迁移能力问题。

Method: 该框架创新性地将Proper Orthogonal Decomposition (POD)降维与预训练LLM的微调策略相结合。

Result: 实验结果表明，该框架在少样本学习场景中优于传统的Transformer模型，同时在各种流入条件和翼型几何形状中表现出卓越的泛化能力。与需要数小时计算的传统Navier-Stokes方程求解器相比，该方法将预测时间缩短到几秒，同时保持90%以上的精度。

Conclusion: 该研究提出的知识迁移框架为快速流体动力学预测开辟了新方向，并具有应用于气动优化、流动控制和其他工程领域的潜力。

Abstract: 本研究提出了一种基于大型语言模型(LLM)知识迁移的通用流场预测框架，旨在解决传统计算流体动力学(CFD)方法的高计算成本和现有深度学习模型的有限跨条件迁移能力问题。该框架创新性地将Proper Orthogonal Decomposition (POD)降维与预训练LLM的微调策略相结合，其中POD有助于压缩表示流场特征，而微调后的模型学习在状态空间中编码系统动力学。为了提高模型对流场数据的适应性，我们专门设计了面向流体动力学的文本模板，通过丰富的上下文语义信息来提高预测性能。实验结果表明，我们的框架在少样本学习场景中优于传统的Transformer模型，同时在各种流入条件和翼型几何形状中表现出卓越的泛化能力。消融研究揭示了FlowBERT架构中关键组件的贡献。与需要数小时计算的传统Navier-Stokes方程求解器相比，我们的方法将预测时间缩短到几秒，同时保持90%以上的精度。所开发的知识转移范式为快速流体动力学预测建立了一个新的方向，其潜在应用可扩展到气动优化、流动控制和其他工程领域。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08021) | **Categories:** cs.LG, physics.flu-dyn

---

### [3] [Time-Aware World Model for Adaptive Prediction and Control](https://arxiv.org/abs/2506.08441)
*Anh N. Nhu, Sanghyun Son, Ming Lin*

Main category: cs.LG

TL;DR: TAWM通过显式地结合时间动态，学习了各种控制问题中的高频和低频任务动态，从而提高了性能和数据效率。


<details>
  <summary>Details</summary>
Motivation: 传统的模型通常以固定的时间步长进行采样，忽略了系统底层动态对最佳采样率的影响。为了解决这个问题，本文提出了时间感知世界模型(TAWM)。

Method: 提出了一种时间感知世界模型(TAWM)，该模型显式地结合了时间动态。通过调节时间步长{\Delta}t，并在不同的{\Delta}t值范围内进行训练，TAWM学习了各种控制问题中的高频和低频任务动态。

Result: 实验结果表明，TAWM在各种控制任务中，使用相同数量的训练样本和迭代次数，并且在不同的观察率下表现优于传统模型。

Conclusion: 时间感知世界模型(TAWM)在各种控制任务中始终优于传统模型，使用相同数量的训练样本和迭代次数，并且在不同的观察率下表现出色。

Abstract: 本文介绍了一种时间感知世界模型(TAWM)，这是一种基于模型的方法，它显式地结合了时间动态。通过调节时间步长{\Delta}t，并在不同的{\Delta}t值范围内进行训练，TAWM学习了各种控制问题中的高频和低频任务动态。基于信息论的洞察，即最佳采样率取决于系统的底层动态，这种时间感知公式提高了性能和数据效率。实验评估表明，TAWM在各种控制任务中，使用相同数量的训练样本和迭代次数，并且在不同的观察率下始终优于传统模型。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08441) | **Categories:** cs.LG, cs.AI, cs.SY, eess.SY

---

### [4] [Intention-Conditioned Flow Occupancy Models](https://arxiv.org/abs/2506.08902)
*Chongyi Zheng, Seohong Park, Sergey Levine, Benjamin Eysenbach*

Main category: cs.LG

TL;DR: 该论文提出了一种名为 InFOM 的意图条件流占用模型，用于强化学习的预训练，该模型通过预测未来状态和利用用户意图，显著提高了回报和成功率。


<details>
  <summary>Details</summary>
Motivation: Applying large-scale pre-training to reinforcement learning can address core challenges like sample efficiency and robustness, but pre-training models that reason across time remains a challenge.

Method: The paper builds a probabilistic model using flow matching to predict future states, incorporating a latent variable to capture user intention and enable adaptation with generalized policy improvement.

Result: Experiments on 36 state-based and 4 image-based benchmark tasks demonstrate that InFOM achieves 1.8x median improvement in returns and increases success rates by 36%.

Conclusion: The proposed InFOM method achieves significant improvements in returns and success rates compared to alternative pre-training methods on a range of benchmark tasks.

Abstract: 大规模预训练从根本上改变了当今机器学习的研究方式：大型基础模型经过一次训练后，社区中的任何人都可以使用它们（包括那些没有数据或计算资源从头开始训练模型的人）来适应和微调特定任务。将这种相同的框架应用于强化学习 (RL) 很有吸引力，因为它为解决 RL 中的核心挑战（包括样本效率和鲁棒性）提供了令人信服的途径。然而，在 RL 的背景下预训练大型模型仍然存在一个根本性的挑战：动作具有长期依赖性，因此训练一个能够跨时间推理的基础模型非常重要。生成式人工智能的最新进展为建模高度复杂的分布提供了新的工具。在本文中，我们构建了一个概率模型，以使用流匹配预测智能体在遥远的将来将访问哪些状态（即，占用率度量）。由于大型数据集通常由执行不同任务的许多不同用户构建，因此我们在模型中包含一个潜在变量，用于捕获用户意图。这种意图提高了我们模型的表达能力，并能够通过广义策略改进进行适应。我们称我们提出的方法为意图条件流占用模型 (InFOM)。与替代的预训练方法相比，我们在 36 个基于状态和 4 个基于图像的基准任务上的实验表明，所提出的方法实现了 1.8 倍的回报中位数改进，并将成功率提高了 36%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08902) | **Categories:** cs.LG, cs.AI

---

### [5] [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007)
*Sophia Tang, Yinuo Zhang, Alexander Tong, Pranam Chatterjee*

Main category: cs.LG

TL;DR: BranchSBM 学习分支薛定谔桥，能够表示群体水平发散到多个终端分布，从而解决了现有方法无法捕获分支或发散演化的问题。


<details>
  <summary>Details</summary>
Motivation: 现有的方法，如流匹配和Schr"odinger Bridge Matching，通过建模单个随机路径来有效地学习两个分布之间的映射。然而，这些方法本质上仅限于单峰转换，无法捕获从共同起源到多个不同结果的分支或发散演化。

Method: BranchSBM参数化多个随时间变化的速度场和增长过程，从而能够表示群体水平发散到多个终端分布。

Result: 我们证明了BranchSBM不仅更具表现力，而且对于涉及多路径表面导航、模拟来自同质祖细胞状态的细胞命运分叉以及模拟对扰动的不同细胞反应的任务至关重要。

Conclusion: BranchSBM不仅更具表现力，而且对于涉及多路径表面导航、模拟来自同质祖细胞状态的细胞命运分叉以及模拟对扰动的不同细胞反应的任务至关重要。

Abstract: 预测初始分布和目标分布之间的中间轨迹是生成建模中的一个核心问题。现有的方法，如流匹配和Schr"odinger Bridge Matching，通过建模单个随机路径来有效地学习两个分布之间的映射。然而，这些方法本质上仅限于单峰转换，无法捕获从共同起源到多个不同结果的分支或发散演化。为了解决这个问题，我们引入了Branched Schr"odinger Bridge Matching (BranchSBM)，这是一个学习分支Schr"odinger bridge的新框架。BranchSBM参数化多个随时间变化的速度场和增长过程，从而能够表示群体水平发散到多个终端分布。我们证明了BranchSBM不仅更具表现力，而且对于涉及多路径表面导航、模拟来自同质祖细胞状态的细胞命运分叉以及模拟对扰动的不同细胞反应的任务至关重要。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09007) | **Categories:** cs.LG

---

### [6] [Agentic Neural Networks: Self-Evolving Multi-Agent Systems via Textual Backpropagation](https://arxiv.org/abs/2506.09046)
*Xiaowen Ma, Chenyang Lin, Yao Zhang, Volker Tresp, Yunpu Ma*

Main category: cs.LG

TL;DR: Agentic Neural Network (ANN) 将多智能体协作构建为神经架构，通过动态分解任务和迭代优化协作，实现了准确性和适应性的显著提升。


<details>
  <summary>Details</summary>
Motivation: 现有方法通常依赖于静态的、手动设计的多智能体配置，为了克服这些限制。

Method: Agentic Neural Network (ANN) 遵循两阶段优化策略：(1) 前向阶段：任务被动态分解为子任务，并逐层构建具有合适聚合方法的协作智能体团队。(2) 后向阶段：通过迭代反馈来优化全局和局部协作，允许智能体自我进化其角色、提示和协调。

Result: Agentic Neural Network (ANN) 能够创建新的或专门的智能体团队，从而在准确性和适应性方面实现了显著的提升。在四个基准数据集上，ANN 在相同的配置下超越了领先的多智能体基线。

Conclusion: Agentic Neural Network (ANN) 在四个基准数据集上超越了领先的多智能体基线，表明 ANN 为多智能体系统提供了一个可扩展的、数据驱动的框架。

Abstract: 利用多个大型语言模型（LLM）已被证明可以有效地解决复杂的高维任务，但目前的方法通常依赖于静态的、手动设计的多智能体配置。为了克服这些限制，我们提出了 Agentic Neural Network（ANN），这是一个将多智能体协作概念化为分层神经网络架构的框架。在这种设计中，每个智能体作为一个节点运行，每一层形成一个专注于特定子任务的协作“团队”。Agentic Neural Network 遵循两阶段优化策略：（1）前向阶段——从神经网络前向传递中获得灵感，任务被动态分解为子任务，并逐层构建具有合适聚合方法的协作智能体团队。（2）后向阶段——镜像反向传播，我们通过迭代反馈来优化全局和局部协作，允许智能体自我进化其角色、提示和协调。这种神经符号方法使 ANN 能够在训练后创建新的或专门的智能体团队，从而在准确性和适应性方面实现了显著的提升。在四个基准数据集上，ANN 在相同的配置下超越了领先的多智能体基线，显示出持续的性能改进。我们的研究结果表明，ANN 为多智能体系统提供了一个可扩展的、数据驱动的框架，结合了 LLM 的协作能力和神经网络原理的效率和灵活性。我们计划开源整个框架。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.09046) | **Categories:** cs.LG, cs.AI, cs.MA

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Ego-centric Learning of Communicative World Models for Autonomous Driving](https://arxiv.org/abs/2506.08149)
*Hang Wang, Dechen Gao, Junshan Zhang*

Main category: cs.RO

TL;DR: CALL：一种用于多智能体强化学习的通信世界模型，通过轻量级信息共享提高预测精度和性能。


<details>
  <summary>Details</summary>
Motivation: 多智能体强化学习（MARL）在复杂高维环境（如自动驾驶）中面临部分可观测性和非平稳性问题，信息共享受到通信开销和可扩展性问题的阻碍。

Method: 提出了一种名为CALL的MARL方法，该方法利用世界模型及其潜在表示进行通信。

Result: 通过信息共享，提高了预测精度，缩小了性能差距。

Conclusion: 在CARLA平台上进行的实验表明，使用CALL可以提高性能。

Abstract: 我们研究了复杂高维环境（如自动驾驶）中的多智能体强化学习（MARL）。众所周知，MARL 存在部分可观测性和非平稳性问题。为了应对这些挑战，通常采用信息共享，但在实践中面临主要障碍，包括巨大的通信开销和可扩展性问题。通过利用世界模型中包含的生成式人工智能及其潜在表示，我们开发了用于 MARL 的通信世界模型 CALL，其中 1) 每个智能体首先学习其世界模型，该模型将其状态和意图编码为具有较小内存占用的低维潜在表示，可以通过轻量级通信与感兴趣的其他智能体共享；2) 每个智能体在进行以自我为中心的学习时，利用轻量级信息共享来丰富其世界模型，然后利用其泛化能力来改进预测，从而更好地进行规划。我们描述了信息共享对预测精度的提升及其对性能差距的影响。在 CARLA 平台上具有挑战性的本地轨迹规划任务上进行了大量实验，以证明使用 CALL 的性能提升。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08149) | **Categories:** cs.RO, cs.AI

---

### [2] [Re4MPC: Reactive Nonlinear MPC for Multi-model Motion Planning via Deep Reinforcement Learning](https://arxiv.org/abs/2506.08344)
*Neşet Ünver Akmandor, Sarvesh Prajapati, Mark Zolotas, Taşkın Padır*

Main category: cs.RO

TL;DR: Re4MPC通过深度强化学习反应性地选择NMPC问题的模型，从而高效地生成轨迹，提高了计算效率和成功率。


<details>
  <summary>Details</summary>
Motivation: 传统的多自由度机器人运动规划方法计算量过大，不适用于实际环境。

Method: 提出了一种新颖的多模型运动规划流程Re4MPC，它使用非线性模型预测控制（NMPC）计算轨迹，并使用深度强化学习（DRL）框架学习反应决策策略。

Result: 实验结果表明，Re4MPC比NMPC基线在计算效率更高，并且在达到末端执行器目标方面取得了更高的成功率。

Conclusion: Re4MPC在计算效率和成功率方面优于NMPC基线。

Abstract: 针对多自由度机器人（如移动机械臂）的传统运动规划方法通常计算量过大，难以应用于实际环境。本文提出了一种新颖的多模型运动规划流程Re4MPC，该流程使用非线性模型预测控制（NMPC）计算轨迹。Re4MPC通过根据任务和机器人状态的复杂性，反应性地选择NMPC问题的模型、成本和约束，从而高效地生成轨迹。这种反应性决策的策略是通过深度强化学习（DRL）框架学习的。我们引入了一个数学公式，将NMPC集成到这个DRL框架中。为了验证我们的方法和设计选择，我们在一个涉及移动机械臂的基于物理的仿真环境中评估了DRL训练和测试结果。实验结果表明，Re4MPC比NMPC基线在计算效率更高，并且在达到末端执行器目标方面取得了更高的成功率。NMPC基线在没有我们的学习机制的情况下计算全身轨迹。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08344) | **Categories:** cs.RO, cs.AI, cs.LG, cs.SY, eess.SY

---

### [3] [TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization](https://arxiv.org/abs/2506.08440)
*Zengjue Chen, Runliang Niu, He Kong, Qi Wang*

Main category: cs.RO

TL;DR: TGRPO通过融合步级和轨迹级优势信号，改进了VLA模型在线强化学习训练的群体级优势估计，从而在操作任务中表现出色。


<details>
  <summary>Details</summary>
Motivation: VLA模型需要在新环境中进行特定于任务的微调，并且严重依赖于静态轨迹数据集的大小和质量。此外，它们无法与环境交互或利用实时执行的反馈。

Method: 提出了一种轨迹式群体相对策略优化（TGRPO）方法，融合了步级和轨迹级优势信号，从而改进了GRPO的群体级优势估计。

Result: 在libero-object基准测试的十个操作任务上的实验结果表明，TGRPO始终优于各种基线方法。

Conclusion: TGRPO在多个操作任务中优于基线方法，能够生成更稳健和高效的策略。

Abstract: 视觉-语言-动作（VLA）模型的最新进展表明，当在大规模数据集上进行预训练时，该模型在不同的场景、任务和机器人平台中具有强大的泛化能力。然而，这些模型仍然需要在新的环境中进行特定于任务的微调，这个过程几乎完全依赖于使用静态轨迹数据集进行监督微调（SFT）。这些方法既不允许机器人与环境交互，也不利用实时执行的反馈。此外，它们的成功还严重依赖于收集到的轨迹的大小和质量。强化学习（RL）通过实现闭环交互并将学习到的策略直接与任务目标对齐，提供了一种有希望的替代方案。在这项工作中，我们从GRPO的思想中汲取灵感，并提出了轨迹式群体相对策略优化（TGRPO）方法。通过融合步级和轨迹级优势信号，该方法改进了GRPO的群体级优势估计，从而使该算法更适合VLA的在线强化学习训练。在libero-object基准测试的十个操作任务上的实验结果表明，TGRPO始终优于各种基线方法，能够生成更稳健和高效的策略。我们的源代码可在以下网址获得：https://github.com/hahans/TGRPO

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08440) | **Categories:** cs.RO

---

### [4] [Diffusion Models for Safety Validation of Autonomous Driving Systems](https://arxiv.org/abs/2506.08459)
*Juanran Wang, Marc R. Schlichting, Harrison Delecki, Mykel J. Kochenderfer*

Main category: cs.RO

TL;DR: 本文提出了一种基于去噪扩散模型的自动驾驶故障生成方法，用于解决自动驾驶系统安全验证的难题。


<details>
  <summary>Details</summary>
Motivation: 由于现实世界测试的高风险和成本以及潜在故障的罕见性和多样性，自动驾驶系统的安全验证极具挑战性。

Method: 我们训练了一个去噪扩散模型，以生成给定任何初始交通状态的自动驾驶汽车的潜在故障案例。

Result: 在四向交叉口问题的实验表明，在各种场景中，扩散模型可以生成真实的故障样本，同时捕获各种潜在故障。

Conclusion: 该扩散模型无需外部训练数据集，可以使用适度的计算资源执行训练和推理，并且不假设任何关于被测系统的先验知识，适用于交通路口的安全验证。

Abstract: 自动驾驶系统（ADS）的安全验证极具挑战，这是由于真实路况测试具有高风险和高成本，并且潜在的失效情况非常少见和多样。为了解决这些问题，我们训练了一个去噪扩散模型，该模型可以在给定任何初始交通状态下生成自动驾驶车辆（AV）的潜在失效案例。在四岔路口问题上的实验表明，在各种场景下，该扩散模型能够生成真实的失效样本，同时捕捉各种各样的潜在失效情况。我们的模型不需要任何外部训练数据集，可以使用适度的计算资源执行训练和推理，并且不假设任何关于被测系统的先验知识，适用于交通路口的安全验证。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08459) | **Categories:** cs.RO, cs.AI

---

### [5] [Deploying SICNav in the Field: Safe and Interactive Crowd Navigation using MPC and Bilevel Optimization](https://arxiv.org/abs/2506.08851)
*Sepehr Samavi, Garvish Bhutani, Florian Shkurti, Angela P. Schoellig*

Main category: cs.RO

TL;DR: 该论文提出了一种安全和交互式的拥挤导航方法，通过显式建模智能体之间的互动，实现了在拥挤环境中的安全高效导航。


<details>
  <summary>Details</summary>
Motivation: 在拥挤环境中安全高效的导航仍然是机器人的一个关键挑战，传统的机器人拥挤导航方法将人类运动预测与机器人运动规划分离，忽略了人类与机器人之间的闭环互动，导致机器人容易陷入困境。

Method: 提出了一种双层模型预测控制（MPC）框架，用于安全和交互式的拥挤导航。

Result: 在室内和室外环境中进行了近7公里的自主导航实验，初步分析了系统的运行情况。

Conclusion: 该论文提出了一个安全和交互式的拥挤导航（SICNav）方法，通过双层模型预测控制（MPC）框架，将预测和规划结合到一个优化问题中，显式地建模了智能体之间的交互。

Abstract: 在拥挤环境中安全高效的导航仍然是机器人面临的一个关键挑战，这些机器人执行各种服务任务，例如食物运送或自主轮椅移动。经典的机器人拥挤导航方法将人类运动预测与机器人运动规划分离，忽略了人类与机器人之间的闭环互动。由于缺乏对人类对机器人计划的反应模型（例如，让路），可能导致机器人陷入困境。我们提出的安全和交互式拥挤导航（SICNav）方法是一个双层模型预测控制（MPC）框架，它将预测和规划结合到一个优化问题中，显式地建模了智能体之间的互动。在本文中，我们对拥挤导航平台进行了系统概述，我们使用该平台在以前未见过的室内和室外环境中部署SICNav。我们对系统在室内和室外环境中超过2小时的近7公里自主导航过程中的运行情况进行了初步分析。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.08851) | **Categories:** cs.RO

---
