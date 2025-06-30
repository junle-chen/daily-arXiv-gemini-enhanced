# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-01

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算语言学 (Computation and Language) (2)](#cs-cl)
- [计算机视觉 (Computer Vision) (5)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (3)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [MobiVerse: Scaling Urban Mobility Simulation with Hybrid Lightweight Domain-Specific Generator and Large Language Models](https://arxiv.org/abs/2506.21784)
*Yifan Liu, Xishun Liao, Haoxuan Ma, Jonathan Liu, Rohan Jadhav, Jiaqi Ma*

Main category: cs.AI

TL;DR: MobiVerse是一个混合框架，它结合了领域特定生成器和LLM，用于大规模城市交通仿真，能够模拟个体对环境变化的反应。


<details>
  <summary>Details</summary>
Motivation: 现有的交通仿真平台在算法开发、政策实施和大规模评估方面存在不足。传统的基于活动的模型需要大量数据和手动校准，机器学习方法难以适应动态条件，而基于LLM的实现面临计算限制。

Method: MobiVerse是一个混合框架，它利用轻量级的领域特定生成器生成基础活动链，并使用LLM进行上下文感知的修改。

Result: 在洛杉矶西木区的案例研究中，MobiVerse成功地为约53,000名智能体生成并动态调整了行程安排，使其能够对道路封闭、大型集会和拥堵等环境反馈做出反应。实验表明，该方法在保持计算效率的同时，提高了行为的真实性。

Conclusion: MobiVerse通过结合领域特定生成器的效率和LLM的适应性，为大规模城市交通仿真提供了一个可定制的平台，能够模拟个体对环境变化的反应，并支持各种交通算法的测试。

Abstract: 理解和建模人类移动模式对于有效的交通规划和城市发展至关重要。尽管移动研究取得了显著进展，但在仿真平台方面仍然存在一个关键缺口，这些平台允许算法开发、政策实施和大规模综合评估。传统的基于活动的模型需要大量数据收集和手动校准，机器学习方法难以适应动态条件，并且新兴的基于代理的大型语言模型（LLM）实现在大规模仿真中面临计算约束。为了应对这些挑战，我们提出了MobiVerse，这是一个混合框架，它利用轻量级领域特定生成器的效率来生成基础活动链，并利用LLM的适应性来进行上下文感知的修改。在洛杉矶西木区进行了一项案例研究，在该研究中，我们在一台标准PC上有效地生成并动态调整了约53,000名智能体的整个行程安排。我们的实验表明，通过我们的混合框架，MobiVerse成功地使智能体能够对环境反馈做出反应，包括道路封闭、大型聚会活动（如足球比赛）和拥堵。其模块化设计有助于在交通系统和智能体级别测试各种移动算法。结果表明，我们的方法在保持计算效率的同时，增强了行为的真实性。MobiVerse通过提供一个可定制的平台，用于移动系统规划和运营以及基准算法，弥合了移动仿真方面的差距。代码和视频可在https://github.com/ucla-mobility/MobiVerse上找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21784) | **Categories:** cs.AI

---

### [2] [CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation](https://arxiv.org/abs/2506.21805)
*Nicolas Bougie, Narimasa Watanabe*

Main category: cs.AI

TL;DR: CitySim利用大型语言模型，通过递归价值驱动方法模拟城市环境中具有信念和长期目标的人类行为，更贴近真实人类，并可用于预测城市现象。


<details>
  <summary>Details</summary>
Motivation: 现有的城市环境中人类行为建模方法依赖于死板的手工规则，限制了模拟细微意图、计划和适应性行为的能力。

Method: 利用递归价值驱动方法，平衡强制性活动、个人习惯和情境因素，生成真实的每日计划。

Result: CitySim在微观和宏观层面都比以前的工作更符合真实的人类行为。通过模拟数万名智能体，并在各种现实场景下评估他们的集体行为，包括估计人群密度、预测地点受欢迎程度和评估幸福感，展示了CitySim的有效性。

Conclusion: CitySim是一个可扩展、灵活的测试平台，可用于理解和预测城市现象。

Abstract: 在城市环境中建模人类行为对于社会科学、行为研究和城市规划至关重要。先前的工作通常依赖于僵化的、手工制作的规则，限制了它们模拟细微的意图、计划和适应性行为的能力。为了应对这些挑战，我们设想了一个城市模拟器（CitySim），它利用了大型语言模型所展现的人类水平智能的突破。在CitySim中，智能体使用递归价值驱动方法生成真实的每日时间表，该方法平衡了强制性活动、个人习惯和情境因素。为了实现长期的、逼真的模拟，我们赋予智能体信念、长期目标和用于导航的空间记忆。CitySim在微观和宏观层面都比以前的工作更符合真实的人类行为。此外，我们通过模拟数万名智能体，并在各种现实场景下评估他们的集体行为，包括估计人群密度、预测地点受欢迎程度和评估幸福感，从而进行了有见地的实验。我们的结果表明，CitySim是一个可扩展、灵活的测试平台，可用于理解和预测城市现象。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21805) | **Categories:** cs.AI, cs.CL

---

### [3] [Universal Retrieval for Multimodal Trajectory Modeling](https://arxiv.org/abs/2506.22056)
*Xuan Zhang, Ziyan Jiang, Rui Meng, Yifei Leng, Zhenbang Xiao, Zora Zhiruo Wang, Yanyi Shang, Dehan Kong*

Main category: cs.AI

TL;DR: 本文提出了GAE-Retriever，用于解决多模态轨迹检索问题，并在多个数据集上取得了优异的性能。


<details>
  <summary>Details</summary>
Motivation: 轨迹数据在增强AI代理能力方面具有巨大潜力，但如何建模轨迹级别数据的表示仍然是一个挑战。

Method: 提出了GAE-Retriever，一个多模态检索框架，它采用了视觉-语言模型，并通过令牌选择和GradCache机制结合了优化的对比学习。

Result: 在多个数据集上的综合评估表明，GAE-Retriever在检索召回率方面始终优于强大的基线。

Conclusion: GAE-Retriever在多模态轨迹检索方面优于现有方法，证明了其有效性。

Abstract: 轨迹数据在捕捉人类行为和环境状态方面具有重要潜力，尤其是在GUI环境中，可以增强AI代理的能力。然而，在轨迹数据爆炸式增长的背景下，如何建模轨迹级别数据的表示仍然是一个尚未得到系统解决的重大挑战。本文提出了多模态轨迹检索，弥合了通用检索和以代理为中心的轨迹建模之间的差距。我们构建了统一代理轨迹数据集（UATD），该数据集来自各种真实场景中的带注释的演示和状态。在此基础上，我们提出了GAE-Bench，这是一个包含大量基于轨迹的检索对的基准。此外，我们提出了GAE-Retriever，一个多模态检索框架，它采用了视觉-语言模型，并通过令牌选择和GradCache机制结合了优化的对比学习。在多个数据集上的综合评估表明，GAE-Retriever在检索召回率方面始终优于强大的基线，突显了其在推进多模态轨迹检索方面的有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22056) | **Categories:** cs.AI

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [FloorPlan-DeepSeek (FPDS): A multimodal approach to floorplan generation using vector-based next room prediction](https://arxiv.org/abs/2506.21562)
*Jun Yin, Pengyu Zeng, Jing Zhong, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In the architectural design process, floor plan generation is inherently progressive and iterative. However, existing generative models for floor plans are predominantly end-to-end generation that produce an entire pixel-based layout in a single pass. This paradigm is often incompatible with the incremental workflows observed in real-world architectural practice. To address this issue, we draw inspiration from the autoregressive 'next token prediction' mechanism commonly used in large language models, and propose a novel 'next room prediction' paradigm tailored to architectural floor plan modeling. Experimental evaluation indicates that FPDS demonstrates competitive performance in comparison to diffusion models and Tell2Design in the text-to-floorplan task, indicating its potential applicability in supporting future intelligent architectural design.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21562) | **Categories:** cs.CL, cs.AI, cs.AR

---

### [2] [Random Initialization Can't Catch Up: The Advantage of Language Model Transfer for Time Series Forecasting](https://arxiv.org/abs/2506.21570)
*Roland Riachi, Kashif Rasul, Arjun Ashok, Prateek Humane, Alexis Roger, Andrew R. Williams, Yuriy Nevmyvaka, Irina Rish*

Main category: cs.CL

TL;DR: 该论文分析了在低数据情况下，语言模型迁移到时间序列预测的有效性，并观察到持续存在的迁移差距，为时间序列预测和模态无关属性研究开辟了道路。


<details>
  <summary>Details</summary>
Motivation: 在低数据情况下，调整预训练语言模型（LMs）以预测时间序列是有效的。本研究旨在分析语言模型到时间序列预测的有效迁移。

Method: 分析了在各种设计选择下，从语言模型到时间序列预测的有效迁移，包括上游后训练、时间序列分词器和语言主干大小。

Result: 在低数据情况下，这些设计选择对验证损失有显著影响，存在明显优于其他选择的方案。与Hernandez等人（2021）的结论相反，我们观察到语言模型的验证损失在随机初始化模型收敛后持续平稳下降，导致在各种设计选择中都存在持续存在的迁移差距。

Conclusion: 语言模型的验证损失在随机初始化模型收敛后持续平稳下降，导致持续存在的迁移差距。

Abstract: 最近的研究表明，调整预训练语言模型（LMs）对于在低数据情况下预测时间序列是有效的。我们在此基础上，分析了在各种设计选择下，从语言模型到时间序列预测的有效迁移，包括上游后训练、时间序列分词器和语言主干大小。在低数据情况下，这些设计选择对验证损失有显著影响，存在明显优于其他选择的方案。与Hernandez等人（2021）的结论相反，我们观察到语言模型的验证损失在随机初始化模型收敛后持续平稳下降，导致在各种设计选择中都存在持续存在的迁移差距。这些发现不仅有助于阐明如何有效地利用计算高效的训练来进行时间序列预测，而且还为研究这些模型所利用的数据分布的模态无关属性开辟了道路。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21570) | **Categories:** cs.CL, cs.AI, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [R1-Track: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning](https://arxiv.org/abs/2506.21980)
*Biao Wang, Wenwen Li*

Main category: cs.CV

TL;DR: 通过GRPO强化学习微调Qwen2.5-VL，R1-Track在视觉单目标跟踪中实现了显著性能，并支持灵活初始化。


<details>
  <summary>Details</summary>
Motivation: 现有的视觉单目标跟踪方法通常需要显式的分类和回归建模，依赖于大规模数据集的监督训练，并且仅限于跟踪的单一任务，缺乏灵活性。大型多模态语言模型在定位任务中表现出色，但直接应用于视觉跟踪时，模板匹配效果不佳。

Method: 使用基于规则奖励函数的GRPO强化学习方法对Qwen2.5-VL进行微调。

Result: R1-Track在GOT-10k基准测试中取得了显著的性能，并支持通过边界框或文本描述进行灵活初始化，同时保留了原始模型的大部分通用能力。

Conclusion: R1-Track通过在小规模数据集上使用基于规则奖励函数的GRPO强化学习方法对Qwen2.5-VL进行微调，在GOT-10k基准测试中取得了显著的性能，并支持通过边界框或文本描述进行灵活初始化，同时保留了原始模型的大部分通用能力。

Abstract: 视觉单目标跟踪旨在给定第一帧中的初始状态后，连续定位和估计后续视频帧中目标的大小。这项任务传统上被视为一个模板匹配问题，经历了包括相关滤波、双流网络和单流网络在内的主要阶段，并取得了显著进展。然而，这些方法通常需要显式的分类和回归建模，依赖于大规模数据集的监督训练，并且仅限于跟踪的单一任务，缺乏灵活性。近年来，多模态大型语言模型（MLLM）发展迅速。像Qwen2.5-VL这样的开源模型，作为具有强大基础能力的主流MLLM，在定位任务中表现出色。这激发了人们对将此类模型直接应用于视觉跟踪的兴趣。然而，实验表明，Qwen2.5-VL在图像对之间的模板匹配（即跟踪任务）方面表现不佳。受到deepseek-R1的启发，我们使用基于规则奖励函数的GRPO强化学习方法在小规模数据集上对Qwen2.5-VL进行了微调。由此产生的模型R1-Track在GOT-10k基准测试中取得了显著的性能。R1-Track支持通过边界框或文本描述进行灵活初始化，同时保留了原始模型的大部分通用能力。我们进一步讨论了R1-Track的潜在改进。这份粗略的技术报告总结了我们截至2025年5月的发现。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21980) | **Categories:** cs.CV

---

### [2] [Integrating Multi-Modal Sensors: A Review of Fusion Techniques for Intelligent Vehicles](https://arxiv.org/abs/2506.21885)
*Chuheng Wei, Ziye Qin, Ziyan Zhang, Guoyuan Wu, Matthew J. Barth*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-sensor fusion plays a critical role in enhancing perception for autonomous driving, overcoming individual sensor limitations, and enabling comprehensive environmental understanding. This paper first formalizes multi-sensor fusion strategies into data-level, feature-level, and decision-level categories and then provides a systematic review of deep learning-based methods corresponding to each strategy. We present key multi-modal datasets and discuss their applicability in addressing real-world challenges, particularly in adverse weather conditions and complex urban environments. Additionally, we explore emerging trends, including the integration of Vision-Language Models (VLMs), Large Language Models (LLMs), and the role of sensor fusion in end-to-end autonomous driving, highlighting its potential to enhance system adaptability and robustness. Our work offers valuable insights into current methods and future directions for multi-sensor fusion in autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21885) | **Categories:** cs.CV, cs.MM, cs.RO

---

### [3] [Generating Attribute-Aware Human Motions from Textual Prompt](https://arxiv.org/abs/2506.21912)
*Xinghan Wang, Kun Xu, Fei Li, Cao Sheng, Jiazhong Yu, Yadong Mu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Text-driven human motion generation has recently attracted considerable attention, allowing models to generate human motions based on textual descriptions. However, current methods neglect the influence of human attributes (such as age, gender, weight, and height) which are key factors shaping human motion patterns. This work represents a pilot exploration for bridging this gap. We conceptualize each motion as comprising both attribute information and action semantics, where textual descriptions align exclusively with action semantics. To achieve this, a new framework inspired by Structural Causal Models is proposed to decouple action semantics from human attributes, enabling text-to-semantics prediction and attribute-controlled generation. The resulting model is capable of generating realistic, attribute-aware motion aligned with the user's text and attribute inputs. For evaluation, we introduce HumanAttr, a comprehensive dataset containing attribute annotations for text-motion pairs, setting the first benchmark for attribute-aware text-to-motion generation. Extensive experiments on the new dataset validate our model's effectiveness.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21912) | **Categories:** cs.CV, cs.MM

---

### [4] [SPAZER: Spatial-Semantic Progressive Reasoning Agent for Zero-shot 3D Visual Grounding](https://arxiv.org/abs/2506.21924)
*Zhao Jin, Rong-Cheng Tu, Jingyi Liao, Wenhao Sun, Xiao Luo, Shunyu Liu, Dacheng Tao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: 3D Visual Grounding (3DVG) aims to localize target objects within a 3D scene based on natural language queries. To alleviate the reliance on costly 3D training data, recent studies have explored zero-shot 3DVG by leveraging the extensive knowledge and powerful reasoning capabilities of pre-trained LLMs and VLMs. However, existing paradigms tend to emphasize either spatial (3D-based) or semantic (2D-based) understanding, limiting their effectiveness in complex real-world applications. In this work, we introduce SPAZER - a VLM-driven agent that combines both modalities in a progressive reasoning framework. It first holistically analyzes the scene and produces a 3D rendering from the optimal viewpoint. Based on this, anchor-guided candidate screening is conducted to perform a coarse-level localization of potential objects. Furthermore, leveraging retrieved relevant 2D camera images, 3D-2D joint decision-making is efficiently performed to determine the best-matching object. By bridging spatial and semantic reasoning neural streams, SPAZER achieves robust zero-shot grounding without training on 3D-labeled data. Extensive experiments on ScanRefer and Nr3D benchmarks demonstrate that SPAZER significantly outperforms previous state-of-the-art zero-shot methods, achieving notable gains of 9.0% and 10.9% in accuracy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21924) | **Categories:** cs.CV

---

### [5] [RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation](https://arxiv.org/abs/2506.22007)
*Liudi Yang, Yang Bai, George Eskandar, Fengyi Shen, Mohammad Altillawi, Dong Chen, Soumajit Majumder, Ziyuan Liu, Gitta Kutyniok, Abhinav Valada*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We address the problem of generating long-horizon videos for robotic manipulation tasks. Text-to-video diffusion models have made significant progress in photorealism, language understanding, and motion generation but struggle with long-horizon robotic tasks. Recent works use video diffusion models for high-quality simulation data and predictive rollouts in robot planning. However, these works predict short sequences of the robot achieving one task and employ an autoregressive paradigm to extend to the long horizon, leading to error accumulations in the generated video and in the execution. To overcome these limitations, we propose a novel pipeline that bypasses the need for autoregressive generation. We achieve this through a threefold contribution: 1) we first decompose the high-level goals into smaller atomic tasks and generate keyframes aligned with these instructions. A second diffusion model then interpolates between each of the two generated frames, achieving the long-horizon video. 2) We propose a semantics preserving attention module to maintain consistency between the keyframes. 3) We design a lightweight policy model to regress the robot joint states from generated videos. Our approach achieves state-of-the-art results on two benchmarks in video quality and consistency while outperforming previous policy models on long-horizon tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22007) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](https://arxiv.org/abs/2506.21976)
*Shuhan Tan, John Lambert, Hong Jeon, Sakshum Kulshrestha, Yijing Bai, Jing Luo, Dragomir Anguelov, Mingxing Tan, Chiyu Max Jiang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The goal of traffic simulation is to augment a potentially limited amount of manually-driven miles that is available for testing and validation, with a much larger amount of simulated synthetic miles. The culmination of this vision would be a generative simulated city, where given a map of the city and an autonomous vehicle (AV) software stack, the simulator can seamlessly simulate the trip from point A to point B by populating the city around the AV and controlling all aspects of the scene, from animating the dynamic agents (e.g., vehicles, pedestrians) to controlling the traffic light states. We refer to this vision as CitySim, which requires an agglomeration of simulation technologies: scene generation to populate the initial scene, agent behavior modeling to animate the scene, occlusion reasoning, dynamic scene generation to seamlessly spawn and remove agents, and environment simulation for factors such as traffic lights. While some key technologies have been separately studied in various works, others such as dynamic scene generation and environment simulation have received less attention in the research community. We propose SceneDiffuser++, the first end-to-end generative world model trained on a single loss function capable of point A-to-B simulation on a city scale integrating all the requirements above. We demonstrate the city-scale traffic simulation capability of SceneDiffuser++ and study its superior realism under long simulation conditions. We evaluate the simulation quality on an augmented version of the Waymo Open Motion Dataset (WOMD) with larger map regions to support trip-level simulation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21976) | **Categories:** cs.LG, cs.AI, cs.CV, cs.MA, cs.RO

---

### [2] [UniCA: Adapting Time Series Foundation Model to General Covariate-Aware Forecasting](https://arxiv.org/abs/2506.22039)
*Lu Han, Yu Liu, Qiwen Deng, Jian Jiang, Yinbo Sun, Zhe Yu, Binfeng Wang, Xingyu Lu, Lintao Ma, Han-Jia Ye, De-Chuan Zhan*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time Series Foundation Models (TSFMs) have achieved remarkable success through large-scale pretraining. However, their design primarily targets real-valued series, limiting their ability to handle general forecasting tasks involving diverse and often heterogeneous covariates--such as categorical variables and multimodal data (e.g., images, text)--which are typically task-specific and difficult to leverage during pretraining. To address this gap, we propose Unified Covariate Adaptation (UniCA), a framework to bridge TSFMs with general covariate-aware forecasting. UniCA first performs covariate homogenization to transform heterogeneous covariates into high-level homogeneous series representations and then fuses them via a unified attention-based fusion mechanism. UniCA is compatible and universal for adaptation with both homogeneous and heterogeneous covariates, incorporating extra covariate information while preserving the generalization ability of TSFMs.Extensive experiments on multiple unimodal and multimodal covariate-aware forecasting benchmarks demonstrate the superiority of UniCA, highlighting the promise of covariate-aware TSFM adaptation in real-world forecasting scenarios. Codes are released on https://github.com/hanlu-nju/UniCA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22039) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Skill-Nav: Enhanced Navigation with Versatile Quadrupedal Locomotion via Waypoint Interface](https://arxiv.org/abs/2506.21853)
*Dewei Wang, Chenjia Ba, Chenhui Li, Jiyuan Shi, Yan Ding, Chi Zhang, Bin Zhao*

Main category: cs.RO

TL;DR: Skill-Nav 通过将四足运动技能整合到使用航点接口的分层导航框架中，增强了机器人在复杂地形中的导航能力。


<details>
  <summary>Details</summary>
Motivation: 四足机器人已经通过强化学习展示了卓越的运动能力，但将运动技能与导航相结合尚未充分研究。

Method: 提出 Skill-Nav，一种将四足机器人运动技能融入分层导航框架的方法，使用航点作为接口。

Result: 在模拟和真实场景中进行的大量实验表明，Skill-Nav 能够有效地穿越复杂地形并完成具有挑战性的导航任务。

Conclusion: Skill-Nav 能够有效地穿越复杂地形并完成具有挑战性的导航任务。

Abstract: 四足机器人已经通过强化学习展示了卓越的运动能力，包括极限跑酷动作。然而，将运动技能与导航在四足机器人中结合尚未充分研究，这为增强长距离移动能力带来了希望。在本文中，我们提出 Skill-Nav，一种将四足机器人运动技能融入分层导航框架的方法，使用航点作为接口。具体来说，我们使用深度强化学习训练了一个航点引导的运动策略，使机器人能够自主调整其运动技能以到达目标位置，同时避开障碍物。与直接速度命令相比，航点为高级规划和低级控制提供了一个更简单但更灵活的接口。利用航点作为接口允许应用各种通用规划工具，例如大型语言模型 (LLM) 和路径规划算法，以指导我们的运动策略穿越具有不同障碍物的地形。在模拟和真实场景中进行的大量实验表明，Skill-Nav 能够有效地穿越复杂地形并完成具有挑战性的导航任务。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21853) | **Categories:** cs.RO

---

### [2] [A MILP-Based Solution to Multi-Agent Motion Planning and Collision Avoidance in Constrained Environments](https://arxiv.org/abs/2506.21982)
*Akshay Jaitly, Jack Cline, Siavash Farzan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose a mixed-integer linear program (MILP) for multi-agent motion planning that embeds Polytopic Action-based Motion Planning (PAAMP) into a sequence-then-solve pipeline. Region sequences confine each agent to adjacent convex polytopes, while a big-M hyperplane model enforces inter-agent separation. Collision constraints are applied only to agents sharing or neighboring a region, which reduces binary variables exponentially compared with naive formulations. An L1 path-length-plus-acceleration cost yields smooth trajectories. We prove finite-time convergence and demonstrate on representative multi-agent scenarios with obstacles that our formulation produces collision-free trajectories an order of magnitude faster than an unstructured MILP baseline.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.21982) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [3] [An Introduction to Zero-Order Optimization Techniques for Robotics](https://arxiv.org/abs/2506.22087)
*Armand Jordana, Jianghan Zhang, Joseph Amigo, Ludovic Righetti*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Zero-order optimization techniques are becoming increasingly popular in robotics due to their ability to handle non-differentiable functions and escape local minima. These advantages make them particularly useful for trajectory optimization and policy optimization. In this work, we propose a mathematical tutorial on random search. It offers a simple and unifying perspective for understanding a wide range of algorithms commonly used in robotics. Leveraging this viewpoint, we classify many trajectory optimization methods under a common framework and derive novel competitive RL algorithms.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22087) | **Categories:** cs.RO

---
