# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-28

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (7)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning](https://arxiv.org/abs/2511.21460)
*Junjian Wang, Lidan Zhao, Xi Sheryl Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring the safety of embodied AI agents during task planning is critical for real-world deployment, especially in household environments where dangerous instructions pose significant risks. Existing methods often suffer from either high computational costs due to preference alignment training or over-rejection when using single-agent safety prompts. To address these limitations, we propose MADRA, a training-free Multi-Agent Debate Risk Assessment framework that leverages collective reasoning to enhance safety awareness without sacrificing task performance. MADRA employs multiple LLM-based agents to debate the safety of a given instruction, guided by a critical evaluator that scores responses based on logical soundness, risk identification, evidence quality, and clarity. Through iterative deliberation and consensus voting, MADRA significantly reduces false rejections while maintaining high sensitivity to dangerous tasks. Additionally, we introduce a hierarchical cognitive collaborative planning framework that integrates safety, memory, planning, and self-evolution mechanisms to improve task success rates through continuous learning. We also contribute SafeAware-VH, a benchmark dataset for safety-aware task planning in VirtualHome, containing 800 annotated instructions. Extensive experiments on AI2-THOR and VirtualHome demonstrate that our approach achieves over 90% rejection of unsafe tasks while ensuring that safe-task rejection is low, outperforming existing methods in both safety and execution efficiency. Our work provides a scalable, model-agnostic solution for building trustworthy embodied agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21460) | **Categories:** cs.AI

---

### [2] [SpatialBench: Benchmarking Multimodal Large Language Models for Spatial Cognition](https://arxiv.org/abs/2511.21471)
*Peiran Xu, Sudong Wang, Yao Zhu, Jianing Li, Yunjian Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Spatial cognition is fundamental to real-world multimodal intelligence, allowing models to effectively interact with the physical environment. While multimodal large language models (MLLMs) have made significant strides, existing benchmarks often oversimplify spatial cognition, reducing it to a single-dimensional metric, which fails to capture the hierarchical structure and interdependence of spatial abilities. To address this gap, we propose a hierarchical spatial cognition framework that decomposes spatial intelligence into five progressively complex levels from basic observation to high-level planning. Building upon this taxonomy, we construct SpatialBench, a large-scale, fine-grained benchmark covering 15 tasks aligned with these cognitive levels. To provide a unified evaluation across heterogeneous tasks, we further introduce a high-level capability-oriented metric that reliably assesses a model's overall spatial reasoning ability. Extensive experiments over massive MLLMs reveal distinct performance stratification across cognitive levels: models exhibit strong perceptual grounding yet remain limited in symbolic reasoning, causal inference, and planning. Additional human tests demonstrate that humans perform selective, goal-directed abstraction, while MLLMs tend to over-attend to surface details without coherent spatial intent. Our work establishes the first systematic framework for measuring hierarchical spatial cognition in MLLMs, laying the foundation for future spatially intelligent systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21471) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving](https://arxiv.org/abs/2511.20720)
*Haibo HU, Lianming Huang, Nan Guan, Chun Jason Xue*

Main category: cs.CV

TL;DR: DeeAD通过评估中间轨迹的物理可行性来加速VLA规划，无需重新训练即可集成到现有VLA模型中。


<details>
  <summary>Details</summary>
Motivation: VLA模型在自主驾驶中统一了感知、推理和轨迹生成，但由于深度transformer堆栈而存在显著的推理延迟。

Method: 提出了一种免训练的、动作引导的早退框架DeeAD，通过评估中间轨迹的物理可行性来加速VLA规划。引入多跳控制器，自适应地跳过冗余层。

Result: 在Bench2Drive基准测试中，DeeAD展示了高达28%的transformer层稀疏性和29%的延迟降低，同时保持了规划质量和安全性。

Conclusion: DeeAD通过在VLA模型中集成动作引导的早退框架，实现了显著的推理加速，同时保持了规划质量和安全性，且无需重新训练。

Abstract: 视觉-语言动作（VLA）模型统一了感知、推理和轨迹生成，用于自动驾驶，但由于深度transformer堆栈而存在显著的推理延迟。我们提出了DeeAD，一个免训练的、动作引导的早退框架，通过评估中间轨迹的物理可行性来加速VLA规划。DeeAD不是依赖于置信度分数，而是在预测的轨迹与轻量级规划先验（例如，导航或低精度规划）在可容忍的偏差（<2m）内对齐时，终止推理。为了提高效率，我们引入了一个多跳控制器，该控制器基于分数的变化率自适应地跳过冗余层。DeeAD集成到现有的VLA模型中，例如ORION，无需重新训练。在Bench2Drive基准测试上的实验表明，高达28%的transformer层稀疏性和29%的延迟降低，同时保持了规划质量和安全性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20720) | **Categories:** cs.CV, cs.AI, cs.LG, cs.RO

---

### [2] [TrafficLens: Multi-Camera Traffic Video Analysis Using LLMs](https://arxiv.org/abs/2511.20965)
*Md Adnan Arefeen, Biplob Debnath, Srimat Chakradhar*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traffic cameras are essential in urban areas, playing a crucial role in intelligent transportation systems. Multiple cameras at intersections enhance law enforcement capabilities, traffic management, and pedestrian safety. However, efficiently managing and analyzing multi-camera feeds poses challenges due to the vast amount of data. Analyzing such huge video data requires advanced analytical tools. While Large Language Models (LLMs) like ChatGPT, equipped with retrieval-augmented generation (RAG) systems, excel in text-based tasks, integrating them into traffic video analysis demands converting video data into text using a Vision-Language Model (VLM), which is time-consuming and delays the timely utilization of traffic videos for generating insights and investigating incidents. To address these challenges, we propose TrafficLens, a tailored algorithm for multi-camera traffic intersections. TrafficLens employs a sequential approach, utilizing overlapping coverage areas of cameras. It iteratively applies VLMs with varying token limits, using previous outputs as prompts for subsequent cameras, enabling rapid generation of detailed textual descriptions while reducing processing time. Additionally, TrafficLens intelligently bypasses redundant VLM invocations through an object-level similarity detector. Experimental results with real-world datasets demonstrate that TrafficLens reduces video-to-text conversion time by up to $4\times$ while maintaining information accuracy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20965) | **Categories:** cs.CV, cs.CL

---

### [3] [Scaling Foundation Models for Radar Scene Understanding](https://arxiv.org/abs/2511.21105)
*Pushkal Mishra, Kshitiz Bansal, Dinesh Bharadia*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Radar sensors provide reliable perception across adverse weather, lighting, and long-range conditions. Recent advances in foundation models have transformed visual and language understanding, yet their integration with radar sensing remains largely underexplored. Existing radar approaches are fragmented and task-specific; each downstream task employs distinct architectures and training objectives, preventing transfer across tasks. In this work, we introduce RadarFM: a radar foundation model that learns unified scene-level representations through structured spatial language supervision. We make two key contributions: (1) a structured caption framework that encodes vehicle distributions in native radar coordinates, and (2) a hash-aware contrastive learning objective that quantifies continuous scene similarity rather than binary matching, enabling fine-grained spatial reasoning. Leveraging the CARLA simulator, we generate large-scale, well-annotated radar datasets across diverse driving scenarios. We also propose localization-aware metrics that assess spatial accuracy beyond traditional detection measures.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21105) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge](https://arxiv.org/abs/2511.20726)
*Yuhang Wang, Heye Huang, Zhenhua Xu, Kailai Sun, Baoshen Guo, Jinhua Zhao*

Main category: cs.LG

TL;DR: 本文提出了一种结合条件变分自编码器（CVAE）和大型语言模型（LLM）的高保真场景生成框架，用于解决自动驾驶在罕见长尾事件和复杂多智能体交互中的安全验证问题。


<details>
  <summary>Details</summary>
Motivation: 自动驾驶在罕见长尾事件和复杂多智能体交互中面临关键挑战，而这些在真实世界数据中稀缺，但对于稳健的安全验证至关重要。

Method: 该框架集成了CVAE和LLM。CVAE用于学习潜在的交通结构，生成物理上一致的基础场景；LLM作为对抗推理引擎，将非结构化的场景描述解析为特定领域的损失函数，并动态地指导跨不同风险级别的场景生成。

Result: 在CARLA和SMARTS中的大量实验表明，该框架显著增加了高风险和长尾事件的覆盖范围，提高了模拟和真实世界交通分布之间的一致性，并使自动驾驶系统暴露于比现有方法更具挑战性的交互。

Conclusion: 该研究建立了一种新的安全验证途径，能够在罕见但重要的事件下对自动驾驶系统进行有原则的压力测试。

Abstract: 自动驾驶在罕见的长尾事件和复杂的多智能体交互中面临着严峻的挑战。这些情况在现实世界的数据中非常少见，但对于确保安全验证的可靠性至关重要。本文提出了一种高保真场景生成框架，该框架结合了条件变分自编码器（CVAE）与大型语言模型（LLM）。CVAE 用于编码来自大规模自然数据集的历史轨迹和地图信息，从而学习潜在的交通结构，并生成符合物理规律的基础场景。在此基础上，LLM 充当对抗性推理引擎，将非结构化的场景描述解析为特定领域的损失函数，并动态地指导跨不同风险级别的场景生成。这种知识驱动的优化平衡了真实性与可控性，确保生成的场景既合理又对风险敏感。在 CARLA 和 SMARTS 中进行的大量实验表明，我们的框架显著增加了高风险和长尾事件的覆盖范围，提高了模拟和真实世界交通分布之间的一致性，并且使自动驾驶系统面临的交互挑战远大于现有规则或数据驱动方法所产生的挑战。这些结果为安全验证建立了一条新途径，从而能够在罕见但重要的事件下对自动驾驶系统进行有原则的压力测试。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20726) | **Categories:** cs.LG, cs.AI

---

### [2] [Spatio-Temporal Trajectory Foundation Model - Recent Advances and Future Directions](https://arxiv.org/abs/2511.20729)
*Sean Bin Yang, Ying Sun, Yunyao Cheng, Yan Lin, Kristian Torp, Jilin Hu*

Main category: cs.LG

TL;DR: 本文综述了轨迹基础模型（TFMs）的最新进展，为时空通用智能的发展提供方向。


<details>
  <summary>Details</summary>
Motivation: 现有研究缺乏对轨迹基础模型（TFMs）的系统性研究，而TFMs是时空基础模型（STFMs）的一个重要子类。

Method: 本文对现有TFMs方法进行了分类，并对其优缺点进行了 критический 分析。

Result: 本文重点介绍了开放性挑战，并概述了有希望的研究方向。

Conclusion: 通过开发稳健、负责和可转移的TFMs，可以推进时空通用智能的发展。

Abstract: 基础模型（FMs）已成为一种强大的范例，能够在各个科学领域实现各种数据分析和知识发现任务。受到FMs，特别是大型语言模型的成功启发，研究人员最近开始探索时空基础模型（STFMs），以提高各种时空（ST）任务的适应性和泛化能力。尽管进展迅速，但对轨迹基础模型（TFMs）（STFMs 的一个关键子类）的系统研究在很大程度上仍然缺乏。本教程通过全面概述 TFM 的最新进展来解决这一差距，包括现有方法的分类以及对其优势和局限性的 критический 分析。此外，本教程重点介绍了开放性挑战，并概述了有希望的研究方向，以通过开发稳健、负责和可转移的 TFM 来推进时空通用智能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20729) | **Categories:** cs.LG, cs.AI

---

### [3] [Active Slice Discovery in Large Language Models](https://arxiv.org/abs/2511.20713)
*Minhui Zhang, Prahar Ijner, Yoav Wald, Elliot Creager*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) often exhibit systematic errors on specific subsets of data, known as error slices. For instance, a slice can correspond to a certain demographic, where a model does poorly in identifying toxic comments regarding that demographic. Identifying error slices is crucial to understanding and improving models, but it is also challenging. An appealing approach to reduce the amount of manual annotation required is to actively group errors that are likely to belong to the same slice, while using limited access to an annotator to verify whether the chosen samples share the same pattern of model mistake. In this paper, we formalize this approach as Active Slice Discovery and explore it empirically on a problem of discovering human-defined slices in toxicity classification. We examine the efficacy of active slice discovery under different choices of feature representations and active learning algorithms. On several slices, we find that uncertainty-based active learning algorithms are most effective, achieving competitive accuracy using 2-10% of the available slice membership information, while significantly outperforming baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20713) | **Categories:** cs.LG, cs.AI

---

### [4] [Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning](https://arxiv.org/abs/2511.20993)
*Shanwei Fan*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20993) | **Categories:** cs.LG, cs.AI

---

### [5] [Efficient Diffusion Planning with Temporal Diffusion](https://arxiv.org/abs/2511.21054)
*Jiaming Guo, Rui Zhang, Zerun Li, Yunkai Gao, Shaohui Peng, Siming Lan, Xing Hu, Zidong Du, Xishan Zhang, Ling Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion planning is a promising method for learning high-performance policies from offline data. To avoid the impact of discrepancies between planning and reality on performance, previous works generate new plans at each time step. However, this incurs significant computational overhead and leads to lower decision frequencies, and frequent plan switching may also affect performance. In contrast, humans might create detailed short-term plans and more general, sometimes vague, long-term plans, and adjust them over time. Inspired by this, we propose the Temporal Diffusion Planner (TDP) which improves decision efficiency by distributing the denoising steps across the time dimension. TDP begins by generating an initial plan that becomes progressively more vague over time. At each subsequent time step, rather than generating an entirely new plan, TDP updates the previous one with a small number of denoising steps. This reduces the average number of denoising steps, improving decision efficiency. Additionally, we introduce an automated replanning mechanism to prevent significant deviations between the plan and reality. Experiments on D4RL show that, compared to previous works that generate new plans every time step, TDP improves the decision-making frequency by 11-24.8 times while achieving higher or comparable performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21054) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation](https://arxiv.org/abs/2511.21135)
*Ziyi Chen, Yingnan Guo, Zedong Chu, Minghua Luo, Yanfen Shen, Mingchao Sun, Junjun Hu, Shichao Xie, Kuan Yang, Pei Shi, Zhining Gu, Lu Liu, Honglin Han, Xiaolong Wu, Mu Xu, Yu Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied navigation that adheres to social norms remains an open research challenge. Our \textbf{SocialNav} is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: https://amap-eai.github.io/SocialNav/

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21135) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [2] [TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos](https://arxiv.org/abs/2511.21690)
*Seungjae Lee, Yoonkyo Jung, Inkook Chun, Yao-Chih Lee, Zikui Cai, Hongjia Huang, Aayush Talreja, Tan Dat Dao, Yongyuan Liang, Jia-Bin Huang, Furong Huang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning new robot tasks on new platforms and in new scenes from only a handful of demonstrations remains challenging. While videos of other embodiments - humans and different robots - are abundant, differences in embodiment, camera, and environment hinder their direct use. We address the small-data problem by introducing a unifying, symbolic representation - a compact 3D "trace-space" of scene-level trajectories - that enables learning from cross-embodiment, cross-environment, and cross-task videos. We present TraceGen, a world model that predicts future motion in trace-space rather than pixel space, abstracting away appearance while retaining the geometric structure needed for manipulation. To train TraceGen at scale, we develop TraceForge, a data pipeline that transforms heterogeneous human and robot videos into consistent 3D traces, yielding a corpus of 123K videos and 1.8M observation-trace-language triplets. Pretraining on this corpus produces a transferable 3D motion prior that adapts efficiently: with just five target robot videos, TraceGen attains 80% success across four tasks while offering 50-600x faster inference than state-of-the-art video-based world models. In the more challenging case where only five uncalibrated human demonstration videos captured on a handheld phone are available, it still reaches 67.5% success on a real robot, highlighting TraceGen's ability to adapt across embodiments without relying on object detectors or heavy pixel-space generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21690) | **Categories:** cs.RO, cs.CV, cs.LG

---

### [3] [NOIR 2.0: Neural Signal Operated Intelligent Robots for Everyday Activities](https://arxiv.org/abs/2511.20848)
*Tasha Kim, Yingke Wang, Hanvit Cho, Alex Hodges*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Neural Signal Operated Intelligent Robots (NOIR) system is a versatile brain-robot interface that allows humans to control robots for daily tasks using their brain signals. This interface utilizes electroencephalography (EEG) to translate human intentions regarding specific objects and desired actions directly into commands that robots can execute. We present NOIR 2.0, an enhanced version of NOIR. NOIR 2.0 includes faster and more accurate brain decoding algorithms, which reduce task completion time by 46%. NOIR 2.0 uses few-shot robot learning algorithms to adapt to individual users and predict their intentions. The new learning algorithms leverage foundation models for more sample-efficient learning and adaptation (15 demos vs. a single demo), significantly reducing overall human time by 65%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.20848) | **Categories:** cs.RO, cs.AI, cs.HC, cs.LG, eess.SY

---

### [4] [AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](https://arxiv.org/abs/2511.21053)
*Chenglizhao Chen, Shaofeng Liang, Runwei Guan, Xiaolou Sun, Haocheng Zhao, Haiyun Jiang, Tao Huang, Henghui Ding, Qing-Long Han*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Referring Multi-Object Tracking (RMOT) aims to achieve precise object detection and tracking through natural language instructions, representing a fundamental capability for intelligent robotic systems. However, current RMOT research remains mostly confined to ground-level scenarios, which constrains their ability to capture broad-scale scene contexts and perform comprehensive tracking and path planning. In contrast, Unmanned Aerial Vehicles (UAVs) leverage their expansive aerial perspectives and superior maneuverability to enable wide-area surveillance. Moreover, UAVs have emerged as critical platforms for Embodied Intelligence, which has given rise to an unprecedented demand for intelligent aerial systems capable of natural language interaction. To this end, we introduce AerialMind, the first large-scale RMOT benchmark in UAV scenarios, which aims to bridge this research gap. To facilitate its construction, we develop an innovative semi-automated collaborative agent-based labeling assistant (COALA) framework that significantly reduces labor costs while maintaining annotation quality. Furthermore, we propose HawkEyeTrack (HETrack), a novel method that collaboratively enhances vision-language representation learning and improves the perception of UAV scenarios. Comprehensive experiments validated the challenging nature of our dataset and the effectiveness of our method.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21053) | **Categories:** cs.RO, cs.CV

---

### [5] [Improvement of Collision Avoidance in Cut-In Maneuvers Using Time-to-Collision Metrics](https://arxiv.org/abs/2511.21280)
*Jamal Raiyn*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes a new strategy for collision avoidance system leveraging Time-to-Collision (TTC) metrics for handling cut-in scenarios, which are particularly challenging for autonomous vehicles (AVs). By integrating a deep learning with TTC calculations, the system predicts potential collisions and determines appropriate evasive actions compared to traditional TTC -based approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21280) | **Categories:** cs.RO, cs.AI

---

### [6] [$\mathcal{E}_0$: Enhancing Generalization and Fine-Grained Control in VLA Models via Continuized Discrete Diffusion](https://arxiv.org/abs/2511.21542)
*Zhihao Zhan, Jiaying Zhou, Likui Zhang, Qinhan Lv, Hao Liu, Jusheng Zhang, Weizheng Li, Ziliang Chen, Tianshui Chen, Keze Wang, Liang Lin, Guangrun Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. Yet existing VLA models still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We introduce E0, a continuized discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. Compared with continuous diffusion policies, E0 offers two key advantages: (1) discrete action tokens align naturally with the symbolic structure of pretrained VLM/VLA backbones, enabling stronger semantic conditioning; and 2. discrete diffusion matches the true quantized nature of real-world robot control-whose hardware constraints (e.g., encoder resolution, control frequency, actuation latency) inherently discretize continuous signals-and therefore benefits from a Bayes-optimal denoiser that models the correct discrete action distribution, leading to stronger generalization. Compared with discrete autoregressive and mask-based discrete diffusion models, E0 supports a significantly larger and finer-grained action vocabulary and avoids the distributional mismatch introduced by masking-based corruptions-yielding more accurate fine-grained action control. We further introduce a spherical viewpoint perturbation augmentation method to improve robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, and ManiSkill show that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average. Real-world evaluation on a Franka arm confirms that E0 delivers precise, robust, and transferable manipulation, establishing discrete diffusion as a promising direction for generalizable VLA policy learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21542) | **Categories:** cs.RO

---

### [7] [Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](https://arxiv.org/abs/2511.21584)
*Haohong Lin, Yunzhi Zhang, Wenhao Ding, Jiajun Wu, Ding Zhao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end (E2E) autonomous driving models have demonstrated strong performance in open-loop evaluations but often suffer from cascading errors and poor generalization in closed-loop settings. To address this gap, we propose Model-based Policy Adaptation (MPA), a general framework that enhances the robustness and safety of pretrained E2E driving agents during deployment. MPA first generates diverse counterfactual trajectories using a geometry-consistent simulation engine, exposing the agent to scenarios beyond the original dataset. Based on this generated data, MPA trains a diffusion-based policy adapter to refine the base policy's predictions and a multi-step Q value model to evaluate long-term outcomes. At inference time, the adapter proposes multiple trajectory candidates, and the Q value model selects the one with the highest expected utility. Experiments on the nuScenes benchmark using a photorealistic closed-loop simulator demonstrate that MPA significantly improves performance across in-domain, out-of-domain, and safety-critical scenarios. We further investigate how the scale of counterfactual data and inference-time guidance strategies affect overall effectiveness.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21584) | **Categories:** cs.RO, cs.AI

---
