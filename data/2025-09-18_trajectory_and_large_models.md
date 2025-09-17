# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-18

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (6)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences](https://arxiv.org/abs/2509.12273)
*Liangqi Yuan, Dong-Jun Han, Christopher G. Brinton, Sabine Brunswicker*

Main category: cs.AI

TL;DR: 本文提出了一种新颖的LLM辅助路线规划系统（LLMAP），该系统结合了LLM-as-Parser和多步图构建迭代搜索（MSGS）算法，以实现考虑多重约束条件下的最优路线规划。


<details>
  <summary>Details</summary>
Motivation: 现有研究中，LLM-as-Agent方法难以处理大规模地图数据，而基于图搜索的方法在理解自然语言偏好方面存在局限性。此外，用户在全球范围内呈现高度异构和不可预测的时空分布，这带来了更大的挑战。

Method: 本文提出了一种LLM辅助路线规划系统（LLMAP），该系统利用LLM-as-Parser理解自然语言、识别任务、提取用户偏好并识别任务依赖关系，并结合多步图构建迭代搜索（MSGS）算法作为底层求解器，以寻找最优路线。

Result: 在包含全球14个国家和27个城市的1000个具有不同复杂度的路线规划提示的实验中，结果表明该方法在多个约束条件下实现了卓越的性能。

Conclusion: 本文提出的LLMAP系统能够有效地解决现有方法在自然语言路线规划中存在的局限性，并在考虑多重约束条件下实现最优路线规划。

Abstract: 大型语言模型（LLM）的兴起使得自然语言驱动的路线规划成为一个新兴的研究领域，该领域涵盖了丰富的用户目标。目前的研究主要有两种不同的方法：使用LLM-as-Agent的直接路线规划和基于图的搜索策略。然而，前一种方法中的LLM难以处理大量的地图数据，而后一种方法在理解自然语言偏好方面的能力有限。此外，一个更严峻的挑战来自于全球用户高度异构且不可预测的时空分布。在本文中，我们介绍了一种新颖的LLM辅助路线规划（LLMAP）系统，该系统采用LLM-as-Parser来理解自然语言，识别任务，提取用户偏好，并识别任务依赖关系，同时结合多步图构建迭代搜索（MSGS）算法作为底层求解器，以寻找最优路线。我们的多目标优化方法自适应地调整目标权重，以最大限度地提高兴趣点（POI）的质量和任务完成率，同时最大限度地减少路线距离，并受限于三个关键约束：用户时间限制、POI开放时间和任务依赖关系。我们使用1000个在世界范围内14个国家和27个城市采样的具有不同复杂度的路线规划提示进行了广泛的实验。结果表明，我们的方法在多个约束条件下实现了卓越的性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12273) | **Categories:** cs.AI, cs.CL, cs.LG

---

### [2] [Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition](https://arxiv.org/abs/2509.12423)
*Danielle Cohen, Yoni Halpern, Noam Kahlon, Joel Oren, Omri Berkovitch, Sapir Caduri, Ido Dagan, Anatoly Efros*

Main category: cs.AI

TL;DR: 该论文提出了一种分解方法，通过结构化交互摘要和意图提取，提高了资源受限模型对用户界面交互轨迹的意图理解能力。


<details>
  <summary>Details</summary>
Motivation: 现有在设备上运行的小型模型难以准确推断用户界面交互轨迹中的意图，而大型多模态语言模型虽然能力更强，但成本高且无法保护用户隐私。

Method: 该方法首先执行结构化交互摘要，捕捉每个用户动作的关键信息，然后使用微调模型对汇总的摘要进行意图提取。

Result: 该方法提高了资源受限模型中的意图理解能力，甚至超过了大型多模态语言模型的基线性能。

Conclusion: 通过分解方法，资源受限模型可以在用户意图理解方面达到甚至超过大型多模态语言模型的性能。

Abstract: 从用户界面交互轨迹中理解用户意图仍然是智能体开发中一个具有挑战性但至关重要的前沿领域。虽然大型数据中心的多模态大型语言模型（MLLM）具有更强的处理此类序列复杂性的能力，但可以在设备上运行以提供保护隐私、低成本和低延迟用户体验的小型模型，难以准确地进行意图推断。我们通过引入一种新颖的分解方法来解决这些限制：首先，我们执行结构化交互摘要，从每个用户动作中捕获关键信息。其次，我们使用在聚合摘要上运行的微调模型执行意图提取。这种方法提高了资源受限模型中的意图理解能力，甚至超过了大型 MLLM 的基本性能。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12423) | **Categories:** cs.AI, cs.CL

---

### [3] [InPhyRe Discovers: Large Multimodal Models Struggle in Inductive Physical Reasoning](https://arxiv.org/abs/2509.12263)
*Gautam Sreekumar, Vishnu Naresh Boddeti*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large multimodal models (LMMs) encode universal physical laws observed during training, such as momentum conservation, as parametric knowledge. It allows LMMs to answer physical reasoning queries, such as the outcome of a potential collision event from visual input. However, since parametric knowledge includes only the physical laws seen during training, it is insufficient for reasoning when the inference scenario violates these physical laws. In contrast, humans possess the skill to adapt their physical reasoning to unseen physical environments from a few visual examples. This ability, which we refer to as inductive physical reasoning, is indispensable for LMMs if they are to replace human agents in safety-critical applications. Despite its importance, existing visual benchmarks evaluate only the parametric knowledge in LMMs, and not inductive physical reasoning. To this end, we propose InPhyRe, the first visual question answering benchmark to measure inductive physical reasoning in LMMs. InPhyRe evaluates LMMs on their ability to predict the outcome of collision events in algorithmically generated synthetic collision videos. By inspecting 13 LMMs, InPhyRe informs us that (1) LMMs struggle to apply their limited parametric knowledge about universal physical laws to reasoning, (2) inductive physical reasoning in LMMs is weak when demonstration samples violate universal physical laws, and (3) inductive physical reasoning in LMMs suffers from language bias and largely ignores the visual inputs, questioning the trustworthiness of LMMs regarding visual inputs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12263) | **Categories:** cs.AI, cs.LG

---

### [4] [Enhancing Physical Consistency in Lightweight World Models](https://arxiv.org/abs/2509.12437)
*Dingrui Wang, Zhexiao Sun, Zhouheng Li, Cheng Wang, Youlun Peng, Hongyuan Ye, Baha Zarrouki, Wei Li, Mattia Piccinini, Lei Xie, Johannes Betz*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A major challenge in deploying world models is the trade-off between size and performance. Large world models can capture rich physical dynamics but require massive computing resources, making them impractical for edge devices. Small world models are easier to deploy but often struggle to learn accurate physics, leading to poor predictions. We propose the Physics-Informed BEV World Model (PIWM), a compact model designed to efficiently capture physical interactions in bird's-eye-view (BEV) representations. PIWM uses Soft Mask during training to improve dynamic object modeling and future prediction. We also introduce a simple yet effective technique, Warm Start, for inference to enhance prediction quality with a zero-shot model. Experiments show that at the same parameter scale (400M), PIWM surpasses the baseline by 60.6% in weighted overall score. Moreover, even when compared with the largest baseline model (400M), the smallest PIWM (130M Soft Mask) achieves a 7.4% higher weighted overall score with a 28% faster inference speed.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12437) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Scalable RF Simulation in Generative 4D Worlds](https://arxiv.org/abs/2508.12176)
*Zhiwei Zheng, Dongyin Hu, Mingmin Zhao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Radio Frequency (RF) sensing has emerged as a powerful, privacy-preserving alternative to vision-based methods for indoor perception tasks. However, collecting high-quality RF data in dynamic and diverse indoor environments remains a major challenge. To address this, we introduce WaveVerse, a prompt-based, scalable framework that simulates realistic RF signals from generated indoor scenes with human motions. WaveVerse introduces a language-guided 4D world generator, which includes a state-aware causal transformer for human motion generation conditioned on spatial constraints and texts, and a phase-coherent ray tracing simulator that enables the simulation of accurate and coherent RF signals. Experiments demonstrate the effectiveness of our approach in conditioned human motion generation and highlight how phase coherence is applied to beamforming and respiration monitoring. We further present two case studies in ML-based high-resolution imaging and human activity recognition, demonstrating that WaveVerse not only enables data generation for RF imaging for the first time, but also consistently achieves performance gain in both data-limited and data-adequate scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2508.12176) | **Categories:** cs.CV, cs.AI, eess.SP

---

### [2] [DYNAMO: Dependency-Aware Deep Learning Framework for Articulated Assembly Motion Prediction](https://arxiv.org/abs/2509.12430)
*Mayank Patel, Rahul Jain, Asim Unmesh, Karthik Ramani*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Understanding the motion of articulated mechanical assemblies from static geometry remains a core challenge in 3D perception and design automation. Prior work on everyday articulated objects such as doors and laptops typically assumes simplified kinematic structures or relies on joint annotations. However, in mechanical assemblies like gears, motion arises from geometric coupling, through meshing teeth or aligned axes, making it difficult for existing methods to reason about relational motion from geometry alone. To address this gap, we introduce MechBench, a benchmark dataset of 693 diverse synthetic gear assemblies with part-wise ground-truth motion trajectories. MechBench provides a structured setting to study coupled motion, where part dynamics are induced by contact and transmission rather than predefined joints. Building on this, we propose DYNAMO, a dependency-aware neural model that predicts per-part SE(3) motion trajectories directly from segmented CAD point clouds. Experiments show that DYNAMO outperforms strong baselines, achieving accurate and temporally consistent predictions across varied gear configurations. Together, MechBench and DYNAMO establish a novel systematic framework for data-driven learning of coupled mechanical motion in CAD assemblies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12430) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [C3DE: Causal-Aware Collaborative Neural Controlled Differential Equation for Long-Term Urban Crowd Flow Prediction](https://arxiv.org/abs/2509.12289)
*Yuting Liu, Qiang Zhou, Hanzhe Li, Chenqi Gong, Jingjing Gu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Long-term urban crowd flow prediction suffers significantly from cumulative sampling errors, due to increased sequence lengths and sampling intervals, which inspired us to leverage Neural Controlled Differential Equations (NCDEs) to mitigate this issue. However, regarding the crucial influence of Points of Interest (POIs) evolution on long-term crowd flow, the multi-timescale asynchronous dynamics between crowd flow and POI distribution, coupled with latent spurious causality, poses challenges to applying NCDEs for long-term urban crowd flow prediction. To this end, we propose Causal-aware Collaborative neural CDE (C3DE) to model the long-term dynamic of crowd flow. Specifically, we introduce a dual-path NCDE as the backbone to effectively capture the asynchronous evolution of collaborative signals across multiple time scales. Then, we design a dynamic correction mechanism with the counterfactual-based causal effect estimator to quantify the causal impact of POIs on crowd flow and minimize the accumulation of spurious correlations. Finally, we leverage a predictor for long-term prediction with the fused collaborative signals of POI and crowd flow. Extensive experiments on three real-world datasets demonstrate the superior performance of C3DE, particularly in cities with notable flow fluctuations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12289) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [An Uncertainty-Weighted Decision Transformer for Navigation in Dense, Complex Driving Scenarios](https://arxiv.org/abs/2509.13132)
*Zhihao Zhang, Chengyang Peng, Minghao Zhu, Ekim Yurtsever, Keith A. Redmill*

Main category: cs.RO

TL;DR: 该论文提出了一种不确定性加权决策Transformer（UWDT）框架，用于在复杂环岛场景中实现更安全、高效的自动驾驶决策。


<details>
  <summary>Details</summary>
Motivation: 在密集、动态环境中，自动驾驶需要能够利用空间结构和长期时间依赖性，同时对不确定性保持鲁棒性的决策系统。

Method: 该方法集成了多通道鸟瞰图占用栅格和基于Transformer的序列建模，并提出UWDT，利用冻结的教师Transformer估计每个token的预测熵，作为学生模型损失函数的权重。

Result: 在不同交通密度的环岛模拟器实验中，UWDT在奖励、碰撞率和行为稳定性方面始终优于其他基线。

Conclusion: 不确定性感知的时空Transformer可以为复杂交通环境中的自动驾驶提供更安全、更高效的决策。

Abstract: 在密集、动态环境中，自动驾驶需要能够利用空间结构和长期时间依赖性，同时对不确定性保持鲁棒性的决策系统。本文提出了一种新颖的框架，该框架集成了多通道鸟瞰图占用栅格和基于Transformer的序列建模，用于复杂环岛场景中的战术驾驶。为了解决频繁的低风险状态和罕见的安全性关键决策之间的不平衡，我们提出了不确定性加权决策Transformer（UWDT）。UWDT采用冻结的教师Transformer来估计每个token的预测熵，然后将其用作学生模型损失函数中的权重。这种机制增强了从不确定、高影响状态的学习，同时保持了常见低风险转换的稳定性。在环岛模拟器中，跨不同交通密度的实验表明，UWDT在奖励、碰撞率和行为稳定性方面始终优于其他基线。结果表明，不确定性感知的时空Transformer可以为复杂交通环境中的自动驾驶提供更安全、更高效的决策。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13132) | **Categories:** cs.RO, cs.AI

---

### [2] [An integrated process for design and control of lunar robotics using AI and simulation](https://arxiv.org/abs/2509.12367)
*Daniel Lindmark, Jonas Andersson, Kenneth Bodin, Tora Bodin, Hugo Börjesson, Fredrik Nordfeldth, Martin Servin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We envision an integrated process for developing lunar construction equipment, where physical design and control are explored in parallel. In this paper, we describe a technical framework that supports this process. It relies on OpenPLX, a readable/writable declarative language that links CAD-models and autonomous systems to high-fidelity, real-time 3D simulations of contacting multibody dynamics, machine regolith interaction forces, and non-ideal sensors. To demonstrate its capabilities, we present two case studies, including an autonomous lunar rover that combines a vision-language model for navigation with a reinforcement learning-based control policy for locomotion.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12367) | **Categories:** cs.RO, cs.AI

---

### [3] [Multi-Robot Task Planning for Multi-Object Retrieval Tasks with Distributed On-Site Knowledge via Large Language Models](https://arxiv.org/abs/2509.12838)
*Kento Murata, Shoichi Hasegawa, Tomochika Ishikawa, Yoshinobu Hagiwara, Akira Taniguchi, Lotfi El Hafi, Tadahiro Taniguchi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: It is crucial to efficiently execute instructions such as "Find an apple and a banana" or "Get ready for a field trip," which require searching for multiple objects or understanding context-dependent commands. This study addresses the challenging problem of determining which robot should be assigned to which part of a task when each robot possesses different situational on-site knowledge-specifically, spatial concepts learned from the area designated to it by the user. We propose a task planning framework that leverages large language models (LLMs) and spatial concepts to decompose natural language instructions into subtasks and allocate them to multiple robots. We designed a novel few-shot prompting strategy that enables LLMs to infer required objects from ambiguous commands and decompose them into appropriate subtasks. In our experiments, the proposed method achieved 47/50 successful assignments, outperforming random (28/50) and commonsense-based assignment (26/50). Furthermore, we conducted qualitative evaluations using two actual mobile manipulators. The results demonstrated that our framework could handle instructions, including those involving ad hoc categories such as "Get ready for a field trip," by successfully performing task decomposition, assignment, sequential planning, and execution.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12838) | **Categories:** cs.RO, cs.AI, cs.MA

---

### [4] [Out of Distribution Detection in Self-adaptive Robots with AI-powered Digital Twins](https://arxiv.org/abs/2509.12982)
*Erblin Isaku, Hassan Sartaj, Shaukat Ali, Beatriz Sanguino, Tongtong Wang, Guoyuan Li, Houxiang Zhang, Thomas Peyrucain*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Self-adaptive robots (SARs) in complex, uncertain environments must proactively detect and address abnormal behaviors, including out-of-distribution (OOD) cases. To this end, digital twins offer a valuable solution for OOD detection. Thus, we present a digital twin-based approach for OOD detection (ODiSAR) in SARs. ODiSAR uses a Transformer-based digital twin to forecast SAR states and employs reconstruction error and Monte Carlo dropout for uncertainty quantification. By combining reconstruction error with predictive variance, the digital twin effectively detects OOD behaviors, even in previously unseen conditions. The digital twin also includes an explainability layer that links potential OOD to specific SAR states, offering insights for self-adaptation. We evaluated ODiSAR by creating digital twins of two industrial robots: one navigating an office environment, and another performing maritime ship navigation. In both cases, ODiSAR forecasts SAR behaviors (i.e., robot trajectories and vessel motion) and proactively detects OOD events. Our results showed that ODiSAR achieved high detection performance -- up to 98\% AUROC, 96\% TNR@TPR95, and 95\% F1-score -- while providing interpretable insights to support self-adaptation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.12982) | **Categories:** cs.RO, cs.AI, cs.SE

---

### [5] [DVDP: An End-to-End Policy for Mobile Robot Visual Docking with RGB-D Perception](https://arxiv.org/abs/2509.13024)
*Haohan Min, Zhoujian Li, Yu Yang, Jinyu Chen, Shenghai Yuan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Automatic docking has long been a significant challenge in the field of mobile robotics. Compared to other automatic docking methods, visual docking methods offer higher precision and lower deployment costs, making them an efficient and promising choice for this task. However, visual docking methods impose strict requirements on the robot's initial position at the start of the docking process. To overcome the limitations of current vision-based methods, we propose an innovative end-to-end visual docking method named DVDP(direct visual docking policy). This approach requires only a binocular RGB-D camera installed on the mobile robot to directly output the robot's docking path, achieving end-to-end automatic docking. Furthermore, we have collected a large-scale dataset of mobile robot visual automatic docking dataset through a combination of virtual and real environments using the Unity 3D platform and actual mobile robot setups. We developed a series of evaluation metrics to quantify the performance of the end-to-end visual docking method. Extensive experiments, including benchmarks against leading perception backbones adapted into our framework, demonstrate that our method achieves superior performance. Finally, real-world deployment on the SCOUT Mini confirmed DVDP's efficacy, with our model generating smooth, feasible docking trajectories that meet physical constraints and reach the target pose.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13024) | **Categories:** cs.RO

---

### [6] [TeraSim-World: Worldwide Safety-Critical Data Synthesis for End-to-End Autonomous Driving](https://arxiv.org/abs/2509.13164)
*Jiawei Wang, Haowei Sun, Xintao Yan, Shuo Feng, Jun Gao, Henry X. Liu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe and scalable deployment of end-to-end (E2E) autonomous driving requires extensive and diverse data, particularly safety-critical events. Existing data are mostly generated from simulators with a significant sim-to-real gap or collected from on-road testing that is costly and unsafe. This paper presents TeraSim-World, an automated pipeline that synthesizes realistic and geographically diverse safety-critical data for E2E autonomous driving at anywhere in the world. Starting from an arbitrary location, TeraSim-World retrieves real-world maps and traffic demand from geospatial data sources. Then, it simulates agent behaviors from naturalistic driving datasets, and orchestrates diverse adversities to create corner cases. Informed by street views of the same location, it achieves photorealistic, geographically grounded sensor rendering via the frontier video generation model Cosmos-Drive. By bridging agent and sensor simulations, TeraSim-World provides a scalable and critical~data synthesis framework for training and evaluation of E2E autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13164) | **Categories:** cs.RO, cs.SY, eess.SY

---
