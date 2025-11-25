# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-26

## 目录

- [人工智能 (Artificial Intelligence) (5)](#cs-ai)
- [计算机视觉 (Computer Vision) (8)](#cs-cv)
- [cs.NI (2)](#cs-ni)
- [机器人学 (Robotics) (12)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents](https://arxiv.org/abs/2511.17855)
*Jordan Abi Nader, David Lee, Nathaniel Dennler, Andreea Bobu*

Main category: cs.AI

TL;DR: QuickLAP提出了一种贝叶斯框架，融合物理和语言反馈，实时推断奖励函数。


<details>
  <summary>Details</summary>
Motivation: 机器人需要从人类的行为和语言中学习，但单一模态信息不完整；物理纠正有实际基础但意图模糊，语言表达高层目标但缺乏物理基础。

Method: QuickLAP利用大型语言模型（LLM）从自由形式的语句中提取奖励特征注意力掩码和偏好转移，并将其与物理反馈集成，实现快速、实时和鲁棒的奖励学习。

Result: 在半自动驾驶模拟器中，QuickLAP将奖励学习误差降低了70%以上；用户研究表明，参与者认为QuickLAP更易于理解和协作，并且更喜欢其学习到的行为。

Conclusion: QuickLAP通过将语言视为用户潜在偏好的概率观察，能够有效融合物理和语言反馈，实现快速、实时和鲁棒的奖励学习，优于仅依赖物理反馈或其他多模态基线方法。

Abstract: 机器人需要学习人类的行为和语言，但单一模态信息往往不完整：物理纠正虽然有实际基础，但意图模糊；语言可以表达高层目标，但缺乏物理基础。我们引入了QuickLAP：快速语言-动作偏好学习，这是一种贝叶斯框架，可以融合物理和语言反馈，实时推断奖励函数。我们的关键在于将语言视为用户潜在偏好的概率观察，从而明确哪些奖励特征重要以及如何解释物理纠正。QuickLAP使用大型语言模型（LLM）从自由形式的语句中提取奖励特征注意力掩码和偏好转移，并将其与物理反馈集成在一个闭式更新规则中。这使得能够进行快速、实时和鲁棒的奖励学习，从而处理模糊的反馈。在半自动驾驶模拟器中，与仅使用物理反馈和启发式多模态基线方法相比，QuickLAP将奖励学习误差降低了70%以上。一项有15名参与者的用户研究进一步验证了我们的方法：参与者发现QuickLAP明显更易于理解和协作，并且更喜欢它学习到的行为。代码可在https://github.com/MIT-CLEAR-Lab/QuickLAP获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17855) | **Categories:** cs.AI, cs.RO

---

### [2] [GContextFormer: A global context-aware hybrid multi-head attention approach with scaled additive aggregation for multimodal trajectory prediction](https://arxiv.org/abs/2511.18874)
*Yuzhi Chen, Yuanchang Xie, Lei Zhao, Pan Liu, Yajie Zou, Chen Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal trajectory prediction generates multiple plausible future trajectories to address vehicle motion uncertainty from intention ambiguity and execution variability. However, HD map-dependent models suffer from costly data acquisition, delayed updates, and vulnerability to corrupted inputs, causing prediction failures. Map-free approaches lack global context, with pairwise attention over-amplifying straight patterns while suppressing transitional patterns, resulting in motion-intention misalignment. This paper proposes GContextFormer, a plug-and-play encoder-decoder architecture with global context-aware hybrid attention and scaled additive aggregation achieving intention-aligned multimodal prediction without map reliance. The Motion-Aware Encoder builds scene-level intention prior via bounded scaled additive aggregation over mode-embedded trajectory tokens and refines per-mode representations under shared global context, mitigating inter-mode suppression and promoting intention alignment. The Hierarchical Interaction Decoder decomposes social reasoning into dual-pathway cross-attention: a standard pathway ensures uniform geometric coverage over agent-mode pairs while a neighbor-context-enhanced pathway emphasizes salient interactions, with gating module mediating their contributions to maintain coverage-focus balance. Experiments on eight highway-ramp scenarios from TOD-VT dataset show GContextFormer outperforms state-of-the-art baselines. Compared to existing transformer models, GContextFormer achieves greater robustness and concentrated improvements in high-curvature and transition zones via spatial distributions. Interpretability is achieved through motion mode distinctions and neighbor context modulation exposing reasoning attribution. The modular architecture supports extensibility toward cross-domain multimodal reasoning tasks. Source: https://fenghy-chen.github.io/sources/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18874) | **Categories:** cs.AI, cs.CV, cs.LG, cs.MA, cs.RO, cs.SI

---

### [3] [UNeMo: Collaborative Visual-Language Reasoning and Navigation via a Multimodal World Model](https://arxiv.org/abs/2511.18845)
*Changxin Huang, Lv Tang, Zhaohuan Zhan, Lisha Yu, Runhao Zeng, Zun Liu, Zhengjie Wang, Jianqiang Li*

Main category: cs.AI

TL;DR: UNeMo通过多模态世界模型和分层预测反馈机制，实现了视觉状态推理和导航决策的协同优化，从而提升了视觉语言导航的性能。


<details>
  <summary>Details</summary>
Motivation: 现有方法缺乏视觉推理能力，且推理模块与导航策略优化分离，导致不兼容和优化目标冲突。

Method: 提出UNeMo框架，利用多模态世界模型（MWM）联合预测后续视觉状态，并通过分层预测反馈（HPN）机制协同优化视觉状态推理和导航决策。

Result: 在R2R和REVERIE数据集上的实验表明，UNeMo在未见场景的导航准确率上分别超过现有最佳方法2.1%和0.7%。

Conclusion: UNeMo框架能够有效提升视觉语言导航的性能，验证了其有效性。

Abstract: 视觉语言导航（VLN）要求智能体通过视觉图像和自然语言指令自主导航复杂环境，这仍然极具挑战性。最近关于使用预训练大型语言模型（LLM）来增强语言引导导航推理的研究显示出很有希望的前景。然而，这些方法的推理仅限于语言模态，缺乏视觉推理能力。此外，现有的推理模块与导航策略分开优化，导致不兼容和优化目标可能冲突。为了应对这些挑战，我们介绍UNeMo，一个为视觉状态推理和导航决策的协同优化而设计的新颖框架。它引入了一个多模态世界模型（MWM），该模型将视觉特征、语言指令和导航动作作为输入，以联合预测后续的视觉状态，从而实现跨模态推理。通过分层预测-反馈（HPN）机制，MWM与导航策略协同工作：第一层使用当前的视觉-语言特征生成动作；然后MWM推断动作后的视觉状态，以指导第二层的细粒度决策。这形成了一个动态的双向促进机制，其中MWM推理优化导航策略，而策略决策反馈以提高MWM的推理准确性。在R2R和REVERIE数据集上的实验表明，对于未见场景，UNeMo的导航准确率优于现有最佳方法2.1%和0.7%，验证了其有效性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18845) | **Categories:** cs.AI

---

### [4] [Wireless Power Transfer and Intent-Driven Network Optimization in AAVs-assisted IoT for 6G Sustainable Connectivity](https://arxiv.org/abs/2511.18368)
*Yue Hu, Xiaoming He, Rui Yuan, Shahid Mumtaz*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous Aerial Vehicle (AAV)-assisted Internet of Things (IoT) represents a collaborative architecture in which AAV allocate resources over 6G links to jointly enhance user-intent interpretation and overall network performance. Owing to this mutual dependence, improvements in intent inference and policy decisions on one component reinforce the efficiency of others, making highly reliable intent prediction and low-latency action execution essential. Although numerous approaches can model intent relationships, they encounter severe obstacles when scaling to high-dimensional action sequences and managing intensive on-board computation. We propose an Intent-Driven Framework for Autonomous Network Optimization comprising prediction and decision modules. First, implicit intent modeling is adopted to mitigate inaccuracies arising from ambiguous user expressions. For prediction, we introduce Hyperdimensional Transformer (HDT), which embeds data into a Hyperdimensional space via Hyperdimensional vector encoding and replaces standard matrix and attention operations with symbolic Hyperdimensional computations. For decision-making, where AAV must respond to user intent while planning trajectories, we design Double Actions based Multi-Agent Proximal Policy Optimization (DA-MAPPO). Building upon MAPPO, it samples actions through two independently parameterized networks and cascades the user-intent network into the trajectory network to maintain action dependencies. We evaluate our framework on a real IoT action dataset with authentic wireless data. Experimental results demonstrate that HDT and DA-MAPPO achieve superior performance across diverse scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18368) | **Categories:** cs.AI

---

### [5] [ORIGAMISPACE: Benchmarking Multimodal LLMs in Multi-Step Spatial Reasoning with Mathematical Constraints](https://arxiv.org/abs/2511.18450)
*Rui Xu, Dakuan Lu, Zicheng Zhao, Xiaoyu Tan, Xintao Wang, Siyu Yuan, Jiangjie Chen, Yinghui Xu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Spatial reasoning is a key capability in the field of artificial intelligence, especially crucial in areas such as robotics, computer vision, and natural language understanding. However, evaluating the ability of multimodal large language models(MLLMs) in complex spatial reasoning still faces challenges, particularly in scenarios requiring multi-step reasoning and precise mathematical constraints. This paper introduces ORIGAMISPACE, a new dataset and benchmark designed to evaluate the multi-step spatial reasoning ability and the capacity to handle mathematical constraints of MLLMs through origami tasks. The dataset contains 350 data instances,each comprising a strictly formatted crease pattern (CP diagram), the Compiled Flat Pattern, the complete Folding Process, and the final Folded Shape Image. We propose four evaluation tasks: Pattern Prediction, Multi-step Spatial Reasoning, Spatial Relationship Prediction, and End-to-End CP Code Generation. For the CP code generation task, we design an interactive environment and explore the possibility of using reinforcement learning methods to train MLLMs. Through experiments on existing MLLMs, we initially reveal the strengths and weaknesses of these models in handling complex spatial reasoning tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18450) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Vision-Motion-Reference Alignment for Referring Multi-Object Tracking via Multi-Modal Large Language Models](https://arxiv.org/abs/2511.17681)
*Weiyi Lv, Ning Zhang, Hanyang Sun, Haoran Jiang, Kai Zhao, Jing Xiao, Dan Zeng*

Main category: cs.CV

TL;DR: VMRMOT通过整合运动模态和多模态大型语言模型，提升了视觉模态与语言参考之间的对齐，从而改进了Referring Multi-Object Tracking (RMOT) 的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的RMOT基准测试仅描述了物体的外观、相对位置和初始运动状态，未能捕捉物体运动的动态变化，导致静态参考和动态视觉模态之间存在时间差异，限制了多模态跟踪性能。

Method: 提出了一个名为VMRMOT的视觉-运动-参考对齐的RMOT框架，该框架集成了从物体动态中提取的运动模态，通过多模态大型语言模型 (MLLM) 增强视觉模态和语言参考之间的对齐。具体来说，引入了从物体动态行为中导出的运动感知描述，并利用 MLLM 强大的时间推理能力，提取运动特征作为运动模态。此外，设计了一个视觉-运动-参考对齐 (VMRA) 模块，以分层方式将视觉查询与运动和参考线索对齐，从而增强它们的跨模态一致性。此外，还开发了一个运动引导预测头 (MGPH)，以探索运动模态，从而提高预测头的性能。

Result: 在多个RMOT基准测试上进行的大量实验表明，VMRMOT 的性能优于现有的最先进方法。

Conclusion: VMRMOT是第一个在RMOT任务中采用MLLM进行视觉-参考对齐的方法，通过整合运动模态和多模态大型语言模型，有效地提升了跟踪性能。

Abstract: 本文提出了一种新的视觉-运动-参考对齐的RMOT框架，名为VMRMOT，旨在解决现有RMOT方法中静态参考和动态视觉模态之间的时间差异问题。该框架通过整合从物体动态中提取的运动模态，并利用多模态大型语言模型 (MLLM) 增强视觉模态和语言参考之间的对齐。此外，还设计了一个视觉-运动-参考对齐 (VMRA) 模块和一个运动引导预测头 (MGPH)，以进一步提高跟踪性能。在多个RMOT基准测试上进行的大量实验表明，VMRMOT 的性能优于现有的最先进方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17681) | **Categories:** cs.CV

---

### [2] [V2X-RECT: An Efficient V2X Trajectory Prediction Framework via Redundant Interaction Filtering and Tracking Error Correction](https://arxiv.org/abs/2511.17941)
*Xiangyan Kong, Xuecheng Wu, Xiongwei Zhao, Xiaodong Li, Yunyun Shi, Gang Wang, Dingkang Yang, Yang Liu, Hong Chen, Yulong Gao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: V2X prediction can alleviate perception incompleteness caused by limited line of sight through fusing trajectory data from infrastructure and vehicles, which is crucial to traffic safety and efficiency. However, in dense traffic scenarios, frequent identity switching of targets hinders cross-view association and fusion. Meanwhile, multi-source information tends to generate redundant interactions during the encoding stage, and traditional vehicle-centric encoding leads to large amounts of repetitive historical trajectory feature encoding, degrading real-time inference performance. To address these challenges, we propose V2X-RECT, a trajectory prediction framework designed for high-density environments. It enhances data association consistency, reduces redundant interactions, and reuses historical information to enable more efficient and accurate prediction. Specifically, we design a multi-source identity matching and correction module that leverages multi-view spatiotemporal relationships to achieve stable and consistent target association, mitigating the adverse effects of mismatches on trajectory encoding and cross-view feature fusion. Then we introduce traffic signal-guided interaction module, encoding trend of traffic light changes as features and exploiting their role in constraining spatiotemporal passage rights to accurately filter key interacting vehicles, while capturing the dynamic impact of signal changes on interaction patterns. Furthermore, a local spatiotemporal coordinate encoding enables reusable features of historical trajectories and map, supporting parallel decoding and significantly improving inference efficiency. Extensive experimental results across V2X-Seq and V2X-Traj datasets demonstrate that our V2X-RECT achieves significant improvements compared to SOTA methods, while also enhancing robustness and inference efficiency across diverse traffic densities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17941) | **Categories:** cs.CV

---

### [3] [Target-Bench: Can World Models Achieve Mapless Path Planning with Semantic Targets?](https://arxiv.org/abs/2511.17792)
*Dingrui Wang, Hongyuan Ye, Zhihao Liang, Zhexiao Sun, Zhaowei Lu, Yuchen Zhang, Yuyu Zhao, Yuan Gao, Marvin Seegert, Finn Schäfer, Haotong Qin, Wei Li, Luigi Palmieri, Felix Jahncke, Mattia Piccinini, Johannes Betz*

Main category: cs.CV

TL;DR: 该论文提出了Target-Bench，一个专门用于评估世界模型在真实环境中无地图路径规划能力的基准。


<details>
  <summary>Details</summary>
Motivation: 评估现有世界模型在真实环境中无地图路径规划能力。

Method: 提出Target-Bench基准，包含450个机器人收集的视频序列，涵盖45个语义类别，并使用SLAM提供ground truth轨迹。评估流程从生成的视频中恢复相机运动，并使用五个互补指标来量化目标到达能力、轨迹准确性和方向一致性。

Result: 现有最佳世界模型（Wan2.2-Flash）的总体得分仅为0.299，表明当前世界模型在机器人规划任务中存在显著局限性。在我们的数据集上微调一个开源的5B参数模型，仅使用325个场景，即可实现0.345的总体得分，比其基础版本（0.066）提高了400%以上，比最好的现有模型高出15%。

Conclusion: 当前世界模型在机器人规划任务中存在显著局限性，通过在特定数据集上进行微调可以显著提高其性能。

Abstract: 尽管最近的世界模型生成了高度逼真的视频，但它们在执行机器人路径规划方面的能力仍不清楚且未被量化。我们推出了 Target-Bench，这是第一个专门设计用于评估世界模型在真实环境中针对语义目标的无地图路径规划的基准。Target-Bench 提供了 450 个机器人收集的视频序列，涵盖 45 个语义类别，并具有基于 SLAM 的地面实况轨迹。我们的评估流程从生成的视频中恢复相机运动，并使用五个互补指标来量化目标到达能力、轨迹准确性和方向一致性。我们评估了最先进的模型，包括 Sora 2、Veo 3.1 和 Wan 系列。最好的现成模型 (Wan2.2-Flash) 仅获得 0.299 的总分，这表明当前世界模型在机器人规划任务中存在显着局限性。我们表明，仅在我们数据集中的 325 个场景上微调一个开源的 5B 参数模型即可获得 0.345 的总分——比其基本版本 (0.066) 提高了 400% 以上，并且比最好的现成模型高 15%。我们将开源代码和数据集。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17792) | **Categories:** cs.CV, cs.RO

---

### [4] [3D Ground Truth Reconstruction from Multi-Camera Annotations Using UKF](https://arxiv.org/abs/2511.17609)
*Linh Van Ma, Unse Fatima, Tepy Sokun Chriv, Haroon Imran, Moongu Jeon*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate 3D ground truth estimation is critical for applications such as autonomous navigation, surveillance, and robotics. This paper introduces a novel method that uses an Unscented Kalman Filter (UKF) to fuse 2D bounding box or pose keypoint ground truth annotations from multiple calibrated cameras into accurate 3D ground truth. By leveraging human-annotated ground-truth 2D, our proposed method, a multi-camera single-object tracking algorithm, transforms 2D image coordinates into robust 3D world coordinates through homography-based projection and UKF-based fusion. Our proposed algorithm processes multi-view data to estimate object positions and shapes while effectively handling challenges such as occlusion. We evaluate our method on the CMC, Wildtrack, and Panoptic datasets, demonstrating high accuracy in 3D localization compared to the available 3D ground truth. Unlike existing approaches that provide only ground-plane information, our method also outputs the full 3D shape of each object. Additionally, the algorithm offers a scalable and fully automatic solution for multi-camera systems using only 2D image annotations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17609) | **Categories:** cs.CV

---

### [5] [JigsawComm: Joint Semantic Feature Encoding and Transmission for Communication-Efficient Cooperative Perception](https://arxiv.org/abs/2511.17843)
*Chenyi Wang, Zhaowei Li, Ming F. Li, Wujie Wen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-agent cooperative perception (CP) promises to overcome the inherent occlusion and sensing-range limitations of single-agent systems (e.g., autonomous driving). However, its practicality is severely constrained by the limited communication bandwidth. Existing approaches attempt to improve bandwidth efficiency via compression or heuristic message selection, without considering the semantic relevance or cross-agent redundancy of sensory data. We argue that a practical CP system must maximize the contribution of every transmitted bit to the final perception task, by extracting and transmitting semantically essential and non-redundant data. In this paper, we formulate a joint semantic feature encoding and transmission problem, which aims to maximize CP accuracy under limited bandwidth. To solve this problem, we introduce JigsawComm, an end-to-end trained, semantic-aware, and communication-efficient CP framework that learns to ``assemble the puzzle'' of multi-agent feature transmission. It uses a regularized encoder to extract semantically-relevant and sparse features, and a lightweight Feature Utility Estimator to predict the contribution of each agent's features to the final perception task. The resulting meta utility maps are exchanged among agents and leveraged to compute a provably optimal transmission policy, which selects features from agents with the highest utility score for each location. This policy inherently eliminates redundancy and achieves a scalable $\mathcal{O}(1)$ communication cost as the number of agents increases. On the benchmarks OPV2V and DAIR-V2X, JigsawComm reduces the total data volume by up to $>$500$\times$ while achieving matching or superior accuracy compared to state-of-the-art methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17843) | **Categories:** cs.CV

---

### [6] [FastMMoE: Accelerating Multimodal Large Language Models through Dynamic Expert Activation and Routing-Aware Token Pruning](https://arxiv.org/abs/2511.17885)
*Guoyang Xia, Yifeng Ding, Fengfa Li, Lei Ren, Wei Chen, Fangxiang Feng, Xiaojie Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal large language models (MLLMs) have achieved impressive performance, but high-resolution visual inputs result in long sequences of visual tokens and substantial inference latency. Reducing redundant visual tokens is critical to ease computational/memory burdens while preserving performance, enabling MLLM deployment in resource-constrained or latency-sensitive scenarios. Current visual token pruning methods mainly rely on attention-based redundancy analysis and are tailored to dense architectures. We propose Fast Multimodal Mixture-of-Experts (FastMMoE), a training-free acceleration framework for mixture-of-experts (MoE) based MLLMs, developed from a routing analysis perspective. FastMMoE combines two complementary strategies: (i) expert activation reduction for visual tokens to minimize unnecessary expert computation; and (ii) routing-aware token pruning that leverages similarity in routing probability distributions to identify and remove highly redundant visual tokens. Experiments on large-scale MoE-MLLMs such as DeepSeek-VL2 and InternVL3.5 demonstrate that FastMMoE can reduce FLOPs by up to 55.0% while retaining approximately 95.5% of the original performance, consistently outperforming dense-model pruning baselines including FastV and SparseVLM across multiple retention rates.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17885) | **Categories:** cs.CV, cs.LG

---

### [7] [Multi-speaker Attention Alignment for Multimodal Social Interaction](https://arxiv.org/abs/2511.17952)
*Liangyang Ouyang, Yifei Huang, Mingfang Zhang, Caixin Kang, Ryosuke Furuta, Yoichi Sato*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Understanding social interaction in video requires reasoning over a dynamic interplay of verbal and non-verbal cues: who is speaking, to whom, and with what gaze or gestures. While Multimodal Large Language Models (MLLMs) are natural candidates, simply adding visual inputs yields surprisingly inconsistent gains on social tasks. Our quantitative analysis of cross-modal attention inside state-of-the-art MLLMs reveals a core failure mode: in multi-speaker scenes, visual and textual tokens lack speaker-consistent alignment, exhibiting substantially weaker cross-modal attention than in object-centric images. To address this, we propose a multimodal multi-speaker attention alignment method that can be integrated into existing MLLMs. First, we introduce dynamic cross-modal head selection to identify attention heads most responsible for grounding. Then, an adaptive social-aware attention bias, computed from existing attention patterns and speaker locations, is injected into the attention mechanism. This bias reinforces alignment between a speaker's visual representation and their utterances without introducing trainable parameters or architectural changes. We integrate our method into three distinct MLLMs (LLaVA-NeXT-Video, Qwen2.5-VL, and InternVL3) and evaluate on three benchmarks (TVQA+, MMSI, OnlineMMSI). Across four social tasks, results demonstrate that our approach improves the ability of MLLMs and achieves state-of-the-art results. Attention visualizations confirm our method successfully focuses the model on speaker-relevant regions, enabling more robust multi-party social reasoning. Our implementation and model will be available at https://github.com/ut-vision/SocialInteraction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17952) | **Categories:** cs.CV

---

### [8] [Plan-X: Instruct Video Generation via Semantic Planning](https://arxiv.org/abs/2511.17986)
*Lun Huang, You Xie, Hongyi Xu, Tianpei Gu, Chenxu Zhang, Guoxian Song, Zenan Li, Xiaochen Zhao, Linjie Luo, Guillermo Sapiro*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion Transformers have demonstrated remarkable capabilities in visual synthesis, yet they often struggle with high-level semantic reasoning and long-horizon planning. This limitation frequently leads to visual hallucinations and mis-alignments with user instructions, especially in scenarios involving complex scene understanding, human-object interactions, multi-stage actions, and in-context motion reasoning. To address these challenges, we propose Plan-X, a framework that explicitly enforces high-level semantic planning to instruct video generation process. At its core lies a Semantic Planner, a learnable multimodal language model that reasons over the user's intent from both text prompts and visual context, and autoregressively generates a sequence of text-grounded spatio-temporal semantic tokens. These semantic tokens, complementary to high-level text prompt guidance, serve as structured "semantic sketches" over time for the video diffusion model, which has its strength at synthesizing high-fidelity visual details. Plan-X effectively integrates the strength of language models in multimodal in-context reasoning and planning, together with the strength of diffusion models in photorealistic video synthesis. Extensive experiments demonstrate that our framework substantially reduces visual hallucinations and enables fine-grained, instruction-aligned video generation consistent with multimodal context.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17986) | **Categories:** cs.CV, cs.AI

---


## cs.NI [cs.NI]
### [1] [AURA: Adaptive Unified Reasoning and Automation with LLM-Guided MARL for NextG Cellular Networks](https://arxiv.org/abs/2511.17506)
*Narjes Nourzad, Mingyu Zong, Bhaskar Krishnamachari*

Main category: cs.NI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Next-generation (NextG) cellular networks are expected to manage dynamic traffic while sustaining high performance. Large language models (LLMs) provide strategic reasoning for 6G planning, but their computational cost and latency limit real-time use. Multi-agent reinforcement learning (MARL) supports localized adaptation, yet coordination at scale remains challenging. We present AURA, a framework that integrates cloud-based LLMs for high-level planning with base stations modeled as MARL agents for local decision-making. The LLM generates objectives and subgoals from its understanding of the environment and reasoning capabilities, while agents at base stations execute these objectives autonomously, guided by a trust mechanism that balances local learning with external input. To reduce latency, AURA employs batched communication so that agents update the LLM's view of the environment and receive improved feedback. In a simulated 6G scenario, AURA improves resilience, reducing dropped handoff requests by more than half under normal and high traffic and lowering system failures. Agents use LLM input in fewer than 60\% of cases, showing that guidance augments rather than replaces local adaptability, thereby mitigating latency and hallucination risks. These results highlight the promise of combining LLM reasoning with MARL adaptability for scalable, real-time NextG network management.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17506) | **Categories:** cs.NI, cs.AI

---

### [2] [RadioMapMotion: A Dataset and Baseline for Proactive Spatio-Temporal Radio Environment Prediction](https://arxiv.org/abs/2511.17526)
*Honggang Jia, Nan Cheng, Xiucheng Wang*

Main category: cs.NI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Radio maps (RMs), which provide location-based pathloss estimations, are fundamental to enabling proactive, environment-aware communication in 6G networks. However, existing deep learning-based methods for RM construction often model dynamic environments as a series of independent static snapshots, thereby omitting the temporal continuity inherent in signal propagation changes caused by the motion of dynamic entities. To address this limitation, we propose the task of spatio-temporal RM prediction, which involves forecasting a sequence of future maps from historical observations. A key barrier to this predictive approach has been the lack of datasets capturing continuous environmental evolution. To fill this gap, we introduce RadioMapMotion, the first large-scale public dataset of continuous RM sequences generated from physically consistent vehicle trajectories. As a baseline for this task, we propose RadioLSTM, a UNet architecture based on Convolutional Long Short-Term Memory (ConvLSTM) and designed for multi-step sequence forecasting. Experimental evaluations show that RadioLSTM achieves higher prediction accuracy and structural fidelity compared to representative baseline methods. Furthermore, the model exhibits a low inference latency, indicating its potential suitability for real-time network operations. Our project will be publicly released at: https://github.com/UNIC-Lab/RadioMapMotion upon paper acceptance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17526) | **Categories:** cs.NI, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Skypilot: Fine-Tuning LLM with Physical Grounding for AAV Coverage Search](https://arxiv.org/abs/2511.18270)
*Zhongkai Chen, Yihao Sun, Chao Yan, Han Zhou, Xiaojia Xiang, Jie Jiang*

Main category: cs.RO

TL;DR: Skypilot提出了一种LLM增强的两阶段框架，通过集成蒙特卡洛树搜索（MCTS）将语言模型扎根于物理现实，从而提升自主飞行器的智能。


<details>
  <summary>Details</summary>
Motivation: 解决大型语言模型（LLMs）在自主飞行器（AAVs）应用中缺乏物理基础导致的幻觉和可重复性问题，尤其是在空间推理和决策方面。

Method: 提出Skypilot框架，第一阶段通过蒙特卡洛树搜索（MCTS）结合物理信息奖励函数，探索多样化的动作空间（生成、重新生成、微调、评估），第二阶段在23,000个MCTS生成的样本上微调Qwen3-4B，以加速推理。

Result: 通过大量的数值模拟和真实飞行实验验证了所提出方法的效率和优越性。

Conclusion: Skypilot框架有效地将语言模型与物理现实相结合，提高了自主飞行器的智能水平，并在实际应用中表现出优越的性能。

Abstract: 自主飞行器（AAVs）在覆盖作业和搜索任务中发挥了关键作用。大型语言模型（LLMs）的最新进展为增强AAV的智能提供了有希望的机会。这些进展有助于解决诸如区域覆盖优化、动态路径规划和自适应决策等复杂挑战。然而，LLM中缺乏物理基础导致了空间推理和决策中的幻觉和可重复性问题。为了解决这些问题，我们提出了Skypilot，这是一个LLM增强的两阶段框架，通过集成蒙特卡洛树搜索（MCTS）将语言模型扎根于物理现实。在第一阶段，我们引入了一个多样化的动作空间，包括生成、重新生成、微调和评估操作，并结合物理信息奖励函数以确保轨迹可行性。在第二阶段，我们在23,000个MCTS生成的样本上微调Qwen3-4B，在保持解决方案质量的同时实现了显著的推理加速。大量的数值模拟和真实飞行实验验证了我们提出的方法的效率和优越性。详细信息和实验结果可在https://sky-pilot.top上获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18270) | **Categories:** cs.RO

---

### [2] [MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots](https://arxiv.org/abs/2511.17889)
*Ting Huang, Dongjian Li, Rui Yang, Zeyu Zhang, Zida Yang, Hao Tang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Grounding natural-language instructions into continuous control for quadruped robots remains a fundamental challenge in vision language action. Existing methods struggle to bridge high-level semantic reasoning and low-level actuation, leading to unstable grounding and weak generalization in the real world. To address these issues, we present MobileVLA-R1, a unified vision-language-action framework that enables explicit reasoning and continuous control for quadruped robots. We construct MobileVLA-CoT, a large-scale dataset of multi-granularity chain-of-thought (CoT) for embodied trajectories, providing structured reasoning supervision for alignment. Built upon this foundation, we introduce a two-stage training paradigm that combines supervised CoT alignment with GRPO reinforcement learning to enhance reasoning consistency, control stability, and long-horizon execution. Extensive evaluations on VLN and VLA tasks demonstrate superior performance over strong baselines, with approximately a 5% improvement. Real-world deployment on a quadruped robot validates robust performance in complex environments. Code: https://github.com/AIGeeksGroup/MobileVLA-R1. Website: https://aigeeksgroup.github.io/MobileVLA-R1.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17889) | **Categories:** cs.RO, cs.CV

---

### [3] [EchoVLA: Robotic Vision-Language-Action Model with Synergistic Declarative Memory for Mobile Manipulation](https://arxiv.org/abs/2511.18112)
*Min Lin, Xiwen Liang, Bingqian Lin, Liu Jingzhi, Zijian Jiao, Kehan Li, Yuhan Ma, Yuecheng Liu, Shen Zhao, Yuzheng Zhuang, Xiaodan Liang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent progress in Vision-Language-Action (VLA) models has enabled embodied agents to interpret multimodal instructions and perform complex tasks. However, existing VLAs are mostly confined to short-horizon, table-top manipulation, lacking the memory and reasoning capability required for long-horizon mobile manipulation, where agents must coordinate navigation and manipulation under changing spatial contexts. In this work, we present EchoVLA, a memory-aware VLA model for long-horizon mobile manipulation. EchoVLA incorporates a synergistic declarative memory inspired by the human brain, consisting of a scene memory that maintains a collection of spatial-semantic maps and an episodic memory that stores task-level experiences with multimodal contextual features. During both training and inference, the two memories are individually stored, updated, and retrieved based on current observations, task history, and instructions, and their retrieved representations are fused via coarse- and fine-grained attention to guide mobile-arm diffusion policies. To support large-scale training and evaluation, we further introduce MoMani, an automated benchmark that generates expert-level long-horizon trajectories through multimodal large language model (MLLM)-guided planning and feedback-driven refinement, supplemented with real-robot demonstrations. Experiments in simulated and real-world settings show that EchoVLA improves long-horizon performance, reaching 0.52 SR on manipulation/navigation and 0.31 on mobile manipulation, exceeding $π_{0.5}$ by +0.08 and +0.11.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18112) | **Categories:** cs.RO

---

### [4] [SM$^2$ITH: Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control](https://arxiv.org/abs/2511.17798)
*Francesco D'Orazio, Sepehr Samavi, Xintong Du, Siqi Zhou, Giuseppe Oriolo, Angela P. Schoellig*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Mobile manipulators are designed to perform complex sequences of navigation and manipulation tasks in human-centered environments. While recent optimization-based methods such as Hierarchical Task Model Predictive Control (HTMPC) enable efficient multitask execution with strict task priorities, they have so far been applied mainly to static or structured scenarios. Extending these approaches to dynamic human-centered environments requires predictive models that capture how humans react to the actions of the robot. This work introduces Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control (SM$^2$ITH), a unified framework that combines HTMPC with interactive human motion prediction through bilevel optimization that jointly accounts for robot and human dynamics. The framework is validated on two different mobile manipulators, the Stretch 3 and the Ridgeback-UR10, across three experimental settings: (i) delivery tasks with different navigation and manipulation priorities, (ii) sequential pick-and-place tasks with different human motion prediction models, and (iii) interactions involving adversarial human behavior. Our results highlight how interactive prediction enables safe and efficient coordination, outperforming baselines that rely on weighted objectives or open-loop human models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17798) | **Categories:** cs.RO

---

### [5] [Time-aware Motion Planning in Dynamic Environments with Conformal Prediction](https://arxiv.org/abs/2511.18170)
*Kaier Liang, Licheng Luo, Yixuan Wang, Mingyu Cai, Cristian Ioan Vasile*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe navigation in dynamic environments remains challenging due to uncertain obstacle behaviors and the lack of formal prediction guarantees. We propose two motion planning frameworks that leverage conformal prediction (CP): a global planner that integrates Safe Interval Path Planning (SIPP) for uncertainty-aware trajectory generation, and a local planner that performs online reactive planning. The global planner offers distribution-free safety guarantees for long-horizon navigation, while the local planner mitigates inaccuracies in obstacle trajectory predictions through adaptive CP, enabling robust and responsive motion in dynamic environments. To further enhance trajectory feasibility, we introduce an adaptive quantile mechanism in the CP-based uncertainty quantification. Instead of using a fixed confidence level, the quantile is automatically tuned to the optimal value that preserves trajectory feasibility, allowing the planner to adaptively tighten safety margins in regions with higher uncertainty. We validate the proposed framework through numerical experiments conducted in dynamic and cluttered environments. The project page is available at https://time-aware-planning.github.io

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18170) | **Categories:** cs.RO

---

### [6] [Off-Road Navigation via Implicit Neural Representation of Terrain Traversability](https://arxiv.org/abs/2511.18183)
*Yixuan Jia, Qingyuan Li, Jonathan P. How*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous off-road navigation requires robots to estimate terrain traversability from onboard sensors and plan accordingly. Conventional approaches typically rely on sampling-based planners such as MPPI to generate short-term control actions that aim to minimize traversal time and risk measures derived from the traversability estimates. These planners can react quickly but optimize only over a short look-ahead window, limiting their ability to reason about the full path geometry, which is important for navigating in challenging off-road environments. Moreover, they lack the ability to adjust speed based on the terrain bumpiness, which is important for smooth navigation on challenging terrains. In this paper, we introduce TRAIL (Traversability with an Implicit Learned Representation), an off-road navigation framework that leverages an implicit neural representation to continuously parameterize terrain properties. This representation yields spatial gradients that enable integration with a novel gradient-based trajectory optimization method that adapts the path geometry and speed profile based on terrain traversability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18183) | **Categories:** cs.RO

---

### [7] [SafeFall: Learning Protective Control for Humanoid Robots](https://arxiv.org/abs/2511.18509)
*Ziyu Meng, Tengyu Liu, Le Ma, Yingying Wu, Ran Song, Wei Zhang, Siyuan Huang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Bipedal locomotion makes humanoid robots inherently prone to falls, causing catastrophic damage to the expensive sensors, actuators, and structural components of full-scale robots. To address this critical barrier to real-world deployment, we present \method, a framework that learns to predict imminent, unavoidable falls and execute protective maneuvers to minimize hardware damage. SafeFall is designed to operate seamlessly alongside existing nominal controller, ensuring no interference during normal operation. It combines two synergistic components: a lightweight, GRU-based fall predictor that continuously monitors the robot's state, and a reinforcement learning policy for damage mitigation. The protective policy remains dormant until the predictor identifies a fall as unavoidable, at which point it activates to take control and execute a damage-minimizing response. This policy is trained with a novel, damage-aware reward function that incorporates the robot's specific structural vulnerabilities, learning to shield critical components like the head and hands while absorbing energy with more robust parts of its body. Validated on a full-scale Unitree G1 humanoid, SafeFall demonstrated significant performance improvements over unprotected falls. It reduced peak contact forces by 68.3\%, peak joint torques by 78.4\%, and eliminated 99.3\% of collisions with vulnerable components. By enabling humanoids to fail safely, SafeFall provides a crucial safety net that allows for more aggressive experiments and accelerates the deployment of these robots in complex, real-world environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18509) | **Categories:** cs.RO

---

### [8] [Asynchronous Distributed Multi-Robot Motion Planning Under Imperfect Communication](https://arxiv.org/abs/2511.18703)
*Ardalan Tajbakhsh, Augustinos Saravanos, James Zhu, Evangelos A. Theodorou, Lorenz T. Biegler, Aaron M. Johnson*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper addresses the challenge of coordinating multi-robot systems under realistic communication delays using distributed optimization. We focus on consensus ADMM as a scalable framework for generating collision-free, dynamically feasible motion plans in both trajectory optimization and receding-horizon control settings. In practice, however, these algorithms are sensitive to penalty tuning or adaptation schemes (e.g. residual balancing and adaptive parameter heuristics) that do not explicitly consider delays. To address this, we introduce a Delay-Aware ADMM (DA-ADMM) variant that adapts penalty parameters based on real-time delay statistics, allowing agents to down-weight stale information and prioritize recent updates during consensus and dual updates. Through extensive simulations in 2D and 3D environments with double-integrator, Dubins-car, and drone dynamics, we show that DA-ADMM significantly improves robustness, success rate, and solution quality compared to fixed-parameter, residual-balancing, and fixed-constraint baselines. Our results highlight that performance degradation is not solely determined by delay length or frequency, but by the optimizer's ability to contextually reason over delayed information. The proposed DA-ADMM achieves consistently better coordination performance across a wide range of delay conditions, offering a principled and efficient mechanism for resilient multi-robot motion planning under imperfect communication.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18703) | **Categories:** cs.RO

---

### [9] [AIRHILT: A Human-in-the-Loop Testbed for Multimodal Conflict Detection in Aviation](https://arxiv.org/abs/2511.18718)
*Omar Garib, Jayaprakash D. Kambhampaty, Olivia J. Pinon Fischer, Dimitri N. Mavris*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce AIRHILT (Aviation Integrated Reasoning, Human-in-the-Loop Testbed), a modular and lightweight simulation environment designed to evaluate multimodal pilot and air traffic control (ATC) assistance systems for aviation conflict detection. Built on the open-source Godot engine, AIRHILT synchronizes pilot and ATC radio communications, visual scene understanding from camera streams, and ADS-B surveillance data within a unified, scalable platform. The environment supports pilot- and controller-in-the-loop interactions, providing a comprehensive scenario suite covering both terminal area and en route operational conflicts, including communication errors and procedural mistakes. AIRHILT offers standardized JSON-based interfaces that enable researchers to easily integrate, swap, and evaluate automatic speech recognition (ASR), visual detection, decision-making, and text-to-speech (TTS) models. We demonstrate AIRHILT through a reference pipeline incorporating fine-tuned Whisper ASR, YOLO-based visual detection, ADS-B-based conflict logic, and GPT-OSS-20B structured reasoning, and present preliminary results from representative runway-overlap scenarios, where the assistant achieves an average time-to-first-warning of approximately 7.7 s, with average ASR and vision latencies of approximately 5.9 s and 0.4 s, respectively. The AIRHILT environment and scenario suite are openly available, supporting reproducible research on multimodal situational awareness and conflict detection in aviation; code and scenarios are available at https://github.com/ogarib3/airhilt.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.18718) | **Categories:** cs.RO, cs.AI

---

### [10] [End-to-end Autonomous Vehicle Following System using Monocular Fisheye Camera](https://arxiv.org/abs/2511.19011)
*Jiale Zhang, Yeqiang Qian, Tong Qin, Mingyang Jiang, Siyuan Chen, Ming Yang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The increase in vehicle ownership has led to increased traffic congestion, more accidents, and higher carbon emissions. Vehicle platooning is a promising solution to address these issues by improving road capacity and reducing fuel consumption. However, existing platooning systems face challenges such as reliance on lane markings and expensive high-precision sensors, which limits their general applicability. To address these issues, we propose a vehicle following framework that expands its capability from restricted scenarios to general scenario applications using only a camera. This is achieved through our newly proposed end-to-end method, which improves overall driving performance. The method incorporates a semantic mask to address causal confusion in multi-frame data fusion. Additionally, we introduce a dynamic sampling mechanism to precisely track the trajectories of preceding vehicles. Extensive closed-loop validation in real-world vehicle experiments demonstrates the system's ability to follow vehicles in various scenarios, outperforming traditional multi-stage algorithms. This makes it a promising solution for cost-effective autonomous vehicle platooning. A complete real-world vehicle experiment is available at https://youtu.be/zL1bcVb9kqQ.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19011) | **Categories:** cs.RO

---

### [11] [Autonomous Docking of Multi-Rotor UAVs on Blimps under the Influence of Wind Gusts](https://arxiv.org/abs/2511.19135)
*Pascal Goldschmid, Aamir Ahmad*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-rotor UAVs face limited flight time due to battery constraints. Autonomous docking on blimps with onboard battery recharging and data offloading offers a promising solution for extended UAV missions. However, the vulnerability of blimps to wind gusts causes trajectory deviations, requiring precise, obstacle-aware docking strategies. To this end, this work introduces two key novelties: (i) a temporal convolutional network that predicts blimp responses to wind gusts, enabling rapid gust detection and estimation of points where the wind gust effect has subsided; (ii) a model predictive controller (MPC) that leverages these predictions to compute collision-free trajectories for docking, enabled by a novel obstacle avoidance method for close-range manoeuvres near the blimp. Simulation results show our method outperforms a baseline constant-velocity model of the blimp significantly across different scenarios. We further validate the approach in real-world experiments, demonstrating the first autonomous multi-rotor docking control strategy on blimps shown outside simulation. Source code is available here https://github.com/robot-perception-group/multi_rotor_airship_docking.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19135) | **Categories:** cs.RO

---

### [12] [Reference-Free Sampling-Based Model Predictive Control](https://arxiv.org/abs/2511.19204)
*Fabian Schramm, Pierre Fabre, Nicolas Perrin-Gilbert, Justin Carpentier*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a sampling-based model predictive control (MPC) framework that enables emergent locomotion without relying on handcrafted gait patterns or predefined contact sequences. Our method discovers diverse motion patterns, ranging from trotting to galloping, robust standing policies, jumping, and handstand balancing, purely through the optimization of high-level objectives. Building on model predictive path integral (MPPI), we propose a dual-space spline parameterization that operates on position and velocity control points. Our approach enables contact-making and contact-breaking strategies that adapt automatically to task requirements, requiring only a limited number of sampled trajectories. This sample efficiency allows us to achieve real-time control on standard CPU hardware, eliminating the need for GPU acceleration typically required by other state-of-the-art MPPI methods. We validate our approach on the Go2 quadrupedal robot, demonstrating various emergent gaits and basic jumping capabilities. In simulation, we further showcase more complex behaviors, such as backflips, dynamic handstand balancing and locomotion on a Humanoid, all without requiring reference tracking or offline pre-training.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19204) | **Categories:** cs.RO, eess.SY

---
