# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-19

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (4)](#cs-lg)
- [机器人学 (Robotics) (14)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning](https://arxiv.org/abs/2509.13352)
*Anis Koubaa, Khaled Gabr*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13352) | **Categories:** cs.AI, cs.RO, 68T07, 68T40, 68T42, I.2.9; I.2.11; I.2.8; I.2.10

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [AdaThinkDrive: Adaptive Thinking via Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/2509.13769)
*Yuechen Luo, Fang Li, Shaoqing Xu, Zhiyi Lai, Lei Yang, Qimao Chen, Ziang Luo, Zixun Xie, Shengyin Jiang, Jiaxin Liu, Long Chen, Bing Wang, Zhi-xin Yang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While reasoning technology like Chain of Thought (CoT) has been widely adopted in Vision Language Action (VLA) models, it demonstrates promising capabilities in end to end autonomous driving. However, recent efforts to integrate CoT reasoning often fall short in simple scenarios, introducing unnecessary computational overhead without improving decision quality. To address this, we propose AdaThinkDrive, a novel VLA framework with a dual mode reasoning mechanism inspired by fast and slow thinking. First, our framework is pretrained on large scale autonomous driving (AD) scenarios using both question answering (QA) and trajectory datasets to acquire world knowledge and driving commonsense. During supervised fine tuning (SFT), we introduce a two mode dataset, fast answering (w/o CoT) and slow thinking (with CoT), enabling the model to distinguish between scenarios that require reasoning. Furthermore, an Adaptive Think Reward strategy is proposed in conjunction with the Group Relative Policy Optimization (GRPO), which rewards the model for selectively applying CoT by comparing trajectory quality across different reasoning modes. Extensive experiments on the Navsim benchmark show that AdaThinkDrive achieves a PDMS of 90.3, surpassing the best vision only baseline by 1.7 points. Moreover, ablations show that AdaThinkDrive surpasses both the never Think and always Think baselines, improving PDMS by 2.0 and 1.4, respectively. It also reduces inference time by 14% compared to the always Think baseline, demonstrating its ability to balance accuracy and efficiency through adaptive reasoning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13769) | **Categories:** cs.CV

---

### [2] [Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles](https://arxiv.org/abs/2509.13577)
*Tongfei Guo, Lili Su*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors -- even on in-distribution samples -- exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13577) | **Categories:** cs.CV, cs.LG, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [ST-LINK: Spatially-Aware Large Language Models for Spatio-Temporal Forecasting](https://arxiv.org/abs/2509.13753)
*Hyotaek Jeon, Hyunwook Lee, Juwon Kim, Sungahn Ko*

Main category: cs.LG

TL;DR: ST-LINK框架通过空间增强注意力和记忆检索前馈网络，提升大型语言模型捕获时空依赖性的能力，从而改进交通预测。


<details>
  <summary>Details</summary>
Motivation: 现有大型语言模型在捕获交通预测中的空间依赖性方面存在不足，无法有效建模空间关系以及与图结构空间数据的不兼容。

Method: 提出ST-LINK框架，包含空间增强注意力（SE-Attention）和记忆检索前馈网络（MRFFN）。SE-Attention通过扩展旋转位置嵌入，将空间相关性整合为注意力机制中的直接旋转变换。MRFFN动态检索和利用关键历史模式，以捕获复杂的时间依赖性。

Result: 在基准数据集上的综合实验表明，ST-LINK超越了传统的深度学习和LLM方法，并有效地捕获了常规交通模式和突变。

Conclusion: ST-LINK框架能够有效提升大型语言模型在交通预测中捕获时空依赖性的能力。

Abstract: 交通预测是智能交通系统中的一个关键问题。在最近的研究中，大型语言模型（LLM）已成为一种很有前途的方法，但它们主要为顺序token处理量身定制的内在设计，在有效捕获空间依赖性方面带来了显著的挑战。具体来说，LLM在建模空间关系方面的固有局限性以及它们与图结构空间数据在架构上的不兼容性在很大程度上仍未得到解决。为了克服这些限制，我们引入了ST-LINK，这是一个新颖的框架，它增强了大型语言模型捕获时空依赖性的能力。它的关键组件是空间增强注意力（SE-Attention）和记忆检索前馈网络（MRFFN）。SE-Attention扩展了旋转位置嵌入，以将空间相关性作为直接旋转变换集成到注意力机制中。这种方法最大限度地提高了空间学习能力，同时保留了LLM固有的顺序处理结构。同时，MRFFN动态检索和利用关键的历史模式来捕获复杂的时间依赖性，并提高长期预测的稳定性。在基准数据集上的综合实验表明，ST-LINK超越了传统的深度学习和LLM方法，并有效地捕获了常规交通模式和突变。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13753) | **Categories:** cs.LG

---

### [2] [Ensemble of Pre-Trained Models for Long-Tailed Trajectory Prediction](https://arxiv.org/abs/2509.13914)
*Divya Thuremella, Yi Yang, Simon Wanna, Lars Kunze, Daniele De Martini*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This work explores the application of ensemble modeling to the multidimensional regression problem of trajectory prediction for vehicles in urban environments. As newer and bigger state-of-the-art prediction models for autonomous driving continue to emerge, an important open challenge is the problem of how to combine the strengths of these big models without the need for costly re-training. We show how, perhaps surprisingly, combining state-of-the-art deep learning models out-of-the-box (without retraining or fine-tuning) with a simple confidence-weighted average method can enhance the overall prediction. Indeed, while combining trajectory prediction models is not straightforward, this simple approach enhances performance by 10% over the best prediction model, especially in the long-tailed metrics. We show that this performance improvement holds on both the NuScenes and Argoverse datasets, and that these improvements are made across the dataset distribution. The code for our work is open source.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13914) | **Categories:** cs.LG, cs.AI

---

### [3] [TFMAdapter: Lightweight Instance-Level Adaptation of Foundation Models for Forecasting with Covariates](https://arxiv.org/abs/2509.13906)
*Afrin Dange, Sunita Sarawagi*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time Series Foundation Models (TSFMs) have recently achieved state-of-the-art performance in univariate forecasting on new time series simply by conditioned on a brief history of past values. Their success demonstrates that large-scale pretraining across diverse domains can acquire the inductive bias to generalize from temporal patterns in a brief history. However, most TSFMs are unable to leverage covariates -- future-available exogenous variables critical for accurate forecasting in many applications -- due to their domain-specific nature and the lack of associated inductive bias. We propose TFMAdapter, a lightweight, instance-level adapter that augments TSFMs with covariate information without fine-tuning. Instead of retraining, TFMAdapter operates on the limited history provided during a single model call, learning a non-parametric cascade that combines covariates with univariate TSFM forecasts. However, such learning would require univariate forecasts at all steps in the history, requiring too many calls to the TSFM. To enable training on the full historical context while limiting TSFM invocations, TFMAdapter uses a two-stage method: (1) generating pseudo-forecasts with a simple regression model, and (2) training a Gaussian Process regressor to refine predictions using both pseudo- and TSFM forecasts alongside covariates. Extensive experiments on real-world datasets demonstrate that TFMAdapter consistently outperforms both foundation models and supervised baselines, achieving a 24-27\% improvement over base foundation models with minimal data and computational overhead. Our results highlight the potential of lightweight adapters to bridge the gap between generic foundation models and domain-specific forecasting needs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13906) | **Categories:** cs.LG, I.2.6

---

### [4] [Deep Temporal Graph Networks for Real-Time Correction of GNSS Jamming-Induced Deviations](https://arxiv.org/abs/2509.14000)
*Ivana Kesić, Aljaž Blatnik, Carolina Fortuna, Blaž Bertalanič*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Global Navigation Satellite Systems (GNSS) are increasingly disrupted by intentional jamming, degrading availability precisely when positioning and timing must remain operational. We address this by reframing jamming mitigation as dynamic graph regression and introducing a receiver-centric deep temporal graph network that predicts, and thus corrects, the receivers horizontal deviation in real time. At each 1 Hz epoch, the satellite receiver environment is represented as a heterogeneous star graph (receiver center, tracked satellites as leaves) with time varying attributes (e.g., SNR, azimuth, elevation, latitude/longitude). A single layer Heterogeneous Graph ConvLSTM (HeteroGCLSTM) aggregates one hop spatial context and temporal dynamics over a short history to output the 2D deviation vector applied for on the fly correction.   We evaluate on datasets from two distinct receivers under three jammer profiles, continuous wave (cw), triple tone (cw3), and wideband FM, each exercised at six power levels between -45 and -70 dBm, with 50 repetitions per scenario (prejam/jam/recovery). Against strong multivariate time series baselines (MLP, uniform CNN, and Seq2Point CNN), our model consistently attains the lowest mean absolute error (MAE). At -45 dBm, it achieves 3.64 cm (GP01/cw), 7.74 cm (GP01/cw3), 4.41 cm (ublox/cw), 4.84 cm (ublox/cw3), and 4.82 cm (ublox/FM), improving to 1.65-2.08 cm by -60 to -70 dBm. On mixed mode datasets pooling all powers, MAE is 3.78 cm (GP01) and 4.25 cm (ublox10), outperforming Seq2Point, MLP, and CNN. A split study shows superior data efficiency: with only 10\% training data our approach remains well ahead of baselines (20 cm vs. 36-42 cm).

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14000) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Language Conditioning Improves Accuracy of Aircraft Goal Prediction in Untowered Airspace](https://arxiv.org/abs/2509.14063)
*Sundhar Vinodh Sangeetha, Chih-Yuan Chiu, Sarah H. Q. Li, Shreyas Kousik*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous aircraft must safely operate in untowered airspace, where coordination relies on voice-based communication among human pilots. Safe operation requires an aircraft to predict the intent, and corresponding goal location, of other aircraft. This paper introduces a multimodal framework for aircraft goal prediction that integrates natural language understanding with spatial reasoning to improve autonomous decision-making in such environments. We leverage automatic speech recognition and large language models to transcribe and interpret pilot radio calls, identify aircraft, and extract discrete intent labels. These intent labels are fused with observed trajectories to condition a temporal convolutional network and Gaussian mixture model for probabilistic goal prediction. Our method significantly reduces goal prediction error compared to baselines that rely solely on motion history, demonstrating that language-conditioned prediction increases prediction accuracy. Experiments on a real-world dataset from an untowered airport validate the approach and highlight its potential to enable socially aware, language-conditioned robotic motion planning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14063) | **Categories:** cs.RO

---

### [2] [Pre-Manipulation Alignment Prediction with Parallel Deep State-Space and Transformer Models](https://arxiv.org/abs/2509.13839)
*Motonari Kambara, Komei Sugiura*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this work, we address the problem of predicting the future success of open-vocabulary object manipulation tasks. Conventional approaches typically determine success or failure after the action has been carried out. However, they make it difficult to prevent potential hazards and rely on failures to trigger replanning, thereby reducing the efficiency of object manipulation sequences. To overcome these challenges, we propose a model, which predicts the alignment between a pre-manipulation egocentric image with the planned trajectory and a given natural language instruction. We introduce a Multi-Level Trajectory Fusion module, which employs a state-of-the-art deep state-space model and a transformer encoder in parallel to capture multi-level time-series self-correlation within the end effector trajectory. Our experimental results indicate that the proposed method outperformed existing methods, including foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13839) | **Categories:** cs.RO

---

### [3] [MAP: End-to-End Autonomous Driving with Map-Assisted Planning](https://arxiv.org/abs/2509.13926)
*Huilin Yin, Yiming Kan, Daniel Watzenig*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In recent years, end-to-end autonomous driving has attracted increasing attention for its ability to jointly model perception, prediction, and planning within a unified framework. However, most existing approaches underutilize the online mapping module, leaving its potential to enhance trajectory planning largely untapped. This paper proposes MAP (Map-Assisted Planning), a novel map-assisted end-to-end trajectory planning framework. MAP explicitly integrates segmentation-based map features and the current ego status through a Plan-enhancing Online Mapping module, an Ego-status-guided Planning module, and a Weight Adapter based on current ego status. Experiments conducted on the DAIR-V2X-seq-SPD dataset demonstrate that the proposed method achieves a 16.6% reduction in L2 displacement error, a 56.2% reduction in off-road rate, and a 44.5% improvement in overall score compared to the UniV2X baseline, even without post-processing. Furthermore, it achieves top ranking in Track 2 of the End-to-End Autonomous Driving through V2X Cooperation Challenge of MEIS Workshop @CVPR2025, outperforming the second-best model by 39.5% in terms of overall score. These results highlight the effectiveness of explicitly leveraging semantic map features in planning and suggest new directions for improving structure design in end-to-end autonomous driving systems. Our code is available at https://gitee.com/kymkym/map.git

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13926) | **Categories:** cs.RO, cs.AI, cs.CV, I.2.9; I.2.10

---

### [4] [Maximizing UAV Cellular Connectivity with Reinforcement Learning for BVLoS Path Planning](https://arxiv.org/abs/2509.13336)
*Mehran Behjati, Rosdiadee Nordin, Nor Fadzilah Abdullah*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a reinforcement learning (RL) based approach for path planning of cellular connected unmanned aerial vehicles (UAVs) operating beyond visual line of sight (BVLoS). The objective is to minimize travel distance while maximizing the quality of cellular link connectivity by considering real world aerial coverage constraints and employing an empirical aerial channel model. The proposed solution employs RL techniques to train an agent, using the quality of communication links between the UAV and base stations (BSs) as the reward function. Simulation results demonstrate the effectiveness of the proposed method in training the agent and generating feasible UAV path plans. The proposed approach addresses the challenges due to limitations in UAV cellular communications, highlighting the need for investigations and considerations in this area. The RL algorithm efficiently identifies optimal paths, ensuring maximum connectivity with ground BSs to ensure safe and reliable BVLoS flight operation. Moreover, the solution can be deployed as an offline path planning module that can be integrated into future ground control systems (GCS) for UAV operations, enhancing their capabilities and safety. The method holds potential for complex long range UAV applications, advancing the technology in the field of cellular connected UAV path planning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13336) | **Categories:** cs.RO, cs.LG

---

### [5] [ASTREA: Introducing Agentic Intelligence for Orbital Thermal Autonomy](https://arxiv.org/abs/2509.13380)
*Alejandro D. Mousist*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents ASTREA, the first agentic system deployed on flight-heritage hardware (TRL 9) for autonomous spacecraft operations. Using thermal control as a representative use case, we integrate a resource-constrained Large Language Model (LLM) agent with a reinforcement learning controller in an asynchronous architecture tailored for space-qualified platforms. Ground experiments show that LLM-guided supervision improves thermal stability and reduces violations, confirming the feasibility of combining semantic reasoning with adaptive control under hardware constraints. However, on-orbit validation aboard the International Space Station (ISS) reveals performance degradation caused by inference latency mismatched with the rapid thermal cycles characteristic of Low Earth Orbit (LEO) satellites. These results highlight both the opportunities and current limitations of agentic LLM-based systems in real flight environments, providing practical design guidelines for future space autonomy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13380) | **Categories:** cs.RO, cs.AI, cs.LG, cs.MA, cs.SY, eess.SY

---

### [6] [VEGA: Electric Vehicle Navigation Agent via Physics-Informed Neural Operator and Proximal Policy Optimization](https://arxiv.org/abs/2509.13386)
*Hansol Lim, Minhyeok Im, Jonathan Boyack, Jee Won Lee, Jongseong Brad Choi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Demands for software-defined vehicles (SDV) are rising and electric vehicles (EVs) are increasingly being equipped with powerful computers. This enables onboard AI systems to optimize charge-aware path optimization customized to reflect vehicle's current condition and environment. We present VEGA, a charge-aware EV navigation agent that plans over a charger-annotated road graph using Proximal Policy Optimization (PPO) with budgeted A* teacher-student guidance under state-of-charge (SoC) feasibility. VEGA consists of two modules. First, a physics-informed neural operator (PINO), trained on real vehicle speed and battery-power logs, uses recent vehicle speed logs to estimate aerodynamic drag, rolling resistance, mass, motor and regenerative-braking efficiencies, and auxiliary load by learning a vehicle-custom dynamics. Second, a Reinforcement Learning (RL) agent uses these dynamics to optimize a path with optimal charging stops and dwell times under SoC constraints. VEGA requires no additional sensors and uses only vehicle speed signals. It may serve as a virtual sensor for power and efficiency to potentially reduce EV cost. In evaluation on long routes like San Francisco to New York, VEGA's stops, dwell times, SoC management, and total travel time closely track Tesla Trip Planner while being slightly more conservative, presumably due to real vehicle conditions such as vehicle parameter drift due to deterioration. Although trained only in U.S. regions, VEGA was able to compute optimal charge-aware paths in France and Japan, demonstrating generalizability. It achieves practical integration of physics-informed learning and RL for EV eco-routing.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13386) | **Categories:** cs.RO, cs.LG

---

### [7] [Dense-Jump Flow Matching with Non-Uniform Time Scheduling for Robotic Policies: Mitigating Multi-Step Inference Degradation](https://arxiv.org/abs/2509.13574)
*Zidong Chen, Zihao Guo, Peng Wang, ThankGod Itua Egbe, Yan Lyu, Chenghao Qian*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Flow matching has emerged as a competitive framework for learning high-quality generative policies in robotics; however, we find that generalisation arises and saturates early along the flow trajectory, in accordance with recent findings in the literature. We further observe that increasing the number of Euler integration steps during inference counter-intuitively and universally degrades policy performance. We attribute this to (i) additional, uniformly spaced integration steps oversample the late-time region, thereby constraining actions towards the training trajectories and reducing generalisation; and (ii) the learned velocity field becoming non-Lipschitz as integration time approaches 1, causing instability. To address these issues, we propose a novel policy that utilises non-uniform time scheduling (e.g., U-shaped) during training, which emphasises both early and late temporal stages to regularise policy training, and a dense-jump integration schedule at inference, which uses a single-step integration to replace the multi-step integration beyond a jump point, to avoid unstable areas around 1. Essentially, our policy is an efficient one-step learner that still pushes forward performance through multi-step integration, yielding up to 23.7% performance gains over state-of-the-art baselines across diverse robotic tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13574) | **Categories:** cs.RO, cs.AI

---

### [8] [TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning](https://arxiv.org/abs/2509.13579)
*Momchil S. Tomov, Sang Uk Lee, Hansford Hendrago, Jinwook Huh, Teawon Han, Forbes Howington, Rafael da Silva, Gianmarco Bernasconi, Marc Heim, Samuel Findler, Xiaonan Ji, Alexander Boule, Michael Napoli, Kuo Chen, Jesse Miller, Boaz Floor, Yunqing Hu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13579) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [9] [Reinforcement Learning for Robotic Insertion of Flexible Cables in Industrial Settings](https://arxiv.org/abs/2509.13731)
*Jeongwoo Park, Seabin Lee, Changmin Park, Wonjong Lee, Changjoo Nam*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The industrial insertion of flexible flat cables (FFCs) into receptacles presents a significant challenge owing to the need for submillimeter precision when handling the deformable cables. In manufacturing processes, FFC insertion with robotic manipulators often requires laborious human-guided trajectory generation. While Reinforcement Learning (RL) offers a solution to automate this task without modeling complex properties of FFCs, the nondeterminism caused by the deformability of FFCs requires significant efforts and time on training. Moreover, training directly in a real environment is dangerous as industrial robots move fast and possess no safety measure. We propose an RL algorithm for FFC insertion that leverages a foundation model-based real-to-sim approach to reduce the training time and eliminate the risk of physical damages to robots and surroundings. Training is done entirely in simulation, allowing for random exploration without the risk of physical damages. Sim-to-real transfer is achieved through semantic segmentation masks which leave only those visual features relevant to the insertion tasks such as the geometric and spatial information of the cables and receptacles. To enhance generality, we use a foundation model, Segment Anything Model 2 (SAM2). To eleminate human intervention, we employ a Vision-Language Model (VLM) to automate the initial prompting of SAM2 to find segmentation masks. In the experiments, our method exhibits zero-shot capabilities, which enable direct deployments to real environments without fine-tuning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13731) | **Categories:** cs.RO

---

### [10] [Motion Adaptation Across Users and Tasks for Exoskeletons via Meta-Learning](https://arxiv.org/abs/2509.13736)
*Muyuan Ma, Long Cheng, Lijun Han, Xiuze Xia, Houcheng Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Wearable exoskeletons can augment human strength and reduce muscle fatigue during specific tasks. However, developing personalized and task-generalizable assistance algorithms remains a critical challenge. To address this, a meta-imitation learning approach is proposed. This approach leverages a task-specific neural network to predict human elbow joint movements, enabling effective assistance while enhancing generalization to new scenarios. To accelerate data collection, full-body keypoint motions are extracted from publicly available RGB video and motion-capture datasets across multiple tasks, and subsequently retargeted in simulation. Elbow flexion trajectories generated in simulation are then used to train the task-specific neural network within the model-agnostic meta-learning (MAML) framework, which allows the network to rapidly adapt to novel tasks and unseen users with only a few gradient updates. The adapted network outputs personalized references tracked by a gravity-compensated PD controller to ensure stable assistance. Experimental results demonstrate that the exoskeleton significantly reduces both muscle activation and metabolic cost for new users performing untrained tasks, compared to performing without exoskeleton assistance. These findings suggest that the proposed framework effectively improves task generalization and user adaptability for wearable exoskeleton systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13736) | **Categories:** cs.RO

---

### [11] [Repulsive Trajectory Modification and Conflict Resolution for Efficient Multi-Manipulator Motion Planning](https://arxiv.org/abs/2509.13882)
*Junhwa Hong, Beomjoon Lee, Woojin Lee, Changjoo Nam*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose an efficient motion planning method designed to efficiently find collision-free trajectories for multiple manipulators. While multi-manipulator systems offer significant advantages, coordinating their motions is computationally challenging owing to the high dimensionality of their composite configuration space. Conflict-Based Search (CBS) addresses this by decoupling motion planning, but suffers from subsequent conflicts incurred by resolving existing conflicts, leading to an exponentially growing constraint tree of CBS. Our proposed method is based on repulsive trajectory modification within the two-level structure of CBS. Unlike conventional CBS variants, the low-level planner applies a gradient descent approach using an Artificial Potential Field. This field generates repulsive forces that guide the trajectory of the conflicting manipulator away from those of other robots. As a result, subsequent conflicts are less likely to occur. Additionally, we develop a strategy that, under a specific condition, directly attempts to find a conflict-free solution in a single step without growing the constraint tree. Through extensive tests including physical robot experiments, we demonstrate that our method consistently reduces the number of expanded nodes in the constraint tree, achieves a higher success rate, and finds a solution faster compared to Enhanced CBS and other state-of-the-art algorithms.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13882) | **Categories:** cs.RO, cs.MA

---

### [12] [PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models](https://arxiv.org/abs/2509.13903)
*Artem Lykov, Jeffrin Sam, Hung Khang Nguyen, Vladislav Kozlovskiy, Yara Mahmoud, Valerii Serpiva, Miguel Altamirano Cabrera, Mikhail Konenkov, Dzmitry Tsetserukou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce PhysicalAgent, an agentic framework for robotic manipulation that integrates iterative reasoning, diffusion-based video generation, and closed-loop execution. Given a textual instruction, our method generates short video demonstrations of candidate trajectories, executes them on the robot, and iteratively re-plans in response to failures. This approach enables robust recovery from execution errors. We evaluate PhysicalAgent across multiple perceptual modalities (egocentric, third-person, and simulated) and robotic embodiments (bimanual UR3, Unitree G1 humanoid, simulated GR1), comparing against state-of-the-art task-specific baselines. Experiments demonstrate that our method consistently outperforms prior approaches, achieving up to 83% success on human-familiar tasks. Physical trials reveal that first-attempt success is limited (20-30%), yet iterative correction increases overall success to 80% across platforms. These results highlight the potential of video-based generative reasoning for general-purpose robotic manipulation and underscore the importance of iterative execution for recovering from initial failures. Our framework paves the way for scalable, adaptable, and robust robot control.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13903) | **Categories:** cs.RO

---

### [13] [MetricNet: Recovering Metric Scale in Generative Navigation Policies](https://arxiv.org/abs/2509.13965)
*Abhijeet Nayak, Débora N. P. Oliveira, Samiran Gode, Cordelia Schmid, Wolfram Burgard*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generative navigation policies have made rapid progress in improving end-to-end learned navigation. Despite their promising results, this paradigm has two structural problems. First, the sampled trajectories exist in an abstract, unscaled space without metric grounding. Second, the control strategy discards the full path, instead moving directly towards a single waypoint. This leads to short-sighted and unsafe actions, moving the robot towards obstacles that a complete and correctly scaled path would circumvent. To address these issues, we propose MetricNet, an effective add-on for generative navigation that predicts the metric distance between waypoints, grounding policy outputs in real-world coordinates. We evaluate our method in simulation with a new benchmarking framework and show that executing MetricNet-scaled waypoints significantly improves both navigation and exploration performance. Beyond simulation, we further validate our approach in real-world experiments. Finally, we propose MetricNav, which integrates MetricNet into a navigation policy to guide the robot away from obstacles while still moving towards the goal.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.13965) | **Categories:** cs.RO, cs.CV

---

### [14] [Prompt2Auto: From Motion Prompt to Automated Control via Geometry-Invariant One-Shot Gaussian Process Learning](https://arxiv.org/abs/2509.14040)
*Zewen Yang, Xiaobing Dai, Dongfa Zhang, Yu Li, Ziyang Meng, Bingkun Huang, Hamid Sadeghian, Sami Haddadin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning from demonstration allows robots to acquire complex skills from human demonstrations, but conventional approaches often require large datasets and fail to generalize across coordinate transformations. In this paper, we propose Prompt2Auto, a geometry-invariant one-shot Gaussian process (GeoGP) learning framework that enables robots to perform human-guided automated control from a single motion prompt. A dataset-construction strategy based on coordinate transformations is introduced that enforces invariance to translation, rotation, and scaling, while supporting multi-step predictions. Moreover, GeoGP is robust to variations in the user's motion prompt and supports multi-skill autonomy. We validate the proposed approach through numerical simulations with the designed user graphical interface and two real-world robotic experiments, which demonstrate that the proposed method is effective, generalizes across tasks, and significantly reduces the demonstration burden. Project page is available at: https://prompt2auto.github.io

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14040) | **Categories:** cs.RO, cs.AI, cs.SY, eess.SY

---
