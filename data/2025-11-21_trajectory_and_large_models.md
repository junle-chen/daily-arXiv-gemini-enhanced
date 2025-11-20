# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-21

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [cs.CR (1)](#cs-cr)
- [计算机视觉 (Computer Vision) (6)](#cs-cv)
- [机器学习 (Machine Learning) (4)](#cs-lg)
- [机器人学 (Robotics) (7)](#cs-ro)
- [stat.AP (1)](#stat-ap)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Learning Human-Like RL Agents Through Trajectory Optimization With Action Quantization](https://arxiv.org/abs/2511.15055)
*Jian-Ting Guo, Yu-Cheng Chen, Ping-Chun Hsieh, Kuo-Hao Ho, Po-Wei Huang, Ti-Rong Wu, I-Chen Wu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human-like agents have long been one of the goals in pursuing artificial intelligence. Although reinforcement learning (RL) has achieved superhuman performance in many domains, relatively little attention has been focused on designing human-like RL agents. As a result, many reward-driven RL agents often exhibit unnatural behaviors compared to humans, raising concerns for both interpretability and trustworthiness. To achieve human-like behavior in RL, this paper first formulates human-likeness as trajectory optimization, where the objective is to find an action sequence that closely aligns with human behavior while also maximizing rewards, and adapts the classic receding-horizon control to human-like learning as a tractable and efficient implementation. To achieve this, we introduce Macro Action Quantization (MAQ), a human-like RL framework that distills human demonstrations into macro actions via Vector-Quantized VAE. Experiments on D4RL Adroit benchmarks show that MAQ significantly improves human-likeness, increasing trajectory similarity scores, and achieving the highest human-likeness rankings among all RL agents in the human evaluation study. Our results also demonstrate that MAQ can be easily integrated into various off-the-shelf RL algorithms, opening a promising direction for learning human-like RL agents. Our code is available at https://rlg.iis.sinica.edu.tw/papers/MAQ.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15055) | **Categories:** cs.AI, cs.LG, cs.RO

---

### [2] [ProRAC: A Neuro-symbolic Method for Reasoning about Actions with LLM-based Progression](https://arxiv.org/abs/2511.15069)
*Haoyong Wu, Yongmei Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this paper, we propose ProRAC (Progression-based Reasoning about Actions and Change), a neuro-symbolic framework that leverages LLMs to tackle RAC problems. ProRAC extracts fundamental RAC elements including actions and questions from the problem, progressively executes each action to derive the final state, and then evaluates the query against the progressed state to arrive at an answer. We evaluate ProRAC on several RAC benchmarks, and the results demonstrate that our approach achieves strong performance across different benchmarks, domains, LLM backbones, and types of RAC tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15069) | **Categories:** cs.AI, cs.CL

---

### [3] [SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making](https://arxiv.org/abs/2511.15202)
*Yinsheng Wang, Tario G You, Léonard Boussioux, Shan Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making), a novel framework that integrates mathematical optimization with the contextual capabilities of large language models (LLMs). SOLID facilitates iterative collaboration between optimization and LLMs agents through dual prices and deviation penalties. This interaction improves the quality of the decisions while maintaining modularity and data privacy. The framework retains theoretical convergence guarantees under convexity assumptions, providing insight into the design of LLMs prompt. To evaluate SOLID, we applied it to a stock portfolio investment case with historical prices and financial news as inputs. Empirical results demonstrate convergence under various scenarios and indicate improved annualized returns compared to a baseline optimizer-only method, validating the synergy of the two agents. SOLID offers a promising framework for advancing automated and intelligent decision-making across diverse domains.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15202) | **Categories:** cs.AI

---


## cs.CR [cs.CR]
### [1] [Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard](https://arxiv.org/abs/2511.14876)
*Henry Wong, Clement Fung, Weiran Lin, Karen Li, Stanley Chen, Lujo Bauer*

Main category: cs.CR

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: To autonomously control vehicles, driving agents use outputs from a combination of machine-learning (ML) models, controller logic, and custom modules. Although numerous prior works have shown that adversarial examples can mislead ML models used in autonomous driving contexts, it remains unclear if these attacks are effective at producing harmful driving actions for various agents, environments, and scenarios.   To assess the risk of adversarial examples to autonomous driving, we evaluate attacks against a variety of driving agents, rather than against ML models in isolation. To support this evaluation, we leverage CARLA, an urban driving simulator, to create and evaluate adversarial examples. We create adversarial patches designed to stop or steer driving agents, stream them into the CARLA simulator at runtime, and evaluate them against agents from the CARLA Leaderboard, a public repository of best-performing autonomous driving agents from an annual research competition. Unlike prior work, we evaluate attacks against autonomous driving systems without creating or modifying any driving-agent code and against all parts of the agent included with the ML model.   We perform a case-study investigation of two attack strategies against three open-source driving agents from the CARLA Leaderboard across multiple driving scenarios, lighting conditions, and locations. Interestingly, we show that, although some attacks can successfully mislead ML models into predicting erroneous stopping or steering commands, some driving agents use modules, such as PID control or GPS-based rules, that can overrule attacker-manipulated predictions from ML models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14876) | **Categories:** cs.CR, cs.CV, cs.LG, cs.RO

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Unsupervised Discovery of Long-Term Spatiotemporal Periodic Workflows in Human Activities](https://arxiv.org/abs/2511.14945)
*Fan Yang, Quanting Xie, Atsunori Moteki, Shoichi Masui, Shan Jiang, Yonatan Bisk, Graham Neubig*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Periodic human activities with implicit workflows are common in manufacturing, sports, and daily life. While short-term periodic activities -- characterized by simple structures and high-contrast patterns -- have been widely studied, long-term periodic workflows with low-contrast patterns remain largely underexplored. To bridge this gap, we introduce the first benchmark comprising 580 multimodal human activity sequences featuring long-term periodic workflows. The benchmark supports three evaluation tasks aligned with real-world applications: unsupervised periodic workflow detection, task completion tracking, and procedural anomaly detection. We also propose a lightweight, training-free baseline for modeling diverse periodic workflow patterns. Experiments show that: (i) our benchmark presents significant challenges to both unsupervised periodic detection methods and zero-shot approaches based on powerful large language models (LLMs); (ii) our baseline outperforms competing methods by a substantial margin in all evaluation tasks; and (iii) in real-world applications, our baseline demonstrates deployment advantages on par with traditional supervised workflow detection approaches, eliminating the need for annotation and retraining. Our project page is https://sites.google.com/view/periodicworkflow.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14945) | **Categories:** cs.CV

---

### [2] [ESA: Energy-Based Shot Assembly Optimization for Automatic Video Editing](https://arxiv.org/abs/2511.02505)
*Yaosen Chen, Wei Wang, Tianheng Zheng, Xuming Wen, Han Yang, Yanru Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Shot assembly is a crucial step in film production and video editing, involving the sequencing and arrangement of shots to construct a narrative, convey information, or evoke emotions. Traditionally, this process has been manually executed by experienced editors. While current intelligent video editing technologies can handle some automated video editing tasks, they often fail to capture the creator's unique artistic expression in shot assembly. To address this challenge, we propose an energy-based optimization method for video shot assembly. Specifically, we first perform visual-semantic matching between the script generated by a large language model and a video library to obtain subsets of candidate shots aligned with the script semantics. Next, we segment and label the shots from reference videos, extracting attributes such as shot size, camera motion, and semantics. We then employ energy-based models to learn from these attributes, scoring candidate shot sequences based on their alignment with reference styles. Finally, we achieve shot assembly optimization by combining multiple syntax rules, producing videos that align with the assembly style of the reference videos. Our method not only automates the arrangement and combination of independent shots according to specific logic, narrative requirements, or artistic styles but also learns the assembly style of reference videos, creating a coherent visual sequence or holistic visual expression. With our system, even users with no prior video editing experience can create visually compelling videos. Project page: https://sobeymil.github.io/esa.com

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02505) | **Categories:** cs.CV, cs.AI

---

### [3] [Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks](https://arxiv.org/abs/2511.15065)
*Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, Chenglin Wu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench -- a comprehensive benchmark designed to systematically evaluate video models' reasoning capabilities. Grounded in maze-solving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10--20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15065) | **Categories:** cs.CV, cs.AI

---

### [4] [MambaTrack3D: A State Space Model Framework for LiDAR-Based Object Tracking under High Temporal Variation](https://arxiv.org/abs/2511.15077)
*Shengjing Tian, Yinan Han, Xiantong Zhao, Xuehu Liu, Qi Lang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Dynamic outdoor environments with high temporal variation (HTV) pose significant challenges for 3D single object tracking in LiDAR point clouds. Existing memory-based trackers often suffer from quadratic computational complexity, temporal redundancy, and insufficient exploitation of geometric priors. To address these issues, we propose MambaTrack3D, a novel HTV-oriented tracking framework built upon the state space model Mamba. Specifically, we design a Mamba-based Inter-frame Propagation (MIP) module that replaces conventional single-frame feature extraction with efficient inter-frame propagation, achieving near-linear complexity while explicitly modeling spatial relations across historical frames. Furthermore, a Grouped Feature Enhancement Module (GFEM) is introduced to separate foreground and background semantics at the channel level, thereby mitigating temporal redundancy in the memory bank. Extensive experiments on KITTI-HTV and nuScenes-HTV benchmarks demonstrate that MambaTrack3D consistently outperforms both HTV-oriented and normal-scenario trackers, achieving improvements of up to 6.5 success and 9.5 precision over HVTrack under moderate temporal gaps. On the standard KITTI dataset, MambaTrack3D remains highly competitive with state-of-the-art normal-scenario trackers, confirming its strong generalization ability. Overall, MambaTrack3D achieves a superior accuracy-efficiency trade-off, delivering robust performance across both specialized HTV and conventional tracking scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15077) | **Categories:** cs.CV

---

### [5] [MambaIO: Global-Coordinate Inertial Odometry for Pedestrians via Multi-Scale Frequency-Decoupled Modeling](https://arxiv.org/abs/2511.15645)
*Shanshan Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Inertial Odometry (IO) enables real-time localization using only acceleration and angular velocity measurements from an Inertial Measurement Unit (IMU), making it a promising solution for localization in consumer-grade applications. Traditionally, IMU measurements in IO have been processed under two coordinate system paradigms: the body coordinate frame and the global coordinate frame, with the latter being widely adopted. However, recent studies in drone scenarios have demonstrated that the body frame can significantly improve localization accuracy, prompting a re-evaluation of the suitability of the global frame for pedestrian IO. To address this issue, this paper systematically evaluates the effectiveness of the global coordinate frame in pedestrian IO through theoretical analysis, qualitative inspection, and quantitative experiments. Building upon these findings, we further propose MambaIO, which decomposes IMU measurements into high-frequency and low-frequency components using a Laplacian pyramid. The low-frequency component is processed by a Mamba architecture to extract implicit contextual motion cues, while the high-frequency component is handled by a convolutional structure to capture fine-grained local motion details. Experiments on multiple public datasets show that MambaIO substantially reduces localization error and achieves state-of-the-art (SOTA) performance. To the best of our knowledge, this is the first application of the Mamba architecture to the inertial odometry task.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15645) | **Categories:** cs.CV, cs.RO

---

### [6] [MMCM: Multimodality-aware Metric using Clustering-based Modes for Probabilistic Human Motion Prediction](https://arxiv.org/abs/2511.15179)
*Kyotaro Tokoro, Hiromu Taketsugu, Norimichi Ukita*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes a novel metric for Human Motion Prediction (HMP). Since a single past sequence can lead to multiple possible futures, a probabilistic HMP method predicts such multiple motions. While a single motion predicted by a deterministic method is evaluated only with the difference from its ground truth motion, multiple predicted motions should also be evaluated based on their distribution. For this evaluation, this paper focuses on the following two criteria. \textbf{(a) Coverage}: motions should be distributed among multiple motion modes to cover diverse possibilities. \textbf{(b) Validity}: motions should be kinematically valid as future motions observable from a given past motion. However, existing metrics simply appreciate widely distributed motions even if these motions are observed in a single mode and kinematically invalid. To resolve these disadvantages, this paper proposes a Multimodality-aware Metric using Clustering-based Modes (MMCM). For (a) coverage, MMCM divides a motion space into several clusters, each of which is regarded as a mode. These modes are used to explicitly evaluate whether predicted motions are distributed among multiple modes. For (b) validity, MMCM identifies valid modes by collecting possible future motions from a motion dataset. Our experiments validate that our clustering yields sensible mode definitions and that MMCM accurately scores multimodal predictions. Code: https://github.com/placerkyo/MMCM

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15179) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control](https://arxiv.org/abs/2511.15248)
*Kai Yang, Xin Xu, Yangkun Chen, Weijie Liu, Jiafei Lyu, Zichuan Lin, Deheng Ye, Saiyong Yang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Long-term training of large language models (LLMs) requires maintaining stable exploration to prevent the model from collapsing into sub-optimal behaviors. Entropy is crucial in this context, as it controls exploration and helps avoid premature convergence to sub-optimal solutions. However, existing reinforcement learning methods struggle to maintain an appropriate level of entropy, as the training process involves a mix of positive and negative samples, each affecting entropy in different ways across steps. To address this, we propose Entropy stablilization via Proportional-Integral Control (EntroPIC), a novel method that adaptively adjusts the influence of positive and negative samples by dynamically tuning their loss coefficients. This approach stabilizes entropy throughout training, ensuring efficient exploration and steady progress. We provide a comprehensive theoretical analysis for both on-policy and off-policy learning settings, demonstrating that EntroPIC is effective at controlling entropy in large-scale LLM training. Experimental results show that our method successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15248) | **Categories:** cs.LG, cs.AI

---

### [2] [Transformer-Guided Deep Reinforcement Learning for Optimal Takeoff Trajectory Design of an eVTOL Drone](https://arxiv.org/abs/2511.14887)
*Nathan M. Roberts, Xiaosong Du*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid advancement of electric vertical take-off and landing (eVTOL) aircraft offers a promising opportunity to alleviate urban traffic congestion. Thus, developing optimal takeoff trajectories for minimum energy consumption becomes essential for broader eVTOL aircraft applications. Conventional optimal control methods (such as dynamic programming and linear quadratic regulator) provide highly efficient and well-established solutions but are limited by problem dimensionality and complexity. Deep reinforcement learning (DRL) emerges as a special type of artificial intelligence tackling complex, nonlinear systems; however, the training difficulty is a key bottleneck that limits DRL applications. To address these challenges, we propose the transformer-guided DRL to alleviate the training difficulty by exploring a realistic state space at each time step using a transformer. The proposed transformer-guided DRL was demonstrated on an optimal takeoff trajectory design of an eVTOL drone for minimal energy consumption while meeting takeoff conditions (i.e., minimum vertical displacement and minimum horizontal velocity) by varying control variables (i.e., power and wing angle to the vertical). Results presented that the transformer-guided DRL agent learned to take off with $4.57\times10^6$ time steps, representing 25% of the $19.79\times10^6$ time steps needed by a vanilla DRL agent. In addition, the transformer-guided DRL achieved 97.2% accuracy on the optimal energy consumption compared against the simulation-based optimal reference while the vanilla DRL achieved 96.3% accuracy. Therefore, the proposed transformer-guided DRL outperformed vanilla DRL in terms of both training efficiency as well as optimal design verification.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14887) | **Categories:** cs.LG

---

### [3] [Reasoning in Diffusion Large Language Models is Concentrated in Dynamic Confusion Zones](https://arxiv.org/abs/2511.15208)
*Ranfei Chen, Ming Chen, Kaifei Wang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion Large Language Models (dLLMs) are rapidly emerging alongside autoregressive models as a powerful paradigm for complex reasoning, with reinforcement learning increasingly used for downstream alignment. Existing trajectory-based RL methods uniformly allocate policy gradients across denoising steps, implicitly treating all steps as equally important. We challenge this assumption by analyzing trajectories with several step-level metrics: entropy-based uncertainty, Confidence-Margin (CM) uncertainty, and Rate of Entropy Change (RoEC). These reveal structured "zones of confusion": transient spikes in uncertainty and instability that strongly predict final success or failure, while most steps remain stable. We propose Adaptive Trajectory Policy Optimization (ATPO), a lightweight step-selection strategy that dynamically reallocates gradient updates to these high-leverage steps without changing the RL objective, rewards, or compute budget. Using a hybrid RoEC+CM rule, ATPO delivers substantial gains in reasoning accuracy and training stability across benchmarks, showing that exploiting trajectory dynamics is key to advancing dLLM RL.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15208) | **Categories:** cs.LG

---

### [4] [DeepThinkVLA: Enhancing Reasoning Capability of Vision-Language-Action Models](https://arxiv.org/abs/2511.15669)
*Cheng Yin, Yankai Lin, Wang Xu, Sikyuen Tam, Xiangrui Zeng, Zhiyuan Liu, Zhouping Yin*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Enabling Vision-Language-Action (VLA) models to "think before acting" via Chain-of-Thought (CoT) is a promising path to overcoming the data-hungry nature of end-to-end robot policies. However, progress is stalled by a fundamental conflict: existing models use a single autoregressive decoder for both sequential CoT reasoning and high-dimensional, parallelizable robot actions. This architectural mismatch degrades motor control and fails to forge a strong causal link between thought and action. We introduce DeepThinkVLA, which resolves this conflict through a tightly integrated architecture and training strategy. Architecturally, our hybrid-attention decoder generates sequential CoT with causal attention and then switches to bidirectional attention for fast, parallel decoding of action vectors. This design is complemented by a two-stage training pipeline: we first use Supervised Fine-Tuning (SFT) to teach the model foundational reasoning, then apply Reinforcement Learning (RL) with task-success rewards to causally align the full reasoning-action sequence with desired outcomes. This synergy leads to state-of-the-art performance, achieving a 97.0% success rate on the LIBERO benchmark. Our ablations confirm the design's effectiveness: the hybrid architecture alone outperforms standard decoders by 15.5%, and the final RL stage provides a crucial 2% boost to secure top performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15669) | **Categories:** cs.LG, cs.AI, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification](https://arxiv.org/abs/2511.14977)
*Xiangyu Li, Zhaomiao Guo*

Main category: cs.RO

TL;DR: SVBRD-LLM框架通过零样本提示工程自动发现、验证和应用可解释的交通行为规则，从而理解自动驾驶车辆的真实行为。


<details>
  <summary>Details</summary>
Motivation: 为了理解自动驾驶车辆在真实交通环境中的行为，从而服务于交通安全分析、政策制定和公众接受度。

Method: 提出SVBRD-LLM框架，该框架利用YOLOv8和ByteTrack提取车辆轨迹，计算运动学特征，并采用GPT-5零样本提示来比较自动驾驶车辆和人类驾驶车辆，生成35个结构化的行为规则假设。然后，在验证集上测试这些规则，并根据失败案例迭代改进，以过滤掉虚假相关性，并编译成一个高置信度的规则库。

Result: 在超过1500小时的真实交通视频上的实验表明，该框架在自动驾驶车辆识别中实现了90.0%的准确率和93.3%的F1分数。发现的规则清晰地揭示了自动驾驶车辆在速度控制平稳性、车道变换保守性和加速稳定性方面的独特特征。

Conclusion: 该研究提出的SVBRD-LLM框架能够有效地从真实交通视频中提取并验证自动驾驶车辆的行为规则，这些规则可以用于识别自动驾驶车辆，并深入了解其行为特性。

Abstract: 随着越来越多的自动驾驶车辆在公共道路上行驶，理解自动驾驶车辆的真实世界行为对于分析交通安全、制定政策和提高公众接受度至关重要。本文提出了SVBRD-LLM，该框架通过零样本提示工程自动发现、验证和应用可解释的行为规则，从真实的交通视频中学习。该框架使用YOLOv8和ByteTrack提取车辆轨迹，计算运动学特征，并采用GPT-5零样本提示来比较自动驾驶车辆和人类驾驶车辆，生成35个结构化的行为规则假设。这些规则在验证集上进行测试，并基于失败案例进行迭代改进，以过滤掉虚假相关性，并编译成一个高置信度的规则库。该框架在一个独立的测试集上进行了速度变化预测、车道变换预测和自动驾驶车辆识别任务的评估。在超过1500小时的真实交通视频上的实验表明，该框架在自动驾驶车辆识别中实现了90.0%的准确率和93.3%的F1分数。发现的规则清晰地揭示了自动驾驶车辆在速度控制平稳性、车道变换保守性和加速稳定性方面的独特特征，每个规则都附带有语义描述、适用上下文和验证置信度。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14977) | **Categories:** cs.RO, cs.AI

---

### [2] [Z-Merge: Multi-Agent Reinforcement Learning for On-Ramp Merging with Zone-Specific V2X Traffic Information](https://arxiv.org/abs/2511.14910)
*Yassine Ibork, Myounggyu Won, Lokesh Das*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ramp merging is a critical and challenging task for autonomous vehicles (AVs), particularly in mixed traffic environments with human-driven vehicles (HVs). Existing approaches typically rely on either lane-changing or inter-vehicle gap creation strategies based solely on local or neighboring information, often leading to suboptimal performance in terms of safety and traffic efficiency. In this paper, we present a V2X (vehicle-to-everything communication)-assisted Multiagent Reinforcement Learning (MARL) framework for on-ramp merging that effectively coordinates the complex interplay between lane-changing and inter-vehicle gap adaptation strategies by utilizing zone-specific global information available from a roadside unit (RSU). The merging control problem is formulated as a Multiagent Partially Observable Markov Decision Process (MA-POMDP), where agents leverage both local and global observations through V2X communication. To support both discrete and continuous control decisions, we design a hybrid action space and adopt a parameterized deep Q-learning approach. Extensive simulations, integrating the SUMO traffic simulator and the MOSAIC V2X simulator, demonstrate that our framework significantly improves merging success rate, traffic efficiency, and road safety across diverse traffic scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14910) | **Categories:** cs.RO

---

### [3] [Communication-Aware Asynchronous Distributed Trajectory Optimization for UAV Swarm](https://arxiv.org/abs/2511.14994)
*Yue Yu, Xiaobo Zheng, Shaoming He*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Distributed optimization offers a promising paradigm for trajectory planning in Unmanned Aerial Vehicle (UAV) swarms, yet its deployment in communication-constrained environments remains challenging due to unreliable links and limited data exchange. This paper addresses this issue via a two-tier architecture explicitly designed for operation under communication constraints. We develop a Communication-Aware Asynchronous Distributed Trajectory Optimization (CA-ADTO) framework that integrates Parameterized Differential Dynamic Programming (PDDP) for local trajectory optimization of individual UAVs with an asynchronous Alternating Direction Method of Multipliers (async-ADMM) for swarm-level coordination. The proposed architecture enables fully distributed optimization while substantially reducing communication overhead, making it suitable for real-world scenarios in which reliable connectivity cannot be guaranteed. The method is particularly effective in handling nonlinear dynamics and spatio-temporal coupling under communication constraints.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14994) | **Categories:** cs.RO

---

### [4] [Symmetry-Breaking in Multi-Agent Navigation: Winding Number-Aware MPC with a Learned Topological Strategy](https://arxiv.org/abs/2511.15239)
*Tomoki Nakao, Kazumi Kasaura, Tadashi Kozuno*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We address the fundamental challenge of resolving symmetry-induced deadlocks in distributed multi-agent navigation by proposing a new hierarchical navigation method. When multiple agents interact, it is inherently difficult for them to autonomously break the symmetry of deciding how to pass each other. To tackle this problem, we introduce an approach that quantifies cooperative symmetry-breaking strategies using a topological invariant called the winding number, and learns the strategies themselves through reinforcement learning. Our method features a hierarchical policy consisting of a learning-based Planner, which plans topological cooperative strategies, and a model-based Controller, which executes them. Through reinforcement learning, the Planner learns to produce two types of parameters for the Controller: one is the topological cooperative strategy represented by winding numbers, and the other is a set of dynamic weights that determine which agent interaction to prioritize in dense scenarios where multiple agents cross simultaneously. The Controller then generates collision-free and efficient motions based on the strategy and weights provided by the Planner. This hierarchical structure combines the flexible decision-making ability of learning-based methods with the reliability of model-based approaches. Simulation and real-world robot experiments demonstrate that our method outperforms existing baselines, particularly in dense environments, by efficiently avoiding collisions and deadlocks while achieving superior navigation performance. The code for the experiments is available at https://github.com/omron-sinicx/WNumMPC.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15239) | **Categories:** cs.RO, cs.MA

---

### [5] [Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception](https://arxiv.org/abs/2511.15279)
*Jiashu Yang, Yifan Han, Yucheng Xie, Ning Guo, Wenzhao Lian*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In embodied AI perception systems, visual perception should be active: the goal is not to passively process static images, but to actively acquire more informative data within pixel and spatial budget constraints. Existing vision models and fixed RGB-D camera systems fundamentally fail to reconcile wide-area coverage with fine-grained detail acquisition, severely limiting their efficacy in open-world robotic applications. To address this issue, we propose EyeVLA, a robotic eyeball for active visual perception that can take proactive actions based on instructions, enabling clear observation of fine-grained target objects and detailed information across a wide spatial extent. EyeVLA discretizes action behaviors into action tokens and integrates them with vision-language models (VLMs) that possess strong open-world understanding capabilities, enabling joint modeling of vision, language, and actions within a single autoregressive sequence. By using the 2D bounding box coordinates to guide the reasoning chain and applying reinforcement learning to refine the viewpoint selection policy, we transfer the open-world scene understanding capability of the VLM to a vision language action (VLA) policy using only minimal real-world data. Experiments show that our system efficiently performs instructed scenes in real-world environments and actively acquires more accurate visual information through instruction-driven actions of rotation and zoom, thereby achieving strong environmental perception capabilities. EyeVLA introduces a novel robotic vision system that leverages detailed and spatially rich, large-scale embodied data, and actively acquires highly informative visual observations for downstream embodied tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15279) | **Categories:** cs.RO, cs.CV

---

### [6] [Path Planning through Multi-Agent Reinforcement Learning in Dynamic Environments](https://arxiv.org/abs/2511.15284)
*Jonas De Maeyer, Hossein Yarahmadi, Moharram Challenger*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Path planning in dynamic environments is a fundamental challenge in intelligent transportation and robotics, where obstacles and conditions change over time, introducing uncertainty and requiring continuous adaptation. While existing approaches often assume complete environmental unpredictability or rely on global planners, these assumptions limit scalability and practical deployment in real-world settings. In this paper, we propose a scalable, region-aware reinforcement learning (RL) framework for path planning in dynamic environments. Our method builds on the observation that environmental changes, although dynamic, are often localized within bounded regions. To exploit this, we introduce a hierarchical decomposition of the environment and deploy distributed RL agents that adapt to changes locally. We further propose a retraining mechanism based on sub-environment success rates to determine when policy updates are necessary. Two training paradigms are explored: single-agent Q-learning and multi-agent federated Q-learning, where local Q-tables are aggregated periodically to accelerate the learning process. Unlike prior work, we evaluate our methods in more realistic settings, where multiple simultaneous obstacle changes and increasing difficulty levels are present. Results show that the federated variants consistently outperform their single-agent counterparts and closely approach the performance of A* Oracle while maintaining shorter adaptation times and robust scalability. Although initial training remains time-consuming in large environments, our decentralized framework eliminates the need for a global planner and lays the groundwork for future improvements using deep RL and flexible environment decomposition.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15284) | **Categories:** cs.RO, cs.AI

---

### [7] [RRT*former: Environment-Aware Sampling-Based Motion Planning using Transformer](https://arxiv.org/abs/2511.15414)
*Mingyang Feng, Shaoyuan Li, Xiang Yin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We investigate the sampling-based optimal path planning problem for robotics in complex and dynamic environments. Most existing sampling-based algorithms neglect environmental information or the information from previous samples. Yet, these pieces of information are highly informative, as leveraging them can provide better heuristics when sampling the next state. In this paper, we propose a novel sampling-based planning algorithm, called \emph{RRT*former}, which integrates the standard RRT* algorithm with a Transformer network in a novel way. Specifically, the Transformer is used to extract features from the environment and leverage information from previous samples to better guide the sampling process. Our extensive experiments demonstrate that, compared to existing sampling-based approaches such as RRT*, Neural RRT*, and their variants, our algorithm achieves considerable improvements in both the optimality of the path and sampling efficiency. The code for our implementation is available on https://github.com/fengmingyang666/RRTformer.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15414) | **Categories:** cs.RO, cs.AI

---


## stat.AP [stat.AP]
### [1] [TacEleven: generative tactic discovery for football open play](https://arxiv.org/abs/2511.13326)
*Siyao Zhao, Hao Ma, Zhiqiang Pu, Jingjing Huang, Yi Pan, Shijie Wang, Zhi Ming*

Main category: stat.AP

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Creating offensive advantages during open play is fundamental to football success. However, due to the highly dynamic and long-sequence nature of open play, the potential tactic space grows exponentially as the sequence progresses, making automated tactic discovery extremely challenging. To address this, we propose TacEleven, a generative framework for football open-play tactic discovery developed in close collaboration with domain experts from AJ Auxerre, designed to assist coaches and analysts in tactical decision-making. TacEleven consists of two core components: a language-controlled tactical generator that produces diverse tactical proposals, and a multimodal large language model-based tactical critic that selects the optimal proposal aligned with a high-level stylistic tactical instruction. The two components enables rapid exploration of tactical proposals and discovery of alternative open-play offensive tactics. We evaluate TacEleven across three tasks with progressive tactical complexity: counterfactual exploration, single-step discovery, and multi-step discovery, through both quantitative metrics and a questionnaire-based qualitative assessment. The results show that the TacEleven-discovered tactics exhibit strong realism and tactical creativity, with 52.50% of the multi-step tactical alternatives rated adoptable in real-world elite football scenarios, highlighting the framework's ability to rapidly generate numerous high-quality tactics for complex long-sequence open-play situations. TacEleven demonstrates the potential of creatively leveraging domain data and generative models to advance tactical analysis in sports.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13326) | **Categories:** stat.AP, cs.AI

---
