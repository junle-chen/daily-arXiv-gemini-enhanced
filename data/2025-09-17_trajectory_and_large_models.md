# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-17

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [HeLoFusion: An Efficient and Scalable Encoder for Modeling Heterogeneous and Multi-Scale Interactions in Trajectory Prediction](https://arxiv.org/abs/2509.11719)
*Bingqing Wei, Lianmin Chen, Zhongyu Xia, Yongtao Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-agent trajectory prediction in autonomous driving requires a comprehensive understanding of complex social dynamics. Existing methods, however, often struggle to capture the full richness of these dynamics, particularly the co-existence of multi-scale interactions and the diverse behaviors of heterogeneous agents. To address these challenges, this paper introduces HeLoFusion, an efficient and scalable encoder for modeling heterogeneous and multi-scale agent interactions. Instead of relying on global context, HeLoFusion constructs local, multi-scale graphs centered on each agent, allowing it to effectively model both direct pairwise dependencies and complex group-wise interactions (\textit{e.g.}, platooning vehicles or pedestrian crowds). Furthermore, HeLoFusion tackles the critical challenge of agent heterogeneity through an aggregation-decomposition message-passing scheme and type-specific feature networks, enabling it to learn nuanced, type-dependent interaction patterns. This locality-focused approach enables a principled representation of multi-level social context, yielding powerful and expressive agent embeddings. On the challenging Waymo Open Motion Dataset, HeLoFusion achieves state-of-the-art performance, setting new benchmarks for key metrics including Soft mAP and minADE. Our work demonstrates that a locality-grounded architecture, which explicitly models multi-scale and heterogeneous interactions, is a highly effective strategy for advancing motion forecasting.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11719) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Traffic-MLLM: A Spatio-Temporal MLLM with Retrieval-Augmented Generation for Causal Inference in Traffic](https://arxiv.org/abs/2509.11165)
*Waikit Xiu, Qiang Lu, Xiying Li, Chen Hu, Shengbo Sun*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: As intelligent transportation systems advance, traffic video understanding plays an increasingly pivotal role in comprehensive scene perception and causal analysis. Yet, existing approaches face notable challenges in accurately modeling spatiotemporal causality and integrating domain-specific knowledge, limiting their effectiveness in complex scenarios. To address these limitations, we propose Traffic-MLLM, a multimodal large language model tailored for fine-grained traffic analysis. Built on the Qwen2.5-VL backbone, our model leverages high-quality traffic-specific multimodal datasets and uses Low-Rank Adaptation (LoRA) for lightweight fine-tuning, significantly enhancing its capacity to model continuous spatiotemporal features in video sequences. Furthermore, we introduce an innovative knowledge prompting module fusing Chain-of-Thought (CoT) reasoning with Retrieval-Augmented Generation (RAG), enabling precise injection of detailed traffic regulations and domain knowledge into the inference process. This design markedly boosts the model's logical reasoning and knowledge adaptation capabilities. Experimental results on TrafficQA and DriveQA benchmarks show Traffic-MLLM achieves state-of-the-art performance, validating its superior ability to process multimodal traffic data. It also exhibits remarkable zero-shot reasoning and cross-scenario generalization capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11165) | **Categories:** cs.CV

---

### [2] [Action Hints: Semantic Typicality and Context Uniqueness for Generalizable Skeleton-based Video Anomaly Detection](https://arxiv.org/abs/2509.11058)
*Canhui Tang, Sanping Zhou, Haoyue Shi, Le Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Zero-Shot Video Anomaly Detection (ZS-VAD) requires temporally localizing anomalies without target domain training data, which is a crucial task due to various practical concerns, e.g., data privacy or new surveillance deployments. Skeleton-based approach has inherent generalizable advantages in achieving ZS-VAD as it eliminates domain disparities both in background and human appearance. However, existing methods only learn low-level skeleton representation and rely on the domain-limited normality boundary, which cannot generalize well to new scenes with different normal and abnormal behavior patterns. In this paper, we propose a novel zero-shot video anomaly detection framework, unlocking the potential of skeleton data via action typicality and uniqueness learning. Firstly, we introduce a language-guided semantic typicality modeling module that projects skeleton snippets into action semantic space and distills LLM's knowledge of typical normal and abnormal behaviors during training. Secondly, we propose a test-time context uniqueness analysis module to finely analyze the spatio-temporal differences between skeleton snippets and then derive scene-adaptive boundaries. Without using any training samples from the target domain, our method achieves state-of-the-art results against skeleton-based methods on four large-scale VAD datasets: ShanghaiTech, UBnormal, NWPU, and UCF-Crime, featuring over 100 unseen surveillance scenes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11058) | **Categories:** cs.CV

---

### [3] [The System Description of CPS Team for Track on Driving with Language of CVPR 2024 Autonomous Grand Challenge](https://arxiv.org/abs/2509.11071)
*Jinghan Peng, Jingwen Wang, Xing Yu, Dehui Du*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This report outlines our approach using vision language model systems for the Driving with Language track of the CVPR 2024 Autonomous Grand Challenge. We have exclusively utilized the DriveLM-nuScenes dataset for training our models. Our systems are built on the LLaVA models, which we enhanced through fine-tuning with the LoRA and DoRA methods. Additionally, we have integrated depth information from open-source depth estimation models to enrich the training and inference processes. For inference, particularly with multiple-choice and yes/no questions, we adopted a Chain-of-Thought reasoning approach to improve the accuracy of the results. This comprehensive methodology enabled us to achieve a top score of 0.7799 on the validation set leaderboard, ranking 1st on the leaderboard.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11071) | **Categories:** cs.CV, cs.AI, cs.CL

---

### [4] [End-to-End Visual Autonomous Parking via Control-Aided Attention](https://arxiv.org/abs/2509.11090)
*Chao Chen, Shunyu Yao, Yuanwu He, Tao Feng, Ruojing Song, Yuliang Guo, Xinyu Huang, Chenxu Wu, Ren Liu, Chen Feng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Precise parking requires an end-to-end system where perception adaptively provides policy-relevant details-especially in critical areas where fine control decisions are essential. End-to-end learning offers a unified framework by directly mapping sensor inputs to control actions, but existing approaches lack effective synergy between perception and control. We find that transformer-based self-attention, when used alone, tends to produce unstable and temporally inconsistent spatial attention, which undermines the reliability of downstream policy decisions over time. Instead, we propose CAA-Policy, an end-to-end imitation learning system that allows control signal to guide the learning of visual attention via a novel Control-Aided Attention (CAA) mechanism. For the first time, we train such an attention module in a self-supervised manner, using backpropagated gradients from the control outputs instead of from the training loss. This strategy encourages the attention to focus on visual features that induce high variance in action outputs, rather than merely minimizing the training loss-a shift we demonstrate leads to a more robust and generalizable policy. To further enhance stability, CAA-Policy integrates short-horizon waypoint prediction as an auxiliary task, and introduces a separately trained motion prediction module to robustly track the target spot over time. Extensive experiments in the CARLA simulator show that \titlevariable~consistently surpasses both the end-to-end learning baseline and the modular BEV segmentation + hybrid A* pipeline, achieving superior accuracy, robustness, and interpretability. Code is released at https://github.com/Joechencc/CAAPolicy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11090) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Multimodal Deep Learning for ATCO Command Lifecycle Modeling and Workload Prediction](https://arxiv.org/abs/2509.10522)
*Kaizhen Tan*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Air traffic controllers (ATCOs) issue high-intensity voice commands in dense airspace, where accurate workload modeling is critical for safety and efficiency. This paper proposes a multimodal deep learning framework that integrates structured data, trajectory sequences, and image features to estimate two key parameters in the ATCO command lifecycle: the time offset between a command and the resulting aircraft maneuver, and the command duration. A high-quality dataset was constructed, with maneuver points detected using sliding window and histogram-based methods. A CNN-Transformer ensemble model was developed for accurate, generalizable, and interpretable predictions. By linking trajectories to voice commands, this work offers the first model of its kind to support intelligent command generation and provides practical value for workload assessment, staffing, and scheduling.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10522) | **Categories:** cs.LG, cs.AI, cs.CV, eess.AS

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Large Foundation Models for Trajectory Prediction in Autonomous Driving: A Comprehensive Survey](https://arxiv.org/abs/2509.10570)
*Wei Dai, Shengen Wu, Wei Wu, Zhenhao Wang, Sisuo Lyu, Haicheng Liao, Limin Yu, Weiping Ding, Runwei Guan, Yutao Yue*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction serves as a critical functionality in autonomous driving, enabling the anticipation of future motion paths for traffic participants such as vehicles and pedestrians, which is essential for driving safety. Although conventional deep learning methods have improved accuracy, they remain hindered by inherent limitations, including lack of interpretability, heavy reliance on large-scale annotated data, and weak generalization in long-tail scenarios. The rise of Large Foundation Models (LFMs) is transforming the research paradigm of trajectory prediction. This survey offers a systematic review of recent advances in LFMs, particularly Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) for trajectory prediction. By integrating linguistic and scene semantics, LFMs facilitate interpretable contextual reasoning, significantly enhancing prediction safety and generalization in complex environments. The article highlights three core methodologies: trajectory-language mapping, multimodal fusion, and constraint-based reasoning. It covers prediction tasks for both vehicles and pedestrians, evaluation metrics, and dataset analyses. Key challenges such as computational latency, data scarcity, and real-world robustness are discussed, along with future research directions including low-latency inference, causality-aware modeling, and motion foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10570) | **Categories:** cs.RO, cs.AI

---

### [2] [DreamNav: A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation](https://arxiv.org/abs/2509.11197)
*Yunheng Wang, Yuetong Fang, Taowen Wang, Yixiao Feng, Yawen Tan, Shuning Zhang, Peiran Liu, Yiding Ji, Renjing Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-and-Language Navigation in Continuous Environments (VLN-CE), which links language instructions to perception and control in the real world, is a core capability of embodied robots. Recently, large-scale pretrained foundation models have been leveraged as shared priors for perception, reasoning, and action, enabling zero-shot VLN without task-specific training. However, existing zero-shot VLN methods depend on costly perception and passive scene understanding, collapsing control to point-level choices. As a result, they are expensive to deploy, misaligned in action semantics, and short-sighted in planning. To address these issues, we present DreamNav that focuses on the following three aspects: (1) for reducing sensory cost, our EgoView Corrector aligns viewpoints and stabilizes egocentric perception; (2) instead of point-level actions, our Trajectory Predictor favors global trajectory-level planning to better align with instruction semantics; and (3) to enable anticipatory and long-horizon planning, we propose an Imagination Predictor to endow the agent with proactive thinking capability. On VLN-CE and real-world tests, DreamNav sets a new zero-shot state-of-the-art (SOTA), outperforming the strongest egocentric baseline with extra information by up to 7.49\% and 18.15\% in terms of SR and SPL metrics. To our knowledge, this is the first zero-shot VLN method to unify trajectory-level planning and active imagination while using only egocentric inputs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11197) | **Categories:** cs.RO, cs.AI, cs.CL, cs.CV

---

### [3] [Follow-Bench: A Unified Motion Planning Benchmark for Socially-Aware Robot Person Following](https://arxiv.org/abs/2509.10796)
*Hanjing Ye, Weixi Situ, Jianwei Peng, Yu Zhan, Bingyi Xia, Kuanqi Cai, Hong Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robot person following (RPF) -- mobile robots that follow and assist a specific person -- has emerging applications in personal assistance, security patrols, eldercare, and logistics. To be effective, such robots must follow the target while ensuring safety and comfort for both the target and surrounding people. In this work, we present the first end-to-end study of RPF, which (i) surveys representative scenarios, motion-planning methods, and evaluation metrics with a focus on safety and comfort; (ii) introduces Follow-Bench, a unified benchmark simulating diverse scenarios, including various target trajectory patterns, dynamic-crowd flows, and environmental layouts; and (iii) re-implements six popular RPF planners, ensuring that both safety and comfort are systematically considered. Moreover, we evaluate the two highest-performing planners from our benchmark on a differential-drive robot to provide insights into real-world deployment. Extensive simulation and real-world experiments provide quantitative insights into the safety-comfort trade-offs of existing planners, while revealing open challenges and future research directions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10796) | **Categories:** cs.RO

---

### [4] [Nav-R1: Reasoning and Navigation in Embodied Scenes](https://arxiv.org/abs/2509.10884)
*Qingxiang Liu, Ting Huang, Zeyu Zhang, Hao Tang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied navigation requires agents to integrate perception, reasoning, and action for robust interaction in complex 3D environments. Existing approaches often suffer from incoherent and unstable reasoning traces that hinder generalization across diverse environments, and difficulty balancing long-horizon semantic reasoning with low-latency control for real-time navigation. To address these challenges, we propose Nav-R1, an embodied foundation model that unifies reasoning in embodied environments. We first construct Nav-CoT-110K, a large-scale dataset of step-by-step Chains-of-Thought (CoT) for embodied tasks, which enables cold-start initialization with structured reasoning. Building on this foundation, we design a GRPO-based reinforcement learning framework with three complementary rewards: format, understanding, and navigation, to improve structural adherence, semantic grounding, and path fidelity. Furthermore, we introduce a Fast-in-Slow reasoning paradigm, decoupling deliberate semantic reasoning from low-latency reactive control for efficient yet coherent navigation. Extensive evaluations on embodied AI benchmarks demonstrate that Nav-R1 consistently outperforms strong baselines, with over 8% average improvement in reasoning and navigation performance. Real-world deployment on a mobile robot further validates its robustness under limited onboard resources. Code: https://github.com/AIGeeksGroup/Nav-R1. Website: https://aigeeksgroup.github.io/Nav-R1.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10884) | **Categories:** cs.RO, cs.CV

---

### [5] [RoVerFly: Robust and Versatile Learning-based Control of Quadrotor Across Payload Configurations](https://arxiv.org/abs/2509.11149)
*Mintae Kim, Jiaze Cai, Koushil Sreenath*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Designing robust controllers for precise, arbitrary trajectory tracking with quadrotors is challenging due to nonlinear dynamics and underactuation, and becomes harder with flexible cable-suspended payloads that introduce extra degrees of freedom and hybridness. Classical model-based methods offer stability guarantees but require extensive tuning and often do not adapt when the configuration changes, such as when a payload is added or removed, or when the payload mass or cable length varies. We present RoVerFly, a unified learning-based control framework in which a reinforcement learning (RL) policy serves as a robust and versatile tracking controller for standard quadrotors and for cable-suspended payload systems across a range of configurations. Trained with task and domain randomization, the controller is resilient to disturbances and varying dynamics. It achieves strong zero-shot generalization across payload settings, including no payload as well as varying mass and cable length, without controller switching or re-tuning, while retaining the interpretability and structure of a feedback tracking controller. Code and supplementary materials are available at https://github.com/mintaeshkim/roverfly

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11149) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [6] [CORB-Planner: Corridor as Observations for RL Planning in High-Speed Flight](https://arxiv.org/abs/2509.11240)
*Yechen Zhang, Bin Gao, Gang Wang, Jian Sun, Zhuo Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) has shown promise in a large number of robotic control tasks. Nevertheless, its deployment on unmanned aerial vehicles (UAVs) remains challenging, mainly because of reliance on accurate dynamic models and platform-specific sensing, which hinders cross-platform transfer. This paper presents the CORB-Planner (Corridor-as-Observations for RL B-spline planner), a real-time, RL-based trajectory planning framework for high-speed autonomous UAV flight across heterogeneous platforms. The key idea is to combine B-spline trajectory generation with the RL policy producing successive control points with a compact safe flight corridor (SFC) representation obtained via heuristic search. The SFC abstracts obstacle information in a low-dimensional form, mitigating overfitting to platform-specific details and reducing sensitivity to model inaccuracies. To narrow the sim-to-real gap, we adopt an easy-to-hard progressive training pipeline in simulation. A value-based soft decomposed-critic Q (SDCQ) algorithm is used to learn effective policies within approximately ten minutes of training. Benchmarks in simulation and real-world tests demonstrate real-time planning on lightweight onboard hardware and support maximum flight speeds up to 8.2m/s in dense, cluttered environments without external positioning. Compatibility with various UAV configurations (quadrotors, hexarotors) and modest onboard compute underlines the generality and robustness of CORB-Planner for practical deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11240) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [7] [ActivePose: Active 6D Object Pose Estimation and Tracking for Robotic Manipulation](https://arxiv.org/abs/2509.11364)
*Sheng Liu, Zhe Li, Weiheng Wang, Han Sun, Heng Zhang, Hongpeng Chen, Yusen Qin, Arash Ajoudani, Yizhao Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate 6-DoF object pose estimation and tracking are critical for reliable robotic manipulation. However, zero-shot methods often fail under viewpoint-induced ambiguities and fixed-camera setups struggle when objects move or become self-occluded. To address these challenges, we propose an active pose estimation pipeline that combines a Vision-Language Model (VLM) with "robotic imagination" to dynamically detect and resolve ambiguities in real time. In an offline stage, we render a dense set of views of the CAD model, compute the FoundationPose entropy for each view, and construct a geometric-aware prompt that includes low-entropy (unambiguous) and high-entropy (ambiguous) examples. At runtime, the system: (1) queries the VLM on the live image for an ambiguity score; (2) if ambiguity is detected, imagines a discrete set of candidate camera poses by rendering virtual views, scores each based on a weighted combination of VLM ambiguity probability and FoundationPose entropy, and then moves the camera to the Next-Best-View (NBV) to obtain a disambiguated pose estimation. Furthermore, since moving objects may leave the camera's field of view, we introduce an active pose tracking module: a diffusion-policy trained via imitation learning, which generates camera trajectories that preserve object visibility and minimize pose ambiguity. Experiments in simulation and real-world show that our approach significantly outperforms classical baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11364) | **Categories:** cs.RO

---

### [8] [Igniting VLMs toward the Embodied Space](https://arxiv.org/abs/2509.11766)
*Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, Lucy Liang, Make Wang, Qian Wang, Roy Gan, Ryan Yu, Shalfun Li, Starrick Liu, Sylas Chen, Vincent Chen, Zach Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While foundation models show remarkable progress in language and vision, existing vision-language models (VLMs) still have limited spatial and embodiment understanding. Transferring VLMs to embodied domains reveals fundamental mismatches between modalities, pretraining distributions, and training objectives, leaving action comprehension and generation as a central bottleneck on the path to AGI.   We introduce WALL-OSS, an end-to-end embodied foundation model that leverages large-scale multimodal pretraining to achieve (1) embodiment-aware vision-language understanding, (2) strong language-action association, and (3) robust manipulation capability.   Our approach employs a tightly coupled architecture and multi-strategies training curriculum that enables Unified Cross-Level CoT-seamlessly unifying instruction reasoning, subgoal decomposition, and fine-grained action synthesis within a single differentiable framework.   Our results show that WALL-OSS attains high success on complex long-horizon manipulations, demonstrates strong instruction-following capabilities, complex understanding and reasoning, and outperforms strong baselines, thereby providing a reliable and scalable path from VLMs to embodied foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11766) | **Categories:** cs.RO

---

### [9] [TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning](https://arxiv.org/abs/2509.11839)
*Jiacheng Liu, Pengxiang Ding, Qihang Zhou, Yuxuan Wu, Da Huang, Zimian Peng, Wei Xiao, Weinan Zhang, Lixin Yang, Cewu Lu, Donglin Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Imitation learning (IL) enables efficient skill acquisition from demonstrations but often struggles with long-horizon tasks and high-precision control due to compounding errors. Residual policy learning offers a promising, model-agnostic solution by refining a base policy through closed-loop corrections. However, existing approaches primarily focus on local corrections to the base policy, lacking a global understanding of state evolution, which limits robustness and generalization to unseen scenarios. To address this, we propose incorporating global dynamics modeling to guide residual policy updates. Specifically, we leverage Koopman operator theory to impose linear time-invariant structure in a learned latent space, enabling reliable state transitions and improved extrapolation for long-horizon prediction and unseen environments. We introduce KORR (Koopman-guided Online Residual Refinement), a simple yet effective framework that conditions residual corrections on Koopman-predicted latent states, enabling globally informed and stable action refinement. We evaluate KORR on long-horizon, fine-grained robotic furniture assembly tasks under various perturbations. Results demonstrate consistent gains in performance, robustness, and generalization over strong baselines. Our findings further highlight the potential of Koopman-based modeling to bridge modern learning methods with classical control theory. For more details, please refer to https://jiachengliu3.github.io/TrajBooster.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.11839) | **Categories:** cs.RO, cs.CV

---
