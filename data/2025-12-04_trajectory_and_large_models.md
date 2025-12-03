# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-12-04

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (7)](#cs-cv)
- [人机交互 (Human-Computer Interaction) (1)](#cs-hc)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (8)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games](https://arxiv.org/abs/2512.02358)
*Ran Zhang, Kun Ouyang, Tiancheng Ma, Yida Yang, Dong Fang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Optimizing numerical systems and mechanism design is crucial for enhancing player experience in Massively Multiplayer Online (MMO) games. Traditional optimization approaches rely on large-scale online experiments or parameter tuning over predefined statistical models, which are costly, time-consuming, and may disrupt player experience. Although simplified offline simulation systems are often adopted as alternatives, their limited fidelity prevents agents from accurately mimicking real player reasoning and reactions to interventions. To address these limitations, we propose a generative agent-based MMO simulation system empowered by Large Language Models (LLMs). By applying Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on large-scale real player behavioral data, we adapt LLMs from general priors to game-specific domains, enabling realistic and interpretable player decision-making. In parallel, a data-driven environment model trained on real gameplay logs reconstructs dynamic in-game systems. Experiments demonstrate strong consistency with real-world player behaviors and plausible causal responses under interventions, providing a reliable, interpretable, and cost-efficient framework for data-driven numerical design optimization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02358) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Multi-Domain Enhanced Map-Free Trajectory Prediction with Selective Attention](https://arxiv.org/abs/2512.02368)
*Wenyi Xiong, Jian Chen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction is crucial for the reliability and safety of autonomous driving systems, yet it remains a challenging task in complex interactive scenarios. Existing methods often struggle to efficiently extract valuable scene information from redundant data, thereby reducing computational efficiency and prediction accuracy, especially when dealing with intricate agent interactions. To address these challenges, we propose a novel map-free trajectory prediction algorithm that achieves trajectory prediction across the temporal, spatial, and frequency domains. Specifically, in temporal information processing, We utilize a Mixture of Experts (MoE) mechanism to adaptively select critical frequency components. Concurrently, we extract these components and integrate multi-scale temporal features. Subsequently, a selective attention module is proposed to filter out redundant information in both temporal sequences and spatial interactions. Finally, we design a multimodal decoder. Under the supervision of patch-level and point-level losses, we obtain reasonable trajectory results. Experiments on Nuscences datasets demonstrate the superiority of our algorithm, validating its effectiveness in handling complex interactive scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02368) | **Categories:** cs.CV, cs.AI

---

### [2] [Progressive Image Restoration via Text-Conditioned Video Generation](https://arxiv.org/abs/2512.02273)
*Peng Kang, Xijun Wang, Yu Yuan*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent text-to-video models have demonstrated strong temporal generation capabilities, yet their potential for image restoration remains underexplored. In this work, we repurpose CogVideo for progressive visual restoration tasks by fine-tuning it to generate restoration trajectories rather than natural video motion. Specifically, we construct synthetic datasets for super-resolution, deblurring, and low-light enhancement, where each sample depicts a gradual transition from degraded to clean frames. Two prompting strategies are compared: a uniform text prompt shared across all samples, and a scene-specific prompting scheme generated via LLaVA multi-modal LLM and refined with ChatGPT. Our fine-tuned model learns to associate temporal progression with restoration quality, producing sequences that improve perceptual metrics such as PSNR, SSIM, and LPIPS across frames. Extensive experiments show that CogVideo effectively restores spatial detail and illumination consistency while maintaining temporal coherence. Moreover, the model generalizes to real-world scenarios on the ReLoBlur dataset without additional training, demonstrating strong zero-shot robustness and interpretability through temporal restoration.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02273) | **Categories:** cs.CV, cs.AI

---

### [3] [nuScenes Revisited: Progress and Challenges in Autonomous Driving](https://arxiv.org/abs/2512.02448)
*Whye Kit Fong, Venice Erin Liong, Kok Seang Tan, Holger Caesar*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization \& mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02448) | **Categories:** cs.CV, cs.RO

---

### [4] [On-the-fly Feedback SfM: Online Explore-and-Exploit UAV Photogrammetry with Incremental Mesh Quality-Aware Indicator and Predictive Path Planning](https://arxiv.org/abs/2512.02375)
*Liyuan Lou, Wanyun Li, Wentian Gan, Yifei Yu, Tengfei Wang, Xin Wang, Zongqian Zhan*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Compared with conventional offline UAV photogrammetry, real-time UAV photogrammetry is essential for time-critical geospatial applications such as disaster response and active digital-twin maintenance. However, most existing methods focus on processing captured images or sequential frames in real time, without explicitly evaluating the quality of the on-the-go 3D reconstruction or providing guided feedback to enhance image acquisition in the target area. This work presents On-the-fly Feedback SfM, an explore-and-exploit framework for real-time UAV photogrammetry, enabling iterative exploration of unseen regions and exploitation of already observed and reconstructed areas in near real time. Built upon SfM on-the-fly , the proposed method integrates three modules: (1) online incremental coarse-mesh generation for dynamically expanding sparse 3D point cloud; (2) online mesh quality assessment with actionable indicators; and (3) predictive path planning for on-the-fly trajectory refinement. Comprehensive experiments demonstrate that our method achieves in-situ reconstruction and evaluation in near real time while providing actionable feedback that markedly reduces coverage gaps and re-flight costs. Via the integration of data collection, processing, 3D reconstruction and assessment, and online feedback, our on the-fly feedback SfM could be an alternative for the transition from traditional passive working mode to a more intelligent and adaptive exploration workflow. Code is now available at https://github.com/IRIS-LAB-whu/OntheflySfMFeedback.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02375) | **Categories:** cs.CV

---

### [5] [From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking](https://arxiv.org/abs/2512.02392)
*Yuqing Shao, Yuchen Yang, Rui Yu, Weilong Li, Xu Guo, Huaicheng Yan, Wei Wang, Xiao Sun*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end multi-object tracking (MOT) methods have recently achieved remarkable progress by unifying detection and association within a single framework. Despite their strong detection performance, these methods suffer from relatively low association accuracy. Through detailed analysis, we observe that object embeddings produced by the shared DETR architecture display excessively high inter-object similarity, as it emphasizes only category-level discrimination within single frames. In contrast, tracking requires instance-level distinction across frames with spatial and temporal continuity, for which current end-to-end approaches insufficiently optimize object embeddings. To address this, we introduce FDTA (From Detection to Association), an explicit feature refinement framework that enhances object discriminativeness across three complementary perspectives. Specifically, we introduce a Spatial Adapter (SA) to integrate depth-aware cues for spatial continuity, a Temporal Adapter (TA) to aggregate historical information for temporal dependencies, and an Identity Adapter (IA) to leverage quality-aware contrastive learning for instance-level separability. Extensive experiments demonstrate that FDTA achieves state-of-the-art performance on multiple challenging MOT benchmarks, including DanceTrack, SportsMOT, and BFT, highlighting the effectiveness of our proposed discriminative embedding enhancement strategy. The code is available at https://github.com/Spongebobbbbbbbb/FDTA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02392) | **Categories:** cs.CV

---

### [6] [Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration](https://arxiv.org/abs/2512.02458)
*Zhongyi Cai, Yi Du, Chen Wang, Yu Kong*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Existing research on indoor embodied tasks typically requires agents to actively explore unknown environments and reason about the scene to achieve a specific goal. However, when deployed in real life, agents often face sequential tasks, where each new sub-task follows the completion of the previous one, and certain sub-tasks may be infeasible, such as searching for a non-existent object. Compared with the single-task setting, the core challenge lies in reusing spatial knowledge accumulated from previous explorations to support subsequent reasoning and exploration. In this work, we investigate this underexplored yet practically significant embodied AI challenge. To evaluate this challenge, we introduce SEER-Bench, a new Sequential Embodied Exploration and Reasoning Benchmark encompassing encompassing two classic embodied tasks: Embodied Question Answering (EQA) and Embodied Multi-modal Navigation (EMN). Building on SEER-Bench, we propose 3DSPMR, a 3D SPatial Memory Reasoning approach that exploits relational, visual, and geometric cues from explored regions to augment Multi-Modal Large Language Models (MLLMs) for reasoning and exploration in sequential embodied tasks. To the best of our knowledge, this is the first work to explicitly incorporate geometric information into MLLM-based spatial understanding and reasoning. Extensive experiments verify that 3DSPMR achieves substantial performance gains on both sequential EQA and EMN tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02458) | **Categories:** cs.CV

---

### [7] [WorldPack: Compressed Memory Improves Spatial Consistency in Video World Modeling](https://arxiv.org/abs/2512.02473)
*Yuta Oshima, Yusuke Iwasawa, Masahiro Suzuki, Yutaka Matsuo, Hiroki Furuta*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video world models have attracted significant attention for their ability to produce high-fidelity future visual observations conditioned on past observations and navigation actions. Temporally- and spatially-consistent, long-term world modeling has been a long-standing problem, unresolved with even recent state-of-the-art models, due to the prohibitively expensive computational costs for long-context inputs. In this paper, we propose WorldPack, a video world model with efficient compressed memory, which significantly improves spatial consistency, fidelity, and quality in long-term generation despite much shorter context length. Our compressed memory consists of trajectory packing and memory retrieval; trajectory packing realizes high context efficiency, and memory retrieval maintains the consistency in rollouts and helps long-term generations that require spatial reasoning. Our performance is evaluated with LoopNav, a benchmark on Minecraft, specialized for the evaluation of long-term consistency, and we verify that WorldPack notably outperforms strong state-of-the-art models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02473) | **Categories:** cs.CV, cs.LG

---


## 人机交互 (Human-Computer Interaction) [cs.HC]
### [1] [Reframing Human-Robot Interaction Through Extended Reality: Unlocking Safer, Smarter, and More Empathic Interactions with Virtual Robots and Foundation Models](https://arxiv.org/abs/2512.02569)
*Yuchong Zhang, Yong Ma, Danica Kragic*

Main category: cs.HC

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This perspective reframes human-robot interaction (HRI) through extended reality (XR), arguing that virtual robots powered by large foundation models (FMs) can serve as cognitively grounded, empathic agents. Unlike physical robots, XR-native agents are unbound by hardware constraints and can be instantiated, adapted, and scaled on demand, while still affording embodiment and co-presence. We synthesize work across XR, HRI, and cognitive AI to show how such agents can support safety-critical scenarios, socially and cognitively empathic interaction across domains, and outreaching physical capabilities with XR and AI integration. We then discuss how multimodal large FMs (e.g., large language model, large vision model, and vision-language model) enable context-aware reasoning, affect-sensitive situations, and long-term adaptation, positioning virtual robots as cognitive and empathic mediators rather than mere simulation assets. At the same time, we highlight challenges and potential risks, including overtrust, cultural and representational bias, privacy concerns around biometric sensing, and data governance and transparency. The paper concludes by outlining a research agenda for human-centered, ethically grounded XR agents - emphasizing multi-layered evaluation frameworks, multi-user ecosystems, mixed virtual-physical embodiment, and societal and ethical design practices to envision XR-based virtual agents powered by FMs as reshaping future HRI into a more efficient and adaptive paradigm.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02569) | **Categories:** cs.HC, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Forecasting MBTA Transit Dynamics: A Performance Benchmarking of Statistical and Machine Learning Models](https://arxiv.org/abs/2512.02336)
*Sai Siddharth Nalamalpu, Kaining Yuan, Aiden Zhou, Eugene Pinsky*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The Massachusetts Bay Transportation Authority (MBTA) is the main public transit provider in Boston, operating multiple means of transport, including trains, subways, and buses. However, the system often faces delays and fluctuations in ridership volume, which negatively affect efficiency and passenger satisfaction. To further understand this phenomenon, this paper compares the performance of existing and unique methods to determine the best approach in predicting gated station entries in the subway system (a proxy for subway usage) and the number of delays in the overall MBTA system. To do so, this research considers factors that tend to affect public transportation, such as day of week, season, pressure, wind speed, average temperature, and precipitation. This paper evaluates the performance of 10 statistical and machine learning models on predicting next-day subway usage. On predicting delay count, the number of models is extended to 11 per day by introducing a self-exciting point process model, representing a unique application of a point-process framework for MBTA delay modeling. This research involves experimenting with the selective inclusion of features to determine feature importance, testing model accuracy via Root Mean Squared Error (RMSE). Remarkably, it is found that providing either day of week or season data has a more substantial benefit to predictive accuracy compared to weather data; in fact, providing weather data generally worsens performance, suggesting a tendency of models to overfit.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02336) | **Categories:** cs.LG

---

### [2] [SeeNav-Agent: Enhancing Vision-Language Navigation with Visual Prompt and Step-Level Policy Optimization](https://arxiv.org/abs/2512.02631)
*Zhengcheng Wang, Zichuan Lin, Yijun Yang, Haobo Fu, Deheng Ye*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Existing Vision-Language Navigation (VLN) agents based on Large Vision-Language Models (LVLMs) often suffer from perception errors, reasoning errors, and planning errors, which significantly hinder their navigation performance. To address these limitations, a novel VLN agent framework, named SeeNav-Agent, is proposed in this work. First, to reduce perception hallucinations of the visual module of the VLN agent, a dual-view Visual Prompt (VP) technique is introduced in the input space, which can also improve the agent's understanding of current spatial states. Subsequently, a novel step-level Reinforcement Fine-Tuning (RFT) method, Step Reward Group Policy Optimization (SRGPO), is designed for the post-training of VLN agents. In SRGPO, we first define verifiable process rewards for the navigation task, and then perform efficient step-level advantage estimation by randomly grouping different navigation steps. SRGPO provides dense reward signals for the reinforcement learning process of the VLN agent and enhances its planning capability. Experimental results on the EmbodiedBench Navigation benchmark indicate that by introducing the zero-shot VP module, the GPT-4.1 achieves a navigation success rate of 86.7%, surpassing the current best LVLM by approximately 20 percentage points (pp). Through post-training based on SRGPO, the Qwen2.5-VL-3B model reaches a navigation success rate of 72.3%, outperforming the best existing LVLM model by 5.6 pp. Moreover, compared to RFT algorithms such as GRPO and GiGPO, the proposed SRGPO demonstrates significant improvements in training stability, convergence efficiency, and generalization capability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02631) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [VLM as Strategist: Adaptive Generation of Safety-critical Testing Scenarios via Guided Diffusion](https://arxiv.org/abs/2512.02844)
*Xinzheng Wu, Junyi Chen, Naiting Zhong, Yong Shen*

Main category: cs.RO

TL;DR: 本文提出了一种结合视觉语言模型（VLM）和自适应引导扩散模型的安全关键测试场景生成框架，能够高效生成逼真、多样且高度交互的安全关键测试场景。


<details>
  <summary>Details</summary>
Motivation: 自动驾驶系统（ADS）的安全部署依赖于全面的测试和评估。然而，能够有效暴露系统漏洞的安全关键场景在现实世界中非常稀疏。现有的场景生成方法在高效构建长尾场景方面面临挑战，难以保证真实性、关键性和互动性，尤其缺乏对被测车辆（VUT）的实时动态响应能力。

Method: 该框架建立了一个三层分层架构，包括用于VLM指导的场景生成目标确定的战略层、用于指导函数制定的战术层和用于引导扩散执行的 операционная 层。首先，建立高质量的基础扩散模型，学习真实驾驶场景的数据分布。其次，设计了一种自适应引导扩散方法，能够实时、精确地控制闭环仿真中的背景车辆（BV）。然后，结合VLM，通过深度场景理解和风险推理，自主生成场景生成目标和指导函数，最终指导扩散模型实现VLM指导的场景生成。

Result: 实验结果表明，该方法能够高效生成逼真、多样且高度交互的安全关键测试场景。案例研究验证了该方法的适应性和VLM指导的生成性能。

Conclusion: 本文提出的安全关键测试场景生成框架，结合了视觉语言模型和自适应引导扩散模型，能够有效解决现有方法在构建安全关键场景时面临的挑战，为自动驾驶系统的安全测试和评估提供了新的解决方案。

Abstract: 本文提出了一种新的安全关键测试场景生成框架，该框架结合了视觉语言模型（VLM）的高层语义理解能力和自适应引导扩散模型的细粒度生成能力。该框架通过三层分层架构实现，包括战略层、战术层和 операционная 层，分别负责场景生成目标确定、指导函数制定和引导扩散执行。实验结果表明，该方法能够高效生成逼真、多样且高度交互的安全关键测试场景。

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02844) | **Categories:** cs.RO, cs.LG

---

### [2] [CogDrive: Cognition-Driven Multimodal Prediction-Planning Fusion for Safe Autonomy](https://arxiv.org/abs/2512.02777)
*Heye Huang, Yibin Yang, Mingfeng Fan, Haoran Wang, Xiaocong Zhao, Jianqiang Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe autonomous driving in mixed traffic requires a unified understanding of multimodal interactions and dynamic planning under uncertainty. Existing learning based approaches struggle to capture rare but safety critical behaviors, while rule based systems often lack adaptability in complex interactions. To address these limitations, CogDrive introduces a cognition driven multimodal prediction and planning framework that integrates explicit modal reasoning with safety aware trajectory optimization. The prediction module adopts cognitive representations of interaction modes based on topological motion semantics and nearest neighbor relational encoding. With a differentiable modal loss and multimodal Gaussian decoding, CogDrive learns sparse and unbalanced interaction behaviors and improves long horizon trajectory prediction. The planning module incorporates an emergency response concept and optimizes safety stabilized trajectories, where short term consistent branches ensure safety during replanning cycles and long term branches support smooth and collision free motion under low probability switching modes. Experiments on Argoverse2 and INTERACTION datasets show that CogDrive achieves strong performance in trajectory accuracy and miss rate, while closed loop simulations confirm adaptive behavior in merge and intersection scenarios. By combining cognitive multimodal prediction with safety oriented planning, CogDrive offers an interpretable and reliable paradigm for safe autonomy in complex traffic.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02777) | **Categories:** cs.RO, cs.MA

---

### [3] [SwarmDiffusion: End-To-End Traversability-Guided Diffusion for Embodiment-Agnostic Navigation of Heterogeneous Robots](https://arxiv.org/abs/2512.02851)
*Iana Zhura, Sausar Karaf, Faryal Batool, Nipun Dhananjaya Weerakkodi Mudalige, Valerii Serpiva, Ali Alridha Abdulkarim, Aleksey Fedoseev, Didar Seyidov, Amjad Hajira, Dzmitry Tsetserukou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Visual traversability estimation is critical for autonomous navigation, but existing VLM-based methods rely on hand-crafted prompts, generalize poorly across embodiments, and output only traversability maps, leaving trajectory generation to slow external planners. We propose SwarmDiffusion, a lightweight end-to-end diffusion model that jointly predicts traversability and generates a feasible trajectory from a single RGB image. To remove the need for annotated or planner-produced paths, we introduce a planner-free trajectory construction pipeline based on randomized waypoint sampling, Bezier smoothing, and regularization enforcing connectivity, safety, directionality, and path thinness. This enables learning stable motion priors without demonstrations. SwarmDiffusion leverages VLM-derived supervision without prompt engineering and conditions the diffusion process on a compact embodiment state, producing physically consistent, traversable paths that transfer across different robot platforms. Across indoor environments and two embodiments (quadruped and aerial), the method achieves 80-100\% navigation success and 0.09 s inference, and adapts to a new robot using only-500 additional visual samples. It generalizes reliably to unseen environments in simulation and real-world trials, offering a scalable, prompt-free approach to unified traversability reasoning and trajectory generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02851) | **Categories:** cs.RO

---

### [4] [Vehicle Dynamics Embedded World Models for Autonomous Driving](https://arxiv.org/abs/2512.02417)
*Huiqian Li, Wei Pan, Haodong Zhang, Jin Huang, Zhihua Zhong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: World models have gained significant attention as a promising approach for autonomous driving. By emulating human-like perception and decision-making processes, these models can predict and adapt to dynamic environments. Existing methods typically map high-dimensional observations into compact latent spaces and learn optimal policies within these latent representations. However, prior work usually jointly learns ego-vehicle dynamics and environmental transition dynamics from the image input, leading to inefficiencies and a lack of robustness to variations in vehicle dynamics. To address these issues, we propose the Vehicle Dynamics embedded Dreamer (VDD) method, which decouples the modeling of ego-vehicle dynamics from environmental transition dynamics. This separation allows the world model to generalize effectively across vehicles with diverse parameters. Additionally, we introduce two strategies to further enhance the robustness of the learned policy: Policy Adjustment during Deployment (PAD) and Policy Augmentation during Training (PAT). Comprehensive experiments in simulated environments demonstrate that the proposed model significantly improves both driving performance and robustness to variations in vehicle dynamics, outperforming existing approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02417) | **Categories:** cs.RO, cs.AI

---

### [5] [AID: Agent Intent from Diffusion for Multi-Agent Informative Path Planning](https://arxiv.org/abs/2512.02535)
*Jeric Lew, Yuhong Cao, Derek Ming Siang Tan, Guillaume Sartoretti*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Information gathering in large-scale or time-critical scenarios (e.g., environmental monitoring, search and rescue) requires broad coverage within limited time budgets, motivating the use of multi-agent systems. These scenarios are commonly formulated as multi-agent informative path planning (MAIPP), where multiple agents must coordinate to maximize information gain while operating under budget constraints. A central challenge in MAIPP is ensuring effective coordination while the belief over the environment evolves with incoming measurements. Recent learning-based approaches address this by using distributions over future positions as "intent" to support coordination. However, these autoregressive intent predictors are computationally expensive and prone to compounding errors. Inspired by the effectiveness of diffusion models as expressive, long-horizon policies, we propose AID, a fully decentralized MAIPP framework that leverages diffusion models to generate long-term trajectories in a non-autoregressive manner. AID first performs behavior cloning on trajectories produced by existing MAIPP planners and then fine-tunes the policy using reinforcement learning via Diffusion Policy Policy Optimization (DPPO). This two-stage pipeline enables the policy to inherit expert behavior while learning improved coordination through online reward feedback. Experiments demonstrate that AID consistently improves upon the MAIPP planners it is trained from, achieving up to 4x faster execution and 17% increased information gain, while scaling effectively to larger numbers of agents. Our implementation is publicly available at https://github.com/marmotlab/AID.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02535) | **Categories:** cs.RO

---

### [6] [SAM2Grasp: Resolve Multi-modal Grasping via Prompt-conditioned Temporal Action Prediction](https://arxiv.org/abs/2512.02609)
*Shengkai Wu, Jinrong Yang, Wenqiu Luo, Linfeng Gao, Chaohui Shang, Meiyu Zhi, Mingshan Sun, Fangping Yang, Liangliang Ren, Yong Zhao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Imitation learning for robotic grasping is often plagued by the multimodal problem: when a scene contains multiple valid targets, demonstrations of grasping different objects create conflicting training signals. Standard imitation learning policies fail by averaging these distinct actions into a single, invalid action. In this paper, we introduce SAM2Grasp, a novel framework that resolves this issue by reformulating the task as a uni-modal, prompt-conditioned prediction problem. Our method leverages the frozen SAM2 model to use its powerful visual temporal tracking capability and introduces a lightweight, trainable action head that operates in parallel with its native segmentation head. This design allows for training only the small action head on pre-computed temporal-visual features from SAM2. During inference, an initial prompt, such as a bounding box provided by an upstream object detection model, designates the specific object to be grasped. This prompt conditions the action head to predict a unique, unambiguous grasp trajectory for that object alone. In all subsequent video frames, SAM2's built-in temporal tracking capability automatically maintains stable tracking of the selected object, enabling our model to continuously predict the grasp trajectory from the video stream without further external guidance. This temporal-prompted approach effectively eliminates ambiguity from the visuomotor policy. We demonstrate through extensive experiments that SAM2Grasp achieves state-of-the-art performance in cluttered, multi-object grasping tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02609) | **Categories:** cs.RO, cs.CV

---

### [7] [Steering Vision-Language-Action Models as Anti-Exploration: A Test-Time Scaling Approach](https://arxiv.org/abs/2512.02834)
*Siyuan Yang, Yang Zhang, Haoran He, Ling Pan, Xiu Li, Chenjia Bai, Xuelong Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models, trained via flow-matching or diffusion objectives, excel at learning complex behaviors from large-scale, multi-modal datasets (e.g., human teleoperation, scripted policies). However, since VLAs incorporate diverse data modes in the pre-training stage, and the finetuning dataset often contains demonstration data collected in a kinematically suboptimal or undesirable way, it exists redundant action modes that are irrelevant to the success action modes of the downstream task. Specifically, we observe a critical inference-time fragility among various sampled noises after supervised finetuning of pre-trained VLAs. In this paper, we attribute this instability to the distribution shift between the VLA policy and the policy induced by stable success modes of the downstream task dataset. Thus, we propose \textbf{TACO}, a test-time-scaling (TTS) framework that applies a lightweight pseudo-count estimator as a high-fidelity verifier of action chunks. The VLA models integrated with TACO can execute the actions with maximum pseudo-count from all sampled action chunks, thereby preventing distribution shifts while preserving the generalization ability of VLAs since the constraint is applied only during inference. Our method resembles the classical anti-exploration principle in offline reinforcement learning (RL), and being gradient-free, it incurs significant computational benefits compared to RL update, especially for flow or diffusion-based VLAs which are difficult to perform RL update due to denoising process. Extensive experiments across four simulation benchmarks (RoboTwin2.0, Robotwin, LIBERO, SimplerEnv) and a dual-arm platform demonstrate that our method significantly improves the inference stability and success rates in downstream-task adaptations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.02834) | **Categories:** cs.RO, cs.AI

---

### [8] [Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling](https://arxiv.org/abs/2512.03044)
*Yueru Jia, Jiaming Liu, Shengbang Liu, Rui Zhou, Wanhe Yu, Yuyang Yan, Xiaowei Chi, Yandong Guo, Boxin Shi, Shanghang Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robust perception and dynamics modeling are fundamental to real-world robotic policy learning. Recent methods employ video diffusion models (VDMs) to enhance robotic policies, improving their understanding and modeling of the physical world. However, existing approaches overlook the coherent and physically consistent motion representations inherently encoded across frames in VDMs. To this end, we propose Video2Act, a framework that efficiently guides robotic action learning by explicitly integrating spatial and motion-aware representations. Building on the inherent representations of VDMs, we extract foreground boundaries and inter-frame motion variations while filtering out background noise and task-irrelevant biases. These refined representations are then used as additional conditioning inputs to a diffusion transformer (DiT) action head, enabling it to reason about what to manipulate and how to move. To mitigate inference inefficiency, we propose an asynchronous dual-system design, where the VDM functions as the slow System 2 and the DiT head as the fast System 1, working collaboratively to generate adaptive actions. By providing motion-aware conditions to System 1, Video2Act maintains stable manipulation even with low-frequency updates from the VDM. For evaluation, Video2Act surpasses previous state-of-the-art VLA methods by 7.7% in simulation and 21.7% in real-world tasks in terms of average success rate, further exhibiting strong generalization capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.03044) | **Categories:** cs.RO

---
