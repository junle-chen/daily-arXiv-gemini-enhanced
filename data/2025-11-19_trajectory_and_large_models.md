# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-19

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [cs.DC (1)](#cs-dc)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (16)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Event-CausNet: Unlocking Causal Knowledge from Text with Large Language Models for Reliable Spatio-Temporal Forecasting](https://arxiv.org/abs/2511.12769)
*Luyao Niu, Zepu Wang, Shuyi Guan, Yang Liu, Peng Sun*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While spatio-temporal Graph Neural Networks (GNNs) excel at modeling recurring traffic patterns, their reliability plummets during non-recurring events like accidents. This failure occurs because GNNs are fundamentally correlational models, learning historical patterns that are invalidated by the new causal factors introduced during disruptions. To address this, we propose Event-CausNet, a framework that uses a Large Language Model to quantify unstructured event reports, builds a causal knowledge base by estimating average treatment effects, and injects this knowledge into a dual-stream GNN-LSTM network using a novel causal attention mechanism to adjust and enhance the forecast. Experiments on a real-world dataset demonstrate that Event-CausNet achieves robust performance, reducing prediction error (MAE) by up to 35.87%, significantly outperforming state-of-the-art baselines. Our framework bridges the gap between correlational models and causal reasoning, providing a solution that is more accurate and transferable, while also offering crucial interpretability, providing a more reliable foundation for real-world traffic management during critical disruptions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12769) | **Categories:** cs.AI, cs.LG

---

### [2] [ViTE: Virtual Graph Trajectory Expert Router for Pedestrian Trajectory Prediction](https://arxiv.org/abs/2511.12214)
*Ruochen Li, Zhanxing Zhu, Tanqiu Qiao, Hubert P. H. Shum*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Pedestrian trajectory prediction is critical for ensuring safety in autonomous driving, surveillance systems, and urban planning applications. While early approaches primarily focus on one-hop pairwise relationships, recent studies attempt to capture high-order interactions by stacking multiple Graph Neural Network (GNN) layers. However, these approaches face a fundamental trade-off: insufficient layers may lead to under-reaching problems that limit the model's receptive field, while excessive depth can result in prohibitive computational costs. We argue that an effective model should be capable of adaptively modeling both explicit one-hop interactions and implicit high-order dependencies, rather than relying solely on architectural depth. To this end, we propose ViTE (Virtual graph Trajectory Expert router), a novel framework for pedestrian trajectory prediction. ViTE consists of two key modules: a Virtual Graph that introduces dynamic virtual nodes to model long-range and high-order interactions without deep GNN stacks, and an Expert Router that adaptively selects interaction experts based on social context using a Mixture-of-Experts design. This combination enables flexible and scalable reasoning across varying interaction patterns. Experiments on three benchmarks (ETH/UCY, NBA, and SDD) demonstrate that our method consistently achieves state-of-the-art performance, validating both its effectiveness and practical efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12214) | **Categories:** cs.AI

---

### [3] [Mobile-Agent-RAG: Driving Smart Multi-Agent Coordination with Contextual Knowledge Empowerment for Long-Horizon Mobile Automation](https://arxiv.org/abs/2511.12254)
*Yuxiang Zhou, Jichang Li, Yanhao Zhang, Haonan Lu, Guanbin Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Mobile agents show immense potential, yet current state-of-the-art (SoTA) agents exhibit inadequate success rates on real-world, long-horizon, cross-application tasks. We attribute this bottleneck to the agents' excessive reliance on static, internal knowledge within MLLMs, which leads to two critical failure points: 1) strategic hallucinations in high-level planning and 2) operational errors during low-level execution on user interfaces (UI). The core insight of this paper is that high-level planning and low-level UI operations require fundamentally distinct types of knowledge. Planning demands high-level, strategy-oriented experiences, whereas operations necessitate low-level, precise instructions closely tied to specific app UIs. Motivated by these insights, we propose Mobile-Agent-RAG, a novel hierarchical multi-agent framework that innovatively integrates dual-level retrieval augmentation. At the planning stage, we introduce Manager-RAG to reduce strategic hallucinations by retrieving human-validated comprehensive task plans that provide high-level guidance. At the execution stage, we develop Operator-RAG to improve execution accuracy by retrieving the most precise low-level guidance for accurate atomic actions, aligned with the current app and subtask. To accurately deliver these knowledge types, we construct two specialized retrieval-oriented knowledge bases. Furthermore, we introduce Mobile-Eval-RAG, a challenging benchmark for evaluating such agents on realistic multi-app, long-horizon tasks. Extensive experiments demonstrate that Mobile-Agent-RAG significantly outperforms SoTA baselines, improving task completion rate by 11.0% and step efficiency by 10.2%, establishing a robust paradigm for context-aware, reliable multi-agent mobile automation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12254) | **Categories:** cs.AI, cs.IR

---

### [4] [Adaptively Coordinating with Novel Partners via Learned Latent Strategies](https://arxiv.org/abs/2511.12754)
*Benjamin Li, Shuyang Shi, Lucia Romero, Huao Li, Yaqi Xie, Woojun Kim, Stefanos Nikolaidis, Michael Lewis, Katia Sycara, Simon Stepputtis*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Adaptation is the cornerstone of effective collaboration among heterogeneous team members. In human-agent teams, artificial agents need to adapt to their human partners in real time, as individuals often have unique preferences and policies that may change dynamically throughout interactions. This becomes particularly challenging in tasks with time pressure and complex strategic spaces, where identifying partner behaviors and selecting suitable responses is difficult. In this work, we introduce a strategy-conditioned cooperator framework that learns to represent, categorize, and adapt to a broad range of potential partner strategies in real-time. Our approach encodes strategies with a variational autoencoder to learn a latent strategy space from agent trajectory data, identifies distinct strategy types through clustering, and trains a cooperator agent conditioned on these clusters by generating partners of each strategy type. For online adaptation to novel partners, we leverage a fixed-share regret minimization algorithm that dynamically infers and adjusts the partner's strategy estimation during interaction. We evaluate our method in a modified version of the Overcooked domain, a complex collaborative cooking environment that requires effective coordination among two players with a diverse potential strategy space. Through these experiments and an online user study, we demonstrate that our proposed agent achieves state of the art performance compared to existing baselines when paired with novel human, and agent teammates.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12754) | **Categories:** cs.AI, cs.LG, cs.MA

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity](https://arxiv.org/abs/2511.12061)
*Zhichen Lai, Hua Lu, Huan Li, Jialiang Li, Christian S. Jensen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory similarity computation is fundamental functionality that is used for, e.g., clustering, prediction, and anomaly detection. However, existing learning-based methods exhibit three key limitations: (1) insufficient modeling of trajectory semantics and hierarchy, lacking both movement dynamics extraction and multi-scale structural representation; (2) high computational costs due to point-wise encoding; and (3) use of physically implausible augmentations that distort trajectory semantics. To address these issues, we propose MovSemCL, a movement-semantics contrastive learning framework for trajectory similarity computation. MovSemCL first transforms raw GPS trajectories into movement-semantics features and then segments them into patches. Next, MovSemCL employs intra- and inter-patch attentions to encode local as well as global trajectory patterns, enabling efficient hierarchical representation and reducing computational costs. Moreover, MovSemCL includes a curvature-guided augmentation strategy that preserves informative segments (e.g., turns and intersections) and masks redundant ones, generating physically plausible augmented views. Experiments on real-world datasets show that MovSemCL is capable of outperforming state-of-the-art methods, achieving mean ranks close to the ideal value of 1 at similarity search tasks and improvements by up to 20.3% at heuristic approximation, while reducing inference latency by up to 43.4%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12061) | **Categories:** cs.CV, cs.AI, cs.DB

---

### [2] [SOTFormer: A Minimal Transformer for Unified Object Tracking and Trajectory Prediction](https://arxiv.org/abs/2511.11824)
*Zhongping Dong, Pengyang Yu, Shuangjian Li, Liming Chen, Mohand Tahar Kechadi*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate single-object tracking and short-term motion forecasting remain challenging under occlusion, scale variation, and temporal drift, which disrupt the temporal coherence required for real-time perception. We introduce \textbf{SOTFormer}, a minimal constant-memory temporal transformer that unifies object detection, tracking, and short-horizon trajectory prediction within a single end-to-end framework. Unlike prior models with recurrent or stacked temporal encoders, SOTFormer achieves stable identity propagation through a ground-truth-primed memory and a burn-in anchor loss that explicitly stabilizes initialization. A single lightweight temporal-attention layer refines embeddings across frames, enabling real-time inference with fixed GPU memory. On the Mini-LaSOT (20%) benchmark, SOTFormer attains 76.3 AUC and 53.7 FPS (AMP, 4.3 GB VRAM), outperforming transformer baselines such as TrackFormer and MOTRv2 under fast motion, scale change, and occlusion.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11824) | **Categories:** cs.CV

---

### [3] [SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images](https://arxiv.org/abs/2511.12040)
*Xinyuan Hu, Changyue Shi, Chuxiao Yang, Minghao Chen, Jiajun Ding, Tao Wei, Chen Wei, Zhou Yu, Min Tan*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Feed-forward 3D reconstruction from sparse, low-resolution (LR) images is a crucial capability for real-world applications, such as autonomous driving and embodied AI. However, existing methods often fail to recover fine texture details. This limitation stems from the inherent lack of high-frequency information in LR inputs. To address this, we propose \textbf{SRSplat}, a feed-forward framework that reconstructs high-resolution 3D scenes from only a few LR views. Our main insight is to compensate for the deficiency of texture information by jointly leveraging external high-quality reference images and internal texture cues. We first construct a scene-specific reference gallery, generated for each scene using Multimodal Large Language Models (MLLMs) and diffusion models. To integrate this external information, we introduce the \textit{Reference-Guided Feature Enhancement (RGFE)} module, which aligns and fuses features from the LR input images and their reference twin image. Subsequently, we train a decoder to predict the Gaussian primitives using the multi-view fused feature obtained from \textit{RGFE}. To further refine predicted Gaussian primitives, we introduce \textit{Texture-Aware Density Control (TADC)}, which adaptively adjusts Gaussian density based on the internal texture richness of the LR inputs. Extensive experiments demonstrate that our SRSplat outperforms existing methods on various datasets, including RealEstate10K, ACID, and DTU, and exhibits strong cross-dataset and cross-resolution generalization capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12040) | **Categories:** cs.CV

---


## cs.DC [cs.DC]
### [1] [Flash-Fusion: Enabling Expressive, Low-Latency Queries on IoT Sensor Streams with LLMs](https://arxiv.org/abs/2511.11885)
*Kausar Patherya, Ashutosh Dhekne, Francisco Romero*

Main category: cs.DC

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Smart cities and pervasive IoT deployments have generated interest in IoT data analysis across transportation and urban planning. At the same time, Large Language Models offer a new interface for exploring IoT data - particularly through natural language. Users today face two key challenges when working with IoT data using LLMs: (1) data collection infrastructure is expensive, producing terabytes of low-level sensor readings that are too granular for direct use, and (2) data analysis is slow, requiring iterative effort and technical expertise. Directly feeding all IoT telemetry to LLMs is impractical due to finite context windows, prohibitive token costs at scale, and non-interactive latencies. What is missing is a system that first parses a user's query to identify the analytical task, then selects the relevant data slices, and finally chooses the right representation before invoking an LLM.   We present Flash-Fusion, an end-to-end edge-cloud system that reduces the IoT data collection and analysis burden on users. Two principles guide its design: (1) edge-based statistical summarization (achieving 73.5% data reduction) to address data volume, and (2) cloud-based query planning that clusters behavioral data and assembles context-rich prompts to address data interpretation. We deploy Flash-Fusion on a university bus fleet and evaluate it against a baseline that feeds raw data to a state-of-the-art LLM. Flash-Fusion achieves a 95% latency reduction and 98% decrease in token usage and cost while maintaining high-quality responses. It enables personas across disciplines - safety officers, urban planners, fleet managers, and data scientists - to efficiently iterate over IoT data without the burden of manual query authoring or preprocessing.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11885) | **Categories:** cs.DC, cs.AI, cs.DB

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [RAGPulse: An Open-Source RAG Workload Trace to Optimize RAG Serving Systems](https://arxiv.org/abs/2511.12979)
*Zhengchao Wang, Yitao Hu, Jianing Ye, Zhuxuan Chang, Jiazheng Yu, Youpeng Deng, Keqiu Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Retrieval-Augmented Generation (RAG) is a critical paradigm for building reliable, knowledge-intensive Large Language Model (LLM) applications. However, the multi-stage pipeline (retrieve, generate) and unique workload characteristics (e.g., knowledge dependency) of RAG systems pose significant challenges for serving performance optimization. Existing generic LLM inference traces fail to capture these RAG-specific dynamics, creating a significant performance gap between academic research and real-world deployment. To bridge this gap, this paper introduces RAGPulse, an open-source RAG workload trace dataset. This dataset was collected from an university-wide Q&A system serving that has served more than 40,000 students and faculties since April 2024. We detail RAGPulse's system architecture, its privacy-preserving hash-based data format, and provide an in-depth statistical analysis. Our analysis reveals that real-world RAG workloads exhibit significant temporal locality and a highly skewed hot document access pattern. RAGPulse provides a high-fidelity foundation for researchers to develop and validate novel optimization strategies for RAG systems, such as content-aware batching and retrieval caching, ultimately enhancing the efficiency and reliability of RAG services. The code is available at https://github.com/flashserve/RAGPulse.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12979) | **Categories:** cs.LG, cs.DB

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Bootstrapped LLM Semantics for Context-Aware Path Planning](https://arxiv.org/abs/2511.11967)
*Mani Amani, Behrad Beheshti, Reza Akhavian*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Prompting robots with natural language (NL) has largely been studied as what task to execute (goal selection, skill sequencing) rather than how to execute that task safely and efficiently in semantically rich, human-centric spaces. We address this gap with a framework that turns a large language model (LLM) into a stochastic semantic sensor whose outputs modulate a classical planner. Given a prompt and a semantic map, we draw multiple LLM "danger" judgments and apply a Bayesian bootstrap to approximate a posterior over per-class risk. Using statistics from the posterior, we create a potential cost to formulate a path planning problem. Across simulated environments and a BIM-backed digital twin, our method adapts how the robot moves in response to explicit prompts and implicit contextual information. We present qualitative and quantitative results.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11967) | **Categories:** cs.RO

---

### [2] [MATT-Diff: Multimodal Active Target Tracking by Diffusion Policy](https://arxiv.org/abs/2511.11931)
*Saida Liu, Nikolay Atanasov, Shumon Koga*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes MATT-Diff: Multi-Modal Active Target Tracking by Diffusion Policy, a control policy that captures multiple behavioral modes - exploration, dedicated tracking, and target reacquisition - for active multi-target tracking. The policy enables agent control without prior knowledge of target numbers, states, or dynamics. Effective target tracking demands balancing exploration for undetected or lost targets with following the motion of detected but uncertain ones. We generate a demonstration dataset from three expert planners including frontier-based exploration, an uncertainty-based hybrid planner switching between frontier-based exploration and RRT* tracking based on target uncertainty, and a time-based hybrid planner switching between exploration and tracking based on target detection time. We design a control policy utilizing a vision transformer for egocentric map tokenization and an attention mechanism to integrate variable target estimates represented by Gaussian densities. Trained as a diffusion model, the policy learns to generate multi-modal action sequences through a denoising process. Evaluations demonstrate MATT-Diff's superior tracking performance against expert and behavior cloning baselines across multiple target motions, empirically validating its advantages in target tracking.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11931) | **Categories:** cs.RO

---

### [3] [SocialNav-Map: Dynamic Mapping with Human Trajectory Prediction for Zero-Shot Social Navigation](https://arxiv.org/abs/2511.12232)
*Lingfeng Zhang, Erjia Xiao, Xiaoshuai Hao, Haoxiang Fu, Zeying Gong, Long Chen, Xiaojun Liang, Renjing Xu, Hangjun Ye, Wenbo Ding*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Social navigation in densely populated dynamic environments poses a significant challenge for autonomous mobile robots, requiring advanced strategies for safe interaction. Existing reinforcement learning (RL)-based methods require over 2000+ hours of extensive training and often struggle to generalize to unfamiliar environments without additional fine-tuning, limiting their practical application in real-world scenarios. To address these limitations, we propose SocialNav-Map, a novel zero-shot social navigation framework that combines dynamic human trajectory prediction with occupancy mapping, enabling safe and efficient navigation without the need for environment-specific training. Specifically, SocialNav-Map first transforms the task goal position into the constructed map coordinate system. Subsequently, it creates a dynamic occupancy map that incorporates predicted human movements as dynamic obstacles. The framework employs two complementary methods for human trajectory prediction: history prediction and orientation prediction. By integrating these predicted trajectories into the occupancy map, the robot can proactively avoid potential collisions with humans while efficiently navigating to its destination. Extensive experiments on the Social-HM3D and Social-MP3D datasets demonstrate that SocialNav-Map significantly outperforms state-of-the-art (SOTA) RL-based methods, which require 2,396 GPU hours of training. Notably, it reduces human collision rates by over 10% without necessitating any training in novel environments. By eliminating the need for environment-specific training, SocialNav-Map achieves superior navigation performance, paving the way for the deployment of social navigation systems in real-world environments characterized by diverse human behaviors. The code is available at: https://github.com/linglingxiansen/SocialNav-Map.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12232) | **Categories:** cs.RO

---

### [4] [Prompt-Driven Domain Adaptation for End-to-End Autonomous Driving via In-Context RL](https://arxiv.org/abs/2511.12755)
*Aleesha Khurram, Amir Moeini, Shangtong Zhang, Rohan Chandra*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite significant progress and advances in autonomous driving, many end-to-end systems still struggle with domain adaptation (DA), such as transferring a policy trained under clear weather to adverse weather conditions. Typical DA strategies in the literature include collecting additional data in the target domain or re-training the model, or both. Both these strategies quickly become impractical as we increase scale and complexity of driving. These limitations have encouraged investigation into few-shot and zero-shot prompt-driven DA at inference time involving LLMs and VLMs. These methods work by adding a few state-action trajectories during inference to the prompt (similar to in-context learning). However, there are two limitations of such an approach: $(i)$ prompt-driven DA methods are currently restricted to perception tasks such as detection and segmentation and $(ii)$ they require expert few-shot data. In this work, we present a new approach to inference-time few-shot prompt-driven DA for closed-loop autonomous driving in adverse weather condition using in-context reinforcement learning (ICRL). Similar to other prompt-driven DA methods, our approach does not require any updates to the model parameters nor does it require additional data collection in adversarial weather regime. Furthermore, our approach advances the state-of-the-art in prompt-driven DA by extending to closed driving using general trajectories observed during inference. Our experiments using the CARLA simulator show that ICRL results in safer, more efficient, and more comfortable driving policies in the target domain compared to state-of-the-art prompt-driven DA baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12755) | **Categories:** cs.RO, cs.LG

---

### [5] [Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos](https://arxiv.org/abs/2511.12882)
*Taiyi Su, Jian Zhu, Yaxuan Li, Chong Ma, Zitai Huang, Yichen Zhu, Hanli Wang, Yi Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied world models aim to predict and interact with the physical world through visual observations and actions. However, existing models struggle to accurately translate low-level actions (e.g., joint positions) into precise robotic movements in predicted frames, leading to inconsistencies with real-world physical interactions. To address these limitations, we propose MTV-World, an embodied world model that introduces Multi-view Trajectory-Video control for precise visuomotor prediction. Specifically, instead of directly using low-level actions for control, we employ trajectory videos obtained through camera intrinsic and extrinsic parameters and Cartesian-space transformation as control signals. However, projecting 3D raw actions onto 2D images inevitably causes a loss of spatial information, making a single view insufficient for accurate interaction modeling. To overcome this, we introduce a multi-view framework that compensates for spatial information loss and ensures high-consistency with physical world. MTV-World forecasts future frames based on multi-view trajectory videos as input and conditioning on an initial frame per view. Furthermore, to systematically evaluate both robotic motion precision and object interaction accuracy, we develop an auto-evaluation pipeline leveraging multimodal large models and referring video object segmentation models. To measure spatial consistency, we formulate it as an object location matching problem and adopt the Jaccard Index as the evaluation metric. Extensive experiments demonstrate that MTV-World achieves precise control execution and accurate physical interaction modeling in complex dual-arm scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12882) | **Categories:** cs.RO, cs.AI

---

### [6] [Hierarchical Federated Graph Attention Networks for Scalable and Resilient UAV Collision Avoidance](https://arxiv.org/abs/2511.11616)
*Rathin Chandra Shit, Sharmila Subudhi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The real-time performance, adversarial resiliency, and privacy preservation are the most important metrics that need to be balanced to practice collision avoidance in large-scale multi-UAV (Unmanned Aerial Vehicle) systems. Current frameworks tend to prescribe monolithic solutions that are not only prohibitively computationally complex with a scaling cost of $O(n^2)$ but simply do not offer Byzantine fault tolerance. The proposed hierarchical framework presented in this paper tries to eliminate such trade-offs by stratifying a three-layered architecture. We spread the intelligence into three layers: an immediate collision avoiding local layer running on dense graph attention with latency of $<10 ms$, a regional layer using sparse attention with $O(nk)$ computational complexity and asynchronous federated learning with coordinate-wise trimmed mean aggregation, and lastly, a global layer using a lightweight Hashgraph-inspired protocol. We have proposed an adaptive differential privacy mechanism, wherein the noise level $(ε\in [0.1, 1.0])$ is dynamically reduced based on an evaluation of the measured real-time threat that in turn maximized the privacy-utility tradeoff. Through the use of Distributed Hash Table (DHT)-based lightweight audit logging instead of heavyweight blockchain consensus, the median cost of getting a $95^{th}$ percentile decision within 50ms is observed across all tested swarm sizes. This architecture provides a scalable scenario of 500 UAVs with a collision rate of $< 2.0\%$ and the Byzantine fault tolerance of $f < n/3$.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11616) | **Categories:** cs.RO, cs.AI, cs.LG, cs.MA

---

### [7] [ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](https://arxiv.org/abs/2511.11740)
*Haowen Jiang, Xinyu Huang, You Lu, Dingji Wang, Yuheng Cao, Chaofeng Sha, Bihuan Chen, Keyu Chen, Xin Peng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in end-to-end autonomous driving systems (ADSs) underscore their potential for perception and planning capabilities. However, challenges remain. Complex driving scenarios contain rich semantic information, yet ambiguous or noisy semantics can compromise decision reliability, while interference between multiple driving tasks may hinder optimal planning. Furthermore, prolonged inference latency slows decision-making, increasing the risk of unsafe driving behaviors. To address these challenges, we propose ExpertAD, a novel framework that enhances the performance of ADS with Mixture of Experts (MoE) architecture. We introduce a Perception Adapter (PA) to amplify task-critical features, ensuring contextually relevant scene understanding, and a Mixture of Sparse Experts (MoSE) to minimize task interference during prediction, allowing for effective and efficient planning. Our experiments show that ExpertAD reduces average collision rates by up to 20% and inference latency by 25% compared to prior methods. We further evaluate its multi-skill planning capabilities in rare scenarios (e.g., accidents, yielding to emergency vehicles) and demonstrate strong generalization to unseen urban environments. Additionally, we present a case study that illustrates its decision-making process in complex driving scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11740) | **Categories:** cs.RO, cs.AI

---

### [8] [Large Language Models and 3D Vision for Intelligent Robotic Perception and Autonomy: A Review](https://arxiv.org/abs/2511.11777)
*Vinit Mehta, Charu Sharma, Karthick Thiyagarajan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the rapid advancement of artificial intelligence and robotics, the integration of Large Language Models (LLMs) with 3D vision is emerging as a transformative approach to enhancing robotic sensing technologies. This convergence enables machines to perceive, reason and interact with complex environments through natural language and spatial understanding, bridging the gap between linguistic intelligence and spatial perception. This review provides a comprehensive analysis of state-of-the-art methodologies, applications and challenges at the intersection of LLMs and 3D vision, with a focus on next-generation robotic sensing technologies. We first introduce the foundational principles of LLMs and 3D data representations, followed by an in-depth examination of 3D sensing technologies critical for robotics. The review then explores key advancements in scene understanding, text-to-3D generation, object grounding and embodied agents, highlighting cutting-edge techniques such as zero-shot 3D segmentation, dynamic scene synthesis and language-guided manipulation. Furthermore, we discuss multimodal LLMs that integrate 3D data with touch, auditory and thermal inputs, enhancing environmental comprehension and robotic decision-making. To support future research, we catalog benchmark datasets and evaluation metrics tailored for 3D-language and vision tasks. Finally, we identify key challenges and future research directions, including adaptive model architectures, enhanced cross-modal alignment and real-time processing capabilities, which pave the way for more intelligent, context-aware and autonomous robotic sensing systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11777) | **Categories:** cs.RO, cs.CV

---

### [9] [SBAMP: Sampling Based Adaptive Motion Planning](https://arxiv.org/abs/2511.12022)
*Anh-Quan Pham, Kabir Ram Puri, Shreyas Raorane*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous robotic systems must navigate complex, dynamic environments in real time, often facing unpredictable obstacles and rapidly changing conditions. Traditional sampling-based methods, such as RRT*, excel at generating collision-free paths but struggle to adapt to sudden changes without extensive replanning. Conversely, learning-based dynamical systems, such as the Stable Estimator of Dynamical Systems (SEDS), offer smooth, adaptive trajectory tracking but typically rely on pre-collected demonstration data, limiting their generalization to novel scenarios. This paper introduces Sampling-Based Adaptive Motion Planning (SBAMP), a novel framework that overcomes these limitations by integrating RRT* for global path planning with a SEDS-based local controller for continuous, adaptive trajectory adjustment. Our approach requires no pre-trained datasets and ensures smooth transitions between planned waypoints, maintaining stability through Lyapunov-based guarantees. We validate SBAMP in both simulated environments and real hardware using the RoboRacer platform, demonstrating superior performance in dynamic obstacle scenarios, rapid recovery from perturbations, and robust handling of sharp turns. Experimental results highlight SBAMP's ability to adapt in real time without sacrificing global path optimality, providing a scalable solution for dynamic, unstructured environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12022) | **Categories:** cs.RO, eess.SY

---

### [10] [Decoupled Action Head: Confining Task Knowledge to Conditioning Layers](https://arxiv.org/abs/2511.12101)
*Jian Zhou, Sihao Lin, Shuai Fu, Qi WU*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Behavior Cloning (BC) is a data-driven supervised learning approach that has gained increasing attention with the success of scaling laws in language and vision domains. Among its implementations in robotic manipulation, Diffusion Policy (DP), with its two variants DP-CNN (DP-C) and DP-Transformer (DP-T), is one of the most effective and widely adopted models, demonstrating the advantages of predicting continuous action sequences. However, both DP and other BC methods remain constrained by the scarcity of paired training data, and the internal mechanisms underlying DP's effectiveness remain insufficiently understood, leading to limited generalization and a lack of principled design in model development. In this work, we propose a decoupled training recipe that leverages nearly cost-free kinematics-generated trajectories as observation-free data to pretrain a general action head (action generator). The pretrained action head is then frozen and adapted to novel tasks through feature modulation. Our experiments demonstrate the feasibility of this approach in both in-distribution and out-of-distribution scenarios. As an additional benefit, decoupling improves training efficiency; for instance, DP-C achieves up to a 41% speedup. Furthermore, the confinement of task-specific knowledge to the conditioning components under decoupling, combined with the near-identical performance of DP-C in both normal and decoupled training, indicates that the action generation backbone plays a limited role in robotic manipulation. Motivated by this observation, we introduce DP-MLP, which replaces the 244M-parameter U-Net backbone of DP-C with only 4M parameters of simple MLP blocks, achieving a 83.9% faster training speed under normal training and 89.1% under decoupling.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12101) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [11] [Game-Theoretic Safe Multi-Agent Motion Planning with Reachability Analysis for Dynamic and Uncertain Environments (Extended Version)](https://arxiv.org/abs/2511.12160)
*Wenbin Mai, Minghui Liwang, Xinlei Yi, Xiaoyu Xia, Seyyedali Hosseinalipour, Xianbin Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring safe, robust, and scalable motion planning for multi-agent systems in dynamic and uncertain environments is a persistent challenge, driven by complex inter-agent interactions, stochastic disturbances, and model uncertainties. To overcome these challenges, particularly the computational complexity of coupled decision-making and the need for proactive safety guarantees, we propose a Reachability-Enhanced Dynamic Potential Game (RE-DPG) framework, which integrates game-theoretic coordination into reachability analysis. This approach formulates multi-agent coordination as a dynamic potential game, where the Nash equilibrium (NE) defines optimal control strategies across agents. To enable scalability and decentralized execution, we develop a Neighborhood-Dominated iterative Best Response (ND-iBR) scheme, built upon an iterated $\varepsilon$-BR (i$\varepsilon$-BR) process that guarantees finite-step convergence to an $\varepsilon$-NE. This allows agents to compute strategies based on local interactions while ensuring theoretical convergence guarantees. Furthermore, to ensure safety under uncertainty, we integrate a Multi-Agent Forward Reachable Set (MA-FRS) mechanism into the cost function, explicitly modeling uncertainty propagation and enforcing collision avoidance constraints. Through both simulations and real-world experiments in 2D and 3D environments, we validate the effectiveness of RE-DPG across diverse operational scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12160) | **Categories:** cs.RO

---

### [12] [Locally Optimal Solutions to Constraint Displacement Problems via Path-Obstacle Overlaps](https://arxiv.org/abs/2511.12203)
*Antony Thomas, Fulvio Mastrogiovanni, Marco Baglietto*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a unified approach for constraint displacement problems in which a robot finds a feasible path by displacing constraints or obstacles. To this end, we propose a two stage process that returns locally optimal obstacle displacements to enable a feasible path for the robot. The first stage proceeds by computing a trajectory through the obstacles while minimizing an appropriate objective function. In the second stage, these obstacles are displaced to make the computed robot trajectory feasible, that is, collision-free. Several examples are provided that successfully demonstrate our approach on two distinct classes of constraint displacement problems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12203) | **Categories:** cs.RO, cs.AI

---

### [13] [EcoFlight: Finding Low-Energy Paths Through Obstacles for Autonomous Sensing Drones](https://arxiv.org/abs/2511.12618)
*Jordan Leyva, Nahim J. Moran Vera, Yihan Xu, Adrien Durasno, Christopher U. Romero, Tendai Chimuka, Gabriel O. Huezo Ramirez, Ziqian Dong, Roberto Rojas-Cessa*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Obstacle avoidance path planning for uncrewed aerial vehicles (UAVs), or drones, is rarely addressed in most flight path planning schemes, despite obstacles being a realistic condition. Obstacle avoidance can also be energy-intensive, making it a critical factor in efficient point-to-point drone flights. To address these gaps, we propose EcoFlight, an energy-efficient pathfinding algorithm that determines the lowest-energy route in 3D space with obstacles. The algorithm models energy consumption based on the drone propulsion system and flight dynamics. We conduct extensive evaluations, comparing EcoFlight with direct-flight and shortest-distance schemes. The simulation results across various obstacle densities show that EcoFlight consistently finds paths with lower energy consumption than comparable algorithms, particularly in high-density environments. We also demonstrate that a suitable flying speed can further enhance energy savings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12618) | **Categories:** cs.RO

---

### [14] [DR. Nav: Semantic-Geometric Representations for Proactive Dead-End Recovery and Navigation](https://arxiv.org/abs/2511.12778)
*Vignesh Rajagopal, Kasun Weerakoon Kulathun Mudiyanselage, Gershom Devake Seneviratne, Pon Aswin Sankaralingam, Mohamed Elnoor, Jing Liang, Rohan Chandra, Dinesh Manocha*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present DR. Nav (Dead-End Recovery-aware Navigation), a novel approach to autonomous navigation in scenarios where dead-end detection and recovery are critical, particularly in unstructured environments where robots must handle corners, vegetation occlusions, and blocked junctions. DR. Nav introduces a proactive strategy for navigation in unmapped environments without prior assumptions. Our method unifies dead-end prediction and recovery by generating a single, continuous, real-time semantic cost map. Specifically, DR. Nav leverages cross-modal RGB-LiDAR fusion with attention-based filtering to estimate per-cell dead-end likelihoods and recovery points, which are continuously updated through Bayesian inference to enhance robustness. Unlike prior mapping methods that only encode traversability, DR. Nav explicitly incorporates recovery-aware risk into the navigation cost map, enabling robots to anticipate unsafe regions and plan safer alternative trajectories. We evaluate DR. Nav across multiple dense indoor and outdoor scenarios and demonstrate an increase of 83.33% in accuracy in detection, a 52.4% reduction in time-to-goal (path efficiency), compared to state-of-the-art planners such as DWA, MPPI, and Nav2 DWB. Furthermore, the dead-end classifier functions

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.12778) | **Categories:** cs.RO

---

### [15] [Collision-Free Navigation of Mobile Robots via Quadtree-Based Model Predictive Control](https://arxiv.org/abs/2511.13188)
*Osama Al Sheikh Ali, Sotiris Koutsoftas, Ze Zhang, Knut Akesson, Emmanuel Dean*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents an integrated navigation framework for Autonomous Mobile Robots (AMRs) that unifies environment representation, trajectory generation, and Model Predictive Control (MPC). The proposed approach incorporates a quadtree-based method to generate structured, axis-aligned collision-free regions from occupancy maps. These regions serve as both a basis for developing safe corridors and as linear constraints within the MPC formulation, enabling efficient and reliable navigation without requiring direct obstacle encoding. The complete pipeline combines safe-area extraction, connectivity graph construction, trajectory generation, and B-spline smoothing into one coherent system. Experimental results demonstrate consistent success and superior performance compared to baseline approaches across complex environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13188) | **Categories:** cs.RO, eess.SY

---

### [16] [PIGEON: VLM-Driven Object Navigation via Points of Interest Selection](https://arxiv.org/abs/2511.13207)
*Cheng Peng, Zhenzhe Zhang, Cheng Chi, Xiaobao Wei, Yanhao Zhang, Heng Wang, Pengwei Wang, Zhongyuan Wang, Jing Liu, Shanghang Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Navigating to a specified object in an unknown environment is a fundamental yet challenging capability of embodied intelligence. However, current methods struggle to balance decision frequency with intelligence, resulting in decisions lacking foresight or discontinuous actions. In this work, we propose PIGEON: Point of Interest Guided Exploration for Object Navigation with VLM, maintaining a lightweight and semantically aligned snapshot memory during exploration as semantic input for the exploration strategy. We use a large Visual-Language Model (VLM), named PIGEON-VL, to select Points of Interest (PoI) formed during exploration and then employ a lower-level planner for action output, increasing the decision frequency. Additionally, this PoI-based decision-making enables the generation of Reinforcement Learning with Verifiable Reward (RLVR) data suitable for simulators. Experiments on classic object navigation benchmarks demonstrate that our zero-shot transfer method achieves state-of-the-art performance, while RLVR further enhances the model's semantic guidance capabilities, enabling deep reasoning during real-time navigation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13207) | **Categories:** cs.RO, cs.CV

---
