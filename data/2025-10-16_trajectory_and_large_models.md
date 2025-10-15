# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-16

## 目录

- [人工智能 (Artificial Intelligence) (8)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [HiCoTraj:Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory](https://arxiv.org/abs/2510.12067)
*Junyi Xie, Yuankun Jiao, Jina Kim, Yao-Yi Chiang, Lingyi Zhao, Khurram Shafique*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Inferring demographic attributes such as age, sex, or income level from human mobility patterns enables critical applications such as targeted public health interventions, equitable urban planning, and personalized transportation services. Existing mobility-based demographic inference studies heavily rely on large-scale trajectory data with demographic labels, leading to limited interpretability and poor generalizability across different datasets and user groups. We propose HiCoTraj (Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory), a framework that leverages LLMs' zero-shot learning and semantic understanding capabilities to perform demographic inference without labeled training data. HiCoTraj transforms trajectories into semantically rich, natural language representations by creating detailed activity chronicles and multi-scale visiting summaries. Then HiCoTraj uses a novel hierarchical chain of thought reasoning to systematically guide LLMs through three cognitive stages: factual feature extraction, behavioral pattern analysis, and demographic inference with structured output. This approach addresses the scarcity challenge of labeled demographic data while providing transparent reasoning chains. Experimental evaluation on real-world trajectory data demonstrates that HiCoTraj achieves competitive performance across multiple demographic attributes in zero-shot scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12067) | **Categories:** cs.AI

---

### [2] [CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction](https://arxiv.org/abs/2510.12703)
*Mattia Grasselli, Angelo Porrello, Carlo Augusto Grazia*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous driving remains a challenging task, particularly due to safety concerns. Modern vehicles are typically equipped with expensive sensors such as LiDAR, cameras, and radars to reduce the risk of accidents. However, these sensors face inherent limitations: their field of view and line of sight can be obstructed by other vehicles, thereby reducing situational awareness. In this context, vehicle-to-vehicle communication plays a crucial role, as it enables cars to share information and remain aware of each other even when sensors are occluded. One way to achieve this is through the use of Cooperative Awareness Messages (CAMs). In this paper, we investigate the use of CAM data for vehicle trajectory prediction. Specifically, we design and train a neural network, Cooperative Awareness Message-based Graph Neural Network (CAMNet), on a widely used motion forecasting dataset. We then evaluate the model on a second dataset that we created from scratch using Cooperative Awareness Messages, in order to assess whether this type of data can be effectively exploited. Our approach demonstrates promising results, showing that CAMs can indeed support vehicle trajectory prediction. At the same time, we discuss several limitations of the approach, which highlight opportunities for future research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12703) | **Categories:** cs.AI, cs.NI

---

### [3] [EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making](https://arxiv.org/abs/2510.12072)
*Zixing Lei, Sheng Yin, Yichen Xiong, Yuanzhuo Ding, Wenhao Huang, Yuxi Wei, Qingyao Xu, Yiming Li, Weixin Li, Yunhong Wang, Siheng Chen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied decision-making enables agents to translate high-level goals into executable actions through continuous interactions within the physical world, forming a cornerstone of general-purpose embodied intelligence. Large language models (LLMs), with their general decision-making capabilities, offer a promising path to realize this potential; however, LLMs trained solely on language lack exposure to physical environments, limiting their true embodied understanding. To bridge this gap, we propose the concept of a training ground: a comprehensive infrastructure that provides task and scene simulation, embodied interaction, and feedback signals, offering a one-stop solution for LLM acquire genuine embodied decision-making skills. In this work, we present EmboMatrix, the first training ground of its kind, providing massive and diverse tasks with efficient simulation and precise rewards. EmboMatrix incorporates a series of novel techniques: a multi-agent data engine for large-scale task and scene generation, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. Leveraging EmboMatrix, we cultivate EmboBrain, an LLM whose embodied decision-making abilities emerge from extensive embodied interactions. Experiments show that EmboBrain-7B surpasses the 671B DeepSeek-R1 baseline by 9.5\% on two challenging embodied decision-making benchmarks, demonstrating the power of interactive, environment-grounded learning for building truly intelligent embodied agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12072) | **Categories:** cs.AI, cs.RO

---

### [4] [ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning](https://arxiv.org/abs/2510.12693)
*Hanyang Chen, Mark Zhao, Rui Yang, Qinwei Ma, Ke Yang, Jiarui Yao, Kangrui Wang, Hao Bai, Zhenhailong Wang, Rui Pan, Mengchao Zhang, Jose Barreiros, Aykut Onol, ChengXiang Zhai, Heng Ji, Manling Li, Huan Zhang, Tong Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in embodied AI highlight the potential of vision language models (VLMs) as agents capable of perception, reasoning, and interaction in complex environments. However, top-performing systems rely on large-scale models that are costly to deploy, while smaller VLMs lack the necessary knowledge and skills to succeed. To bridge this gap, we present \textit{Embodied Reasoning Agent (ERA)}, a two-stage framework that integrates prior knowledge learning and online reinforcement learning (RL). The first stage, \textit{Embodied Prior Learning}, distills foundational knowledge from three types of data: (1) Trajectory-Augmented Priors, which enrich existing trajectory data with structured reasoning generated by stronger models; (2) Environment-Anchored Priors, which provide in-environment knowledge and grounding supervision; and (3) External Knowledge Priors, which transfer general knowledge from out-of-environment datasets. In the second stage, we develop an online RL pipeline that builds on these priors to further enhance agent performance. To overcome the inherent challenges in agent RL, including long horizons, sparse rewards, and training instability, we introduce three key designs: self-summarization for context management, dense reward shaping, and turn-level policy optimization. Extensive experiments on both high-level planning (EB-ALFRED) and low-level control (EB-Manipulation) tasks demonstrate that ERA-3B surpasses both prompting-based large models and previous training-based baselines. Specifically, it achieves overall improvements of 8.4\% on EB-ALFRED and 19.4\% on EB-Manipulation over GPT-4o, and exhibits strong generalization to unseen tasks. Overall, ERA offers a practical path toward scalable embodied intelligence, providing methodological insights for future embodied AI systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12693) | **Categories:** cs.AI

---

### [5] [Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response](https://arxiv.org/abs/2510.12061)
*Yiheng Chen, Lingyao Li, Zihui Ma, Qikai Hu, Yilun Zhu, Min Deng, Runlong Yu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective disaster response is essential for safeguarding lives and property. Existing statistical approaches often lack semantic context, generalize poorly across events, and offer limited interpretability. While Large language models (LLMs) provide few-shot generalization, they remain text-bound and blind to geography. To bridge this gap, we introduce a Geospatial Awareness Layer (GAL) that grounds LLM agents in structured earth data. Starting from raw wildfire detections, GAL automatically retrieves and integrates infrastructure, demographic, terrain, and weather information from external geodatabases, assembling them into a concise, unit-annotated perception script. This enriched context enables agents to produce evidence-based resource-allocation recommendations (e.g., personnel assignments, budget allocations), further reinforced by historical analogs and daily change signals for incremental updates. We evaluate the framework in real wildfire scenarios across multiple LLM models, showing that geospatially grounded agents can outperform baselines. The proposed framework can generalize to other hazards such as floods and hurricanes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12061) | **Categories:** cs.AI

---

### [6] [BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data](https://arxiv.org/abs/2510.12076)
*Junyi Xie, Jina Kim, Yao-Yi Chiang, Lingyi Zhao, Khurram Shafique*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional anomaly detection in human mobility has primarily focused on trajectory-level analysis, identifying statistical outliers or spatiotemporal inconsistencies across aggregated movement traces. However, detecting individual-level anomalies, i.e., unusual deviations in a person's mobility behavior relative to their own historical patterns, within datasets encompassing large populations remains a significant challenge. In this paper, we present BeSTAD (Behavior-aware Spatio-Temporal Anomaly Detection for Human Mobility Data), an unsupervised framework that captures individualized behavioral signatures across large populations and uncovers fine-grained anomalies by jointly modeling spatial context and temporal dynamics. BeSTAD learns semantically enriched mobility representations that integrate location meaning and temporal patterns, enabling the detection of subtle deviations in individual movement behavior. BeSTAD further employs a behavior-cluster-aware modeling mechanism that builds personalized behavioral profiles from normal activity and identifies anomalies through cross-period behavioral comparison with consistent semantic alignment. Building on prior work in mobility behavior clustering, this approach enables not only the detection of behavioral shifts and deviations from established routines but also the identification of individuals exhibiting such changes within large-scale mobility datasets. By learning individual behaviors directly from unlabeled data, BeSTAD advances anomaly detection toward personalized and interpretable mobility analysis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12076) | **Categories:** cs.AI

---

### [7] [Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey](https://arxiv.org/abs/2510.12178)
*Abdulhady Abas Abdullah, Arkaitz Zubiaga, Seyedali Mirjalili, Amir H. Gandomi, Fatemeh Daneshfar, Mohammadsadra Amini, Alan Salam Mohammed, Hadi Veisi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series - from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12178) | **Categories:** cs.AI, cs.CL

---

### [8] [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635)
*Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, Jitao Sang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models face challenges in long-horizon agentic tasks as their constrained memory is easily overwhelmed by distracting or irrelevant context. Existing working memory methods typically rely on external, heuristic mechanisms that are decoupled from the agent's core policy. In this work, we reframe working memory management as a learnable, intrinsic capability. We propose a novel framework, Memory-as-Action, where an agent actively manages its working memory by executing explicit editing operations as part of a unified policy. This formulation allows an agent, trained via reinforcement learning, to balance memory curation against long-term task objectives under given resource constraints. However, such memory editing actions break the standard assumption of a continuously growing prefix in LLM interactions, leading to what we call trajectory fractures. These non-prefix changes disrupt the causal continuity required by standard policy gradient methods, making those methods inapplicable. To address this, we propose a new algorithm, Dynamic Context Policy Optimization, which enables stable end-to-end reinforcement learning by segmenting trajectories at memory action points and applying trajectory-level advantages to the resulting action segments. Our results demonstrate that jointly optimizing for task reasoning and memory management in an end-to-end fashion not only reduces overall computational consumption but also improves task performance, driven by adaptive context curation strategies tailored to the model's intrinsic capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12635) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Task-Specific Dual-Model Framework for Comprehensive Traffic Safety Video Description and Analysis](https://arxiv.org/abs/2510.11907)
*Blessing Agyei Kyem, Neema Jakisa Owor, Andrews Danyo, Joshua Kofi Asamoah, Eugene Denteh, Tanner Muturi, Anthony Dontoh, Yaw Adu-Gyamfi, Armstrong Aboah*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traffic safety analysis requires complex video understanding to capture fine-grained behavioral patterns and generate comprehensive descriptions for accident prevention. In this work, we present a unique dual-model framework that strategically utilizes the complementary strengths of VideoLLaMA and Qwen2.5-VL through task-specific optimization to address this issue. The core insight behind our approach is that separating training for captioning and visual question answering (VQA) tasks minimizes task interference and allows each model to specialize more effectively. Experimental results demonstrate that VideoLLaMA is particularly effective in temporal reasoning, achieving a CIDEr score of 1.1001, while Qwen2.5-VL excels in visual understanding with a VQA accuracy of 60.80\%. Through extensive experiments on the WTS dataset, our method achieves an S2 score of 45.7572 in the 2025 AI City Challenge Track 2, placing 10th on the challenge leaderboard. Ablation studies validate that our separate training strategy outperforms joint training by 8.6\% in VQA accuracy while maintaining captioning quality.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.11907) | **Categories:** cs.CV

---

### [2] [CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving](https://arxiv.org/abs/2510.12560)
*Xiaoji Zheng, Ziyuan Yang, Yanhao Chen, Yuhang Peng, Yuanrong Tang, Gengyuan Liu, Bokui Chen, Jiangtao Gong*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end autonomous driving models trained solely with imitation learning (IL) often suffer from poor generalization. In contrast, reinforcement learning (RL) promotes exploration through reward maximization but faces challenges such as sample inefficiency and unstable convergence. A natural solution is to combine IL and RL. Moving beyond the conventional two-stage paradigm (IL pretraining followed by RL fine-tuning), we propose CoIRL-AD, a competitive dual-policy framework that enables IL and RL agents to interact during training. CoIRL-AD introduces a competition-based mechanism that facilitates knowledge exchange while preventing gradient conflicts. Experiments on the nuScenes dataset show an 18% reduction in collision rate compared to baselines, along with stronger generalization and improved performance on long-tail scenarios. Code is available at: https://github.com/SEU-zxj/CoIRL-AD.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12560) | **Categories:** cs.CV, cs.LG, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Indoor Localization using Compact, Telemetry-Agnostic, Transfer-Learning Enabled Decoder-Only Transformer](https://arxiv.org/abs/2510.11926)
*Nayan Sanjay Bhatia, Pranay Kocheta, Russell Elliott, Harikrishna S. Kuttivelil, Katia Obraczka*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Indoor Wi-Fi positioning remains a challenging problem due to the high sensitivity of radio signals to environmental dynamics, channel propagation characteristics, and hardware heterogeneity. Conventional fingerprinting and model-based approaches typically require labor-intensive calibration and suffer rapid performance degradation when devices, channel or deployment conditions change. In this paper, we introduce Locaris, a decoder-only large language model (LLM) for indoor localization. Locaris treats each access point (AP) measurement as a token, enabling the ingestion of raw Wi-Fi telemetry without pre-processing. By fine-tuning its LLM on different Wi-Fi datasets, Locaris learns a lightweight and generalizable mapping from raw signals directly to device location. Our experimental study comparing Locaris with state-of-the-art methods consistently shows that Locaris matches or surpasses existing techniques for various types of telemetry. Our results demonstrate that compact LLMs can serve as calibration-free regression models for indoor localization, offering scalable and robust cross-environment performance in heterogeneous Wi-Fi deployments. Few-shot adaptation experiments, using only a handful of calibration points per device, further show that Locaris maintains high accuracy when applied to previously unseen devices and deployment scenarios. This yields sub-meter accuracy with just a few hundred samples, robust performance under missing APs and supports any and all available telemetry. Our findings highlight the practical viability of Locaris for indoor positioning in the real-world scenarios, particularly in large-scale deployments where extensive calibration is infeasible.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.11926) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Controllable Collision Scenario Generation via Collision Pattern Prediction](https://arxiv.org/abs/2510.12206)
*Pin-Lun Chen, Chi-Hsi Kung, Che-Han Chang, Wei-Chen Chiu, Yi-Ting Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Evaluating the safety of autonomous vehicles (AVs) requires diverse, safety-critical scenarios, with collisions being especially important yet rare and unsafe to collect in the real world. Therefore, the community has been focusing on generating safety-critical scenarios in simulation. However, controlling attributes such as collision type and time-to-accident (TTA) remains challenging. We introduce a new task called controllable collision scenario generation, where the goal is to produce trajectories that realize a user-specified collision type and TTA, to investigate the feasibility of automatically generating desired collision scenarios. To support this task, we present COLLIDE, a large-scale collision scenario dataset constructed by transforming real-world driving logs into diverse collisions, balanced across five representative collision types and different TTA intervals. We propose a framework that predicts Collision Pattern, a compact and interpretable representation that captures the spatial configuration of the ego and the adversarial vehicles at impact, before rolling out full adversarial trajectories. Experiments show that our approach outperforms strong baselines in both collision rate and controllability. Furthermore, generated scenarios consistently induce higher planner failure rates, revealing limitations of existing planners. We demonstrate that these scenarios fine-tune planners for robustness improvements, contributing to safer AV deployment in different collision scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12206) | **Categories:** cs.RO, cs.LG

---

### [2] [HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions](https://arxiv.org/abs/2510.12733)
*Hang Yu, Julian Jordan, Julian Schmidt, Silvan Lindner, Alessandro Canevaro, Wilhelm Stork*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12733) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [3] [Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications](https://arxiv.org/abs/2510.12215)
*Chanwoo Kim, Jihwan Yoon, Hyeonseong Kim, Taemoon Jeong, Changwoo Yoo, Seungbeen Lee, Soohwan Byeon, Hoon Chung, Matthew Pan, Jean Oh, Kyungjae Lee, Sungjoon Choi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Mobile robot navigation in dynamic human environments requires policies that balance adaptability to diverse behaviors with compliance to safety constraints. We hypothesize that integrating data-driven rewards with rule-based objectives enables navigation policies to achieve a more effective balance of adaptability and safety. To this end, we develop a framework that learns a density-based reward from positive and negative demonstrations and augments it with rule-based objectives for obstacle avoidance and goal reaching. A sampling-based lookahead controller produces supervisory actions that are both safe and adaptive, which are subsequently distilled into a compact student policy suitable for real-time operation with uncertainty estimates. Experiments in synthetic and elevator co-boarding simulations show consistent gains in success rate and time efficiency over baselines, and real-world demonstrations with human participants confirm the practicality of deployment. A video illustrating this work can be found on our project page https://chanwookim971024.github.io/PioneeR/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12215) | **Categories:** cs.RO

---

### [4] [Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model](https://arxiv.org/abs/2510.12276)
*Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimators.We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action precision.Extensive experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page is at https://spatial-forcing.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12276) | **Categories:** cs.RO

---

### [5] [Controlling Intent Expressiveness in Robot Motion with Diffusion Models](https://arxiv.org/abs/2510.12370)
*Wenli Shi, Clemence Grislain, Olivier Sigaud, Mohamed Chetouani*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Legibility of robot motion is critical in human-robot interaction, as it allows humans to quickly infer a robot's intended goal. Although traditional trajectory generation methods typically prioritize efficiency, they often fail to make the robot's intentions clear to humans. Meanwhile, existing approaches to legible motion usually produce only a single "most legible" trajectory, overlooking the need to modulate intent expressiveness in different contexts. In this work, we propose a novel motion generation framework that enables controllable legibility across the full spectrum, from highly legible to highly ambiguous motions. We introduce a modeling approach based on an Information Potential Field to assign continuous legibility scores to trajectories, and build upon it with a two-stage diffusion framework that first generates paths at specified legibility levels and then translates them into executable robot actions. Experiments in both 2D and 3D reaching tasks demonstrate that our approach produces diverse and controllable motions with varying degrees of legibility, while achieving performance comparable to SOTA. Code and project page: https://legibility-modulator.github.io.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12370) | **Categories:** cs.RO

---
