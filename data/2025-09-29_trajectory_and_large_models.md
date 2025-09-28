# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-29

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (12)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning](https://arxiv.org/abs/2509.20754)
*Yufan Mao, Hanjing Ye, Wenlong Dong, Chengjie Zhang, Hong Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Navigating complex environments requires robots to effectively store observations as memories and leverage them to answer human queries about spatial locations, which is a critical yet underexplored research challenge. While prior work has made progress in constructing robotic memory, few have addressed the principled mechanisms needed for efficient memory retrieval and integration. To bridge this gap, we propose Meta-Memory, a large language model (LLM)-driven agent that constructs a high-density memory representation of the environment. The key innovation of Meta-Memory lies in its capacity to retrieve and integrate relevant memories through joint reasoning over semantic and spatial modalities in response to natural language location queries, thereby empowering robots with robust and accurate spatial reasoning capabilities. To evaluate its performance, we introduce SpaceLocQA, a large-scale dataset encompassing diverse real-world spatial question-answering scenarios. Experimental results show that Meta-Memory significantly outperforms state-of-the-art methods on both the SpaceLocQA and the public NaVQA benchmarks. Furthermore, we successfully deployed Meta-Memory on real-world robotic platforms, demonstrating its practical utility in complex environments. Project page: https://itsbaymax.github.io/meta-memory.github.io/ .

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20754) | **Categories:** cs.AI, cs.RO

---

### [2] [ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective](https://arxiv.org/abs/2509.21134)
*Yiwen Zhang, Ziang Chen, Fanqi Kong, Yizhe Huang, Xue Feng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21134) | **Categories:** cs.AI, cs.MA

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Beyond the Individual: Introducing Group Intention Forecasting with SHOT Dataset](https://arxiv.org/abs/2509.20715)
*Ruixu Zhang, Yuran Wang, Xinyi Hu, Chaoyu Mai, Wenxuan Liu, Danni Xu, Xian Zhong, Zheng Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Intention recognition has traditionally focused on individual intentions, overlooking the complexities of collective intentions in group settings. To address this limitation, we introduce the concept of group intention, which represents shared goals emerging through the actions of multiple individuals, and Group Intention Forecasting (GIF), a novel task that forecasts when group intentions will occur by analyzing individual actions and interactions before the collective goal becomes apparent. To investigate GIF in a specific scenario, we propose SHOT, the first large-scale dataset for GIF, consisting of 1,979 basketball video clips captured from 5 camera views and annotated with 6 types of individual attributes. SHOT is designed with 3 key characteristics: multi-individual information, multi-view adaptability, and multi-level intention, making it well-suited for studying emerging group intentions. Furthermore, we introduce GIFT (Group Intention ForecasTer), a framework that extracts fine-grained individual features and models evolving group dynamics to forecast intention emergence. Experimental results confirm the effectiveness of SHOT and GIFT, establishing a strong foundation for future research in group intention forecasting. The dataset is available at https://xinyi-hu.github.io/SHOT_DATASET.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20715) | **Categories:** cs.CV, cs.AI

---

### [2] [SimDiff: Simulator-constrained Diffusion Model for Physically Plausible Motion Generation](https://arxiv.org/abs/2509.20927)
*Akihisa Watanabe, Jiawei Ren, Li Siyao, Yichen Peng, Erwin Wu, Edgar Simo-Serra*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generating physically plausible human motion is crucial for applications such as character animation and virtual reality. Existing approaches often incorporate a simulator-based motion projection layer to the diffusion process to enforce physical plausibility. However, such methods are computationally expensive due to the sequential nature of the simulator, which prevents parallelization. We show that simulator-based motion projection can be interpreted as a form of guidance, either classifier-based or classifier-free, within the diffusion process. Building on this insight, we propose SimDiff, a Simulator-constrained Diffusion Model that integrates environment parameters (e.g., gravity, wind) directly into the denoising process. By conditioning on these parameters, SimDiff generates physically plausible motions efficiently, without repeated simulator calls at inference, and also provides fine-grained control over different physical coefficients. Moreover, SimDiff successfully generalizes to unseen combinations of environmental parameters, demonstrating compositional generalization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20927) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Beyond Visual Similarity: Rule-Guided Multimodal Clustering with explicit domain rules](https://arxiv.org/abs/2509.20501)
*Kishor Datta Gupta, Mohd Ariful Haque, Marufa Kamal, Ahmed Rafi Hasan, Md. Mahfuzur Rahman, Roy George*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional clustering techniques often rely solely on similarity in the input data, limiting their ability to capture structural or semantic constraints that are critical in many domains. We introduce the Domain Aware Rule Triggered Variational Autoencoder (DARTVAE), a rule guided multimodal clustering framework that incorporates domain specific constraints directly into the representation learning process. DARTVAE extends the VAE architecture by embedding explicit rules, semantic representations, and data driven features into a unified latent space, while enforcing constraint compliance through rule consistency and violation penalties in the loss function. Unlike conventional clustering methods that rely only on visual similarity or apply rules as post hoc filters, DARTVAE treats rules as first class learning signals. The rules are generated by LLMs, structured into knowledge graphs, and enforced through a loss function combining reconstruction, KL divergence, consistency, and violation penalties. Experiments on aircraft and automotive datasets demonstrate that rule guided clustering produces more operationally meaningful and interpretable clusters for example, isolating UAVs, unifying stealth aircraft, or separating SUVs from sedans while improving traditional clustering metrics. However, the framework faces challenges: LLM generated rules may hallucinate or conflict, excessive rules risk overfitting, and scaling to complex domains increases computational and consistency difficulties. By combining rule encodings with learned representations, DARTVAE achieves more meaningful and consistent clustering outcomes than purely data driven models, highlighting the utility of constraint guided multimodal clustering for complex, knowledge intensive settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20501) | **Categories:** cs.LG, cs.CV

---

### [2] [CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](https://arxiv.org/abs/2509.20712)
*Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}ontrolling \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20712) | **Categories:** cs.LG, cs.CL

---

### [3] [CaTS-Bench: Can Language Models Describe Numeric Time Series?](https://arxiv.org/abs/2509.20823)
*Luca Zhou, Pratham Yashwante, Marshall Fisher, Alessio Sampieri, Zihao Zhou, Fabio Galasso, Rose Yu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time series captioning, the task of describing numeric time series in natural language, requires numerical reasoning, trend interpretation, and contextual understanding. Existing benchmarks, however, often rely on synthetic data or overly simplistic captions, and typically neglect metadata and visual representations. To close this gap, we introduce CaTS-Bench, the first large-scale, real-world benchmark for Context-aware Time Series captioning. CaTS-Bench is derived from 11 diverse datasets reframed as captioning and Q&A tasks, comprising roughly 465k training and 105k test timestamps. Each sample includes a numeric series segment, contextual metadata, a line-chart image, and a caption. A key contribution of this work is the scalable pipeline used to generate reference captions: while most references are produced by an oracle LLM and verified through factual checks, human indistinguishability studies, and diversity analyses, we also provide a human-revisited subset of 579 test captions, refined from LLM outputs to ensure accuracy and human-like style. Beyond captioning, CaTS-Bench offers 460 multiple-choice questions targeting deeper aspects of time series reasoning. We further propose new tailored evaluation metrics and benchmark leading VLMs, highlighting both their strengths and persistent limitations. Together, these contributions establish CaTS-Bench and its captioning pipeline as a reliable and extensible foundation for future research at the intersection of time series analysis and foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20823) | **Categories:** cs.LG, cs.AI, cs.CV

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting](https://arxiv.org/abs/2509.20499)
*Boqi Li, Siyuan Li, Weiyi Wang, Anran Li, Zhong Cao, Henry X. Liu*

Main category: cs.RO

TL;DR: 本文提出了一种结合航点预测器和多模态大语言模型（MLLM）的零样本框架，用于解决连续环境下的视觉语言导航（VLN）问题。


<details>
  <summary>Details</summary>
Motivation: 解决连续环境中视觉语言导航（VLN）任务的挑战，该任务要求智能体理解自然语言指令、感知周围环境并规划底层动作。

Method: 提出一个零样本框架，该框架集成了简化的航点预测器与多模态大语言模型（MLLM），利用抽象障碍物地图生成线性可达的航点，并将其整合到动态更新的拓扑图中，通过编码空间结构和探索历史来提示 MLLM，实现局部路径规划和纠错。

Result: 在R2R-CE和RxR-CE数据集上实现了最先进的零样本性能，成功率分别为41%和36%，优于现有方法。

Conclusion: 该方法通过结合航点预测器和多模态大语言模型，有效解决了连续环境下的视觉语言导航问题，并在实验中取得了显著的性能提升。

Abstract: 随着基础模型和机器人技术的快速发展，视觉语言导航（VLN）已成为具身智能体的关键任务，具有广泛的实际应用。我们研究了连续环境中的VLN，这是一个特别具有挑战性的环境，其中智能体必须共同解释自然语言指令、感知其周围环境并规划低级动作。我们提出了一个零样本框架，该框架集成了简化的但有效的航点预测器与多模态大型语言模型（MLLM）。该预测器在抽象障碍物地图上运行，生成线性可达的航点，这些航点被合并到具有显式访问记录的动态更新的拓扑图中。图和访问信息被编码到提示中，从而能够对空间结构和探索历史进行推理，以鼓励探索并使 MLLM 具备用于纠错的本地路径规划。在 R2R-CE 和 RxR-CE 上的大量实验表明，我们的方法实现了最先进的零样本性能，成功率分别为 41% 和 36%，优于先前的最先进方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20499) | **Categories:** cs.RO, cs.AI

---

### [2] [Digital Twin-Guided Robot Path Planning: A Beta-Bernoulli Fusion with Large Language Model as a Sensor](https://arxiv.org/abs/2509.20709)
*Mani Amani, Reza Akhavian*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Integrating natural language (NL) prompts into robotic mission planning has attracted significant interest in recent years. In the construction domain, Building Information Models (BIM) encapsulate rich NL descriptions of the environment. We present a novel framework that fuses NL directives with BIM-derived semantic maps via a Beta-Bernoulli Bayesian fusion by interpreting the LLM as a sensor: each obstacle's design-time repulsive coefficient is treated as a Beta(alpha, beta) random variable and LLM-returned danger scores are incorporated as pseudo-counts to update alpha and beta. The resulting posterior mean yields a continuous, context-aware repulsive gain that augments a Euclidean-distance-based potential field for cost heuristics. By adjusting gains based on sentiment and context inferred from user prompts, our method guides robots along safer, more context-aware paths. This provides a numerically stable method that can chain multiple natural commands and prompts from construction workers and foreman to enable planning while giving flexibility to be integrated in any learned or classical AI framework. Simulation results demonstrate that this Beta-Bernoulli fusion yields both qualitative and quantitative improvements in path robustness and validity.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20709) | **Categories:** cs.RO

---

### [3] [SLAM-Free Visual Navigation with Hierarchical Vision-Language Perception and Coarse-to-Fine Semantic Topological Planning](https://arxiv.org/abs/2509.20739)
*Guoyang Zhao, Yudong Li, Weiqing Qi, Kai Zhang, Bonan Liu, Kai Chen, Haoang Li, Jun Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Conventional SLAM pipelines for legged robot navigation are fragile under rapid motion, calibration demands, and sensor drift, while offering limited semantic reasoning for task-driven exploration. To deal with these issues, we propose a vision-only, SLAM-free navigation framework that replaces dense geometry with semantic reasoning and lightweight topological representations. A hierarchical vision-language perception module fuses scene-level context with object-level cues for robust semantic inference. And a semantic-probabilistic topological map supports coarse-to-fine planning: LLM-based global reasoning for subgoal selection and vision-based local planning for obstacle avoidance. Integrated with reinforcement-learning locomotion controllers, the framework is deployable across diverse legged robot platforms. Experiments in simulation and real-world settings demonstrate consistent improvements in semantic accuracy, planning quality, and navigation success, while ablation studies further showcase the necessity of both hierarchical perception and fine local planning. This work introduces a new paradigm for SLAM-free, vision-language-driven navigation, shifting robotic exploration from geometry-centric mapping to semantics-driven decision making.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20739) | **Categories:** cs.RO, cs.CV

---

### [4] [MTRDrive: Memory-Tool Synergistic Reasoning for Robust Autonomous Driving in Corner Cases](https://arxiv.org/abs/2509.20843)
*Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, Diange Yang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language Models(VLMs) have demonstrated significant potential for end-to-end autonomous driving, yet a substantial gap remains between their current capabilities and the reliability necessary for real-world deployment. A critical challenge is their fragility, characterized by hallucinations and poor generalization in out-of-distribution (OOD) scenarios. To bridge this gap, we introduce MTRDrive, a novel framework that integrates procedural driving experiences with a dynamic toolkit to enhance generalization and proactive decision-making.   MTRDrive addresses these limitations through a closed-loop system that combines a memory-based experience retrieval mechanism with dynamic toolkits. This synergy enables the model to interact more effectively with its environment, improving both reasoning and decision-making capabilities with the help of our memory-tool synergistic reasoning. Additionally, we introduce a new benchmark based on complex Roadwork construction scenarios to rigorously evaluate zero-shot generalization.   Extensive experiments demonstrate the superior effectiveness of our approach. On the public NAVSIM benchmark, our 3B-parameter MTRDrive model achieves an exceptional PDMS of 88.3 without chain-of-thought and sets a state-of-the-art performance bar on high-level planning, with a driving metric score of 79.8\% and a planning accuracy of 82.6\%. Rigorous zero-shot evaluation on the new Roadwork-VLM benchmark shows a strong ability to reason robustly in unseen scenarios, achieving a driving metric score of 80.2\%. These results highlight MTRDrive's potential to advance autonomous driving toward safer and more reliable systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20843) | **Categories:** cs.RO

---

### [5] [Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation](https://arxiv.org/abs/2509.20623)
*Satyajeet Das, Darren Chiu, Zhehui Huang, Lars Lindemann, Gaurav S. Sukhatme*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning has enabled significant progress in complex domains such as coordinating and navigating multiple quadrotors. However, even well-trained policies remain vulnerable to collisions in obstacle-rich environments. Addressing these infrequent but critical safety failures through retraining or fine-tuning is costly and risks degrading previously learned skills. Inspired by activation steering in large language models and latent editing in computer vision, we introduce a framework for inference-time Latent Activation Editing (LAE) that refines the behavior of pre-trained policies without modifying their weights or architecture. The framework operates in two stages: (i) an online classifier monitors intermediate activations to detect states associated with undesired behaviors, and (ii) an activation editing module that selectively modifies flagged activations to shift the policy towards safer regimes. In this work, we focus on improving safety in multi-quadrotor navigation. We hypothesize that amplifying a policy's internal perception of risk can induce safer behaviors. We instantiate this idea through a latent collision world model trained to predict future pre-collision activations, thereby prompting earlier and more cautious avoidance responses. Extensive simulations and real-world Crazyflie experiments demonstrate that LAE achieves statistically significant reduction in collisions (nearly 90% fewer cumulative collisions compared to the unedited baseline) and substantially increases the fraction of collision-free trajectories, while preserving task completion. More broadly, our results establish LAE as a lightweight paradigm, feasible on resource-constrained hardware, for post-deployment refinement of learned robot policies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20623) | **Categories:** cs.RO

---

### [6] [Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation](https://arxiv.org/abs/2509.20681)
*Wei-Teng Chu, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20681) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [7] [Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework](https://arxiv.org/abs/2509.20705)
*Reza Akhavian, Mani Amani, Johannes Mootz, Robert Ashe, Behrad Beheshti*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20705) | **Categories:** cs.RO

---

### [8] [Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement](https://arxiv.org/abs/2509.20938)
*Jianbo Zhao, Taiyu Ban, Xiangjie Li, Xingtai Gui, Hangning Zhou, Lei Liu, Hongwei Zhao, Bin Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The inherent sequential modeling capabilities of autoregressive models make them a formidable baseline for end-to-end planning in autonomous driving. Nevertheless, their performance is constrained by a spatio-temporal misalignment, as the planner must condition future actions on past sensory data. This creates an inconsistent worldview, limiting the upper bound of performance for an otherwise powerful approach. To address this, we propose a Time-Invariant Spatial Alignment (TISA) module that learns to project initial environmental features into a consistent ego-centric frame for each future time step, effectively correcting the agent's worldview without explicit future scene prediction. In addition, we employ a kinematic action prediction head (i.e., acceleration and yaw rate) to ensure physically feasible trajectories. Finally, we introduce a multi-objective post-training stage using Direct Preference Optimization (DPO) to move beyond pure imitation. Our approach provides targeted feedback on specific driving behaviors, offering a more fine-grained learning signal than the single, overall objective used in standard DPO. Our model achieves a state-of-the-art 89.8 PDMS on the NAVSIM dataset among autoregressive models. The video document is available at https://tisa-dpo-e2e.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20938) | **Categories:** cs.RO, cs.CV

---

### [9] [KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models](https://arxiv.org/abs/2509.21027)
*Sibo Li, Qianyue Hao, Yu Shang, Yong Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic world models are a promising paradigm for forecasting future environment states, yet their inference speed and the physical plausibility of generated trajectories remain critical bottlenecks, limiting their real-world applications. This stems from the redundancy of the prevailing frame-to-frame generation approach, where the model conducts costly computation on similar frames, as well as neglecting the semantic importance of key transitions. To address this inefficiency, we propose KeyWorld, a framework that improves text-conditioned robotic world models by concentrating transformers computation on a few semantic key frames while employing a lightweight convolutional model to fill the intermediate frames. Specifically, KeyWorld first identifies significant transitions by iteratively simplifying the robot's motion trajectories, obtaining the ground truth key frames. Then, a DiT model is trained to reason and generate these physically meaningful key frames from textual task descriptions. Finally, a lightweight interpolator efficiently reconstructs the full video by inpainting all intermediate frames. Evaluations on the LIBERO benchmark demonstrate that KeyWorld achieves a 5.68$\times$ acceleration compared to the frame-to-frame generation baseline, and focusing on the motion-aware key frames further contributes to the physical validity of the generated videos, especially on complex tasks. Our approach highlights a practical path toward deploying world models in real-time robotic control and other domains requiring both efficient and effective world models. Code is released at https://anonymous.4open.science/r/Keyworld-E43D.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21027) | **Categories:** cs.RO, cs.CV

---

### [10] [Cross-Modal Instructions for Robot Motion Generation](https://arxiv.org/abs/2509.21107)
*William Barron, Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Teaching robots novel behaviors typically requires motion demonstrations via teleoperation or kinaesthetic teaching, that is, physically guiding the robot. While recent work has explored using human sketches to specify desired behaviors, data collection remains cumbersome, and demonstration datasets are difficult to scale. In this paper, we introduce an alternative paradigm, Learning from Cross-Modal Instructions, where robots are shaped by demonstrations in the form of rough annotations, which can contain free-form text labels, and are used in lieu of physical motion. We introduce the CrossInstruct framework, which integrates cross-modal instructions as examples into the context input to a foundational vision-language model (VLM). The VLM then iteratively queries a smaller, fine-tuned model, and synthesizes the desired motion over multiple 2D views. These are then subsequently fused into a coherent distribution over 3D motion trajectories in the robot's workspace. By incorporating the reasoning of the large VLM with a fine-grained pointing model, CrossInstruct produces executable robot behaviors that generalize beyond the environment of in the limited set of instruction examples. We then introduce a downstream reinforcement learning pipeline that leverages CrossInstruct outputs to efficiently learn policies to complete fine-grained tasks. We rigorously evaluate CrossInstruct on benchmark simulation tasks and real hardware, demonstrating effectiveness without additional fine-tuning and providing a strong initialization for policies subsequently refined via reinforcement learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21107) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [11] [Human-like Navigation in a World Built for Humans](https://arxiv.org/abs/2509.21189)
*Bhargav Chandaka, Gloria X. Wang, Haozhe Chen, Henry Che, Albert J. Zhai, Shenlong Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: When navigating in a man-made environment they haven't visited before--like an office building--humans employ behaviors such as reading signs and asking others for directions. These behaviors help humans reach their destinations efficiently by reducing the need to search through large areas. Existing robot navigation systems lack the ability to execute such behaviors and are thus highly inefficient at navigating within large environments. We present ReasonNav, a modular navigation system which integrates these human-like navigation skills by leveraging the reasoning capabilities of a vision-language model (VLM). We design compact input and output abstractions based on navigation landmarks, allowing the VLM to focus on language understanding and reasoning. We evaluate ReasonNav on real and simulated navigation tasks and show that the agent successfully employs higher-order reasoning to navigate efficiently in large, complex buildings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21189) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [12] [\LARGE GMP$^{3}$: Learning-Driven, Bellman-Guided Trajectory Planning for UAVs in Real-Time on SE(3)](https://arxiv.org/abs/2509.21264)
*Babak Salamat, Dominik Mattern, Sebastian-Sven Olzem, Gerhard Elsbacher, Christian Seidel, Andrea M. Tonello*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose $\text{GMP}^{3}$, a multiphase global path planning framework that generates dynamically feasible three-dimensional trajectories for unmanned aerial vehicles (UAVs) operating in cluttered environments. The framework extends traditional path planning from Euclidean position spaces to the Lie group $\mathrm{SE}(3)$, allowing joint learning of translational motion and rotational dynamics. A modified Bellman-based operator is introduced to support reinforcement learning (RL) policy updates while leveraging prior trajectory information for improved convergence. $\text{GMP}^{3}$ is designed as a distributed framework in which agents influence each other and share policy information along the trajectory: each agent refines its assigned segment and shares with its neighbors via a consensus-based scheme, enabling cooperative policy updates and convergence toward a path shaped globally even under kinematic constraints. We also propose DroneManager, a modular ground control software that interfaces the planner with real UAV platforms via the MAVLink protocol, supporting real-time deployment and feedback. Simulation studies and indoor flight experiments validate the effectiveness of the proposed method in constrained 3D environments, demonstrating reliable obstacle avoidance and smooth, feasible trajectories across both position and orientation. The open-source implementation is available at https://github.com/Domattee/DroneManager

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21264) | **Categories:** cs.RO

---
