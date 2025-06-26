# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-27

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算语言学 (Computation and Language) (1)](#cs-cl)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)
- [eess.AS (1)](#eess-as)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Case-based Reasoning Augmented Large Language Model Framework for Decision Making in Realistic Safety-Critical Driving Scenarios](https://arxiv.org/abs/2506.20531)
*Wenbin Gan, Minh-Son Dao, Koji Zettsu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Driving in safety-critical scenarios requires quick, context-aware decision-making grounded in both situational understanding and experiential reasoning. Large Language Models (LLMs), with their powerful general-purpose reasoning capabilities, offer a promising foundation for such decision-making. However, their direct application to autonomous driving remains limited due to challenges in domain adaptation, contextual grounding, and the lack of experiential knowledge needed to make reliable and interpretable decisions in dynamic, high-risk environments. To address this gap, this paper presents a Case-Based Reasoning Augmented Large Language Model (CBR-LLM) framework for evasive maneuver decision-making in complex risk scenarios. Our approach integrates semantic scene understanding from dashcam video inputs with the retrieval of relevant past driving cases, enabling LLMs to generate maneuver recommendations that are both context-sensitive and human-aligned. Experiments across multiple open-source LLMs show that our framework improves decision accuracy, justification quality, and alignment with human expert behavior. Risk-aware prompting strategies further enhance performance across diverse risk types, while similarity-based case retrieval consistently outperforms random sampling in guiding in-context learning. Case studies further demonstrate the framework's robustness in challenging real-world conditions, underscoring its potential as an adaptive and trustworthy decision-support tool for intelligent driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20531) | **Categories:** cs.AI, cs.CY

---

### [2] [Language Modeling by Language Models](https://arxiv.org/abs/2506.20249)
*Junyan Cheng, Peter Clark, Kyle Richardson*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Can we leverage LLMs to model the process of discovering novel language model (LM) architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system, Genesys, employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M$\sim$350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., $\sim$86\% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous discovery systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20249) | **Categories:** cs.AI, cs.CL, cs.MA

---

### [3] [DiaLLMs: EHR Enhanced Clinical Conversational System for Clinical Test Recommendation and Diagnosis Prediction](https://arxiv.org/abs/2506.20059)
*Weijieying Ren, Tianxiang Zhao, Lei Wang, Tianchun Wang, Vasant Honavar*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in Large Language Models (LLMs) have led to remarkable progresses in medical consultation. However, existing medical LLMs overlook the essential role of Electronic Health Records (EHR) and focus primarily on diagnosis recommendation, limiting their clinical applicability. We propose DiaLLM, the first medical LLM that integrates heterogeneous EHR data into clinically grounded dialogues, enabling clinical test recommendation, result interpretation, and diagnosis prediction to better align with real-world medical practice. To construct clinically grounded dialogues from EHR, we design a Clinical Test Reference (CTR) strategy that maps each clinical code to its corresponding description and classifies test results as "normal" or "abnormal". Additionally, DiaLLM employs a reinforcement learning framework for evidence acquisition and automated diagnosis. To handle the large action space, we introduce a reject sampling strategy to reduce redundancy and improve exploration efficiency. Furthermore, a confirmation reward and a class-sensitive diagnosis reward are designed to guide accurate diagnosis prediction. Extensive experimental results demonstrate that DiaLLM outperforms baselines in clinical test recommendation and diagnosis prediction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20059) | **Categories:** cs.AI

---

### [4] [Mobile-R1: Towards Interactive Reinforcement Learning for VLM-Based Mobile Agent via Task-Level Rewards](https://arxiv.org/abs/2506.20332)
*Jihao Gu, Qihang Ai, Yingyao Wang, Pi Bu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Ziming Wang, Yingxiu Zhao, Ming-Liang Zhang, Jun Song, Yuning Jiang, Bo Zheng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-language model-based mobile agents have gained the ability to not only understand complex instructions and mobile screenshots, but also optimize their action outputs via thinking and reasoning, benefiting from reinforcement learning, such as Group Relative Policy Optimization (GRPO). However, existing research centers on offline reinforcement learning training or online optimization using action-level rewards, which limits the agent's dynamic interaction with the environment. This often results in agents settling into local optima, thereby weakening their ability for exploration and error action correction. To address these challenges, we introduce an approach called Mobile-R1, which employs interactive multi-turn reinforcement learning with task-level rewards for mobile agents. Our training framework consists of three stages: initial format finetuning, single-step online training via action-level reward, followed by online training via task-level reward based on multi-turn trajectories. This strategy is designed to enhance the exploration and error correction capabilities of Mobile-R1, leading to significant performance improvements. Moreover, we have collected a dataset covering 28 Chinese applications with 24,521 high-quality manual annotations and established a new benchmark with 500 trajectories. We will open source all resources, including the dataset, benchmark, model weight, and codes: https://mobile-r1.github.io/Mobile-R1/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20332) | **Categories:** cs.AI

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation](https://arxiv.org/abs/2506.19952)
*Deepon Halder, Thanmay Jayakumar, Raj Dabre*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19952) | **Categories:** cs.CL, cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [BrokenVideos: A Benchmark Dataset for Fine-Grained Artifact Localization in AI-Generated Videos](https://arxiv.org/abs/2506.20103)
*Jiahao Lin, Weixuan Peng, Bojia Zi, Yifeng Gao, Xianbiao Qi, Xingjun Ma, Yu-Gang Jiang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in deep generative models have led to significant progress in video generation, yet the fidelity of AI-generated videos remains limited. Synthesized content often exhibits visual artifacts such as temporally inconsistent motion, physically implausible trajectories, unnatural object deformations, and local blurring that undermine realism and user trust. Accurate detection and spatial localization of these artifacts are crucial for both automated quality control and for guiding the development of improved generative models. However, the research community currently lacks a comprehensive benchmark specifically designed for artifact localization in AI generated videos. Existing datasets either restrict themselves to video or frame level detection or lack the fine-grained spatial annotations necessary for evaluating localization methods. To address this gap, we introduce BrokenVideos, a benchmark dataset of 3,254 AI-generated videos with meticulously annotated, pixel-level masks highlighting regions of visual corruption. Each annotation is validated through detailed human inspection to ensure high quality ground truth. Our experiments show that training state of the art artifact detection models and multi modal large language models (MLLMs) on BrokenVideos significantly improves their ability to localize corrupted regions. Through extensive evaluation, we demonstrate that BrokenVideos establishes a critical foundation for benchmarking and advancing research on artifact localization in generative video models. The dataset is available at: https://broken-video-detection-datetsets.github.io/Broken-Video-Detection-Datasets.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20103) | **Categories:** cs.CV, cs.AI, I.4

---

### [2] [From 2D to 3D Cognition: A Brief Survey of General World Models](https://arxiv.org/abs/2506.20134)
*Ningwei Xie, Zizi Tian, Lei Yang, Xiao-Ping Zhang, Meng Guo, Jie Li*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: World models have garnered increasing attention in the development of artificial general intelligence (AGI), serving as computational frameworks for learning representations of the external world and forecasting future states. While early efforts focused on 2D visual perception and simulation, recent 3D-aware generative world models have demonstrated the ability to synthesize geometrically consistent, interactive 3D environments, marking a shift toward 3D spatial cognition. Despite rapid progress, the field lacks systematic analysis to categorize emerging techniques and clarify their roles in advancing 3D cognitive world models. This survey addresses this need by introducing a conceptual framework, providing a structured and forward-looking review of world models transitioning from 2D perception to 3D cognition. Within this framework, we highlight two key technological drivers, particularly advances in 3D representations and the incorporation of world knowledge, as fundamental pillars. Building on these, we dissect three core cognitive capabilities that underpin 3D world modeling: 3D physical scene generation, 3D spatial reasoning, and 3D spatial interaction. We further examine the deployment of these capabilities in real-world applications, including embodied AI, autonomous driving, digital twin, and gaming/VR. Finally, we identify challenges across data, modeling, and deployment, and outline future directions for advancing more robust and generalizable 3D world models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20134) | **Categories:** cs.CV

---

### [3] [A Transformer Based Handwriting Recognition System Jointly Using Online and Offline Features](https://arxiv.org/abs/2506.20255)
*Ayush Lodh, Ritabrata Chakraborty, Shivakumara Palaiahnakote, Umapada Pal*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We posit that handwriting recognition benefits from complementary cues carried by the rasterized complex glyph and the pen's trajectory, yet most systems exploit only one modality. We introduce an end-to-end network that performs early fusion of offline images and online stroke data within a shared latent space. A patch encoder converts the grayscale crop into fixed-length visual tokens, while a lightweight transformer embeds the $(x, y, \text{pen})$ sequence. Learnable latent queries attend jointly to both token streams, yielding context-enhanced stroke embeddings that are pooled and decoded under a cross-entropy loss objective. Because integration occurs before any high-level classification, temporal cues reinforce each other during representation learning, producing stronger writer independence. Comprehensive experiments on IAMOn-DB and VNOn-DB demonstrate that our approach achieves state-of-the-art accuracy, exceeding previous bests by up to 1\%. Our study also shows adaptation of this pipeline with gesturification on the ISI-Air dataset. Our code can be found here.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20255) | **Categories:** cs.CV, cs.LG

---

### [4] [Video Perception Models for 3D Scene Synthesis](https://arxiv.org/abs/2506.20601)
*Rui Huang, Guangyao Zhai, Zuria Bauer, Marc Pollefeys, Federico Tombari, Leonidas Guibas, Gao Huang, Francis Engelmann*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditionally, 3D scene synthesis requires expert knowledge and significant manual effort. Automating this process could greatly benefit fields such as architectural design, robotics simulation, virtual reality, and gaming. Recent approaches to 3D scene synthesis often rely on the commonsense reasoning of large language models (LLMs) or strong visual priors of modern image generation models. However, current LLMs demonstrate limited 3D spatial reasoning ability, which restricts their ability to generate realistic and coherent 3D scenes. Meanwhile, image generation-based methods often suffer from constraints in viewpoint selection and multi-view inconsistencies. In this work, we present Video Perception models for 3D Scene synthesis (VIPScene), a novel framework that exploits the encoded commonsense knowledge of the 3D physical world in video generation models to ensure coherent scene layouts and consistent object placements across views. VIPScene accepts both text and image prompts and seamlessly integrates video generation, feedforward 3D reconstruction, and open-vocabulary perception models to semantically and geometrically analyze each object in a scene. This enables flexible scene synthesis with high realism and structural consistency. For more precise analysis, we further introduce First-Person View Score (FPVScore) for coherence and plausibility evaluation, utilizing continuous first-person perspective to capitalize on the reasoning ability of multimodal large language models. Extensive experiments show that VIPScene significantly outperforms existing methods and generalizes well across diverse scenarios. The code will be released.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20601) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [FlightKooba: A Fast Interpretable FTP Model](https://arxiv.org/abs/2506.19885)
*Jing Lu, Xuan Wu, Yizhun Tian, Songhan Fan, Yali Fang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The Koopman theory is a powerful and effective modeling tool for converting nonlinear systems into linear representations, and flight trajectory prediction (FTP) is a complex nonlinear system. However, current models applying the Koopman theory to FTP tasks are not very effective, model interpretability is indeed an issue, and the Koopman operators are computationally intensive, resulting in long training times. To address this issue, this paper proposes a new modeling and control framework based on the HIPPO method, the Koopman theory, and state space equations from cybernetics: FlightKooba. Inspired by the idea of structural state space equations, FlightKooba directly constructs the Koopman operators from data. This makes the framework highly interpretable and significantly reduces the number of trainable parameters in the module, thereby greatly reducing training time. Experiments have demonstrated the superiority of the FlightKooba modeling method in terms of time and memory consumption (training time comparable to the Mamba module without using CUDA-level acceleration; memory reduced by more than 50% on most datasets, with a tenfold reduction in the number of parameters), essentially completing the FTP task. It provides a new method for the fast computation of the Koopman operators, opening up new possibilities for the combination of time series forecasting and control.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19885) | **Categories:** cs.LG, cs.AI, cs.SY, eess.SY

---

### [2] [Automated Generation of Diverse Courses of Actions for Multi-Agent Operations using Binary Optimization and Graph Learning](https://arxiv.org/abs/2506.20031)
*Prithvi Poddar, Ehsan Tarkesh Esfahani, Karthik Dantu, Souma Chowdhury*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Operations in disaster response, search \& rescue, and military missions that involve multiple agents demand automated processes to support the planning of the courses of action (COA). Moreover, traverse-affecting changes in the environment (rain, snow, blockades, etc.) may impact the expected performance of a COA, making it desirable to have a pool of COAs that are diverse in task distributions across agents. Further, variations in agent capabilities, which could be human crews and/or autonomous systems, present practical opportunities and computational challenges to the planning process. This paper presents a new theoretical formulation and computational framework to generate such diverse pools of COAs for operations with soft variations in agent-task compatibility. Key to the problem formulation is a graph abstraction of the task space and the pool of COAs itself to quantify its diversity. Formulating the COAs as a centralized multi-robot task allocation problem, a genetic algorithm is used for (order-ignoring) allocations of tasks to each agent that jointly maximize diversity within the COA pool and overall compatibility of the agent-task mappings. A graph neural network is trained using a policy gradient approach to then perform single agent task sequencing in each COA, which maximizes completion rates adaptive to task features. Our tests of the COA generation process in a simulated environment demonstrate significant performance gain over a random walk baseline, small optimality gap in task sequencing, and execution time of about 50 minutes to plan up to 20 COAs for 5 agent/100 task operations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20031) | **Categories:** cs.LG, cs.AI

---

### [3] [Towards Interpretable and Efficient Feature Selection in Trajectory Datasets: A Taxonomic Approach](https://arxiv.org/abs/2506.20359)
*Chanuka Don Samarasinghage, Dhruv Gulabani*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory analysis is not only about obtaining movement data, but it is also of paramount importance in understanding the pattern in which an object moves through space and time, as well as in predicting its next move. Due to the significant interest in the area, data collection has improved substantially, resulting in a large number of features becoming available for training and predicting models. However, this introduces a high-dimensionality-induced feature explosion problem, which reduces the efficiency and interpretability of the data, thereby reducing the accuracy of machine learning models. To overcome this issue, feature selection has become one of the most prevalent tools. Thus, the objective of this paper was to introduce a taxonomy-based feature selection method that categorizes features based on their internal structure. This approach classifies the data into geometric and kinematic features, further categorizing them into curvature, indentation, speed, and acceleration. The comparative analysis indicated that a taxonomy-based approach consistently achieved comparable or superior predictive performance. Furthermore, due to the taxonomic grouping, which reduces combinatorial space, the time taken to select features was drastically reduced. The taxonomy was also used to gain insights into what feature sets each dataset was more sensitive to. Overall, this study provides robust evidence that a taxonomy-based feature selection method can add a layer of interpretability, reduce dimensionality and computational complexity, and contribute to high-level decision-making. It serves as a step toward providing a methodological framework for researchers and practitioners dealing with trajectory datasets and contributing to the broader field of explainable artificial intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20359) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Generating and Customizing Robotic Arm Trajectories using Neural Networks](https://arxiv.org/abs/2506.20259)
*Andrej Lúčny, Matilde Antonj, Carlo Mazzola, Hana Hornáčková, Igor Farkaš*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20259) | **Categories:** cs.RO, cs.AI, 68T40, 93C85, 70E60, I.2.9

---

### [2] [Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles](https://arxiv.org/abs/2506.20311)
*Jingwen Wei*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The growing use of mobile robots in sectors such as automotive, agriculture, and rescue operations reflects progress in robotics and autonomy. In unmanned aerial vehicles (UAVs), most research emphasizes visual SLAM, sensor fusion, and path planning. However, applying UAVs to search and rescue missions in disaster zones remains underexplored, especially for autonomous navigation.   This report develops methods for real-time and secure UAV maneuvering in complex 3D environments, crucial during forest fires. Building upon past research, it focuses on designing navigation algorithms for unfamiliar and hazardous environments, aiming to improve rescue efficiency and safety through UAV-based early warning and rapid response.   The work unfolds in phases. First, a 2D fusion navigation strategy is explored, initially for mobile robots, enabling safe movement in dynamic settings. This sets the stage for advanced features such as adaptive obstacle handling and decision-making enhancements. Next, a novel 3D reactive navigation strategy is introduced for collision-free movement in forest fire simulations, addressing the unique challenges of UAV operations in such scenarios.   Finally, the report proposes a unified control approach that integrates UAVs and unmanned ground vehicles (UGVs) for coordinated rescue missions in forest environments. Each phase presents challenges, proposes control models, and validates them with mathematical and simulation-based evidence. The study offers practical value and academic insights for improving the role of UAVs in natural disaster rescue operations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20311) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [3] [Finding the Easy Way Through -- the Probabilistic Gap Planner for Social Robot Navigation](https://arxiv.org/abs/2506.20320)
*Malte Probst, Raphael Wenzel, Tim Puphal, Monica Dasi, Nico A. Steinhardt, Sango Matsuzaki, Misa Komuro*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In Social Robot Navigation, autonomous agents need to resolve many sequential interactions with other agents. State-of-the art planners can efficiently resolve the next, imminent interaction cooperatively and do not focus on longer planning horizons. This makes it hard to maneuver scenarios where the agent needs to select a good strategy to find gaps or channels in the crowd. We propose to decompose trajectory planning into two separate steps: Conflict avoidance for finding good, macroscopic trajectories, and cooperative collision avoidance (CCA) for resolving the next interaction optimally. We propose the Probabilistic Gap Planner (PGP) as a conflict avoidance planner. PGP modifies an established probabilistic collision risk model to include a general assumption of cooperativity. PGP biases the short-term CCA planner to head towards gaps in the crowd. In extensive simulations with crowds of varying density, we show that using PGP in addition to state-of-the-art CCA planners improves the agents' performance: On average, agents keep more space to others, create less tension, and cause fewer collisions. This typically comes at the expense of slightly longer paths. PGP runs in real-time on WaPOCHI mobile robot by Honda R&D.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20320) | **Categories:** cs.RO

---

### [4] [Communication-Aware Map Compression for Online Path-Planning: A Rate-Distortion Approach](https://arxiv.org/abs/2506.20579)
*Ali Reza Pedram, Evangelos Psomiadis, Dipankar Maity, Panagiotis Tsiotras*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper addresses the problem of collaborative navigation in an unknown environment, where two robots, referred to in the sequel as the Seeker and the Supporter, traverse the space simultaneously. The Supporter assists the Seeker by transmitting a compressed representation of its local map under bandwidth constraints to support the Seeker's path-planning task. We introduce a bit-rate metric based on the expected binary codeword length to quantify communication cost. Using this metric, we formulate the compression design problem as a rate-distortion optimization problem that determines when to communicate, which regions of the map should be included in the compressed representation, and at what resolution (i.e., quantization level) they should be encoded. Our formulation allows different map regions to be encoded at varying quantization levels based on their relevance to the Seeker's path-planning task. We demonstrate that the resulting optimization problem is convex, and admits a closed-form solution known in the information theory literature as reverse water-filling, enabling efficient, low-computation, and real-time implementation. Additionally, we show that the Seeker can infer the compression decisions of the Supporter independently, requiring only the encoded map content and not the encoding policy itself to be transmitted, thereby reducing communication overhead. Simulation results indicate that our method effectively constructs compressed, task-relevant map representations, both in content and resolution, that guide the Seeker's planning decisions even under tight bandwidth limitations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20579) | **Categories:** cs.RO

---

### [5] [DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy](https://arxiv.org/abs/2506.20668)
*Sungjae Park, Homanga Bharadhwaj, Shubham Tulsiani*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose DemoDiffusion, a simple and scalable method for enabling robots to perform manipulation tasks in natural environments by imitating a single human demonstration. Our approach is based on two key insights. First, the hand motion in a human demonstration provides a useful prior for the robot's end-effector trajectory, which we can convert into a rough open-loop robot motion trajectory via kinematic retargeting. Second, while this retargeted motion captures the overall structure of the task, it may not align well with plausible robot actions in-context. To address this, we leverage a pre-trained generalist diffusion policy to modify the trajectory, ensuring it both follows the human motion and remains within the distribution of plausible robot actions. Our approach avoids the need for online reinforcement learning or paired human-robot data, enabling robust adaptation to new tasks and scenes with minimal manual effort. Experiments in both simulation and real-world settings show that DemoDiffusion outperforms both the base policy and the retargeted trajectory, enabling the robot to succeed even on tasks where the pre-trained generalist policy fails entirely. Project page: https://demodiffusion.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.20668) | **Categories:** cs.RO, cs.LG

---


## eess.AS [eess.AS]
### [1] [Speaker Embeddings to Improve Tracking of Intermittent and Moving Speakers](https://arxiv.org/abs/2506.19875)
*Taous Iatariene, Can Cui, Alexandre Guérin, Romain Serizel*

Main category: eess.AS

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Speaker tracking methods often rely on spatial observations to assign coherent track identities over time. This raises limits in scenarios with intermittent and moving speakers, i.e., speakers that may change position when they are inactive, thus leading to discontinuous spatial trajectories. This paper proposes to investigate the use of speaker embeddings, in a simple solution to this issue. We propose to perform identity reassignment post-tracking, using speaker embeddings. We leverage trajectory-related information provided by an initial tracking step and multichannel audio signal. Beamforming is used to enhance the signal towards the speakers' positions in order to compute speaker embeddings. These are then used to assign new track identities based on an enrollment pool. We evaluate the performance of the proposed speaker embedding-based identity reassignment method on a dataset where speakers change position during inactivity periods. Results show that it consistently improves the identity assignment performance of neural and standard tracking systems. In particular, we study the impact of beamforming and input duration for embedding extraction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19875) | **Categories:** eess.AS, cs.AI, cs.SD

---
