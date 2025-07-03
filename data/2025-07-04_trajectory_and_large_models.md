# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-04

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [人机交互 (Human-Computer Interaction) (1)](#cs-hc)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [cs.MA (1)](#cs-ma)
- [机器人学 (Robotics) (11)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [AI Agents and Agentic AI-Navigating a Plethora of Concepts for Future Manufacturing](https://arxiv.org/abs/2507.01376)
*Yinwang Ren, Yangyang Liu, Tang Ji, Xun Xu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: AI agents are autonomous systems designed to perceive, reason, and act within dynamic environments. With the rapid advancements in generative AI (GenAI), large language models (LLMs) and multimodal large language models (MLLMs) have significantly improved AI agents' capabilities in semantic comprehension, complex reasoning, and autonomous decision-making. At the same time, the rise of Agentic AI highlights adaptability and goal-directed autonomy in dynamic and complex environments. LLMs-based AI Agents (LLM-Agents), MLLMs-based AI Agents (MLLM-Agents), and Agentic AI contribute to expanding AI's capabilities in information processing, environmental perception, and autonomous decision-making, opening new avenues for smart manufacturing. However, the definitions, capability boundaries, and practical applications of these emerging AI paradigms in smart manufacturing remain unclear. To address this gap, this study systematically reviews the evolution of AI and AI agent technologies, examines the core concepts and technological advancements of LLM-Agents, MLLM-Agents, and Agentic AI, and explores their potential applications in and integration into manufacturing, along with the potential challenges they may face.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01376) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Geometry-aware 4D Video Generation for Robot Manipulation](https://arxiv.org/abs/2507.01099)
*Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, Shuran Song*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Understanding and predicting the dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of videos by supervising the model with cross-view pointmap alignment during training. This geometric supervision enables the model to learn a shared 3D representation of the scene, allowing it to predict future video sequences from novel viewpoints based solely on the given RGB-D observations, without requiring camera poses as inputs. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, supporting robust robot manipulation and generalization to novel camera viewpoints.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01099) | **Categories:** cs.CV, cs.AI, cs.LG, cs.RO

---


## 人机交互 (Human-Computer Interaction) [cs.HC]
### [1] [AI Meets Maritime Training: Precision Analytics for Enhanced Safety and Performance](https://arxiv.org/abs/2507.01274)
*Vishakha Lall, Yisi Liu*

Main category: cs.HC

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional simulator-based training for maritime professionals is critical for ensuring safety at sea but often depends on subjective trainer assessments of technical skills, behavioral focus, communication, and body language, posing challenges such as subjectivity, difficulty in measuring key features, and cognitive limitations. Addressing these issues, this study develops an AI-driven framework to enhance maritime training by objectively assessing trainee performance through visual focus tracking, speech recognition, and stress detection, improving readiness for high-risk scenarios. The system integrates AI techniques, including visual focus determination using eye tracking, pupil dilation analysis, and computer vision; communication analysis through a maritime-specific speech-to-text model and natural language processing; communication correctness using large language models; and mental stress detection via vocal pitch. Models were evaluated on data from simulated maritime scenarios with seafarers exposed to controlled high-stress events. The AI algorithms achieved high accuracy, with ~92% for visual detection, ~91% for maritime speech recognition, and ~90% for stress detection, surpassing existing benchmarks. The system provides insights into visual attention, adherence to communication checklists, and stress levels under demanding conditions. This study demonstrates how AI can transform maritime training by delivering objective performance analytics, enabling personalized feedback, and improving preparedness for real-world operational challenges.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01274) | **Categories:** cs.HC, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Distributional Soft Actor-Critic with Diffusion Policy](https://arxiv.org/abs/2507.01381)
*Tong Liu, Yinuo Wang, Xujie Song, Wenjun Zou, Liangfa Chen, Likun Wang, Bin Shuai, Jingliang Duan, Shengbo Eben Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning has been proven to be highly effective in handling complex control tasks. Traditional methods typically use unimodal distributions, such as Gaussian distributions, to model the output of value distributions. However, unimodal distribution often and easily causes bias in value function estimation, leading to poor algorithm performance. This paper proposes a distributional reinforcement learning algorithm called DSAC-D (Distributed Soft Actor Critic with Diffusion Policy) to address the challenges of estimating bias in value functions and obtaining multimodal policy representations. A multimodal distributional policy iteration framework that can converge to the optimal policy was established by introducing policy entropy and value distribution function. A diffusion value network that can accurately characterize the distribution of multi peaks was constructed by generating a set of reward samples through reverse sampling using a diffusion model. Based on this, a distributional reinforcement learning algorithm with dual diffusion of the value network and the policy network was derived. MuJoCo testing tasks demonstrate that the proposed algorithm not only learns multimodal policy, but also achieves state-of-the-art (SOTA) performance in all 9 control tasks, with significant suppression of estimation bias and total average return improvement of over 10\% compared to existing mainstream algorithms. The results of real vehicle testing show that DSAC-D can accurately characterize the multimodal distribution of different driving styles, and the diffusion policy network can characterize multimodal trajectories.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01381) | **Categories:** cs.LG, cs.AI

---


## cs.MA [cs.MA]
### [1] [RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms](https://arxiv.org/abs/2507.01378)
*Ziyao Wang, Rongpeng Li, Sizhao Li, Yuming Xiang, Haiping Wang, Zhifeng Zhao, Honggang Zhang*

Main category: cs.MA

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01378) | **Categories:** cs.MA, cs.AI, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [VLAD: A VLM-Augmented Autonomous Driving Framework with Hierarchical Planning and Interpretable Decision Process](https://arxiv.org/abs/2507.01284)
*Cristian Gariboldi, Hayato Tokida, Ken Kinjo, Yuki Asada, Alexander Carballo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in open-source Visual Language Models (VLMs) such as LLaVA, Qwen-VL, and Llama have catalyzed extensive research on their integration with diverse systems. The internet-scale general knowledge encapsulated within these models presents significant opportunities for enhancing autonomous driving perception, prediction, and planning capabilities. In this paper we propose VLAD, a vision-language autonomous driving model, which integrates a fine-tuned VLM with VAD, a state-of-the-art end-to-end system. We implement a specialized fine-tuning approach using custom question-answer datasets designed specifically to improve the spatial reasoning capabilities of the model. The enhanced VLM generates high-level navigational commands that VAD subsequently processes to guide vehicle operation. Additionally, our system produces interpretable natural language explanations of driving decisions, thereby increasing transparency and trustworthiness of the traditionally black-box end-to-end architecture. Comprehensive evaluation on the real-world nuScenes dataset demonstrates that our integrated system reduces average collision rates by 31.82% compared to baseline methodologies, establishing a new benchmark for VLM-augmented autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01284) | **Categories:** cs.RO, cs.AI, cs.CV, cs.ET, cs.LG

---

### [2] [LLM-based Realistic Safety-Critical Driving Video Generation](https://arxiv.org/abs/2507.01264)
*Yongjie Fu, Ruijian Zha, Pei Tian, Xuan Di*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Designing diverse and safety-critical driving scenarios is essential for evaluating autonomous driving systems. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) for few-shot code generation to automatically synthesize driving scenarios within the CARLA simulator, which has flexibility in scenario scripting, efficient code-based control of traffic participants, and enforcement of realistic physical dynamics. Given a few example prompts and code samples, the LLM generates safety-critical scenario scripts that specify the behavior and placement of traffic participants, with a particular focus on collision events. To bridge the gap between simulation and real-world appearance, we integrate a video generation pipeline using Cosmos-Transfer1 with ControlNet, which converts rendered scenes into realistic driving videos. Our approach enables controllable scenario generation and facilitates the creation of rare but critical edge cases, such as pedestrian crossings under occlusion or sudden vehicle cut-ins. Experimental results demonstrate the effectiveness of our method in generating a wide range of realistic, diverse, and safety-critical scenarios, offering a promising tool for simulation-based testing of autonomous vehicles.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01264) | **Categories:** cs.RO, cs.AI

---

### [3] [LANet: A Lane Boundaries-Aware Approach For Robust Trajectory Prediction](https://arxiv.org/abs/2507.01308)
*Muhammad Atta ur Rahman, Dooseop Choi, KyoungWook Min*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate motion forecasting is critical for safe and efficient autonomous driving, enabling vehicles to predict future trajectories and make informed decisions in complex traffic scenarios. Most of the current designs of motion prediction models are based on the major representation of lane centerlines, which limits their capability to capture critical road environments and traffic rules and constraints. In this work, we propose an enhanced motion forecasting model informed by multiple vector map elements, including lane boundaries and road edges, that facilitates a richer and more complete representation of driving environments. An effective feature fusion strategy is developed to merge information in different vector map components, where the model learns holistic information on road structures and their interactions with agents. Since encoding more information about the road environment increases memory usage and is computationally expensive, we developed an effective pruning mechanism that filters the most relevant map connections to the target agent, ensuring computational efficiency while maintaining essential spatial and semantic relationships for accurate trajectory prediction. Overcoming the limitations of lane centerline-based models, our method provides a more informative and efficient representation of the driving environment and advances the state of the art for autonomous vehicle motion forecasting. We verify our approach with extensive experiments on the Argoverse 2 motion forecasting dataset, where our method maintains competitiveness on AV2 while achieving improved performance.   Index Terms-Autonomous driving, trajectory prediction, vector map elements, road topology, connection pruning, Argoverse 2.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01308) | **Categories:** cs.RO, cs.CV

---

### [4] [Large Language Model-Driven Closed-Loop UAV Operation with Semantic Observations](https://arxiv.org/abs/2507.01930)
*Wenhao Wang, Yanyan Li, Long Jiao, Jiawei Yuan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have revolutionized robotic autonomy, including Unmanned Aerial Vehicles (UAVs). Recent studies have demonstrated the potential of LLMs for translating human instructions into executable control code for UAV operations. However, LLMs still face challenges from logical reasoning and complex decision-making, leading to concerns about the reliability of LLM-driven UAV operations. In this paper, we propose a LLM-driven closed-loop control framework that enables reliable UAV operations powered by effective feedback and refinement using two LLM modules, i.e., a Code Generator and an Evaluator. Our framework transforms numerical state observations from UAV operations into natural language trajectory descriptions to enhance the evaluator LLM's understanding of UAV dynamics for precise feedback generation. Our framework also enables a simulation-based refinement process, and hence eliminates the risks to physical UAVs caused by incorrect code execution during the refinement. Extensive experiments on UAV control tasks with different complexities are conducted. The experimental results show that our framework can achieve reliable UAV operations using LLMs, which significantly outperforms baseline approaches in terms of success rate and completeness with the increase of task complexity.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01930) | **Categories:** cs.RO

---

### [5] [2024 NASA SUITS Report: LLM-Driven Immersive Augmented Reality User Interface for Robotics and Space Exploration](https://arxiv.org/abs/2507.01206)
*Kathy Zhuang, Zixun Huang, Yukun Song, Rui Li, Yinuo Zhou, Allen Y. Yang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: As modern computing advances, new interaction paradigms have emerged, particularly in Augmented Reality (AR), which overlays virtual interfaces onto physical objects. This evolution poses challenges in machine perception, especially for tasks like 3D object pose estimation in complex, dynamic environments. Our project addresses critical issues in human-robot interaction within mobile AR, focusing on non-intrusive, spatially aware interfaces. We present URSA, an LLM-driven immersive AR system developed for NASA's 2023-2024 SUITS challenge, targeting future spaceflight needs such as the Artemis missions. URSA integrates three core technologies: a head-mounted AR device (e.g., HoloLens) for intuitive visual feedback, voice control powered by large language models for hands-free interaction, and robot tracking algorithms that enable accurate 3D localization in dynamic settings. To enhance precision, we leverage digital twin localization technologies, using datasets like DTTD-Mobile and specialized hardware such as the ZED2 camera for real-world tracking under noise and occlusion. Our system enables real-time robot control and monitoring via an AR interface, even in the absence of ground-truth sensors--vital for hazardous or remote operations. Key contributions include: (1) a non-intrusive AR interface with LLM-based voice input; (2) a ZED2-based dataset tailored for non-rigid robotic bodies; (3) a Local Mission Control Console (LMCC) for mission visualization; (4) a transformer-based 6DoF pose estimator (DTTDNet) optimized for depth fusion and real-time tracking; and (5) end-to-end integration for astronaut mission support. This work advances digital twin applications in robotics, offering scalable solutions for both aerospace and industrial domains.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01206) | **Categories:** cs.RO, cs.HC

---

### [6] [TriVLA: A Unified Triple-System-Based Unified Vision-Language-Action Model for General Robot Control](https://arxiv.org/abs/2507.01424)
*Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Xiangyang Xue, Yanwei Fu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in vision-language models (VLMs) for common-sense reasoning have led to the development of vision-language-action (VLA) models, enabling robots to perform generalized manipulation. Although existing autoregressive VLA methods design a specific architecture like dual-system to leverage large-scale pretrained knowledge, they tend to capture static information, often neglecting the dynamic aspects vital for embodied tasks. To this end, we propose TriVLA, a unified Vision-Language-Action model with a triple-system architecture for general robot control. The vision-language module (System 2) interprets the environment through vision and language instructions. The dynamics perception module (System 3) inherently produces visual representations that encompass both current static information and predicted future dynamics, thereby providing valuable guidance for policy learning. TriVLA utilizes pre-trained VLM model and fine-tunes pre-trained video foundation model on robot datasets along with internet human manipulation data. The subsequent policy learning module (System 1) generates fluid motor actions in real time. Experimental evaluation demonstrates that TriVLA operates at approximately 36 Hz and surpasses state-of-the-art imitation learning baselines on standard simulation benchmarks as well as challenging real-world manipulation tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01424) | **Categories:** cs.RO

---

### [7] [Quantum-Assisted Automatic Path-Planning for Robotic Quality Inspection in Industry 4.0](https://arxiv.org/abs/2507.01462)
*Eneko Osaba, Estibaliz Garrote, Pablo Miranda-Rodriguez, Alessia Ciacco, Itziar Cabanes, Aitziber Mancisidor*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This work explores the application of hybrid quantum-classical algorithms to optimize robotic inspection trajectories derived from Computer-Aided Design (CAD) models in industrial settings. By modeling the task as a 3D variant of the Traveling Salesman Problem, incorporating incomplete graphs and open-route constraints, this study evaluates the performance of two D-Wave-based solvers against classical methods such as GUROBI and Google OR-Tools. Results across five real-world cases demonstrate competitive solution quality with significantly reduced computation times, highlighting the potential of quantum approaches in automation under Industry 4.0.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01462) | **Categories:** cs.RO, cs.AI, cs.ET

---

### [8] [BioMARS: A Multi-Agent Robotic System for Autonomous Biological Experiments](https://arxiv.org/abs/2507.01485)
*Yibo Qiu, Zan Huang, Zhiyu Wang, Handi Liu, Yiling Qiao, Yifeng Hu, Shu'ang Sun, Hangke Peng, Ronald X Xu, Mingzhai Sun*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) and vision-language models (VLMs) have the potential to transform biological research by enabling autonomous experimentation. Yet, their application remains constrained by rigid protocol design, limited adaptability to dynamic lab conditions, inadequate error handling, and high operational complexity. Here we introduce BioMARS (Biological Multi-Agent Robotic System), an intelligent platform that integrates LLMs, VLMs, and modular robotics to autonomously design, plan, and execute biological experiments. BioMARS uses a hierarchical architecture: the Biologist Agent synthesizes protocols via retrieval-augmented generation; the Technician Agent translates them into executable robotic pseudo-code; and the Inspector Agent ensures procedural integrity through multimodal perception and anomaly detection. The system autonomously conducts cell passaging and culture tasks, matching or exceeding manual performance in viability, consistency, and morphological integrity. It also supports context-aware optimization, outperforming conventional strategies in differentiating retinal pigment epithelial cells. A web interface enables real-time human-AI collaboration, while a modular backend allows scalable integration with laboratory hardware. These results highlight the feasibility of generalizable, AI-driven laboratory automation and the transformative role of language-based reasoning in biological research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01485) | **Categories:** cs.RO, cs.AI, cs.MA, q-bio.QM

---

### [9] [MoIRA: Modular Instruction Routing Architecture for Multi-Task Robotics](https://arxiv.org/abs/2507.01843)
*Dmytro Kuzmenko, Nadiya Shvai*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Mixture-of-Experts (MoE) approaches have recently gained traction in robotics applications due to their ability to dynamically allocate computational resources and specialize sub-networks for distinct tasks or environmental contexts, enabling more efficient decision-making. Such systems often comprise sparsely activated experts combined under a single monolithic architecture and require a well-configured internal routing mechanism, which does not allow for selective low-level expert and router customization and requires additional training. We propose MoIRA, an architecture-agnostic modular MoE framework designed to coordinate existing experts with an external text-based router. MoIRA incorporates two zero-shot routing options: embedding-based similarity and prompt-driven language model inference. In our experiments, we choose large Vision-Language-Action models, gr00t-N1 and $\pi_0$, as the underlying experts, and train low-rank adapters for low-overhead inference. We evaluate MoIRA on various GR1 Humanoid tasks and LIBERO Spatial and Goal benchmarks, where it consistently outperforms generalist models and competes with other MoE pipelines. Additionally, we analyse the robustness of the proposed approach to the variations of the instructions. While relying solely on textual descriptions of tasks and experts, MoIRA demonstrates the practical viability of modular deployment with precise, low-effort routing and provides an alternative, scalable foundation for future multi-expert robotic systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01843) | **Categories:** cs.RO

---

### [10] [TypeTele: Releasing Dexterity in Teleoperation by Dexterous Manipulation Types](https://arxiv.org/abs/2507.01857)
*Yuhao Lin, Yi-Lin Wei, Haoran Liao, Mu Lin, Chengyi Xing, Hao Li, Dandan Zhang, Mark Cutkosky, Wei-Shi Zheng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Dexterous teleoperation plays a crucial role in robotic manipulation for real-world data collection and remote robot control. Previous dexterous teleoperation mostly relies on hand retargeting to closely mimic human hand postures. However, these approaches may fail to fully leverage the inherent dexterity of dexterous hands, which can execute unique actions through their structural advantages compared to human hands. To address this limitation, we propose TypeTele, a type-guided dexterous teleoperation system, which enables dexterous hands to perform actions that are not constrained by human motion patterns. This is achieved by introducing dexterous manipulation types into the teleoperation system, allowing operators to employ appropriate types to complete specific tasks. To support this system, we build an extensible dexterous manipulation type library to cover comprehensive dexterous postures used in manipulation tasks. During teleoperation, we employ a MLLM (Multi-modality Large Language Model)-assisted type retrieval module to identify the most suitable manipulation type based on the specific task and operator commands. Extensive experiments of real-world teleoperation and imitation learning demonstrate that the incorporation of manipulation types significantly takes full advantage of the dexterous robot's ability to perform diverse and complex tasks with higher success rates.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01857) | **Categories:** cs.RO

---

### [11] [A Survey on Vision-Language-Action Models: An Action Tokenization Perspective](https://arxiv.org/abs/2507.01925)
*Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, Zhiquan Qi, Yitao Liang, Yuanpei Chen, Yaodong Yang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The remarkable advancements of vision and language foundation models in multimodal understanding, reasoning, and generation has sparked growing efforts to extend such intelligence to the physical world, fueling the flourishing of vision-language-action (VLA) models. Despite seemingly diverse approaches, we observe that current VLA models can be unified under a single framework: vision and language inputs are processed by a series of VLA modules, producing a chain of \textit{action tokens} that progressively encode more grounded and actionable information, ultimately generating executable actions. We further determine that the primary design choice distinguishing VLA models lies in how action tokens are formulated, which can be categorized into language description, code, affordance, trajectory, goal state, latent representation, raw action, and reasoning. However, there remains a lack of comprehensive understanding regarding action tokens, significantly impeding effective VLA development and obscuring future directions. Therefore, this survey aims to categorize and interpret existing VLA research through the lens of action tokenization, distill the strengths and limitations of each token type, and identify areas for improvement. Through this systematic review and analysis, we offer a synthesized outlook on the broader evolution of VLA models, highlight underexplored yet promising directions, and contribute guidance for future research, hoping to bring the field closer to general-purpose intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01925) | **Categories:** cs.RO

---


## eess.SY [eess.SY]
### [1] [Time-Varying Coverage Control: A Distributed Tracker-Planner MPC Framework](https://arxiv.org/abs/2507.01567)
*Patrick Benito Eberhard, Johannes Köhler, Oliver Hüsser, Melanie N. Zeilinger, Andrea Carron*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time-varying coverage control addresses the challenge of coordinating multiple agents covering an environment where regions of interest change over time. This problem has broad applications, including the deployment of autonomous taxis and coordination in search and rescue operations. The achievement of effective coverage is complicated by the presence of time-varying density functions, nonlinear agent dynamics, and stringent system and safety constraints. In this paper, we present a distributed multi-agent control framework for time-varying coverage under nonlinear constrained dynamics. Our approach integrates a reference trajectory planner and a tracking model predictive control (MPC) scheme, which operate at different frequencies within a multi-rate framework. For periodic density functions, we demonstrate closed-loop convergence to an optimal configuration of trajectories and provide formal guarantees regarding constraint satisfaction, collision avoidance, and recursive feasibility. Additionally, we propose an efficient algorithm capable of handling nonperiodic density functions, making the approach suitable for practical applications. Finally, we validate our method through hardware experiments using a fleet of four miniature race cars.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01567) | **Categories:** eess.SY, cs.RO, cs.SY

---
