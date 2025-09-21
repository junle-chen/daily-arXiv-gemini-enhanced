# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-22

## 目录

- [计算语言学 (Computation and Language) (2)](#cs-cl)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (7)](#cs-ro)

## 计算语言学 (Computation and Language) [cs.CL]
### [1] [LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
*Hai Huang, Yann LeCun, Randall Balestriero*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Model (LLM) pretraining, finetuning, and evaluation rely on input-space reconstruction and generative capabilities. Yet, it has been observed in vision that embedding-space training objectives, e.g., with Joint Embedding Predictive Architectures (JEPAs), are far superior to their input-space counterpart. That mismatch in how training is achieved between language and vision opens up a natural question: {\em can language training methods learn a few tricks from the vision ones?} The lack of JEPA-style LLM is a testimony of the challenge in designing such objectives for language. In this work, we propose a first step in that direction where we develop LLM-JEPA, a JEPA based solution for LLMs applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to outperform the standard LLM training objectives by a significant margin across models, all while being robust to overfiting. Those findings are observed across numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models from the Llama3, OpenELM, Gemma2 and Olmo families. Code: https://github.com/rbalestr-lab/llm-jepa.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14252) | **Categories:** cs.CL, cs.AI

---

### [2] [DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models](https://arxiv.org/abs/2509.14268)
*Jiachen Fu, Chun-Le Guo, Chongyi Li*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid advancement of large language models (LLMs) has drawn urgent attention to the task of machine-generated text detection (MGTD). However, existing approaches struggle in complex real-world scenarios: zero-shot detectors rely heavily on scoring model's output distribution while training-based detectors are often constrained by overfitting to the training data, limiting generalization. We found that the performance bottleneck of training-based detectors stems from the misalignment between training objective and task needs. To address this, we propose Direct Discrepancy Learning (DDL), a novel optimization strategy that directly optimizes the detector with task-oriented knowledge. DDL enables the detector to better capture the core semantics of the detection task, thereby enhancing both robustness and generalization. Built upon this, we introduce DetectAnyLLM, a unified detection framework that achieves state-of-the-art MGTD performance across diverse LLMs. To ensure a reliable evaluation, we construct MIRAGE, the most diverse multi-task MGTD benchmark. MIRAGE samples human-written texts from 10 corpora across 5 text-domains, which are then re-generated or revised using 17 cutting-edge LLMs, covering a wide spectrum of proprietary models and textual styles. Extensive experiments on MIRAGE reveal the limitations of existing methods in complex environment. In contrast, DetectAnyLLM consistently outperforms them, achieving over a 70% performance improvement under the same training data and base scoring model, underscoring the effectiveness of our DDL. Project page: {https://fjc2005.github.io/detectanyllm}.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14268) | **Categories:** cs.CL, cs.AI, cs.CY

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DiffVL: Diffusion-Based Visual Localization on 2D Maps via BEV-Conditioned GPS Denoising](https://arxiv.org/abs/2509.14565)
*Li Gao, Hongyang Sun, Liu Liu, Yunhao Li, Yang Cai*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate visual localization is crucial for autonomous driving, yet existing methods face a fundamental dilemma: While high-definition (HD) maps provide high-precision localization references, their costly construction and maintenance hinder scalability, which drives research toward standard-definition (SD) maps like OpenStreetMap. Current SD-map-based approaches primarily focus on Bird's-Eye View (BEV) matching between images and maps, overlooking a ubiquitous signal-noisy GPS. Although GPS is readily available, it suffers from multipath errors in urban environments. We propose DiffVL, the first framework to reformulate visual localization as a GPS denoising task using diffusion models. Our key insight is that noisy GPS trajectory, when conditioned on visual BEV features and SD maps, implicitly encode the true pose distribution, which can be recovered through iterative diffusion refinement. DiffVL, unlike prior BEV-matching methods (e.g., OrienterNet) or transformer-based registration approaches, learns to reverse GPS noise perturbations by jointly modeling GPS, SD map, and visual signals, achieving sub-meter accuracy without relying on HD maps. Experiments on multiple datasets demonstrate that our method achieves state-of-the-art accuracy compared to BEV-matching baselines. Crucially, our work proves that diffusion models can enable scalable localization by treating noisy GPS as a generative prior-making a paradigm shift from traditional matching-based methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14565) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [STEP: Structured Training and Evaluation Platform for benchmarking trajectory prediction models](https://arxiv.org/abs/2509.14801)
*Julian F. Schumann, Anna Mészáros, Jens Kober, Arkady Zgonnikov*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While trajectory prediction plays a critical role in enabling safe and effective path-planning in automated vehicles, standardized practices for evaluating such models remain underdeveloped. Recent efforts have aimed to unify dataset formats and model interfaces for easier comparisons, yet existing frameworks often fall short in supporting heterogeneous traffic scenarios, joint prediction models, or user documentation. In this work, we introduce STEP -- a new benchmarking framework that addresses these limitations by providing a unified interface for multiple datasets, enforcing consistent training and evaluation conditions, and supporting a wide range of prediction models. We demonstrate the capabilities of STEP in a number of experiments which reveal 1) the limitations of widely-used testing procedures, 2) the importance of joint modeling of agents for better predictions of interactions, and 3) the vulnerability of current state-of-the-art models against both distribution shifts and targeted attacks by adversarial agents. With STEP, we aim to shift the focus from the ``leaderboard'' approach to deeper insights about model behavior and generalization in complex multi-agent settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14801) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [AEGIS: Automated Error Generation and Identification for Multi-Agent Systems](https://arxiv.org/abs/2509.14295)
*Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at https://kfq20.github.io/AEGIS-Website.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14295) | **Categories:** cs.RO

---

### [2] [CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks](https://arxiv.org/abs/2509.14380)
*Seoyeon Choi, Kanghyun Ryu, Jonghoon Ock, Negar Mehr*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-Agent Reinforcement Learning (MARL) provides a powerful framework for learning coordination in multi-agent systems. However, applying MARL to robotics still remains challenging due to high-dimensional continuous joint action spaces, complex reward design, and non-stationary transitions inherent to decentralized settings. On the other hand, humans learn complex coordination through staged curricula, where long-horizon behaviors are progressively built upon simpler skills. Motivated by this, we propose CRAFT: Coaching Reinforcement learning Autonomously using Foundation models for multi-robot coordination Tasks, a framework that leverages the reasoning capabilities of foundation models to act as a "coach" for multi-robot coordination. CRAFT automatically decomposes long-horizon coordination tasks into sequences of subtasks using the planning capability of Large Language Models (LLMs). In what follows, CRAFT trains each subtask using reward functions generated by LLM, and refines them through a Vision Language Model (VLM)-guided reward-refinement loop. We evaluate CRAFT on multi-quadruped navigation and bimanual manipulation tasks, demonstrating its capability to learn complex coordination behaviors. In addition, we validate the multi-quadruped navigation policy in real hardware experiments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14380) | **Categories:** cs.RO

---

### [3] [SimCoachCorpus: A naturalistic dataset with language and trajectories for embodied teaching](https://arxiv.org/abs/2509.14548)
*Emily Sumner, Deepak E. Gopinath, Laporsha Dees, Patricio Reyes Gomez, Xiongyi Cui, Andrew Silva, Jean Costa, Allison Morgan, Mariah Schrum, Tiffany L. Chen, Avinash Balachandran, Guy Rosman*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Curated datasets are essential for training and evaluating AI approaches, but are often lacking in domains where language and physical action are deeply intertwined. In particular, few datasets capture how people acquire embodied skills through verbal instruction over time. To address this gap, we introduce SimCoachCorpus: a unique dataset of race car simulator driving that allows for the investigation of rich interactive phenomena during guided and unguided motor skill acquisition. In this dataset, 29 humans were asked to drive in a simulator around a race track for approximately ninety minutes. Fifteen participants were given personalized one-on-one instruction from a professional performance driving coach, and 14 participants drove without coaching. \name\ includes embodied features such as vehicle state and inputs, map (track boundaries and raceline), and cone landmarks. These are synchronized with concurrent verbal coaching from a professional coach and additional feedback at the end of each lap. We further provide annotations of coaching categories for each concurrent feedback utterance, ratings on students' compliance with coaching advice, and self-reported cognitive load and emotional state of participants (gathered from surveys during the study). The dataset includes over 20,000 concurrent feedback utterances, over 400 terminal feedback utterances, and over 40 hours of vehicle driving data. Our naturalistic dataset can be used for investigating motor learning dynamics, exploring linguistic phenomena, and training computational models of teaching. We demonstrate applications of this dataset for in-context learning, imitation learning, and topic modeling. The dataset introduced in this work will be released publicly upon publication of the peer-reviewed version of this paper. Researchers interested in early access may register at https://tinyurl.com/SimCoachCorpusForm.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14548) | **Categories:** cs.RO, cs.HC

---

### [4] [FlowDrive: Energy Flow Field for End-to-End Autonomous Driving](https://arxiv.org/abs/2509.14303)
*Hao Jiang, Zhipeng Zhang, Yu Gao, Zhigang Sun, Yiru Wang, Yuwen Heng, Shuo Wang, Jinhao Chai, Zhuo Chen, Hao Zhao, Hao Sun, Xi Zhang, Anqing Jiang, Chuan Hu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in end-to-end autonomous driving leverage multi-view images to construct BEV representations for motion planning. In motion planning, autonomous vehicles need considering both hard constraints imposed by geometrically occupied obstacles (e.g., vehicles, pedestrians) and soft, rule-based semantics with no explicit geometry (e.g., lane boundaries, traffic priors). However, existing end-to-end frameworks typically rely on BEV features learned in an implicit manner, lacking explicit modeling of risk and guidance priors for safe and interpretable planning. To address this, we propose FlowDrive, a novel framework that introduces physically interpretable energy-based flow fields-including risk potential and lane attraction fields-to encode semantic priors and safety cues into the BEV space. These flow-aware features enable adaptive refinement of anchor trajectories and serve as interpretable guidance for trajectory generation. Moreover, FlowDrive decouples motion intent prediction from trajectory denoising via a conditional diffusion planner with feature-level gating, alleviating task interference and enhancing multimodal diversity. Experiments on the NAVSIM v2 benchmark demonstrate that FlowDrive achieves state-of-the-art performance with an EPDMS of 86.3, surpassing prior baselines in both safety and planning quality. The project is available at https://astrixdrive.github.io/FlowDrive.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14303) | **Categories:** cs.RO, cs.AI

---

### [5] [GestOS: Advanced Hand Gesture Interpretation via Large Language Models to control Any Type of Robot](https://arxiv.org/abs/2509.14412)
*Artem Lykov, Oleg Kobzarev, Dzmitry Tsetserukou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present GestOS, a gesture-based operating system for high-level control of heterogeneous robot teams. Unlike prior systems that map gestures to fixed commands or single-agent actions, GestOS interprets hand gestures semantically and dynamically distributes tasks across multiple robots based on their capabilities, current state, and supported instruction sets. The system combines lightweight visual perception with large language model (LLM) reasoning: hand poses are converted into structured textual descriptions, which the LLM uses to infer intent and generate robot-specific commands. A robot selection module ensures that each gesture-triggered task is matched to the most suitable agent in real time. This architecture enables context-aware, adaptive control without requiring explicit user specification of targets or commands. By advancing gesture interaction from recognition to intelligent orchestration, GestOS supports scalable, flexible, and user-friendly collaboration with robotic systems in dynamic environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14412) | **Categories:** cs.RO

---

### [6] [Rethinking Reference Trajectories in Agile Drone Racing: A Unified Reference-Free Model-Based Controller via MPPI](https://arxiv.org/abs/2509.14726)
*Fangguo Zhao, Xin Guan, Shuo Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While model-based controllers have demonstrated remarkable performance in autonomous drone racing, their performance is often constrained by the reliance on pre-computed reference trajectories. Conventional approaches, such as trajectory tracking, demand a dynamically feasible, full-state reference, whereas contouring control relaxes this requirement to a geometric path but still necessitates a reference. Recent advancements in reinforcement learning (RL) have revealed that many model-based controllers optimize surrogate objectives, such as trajectory tracking, rather than the primary racing goal of directly maximizing progress through gates. Inspired by these findings, this work introduces a reference-free method for time-optimal racing by incorporating this gate progress objective, derived from RL reward shaping, directly into the Model Predictive Path Integral (MPPI) formulation. The sampling-based nature of MPPI makes it uniquely capable of optimizing the discontinuous and non-differentiable objective in real-time. We also establish a unified framework that leverages MPPI to systematically and fairly compare three distinct objective functions with a consistent dynamics model and parameter set: classical trajectory tracking, contouring control, and the proposed gate progress objective. We compare the performance of these three objectives when solved via both MPPI and a traditional gradient-based solver. Our results demonstrate that the proposed reference-free approach achieves competitive racing performance, rivaling or exceeding reference-based methods. Videos are available at https://zhaofangguo.github.io/racing_mppi/

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14726) | **Categories:** cs.RO

---

### [7] [CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids](https://arxiv.org/abs/2509.14935)
*Punith Reddy Vanteddu, Davide Gorbani, Giuseppe L'Erario, Hosameldin Awadalla Omer Mohamed, Fabio Bergonti, Daniele Pucci*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a CAD-driven co-design framework for optimizing jet-powered aerial humanoid robots to execute dynamically constrained trajectories. Starting from the iRonCub-Mk3 model, a Design of Experiments (DoE) approach is used to generate 5,000 geometrically varied and mechanically feasible designs by modifying limb dimensions, jet interface geometry (e.g., angle and offset), and overall mass distribution. Each model is constructed through CAD assemblies to ensure structural validity and compatibility with simulation tools. To reduce computational cost and enable parameter sensitivity analysis, the models are clustered using K-means, with representative centroids selected for evaluation. A minimum-jerk trajectory is used to assess flight performance, providing position and velocity references for a momentum-based linearized Model Predictive Control (MPC) strategy. A multi-objective optimization is then conducted using the NSGA-II algorithm, jointly exploring the space of design centroids and MPC gain parameters. The objectives are to minimize trajectory tracking error and mechanical energy expenditure. The framework outputs a set of flight-ready humanoid configurations with validated control parameters, offering a structured method for selecting and implementing feasible aerial humanoid designs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.14935) | **Categories:** cs.RO

---
