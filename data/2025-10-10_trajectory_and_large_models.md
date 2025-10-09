# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-10

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)
- [physics.soc-ph (1)](#physics-soc-ph)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory?](https://arxiv.org/abs/2510.06410)
*Aochong Oliver Li, Tanya Goyal*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06410) | **Categories:** cs.AI

---

### [2] [Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them](https://arxiv.org/abs/2510.06534)
*Jiahe Jin, Abhijay Paladugu, Chenyan Xiong*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Agentic search leverages large language models (LLMs) to interpret complex user information needs and execute a multi-step process of planning, searching, and synthesizing information to provide answers. This paradigm introduces unique challenges for LLMs' reasoning and agentic capabilities when interacting with retrieval systems and the broader web. In this paper, we propose a reasoning-driven LLM-based pipeline to study effective reasoning behavior patterns in agentic search. Using this pipeline, we analyze successful agentic search trajectories and identify four beneficial reasoning behaviors: Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery. Based on these findings, we propose a technique called Behavior Priming to train more effective agentic search models. It synthesizes agentic search trajectories that exhibit these four behaviors and integrates them into the agentic search model through supervised fine-tuning (SFT), followed by standard reinforcement learning (RL). Experiments on three benchmarks (GAIA, WebWalker, and HLE) demonstrate that behavior priming yields over 35% gains in Llama3.2-3B and Qwen3-1.7B compared to directly training agentic search models with RL. Crucially, we demonstrate that the desired reasoning behaviors in the SFT data, rather than the correctness of the final answer, is the critical factor for achieving strong final performance after RL: fine-tuning on trajectories with desirable reasoning behaviors but incorrect answers leads to better performance than fine-tuning on trajectories with correct answers. Our analysis further reveals the underlying mechanism: the introduced reasoning behaviors endow models with more effective exploration (higher pass@k and entropy) and test-time scaling (longer trajectories) capabilities, providing a strong foundation for RL. Our code will be released as open source.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06534) | **Categories:** cs.AI, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation](https://arxiv.org/abs/2510.06504)
*Qingxuan Wu, Zhiyang Dou, Chuan Guo, Yiming Huang, Qiao Feng, Bing Zhou, Jian Wang, Lingjie Liu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modeling human-human interactions from text remains challenging because it requires not only realistic individual dynamics but also precise, text-consistent spatiotemporal coupling between agents. Currently, progress is hindered by 1) limited two-person training data, inadequate to capture the diverse intricacies of two-person interactions; and 2) insufficiently fine-grained text-to-interaction modeling, where language conditioning collapses rich, structured prompts into a single sentence embedding. To address these limitations, we propose our Text2Interact framework, designed to generate realistic, text-aligned human-human interactions through a scalable high-fidelity interaction data synthesizer and an effective spatiotemporal coordination pipeline. First, we present InterCompose, a scalable synthesis-by-composition pipeline that aligns LLM-generated interaction descriptions with strong single-person motion priors. Given a prompt and a motion for an agent, InterCompose retrieves candidate single-person motions, trains a conditional reaction generator for another agent, and uses a neural motion evaluator to filter weak or misaligned samples-expanding interaction coverage without extra capture. Second, we propose InterActor, a text-to-interaction model with word-level conditioning that preserves token-level cues (initiation, response, contact ordering) and an adaptive interaction loss that emphasizes contextually relevant inter-person joint pairs, improving coupling and physical plausibility for fine-grained interaction modeling. Extensive experiments show consistent gains in motion diversity, fidelity, and generalization, including out-of-distribution scenarios and user studies. We will release code and models to facilitate reproducibility.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06504) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Traj-Transformer: Diffusion Models with Transformer for GPS Trajectory Generation](https://arxiv.org/abs/2510.06291)
*Zhiyang Zhang, Ningcong Chen, Xin Zhang, Yanhua Li, Shen Su, Hui Lu, Jun Luo*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The widespread use of GPS devices has driven advances in spatiotemporal data mining, enabling machine learning models to simulate human decision making and generate realistic trajectories, addressing both data collection costs and privacy concerns. Recent studies have shown the promise of diffusion models for high-quality trajectory generation. However, most existing methods rely on convolution based architectures (e.g. UNet) to predict noise during the diffusion process, which often results in notable deviations and the loss of fine-grained street-level details due to limited model capacity. In this paper, we propose Trajectory Transformer, a novel model that employs a transformer backbone for both conditional information embedding and noise prediction. We explore two GPS coordinate embedding strategies, location embedding and longitude-latitude embedding, and analyze model performance at different scales. Experiments on two real-world datasets demonstrate that Trajectory Transformer significantly enhances generation quality and effectively alleviates the deviation issues observed in prior approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06291) | **Categories:** cs.LG, cs.AI

---

### [2] [DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning](https://arxiv.org/abs/2510.06913)
*Ke Guo, Haochen Liu, Xiaojun Wu, Chen Lv*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06913) | **Categories:** cs.LG, cs.AI, cs.RO

---

### [3] [Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report](https://arxiv.org/abs/2510.07092)
*Riccardo Mereu, Aidan Scannell, Yuxin Hou, Yi Zhao, Aditya Jitta, Antonio Dominguez, Luigi Acerbi, Amos Storkey, Paul Chang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: World models are a powerful paradigm in AI and robotics, enabling agents to reason about the future by predicting visual observations or compact latent states. The 1X World Model Challenge introduces an open-source benchmark of real-world humanoid interaction, with two complementary tracks: sampling, focused on forecasting future image frames, and compression, focused on predicting future discrete latent codes. For the sampling track, we adapt the video generation foundation model Wan-2.2 TI2V-5B to video-state-conditioned future frame prediction. We condition the video generation on robot states using AdaLN-Zero, and further post-train the model using LoRA. For the compression track, we train a Spatio-Temporal Transformer model from scratch. Our models achieve 23.0 dB PSNR in the sampling task and a Top-500 CE of 6.6386 in the compression task, securing 1st place in both challenges.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07092) | **Categories:** cs.LG, cs.AI, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [SanDRA: Safe Large-Language-Model-Based Decision Making for Automated Vehicles Using Reachability Analysis](https://arxiv.org/abs/2510.06717)
*Yuanfei Lin, Sebastian Illing, Matthias Althoff*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models have been widely applied to knowledge-driven decision-making for automated vehicles due to their strong generalization and reasoning capabilities. However, the safety of the resulting decisions cannot be ensured due to possible hallucinations and the lack of integrated vehicle dynamics. To address this issue, we propose SanDRA, the first safe large-language-model-based decision making framework for automated vehicles using reachability analysis. Our approach starts with a comprehensive description of the driving scenario to prompt large language models to generate and rank feasible driving actions. These actions are translated into temporal logic formulas that incorporate formalized traffic rules, and are subsequently integrated into reachability analysis to eliminate unsafe actions. We validate our approach in both open-loop and closed-loop driving environments using off-the-shelf and finetuned large language models, showing that it can provide provably safe and, where possible, legally compliant driving actions, even under high-density traffic conditions. To ensure transparency and facilitate future research, all code and experimental setups are publicly available at github.com/CommonRoad/SanDRA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06717) | **Categories:** cs.RO

---

### [2] [A Formal gatekeeper Framework for Safe Dual Control with Active Exploration](https://arxiv.org/abs/2510.06351)
*Kaleb Ben Naveed, Devansh R. Agrawal, Dimitra Panagou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planning safe trajectories under model uncertainty is a fundamental challenge. Robust planning ensures safety by considering worst-case realizations, yet ignores uncertainty reduction and leads to overly conservative behavior. Actively reducing uncertainty on-the-fly during a nominal mission defines the dual control problem. Most approaches address this by adding a weighted exploration term to the cost, tuned to trade off the nominal objective and uncertainty reduction, but without formal consideration of when exploration is beneficial. Moreover, safety is enforced in some methods but not in others. We propose a framework that integrates robust planning with active exploration under formal guarantees as follows: The key innovation and contribution is that exploration is pursued only when it provides a verifiable improvement without compromising safety. To achieve this, we utilize our earlier work on gatekeeper as an architecture for safety verification, and extend it so that it generates both safe and informative trajectories that reduce uncertainty and the cost of the mission, or keep it within a user-defined budget. The methodology is evaluated via simulation case studies on the online dual control of a quadrotor under parametric uncertainty.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06351) | **Categories:** cs.RO

---

### [3] [Constrained Natural Language Action Planning for Resilient Embodied Systems](https://arxiv.org/abs/2510.06357)
*Grayson Byrd, Corban Rivera, Bethany Kemp, Meghan Booker, Aurora Schmidt, Celso M de Melo, Lalithkumar Seenivasan, Mathias Unberath*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Replicating human-level intelligence in the execution of embodied tasks remains challenging due to the unconstrained nature of real-world environments. Novel use of large language models (LLMs) for task planning seeks to address the previously intractable state/action space of complex planning tasks, but hallucinations limit their reliability, and thus, viability beyond a research context. Additionally, the prompt engineering required to achieve adequate system performance lacks transparency, and thus, repeatability. In contrast to LLM planning, symbolic planning methods offer strong reliability and repeatability guarantees, but struggle to scale to the complexity and ambiguity of real-world tasks. We introduce a new robotic planning method that augments LLM planners with symbolic planning oversight to improve reliability and repeatability, and provide a transparent approach to defining hard constraints with considerably stronger clarity than traditional prompt engineering. Importantly, these augmentations preserve the reasoning capabilities of LLMs and retain impressive generalization in open-world environments. We demonstrate our approach in simulated and real-world environments. On the ALFWorld planning benchmark, our approach outperforms current state-of-the-art methods, achieving a near-perfect 99% success rate. Deployment of our method to a real-world quadruped robot resulted in 100% task success compared to 50% and 30% for pure LLM and symbolic planners across embodied pick and place tasks. Our approach presents an effective strategy to enhance the reliability, repeatability and transparency of LLM-based robot planners while retaining their key strengths: flexibility and generalizability to complex real-world environments. We hope that this work will contribute to the broad goal of building resilient embodied intelligent systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06357) | **Categories:** cs.RO, cs.AI

---

### [4] [Assist-As-Needed: Adaptive Multimodal Robotic Assistance for Medication Management in Dementia Care](https://arxiv.org/abs/2510.06633)
*Kruthika Gangaraju, Tanmayi Inaparthy, Jiaqi Yang, Yihao Zheng, Fengpei Yuan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: People living with dementia (PLWDs) face progressively declining abilities in medication management-from simple forgetfulness to complete task breakdown-yet most assistive technologies fail to adapt to these changing needs. This one-size-fits-all approach undermines autonomy, accelerates dependence, and increases caregiver burden. Occupational therapy principles emphasize matching assistance levels to individual capabilities: minimal reminders for those who merely forget, spatial guidance for those who misplace items, and comprehensive multimodal support for those requiring step-by-step instruction. However, existing robotic systems lack this adaptive, graduated response framework essential for maintaining PLWD independence. We present an adaptive multimodal robotic framework using the Pepper robot that dynamically adjusts assistance based on real-time assessment of user needs. Our system implements a hierarchical intervention model progressing from (1) simple verbal reminders, to (2) verbal + gestural cues, to (3) full multimodal guidance combining physical navigation to medication locations with step-by-step verbal and gestural instructions. Powered by LLM-driven interaction strategies and multimodal sensing, the system continuously evaluates task states to provide just-enough assistance-preserving autonomy while ensuring medication adherence. We conducted a preliminary study with healthy adults and dementia care stakeholders in a controlled lab setting, evaluating the system's usability, comprehensibility, and appropriateness of adaptive feedback mechanisms. This work contributes: (1) a theoretically grounded adaptive assistance framework translating occupational therapy principles into HRI design, (2) a multimodal robotic implementation that preserves PLWD dignity through graduated support, and (3) empirical insights into stakeholder perceptions of adaptive robotic care.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06633) | **Categories:** cs.RO

---

### [5] [Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications](https://arxiv.org/abs/2510.07077)
*Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, Yuke Zhu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Amid growing efforts to leverage advances in large language models (LLMs) and vision-language models (VLMs) for robotics, Vision-Language-Action (VLA) models have recently gained significant attention. By unifying vision, language, and action data at scale, which have traditionally been studied separately, VLA models aim to learn policies that generalise across diverse tasks, objects, embodiments, and environments. This generalisation capability is expected to enable robots to solve novel downstream tasks with minimal or no additional task-specific data, facilitating more flexible and scalable real-world deployment. Unlike previous surveys that focus narrowly on action representations or high-level model architectures, this work offers a comprehensive, full-stack review, integrating both software and hardware components of VLA systems. In particular, this paper provides a systematic review of VLAs, covering their strategy and architectural transition, architectures and building blocks, modality-specific processing techniques, and learning paradigms. In addition, to support the deployment of VLAs in real-world robotic applications, we also review commonly used robot platforms, data collection strategies, publicly available datasets, data augmentation methods, and evaluation benchmarks. Throughout this comprehensive survey, this paper aims to offer practical guidance for the robotics community in applying VLAs to real-world robotic systems. All references categorized by training approach, evaluation method, modality, and dataset are available in the table on our project website: https://vla-survey.github.io .

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07077) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [6] [A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Model](https://arxiv.org/abs/2510.07133)
*Tony Zhang, Burak Kantarci, Umair Siddique*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring the safety of self-driving cars remains a major challenge due to the complexity and unpredictability of real-world driving environments. Traditional testing methods face significant limitations, such as the oracle problem, which makes it difficult to determine whether a system's behavior is correct, and the inability to cover the full range of scenarios an autonomous vehicle may encounter. In this paper, we introduce a digital twin-driven metamorphic testing framework that addresses these challenges by creating a virtual replica of the self-driving system and its operating environment. By combining digital twin technology with AI-based image generative models such as Stable Diffusion, our approach enables the systematic generation of realistic and diverse driving scenes. This includes variations in weather, road topology, and environmental features, all while maintaining the core semantics of the original scenario. The digital twin provides a synchronized simulation environment where changes can be tested in a controlled and repeatable manner. Within this environment, we define three metamorphic relations inspired by real-world traffic rules and vehicle behavior. We validate our framework in the Udacity self-driving simulator and demonstrate that it significantly enhances test coverage and effectiveness. Our method achieves the highest true positive rate (0.719), F1 score (0.689), and precision (0.662) compared to baseline approaches. This paper highlights the value of integrating digital twins with AI-powered scenario generation to create a scalable, automated, and high-fidelity testing solution for autonomous vehicle safety.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07133) | **Categories:** cs.RO, cs.AI

---

### [7] [TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking](https://arxiv.org/abs/2510.07134)
*Jiahang Liu, Yunpeng Qi, Jiazhao Zhang, Minghan Li, Shaoan Wang, Kui Wu, Hanjing Ye, Hong Zhang, Zhibo Chen, Fangwei Zhong, Zhizheng Zhang, He Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied Visual Tracking (EVT) is a fundamental ability that underpins practical applications, such as companion robots, guidance robots and service assistants, where continuously following moving targets is essential. Recent advances have enabled language-guided tracking in complex and unstructured scenes. However, existing approaches lack explicit spatial reasoning and effective temporal memory, causing failures under severe occlusions or in the presence of similar-looking distractors. To address these challenges, we present TrackVLA++, a novel Vision-Language-Action (VLA) model that enhances embodied visual tracking with two key modules, a spatial reasoning mechanism and a Target Identification Memory (TIM). The reasoning module introduces a Chain-of-Thought paradigm, termed Polar-CoT, which infers the target's relative position and encodes it as a compact polar-coordinate token for action prediction. Guided by these spatial priors, the TIM employs a gated update strategy to preserve long-horizon target memory, ensuring spatiotemporal consistency and mitigating target loss during extended occlusions. Extensive experiments show that TrackVLA++ achieves state-of-the-art performance on public benchmarks across both egocentric and multi-camera settings. On the challenging EVT-Bench DT split, TrackVLA++ surpasses the previous leading approach by 5.1 and 12, respectively. Furthermore, TrackVLA++ exhibits strong zero-shot generalization, enabling robust real-world tracking in dynamic and occluded scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07134) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [8] [TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics](https://arxiv.org/abs/2510.07181)
*Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, Shanghang Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, trajectory generation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07181) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [9] [HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving](https://arxiv.org/abs/2510.07210)
*Donald Pfaffmann, Matthias Klusch, Marcel Steinmetz*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07210) | **Categories:** cs.RO, cs.AI

---


## physics.soc-ph [physics.soc-ph]
### [1] [Generalized Multi-agent Social Simulation Framework](https://arxiv.org/abs/2510.06225)
*Gang Li, Jie Lin, Yining Tang, Ziteng Wang, Yirui Huang, Junyu Zhang, Shuang Luo, Chao Wu, Yike Guo*

Main category: physics.soc-ph

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-agent social interaction has clearly benefited from Large Language Models. However, current simulation systems still face challenges such as difficulties in scaling to diverse scenarios and poor reusability due to a lack of modular design. To address these issues, we designed and developed a modular, object-oriented framework that organically integrates various base classes through a hierarchical structure, harvesting scalability and reusability. We inherited the framework to realize common derived classes. Additionally, a memory summarization mechanism is proposed to filter and distill relevant information from raw memory data, prioritizing contextually salient events and interactions. By selecting and combining some necessary derived classes, we customized a specific simulated environment. Utilizing this simulated environment, we successfully simulated human interactions on social media, replicating real-world online social behaviors. The source code for the project will be released and evolve.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06225) | **Categories:** physics.soc-ph, cs.AI

---
