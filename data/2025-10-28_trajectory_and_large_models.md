# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-28

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (4)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge](https://arxiv.org/abs/2510.21144)
*Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21144) | **Categories:** cs.AI

---

### [2] [Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](https://arxiv.org/abs/2510.21302)
*Sanghyun Ahn, Wonje Choi, Junyong Lee, Jinwoo Park, Honguk Woo*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in large language models (LLMs) have enabled the automatic generation of executable code for task planning and control in embodied agents such as robots, demonstrating the potential of LLM-based embodied intelligence. However, these LLM-based code-as-policies approaches often suffer from limited environmental grounding, particularly in dynamic or partially observable settings, leading to suboptimal task success rates due to incorrect or incomplete code generation. In this work, we propose a neuro-symbolic embodied task planning framework that incorporates explicit symbolic verification and interactive validation processes during code generation. In the validation phase, the framework generates exploratory code that actively interacts with the environment to acquire missing observations while preserving task-relevant states. This integrated process enhances the grounding of generated code, resulting in improved task reliability and success rates in complex environments. We evaluate our framework on RLBench and in real-world settings across dynamic, partially observable scenarios. Experimental results demonstrate that our framework improves task success rates by 46.2% over Code-as-Policies baselines and attains over 86.8% executability of task-relevant actions, thereby enhancing the reliability of task planning in dynamic environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21302) | **Categories:** cs.AI, cs.RO

---

### [3] [A Knowledge-Graph Translation Layer for Mission-Aware Multi-Agent Path Planning in Spatiotemporal Dynamics](https://arxiv.org/abs/2510.21695)
*Edward Holmberg, Elias Ioup, Mahdi Abdelguerfi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The coordination of autonomous agents in dynamic environments is hampered by the semantic gap between high-level mission objectives and low-level planner inputs. To address this, we introduce a framework centered on a Knowledge Graph (KG) that functions as an intelligent translation layer. The KG's two-plane architecture compiles declarative facts into per-agent, mission-aware ``worldviews" and physics-aware traversal rules, decoupling mission semantics from a domain-agnostic planner. This allows complex, coordinated paths to be modified simply by changing facts in the KG. A case study involving Autonomous Underwater Vehicles (AUVs) in the Gulf of Mexico visually demonstrates the end-to-end process and quantitatively proves that different declarative policies produce distinct, high-performing outcomes. This work establishes the KG not merely as a data repository, but as a powerful, stateful orchestrator for creating adaptive and explainable autonomous systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21695) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Generative Point Tracking with Flow Matching](https://arxiv.org/abs/2510.20951)
*Mattie Tesfaldet, Adam W. Harley, Konstantinos G. Derpanis, Derek Nowrouzezahrai, Christopher Pal*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Tracking a point through a video can be a challenging task due to uncertainty arising from visual obfuscations, such as appearance changes and occlusions. Although current state-of-the-art discriminative models excel in regressing long-term point trajectory estimates -- even through occlusions -- they are limited to regressing to a mean (or mode) in the presence of uncertainty, and fail to capture multi-modality. To overcome this limitation, we introduce Generative Point Tracker (GenPT), a generative framework for modelling multi-modal trajectories. GenPT is trained with a novel flow matching formulation that combines the iterative refinement of discriminative trackers, a window-dependent prior for cross-window consistency, and a variance schedule tuned specifically for point coordinates. We show how our model's generative capabilities can be leveraged to improve point trajectory estimates by utilizing a best-first search strategy on generated samples during inference, guided by the model's own confidence of its predictions. Empirically, we evaluate GenPT against the current state of the art on the standard PointOdyssey, Dynamic Replica, and TAP-Vid benchmarks. Further, we introduce a TAP-Vid variant with additional occlusions to assess occluded point tracking performance and highlight our model's ability to capture multi-modality. GenPT is capable of capturing the multi-modality in point trajectories, which translates to state-of-the-art tracking accuracy on occluded points, while maintaining competitive tracking accuracy on visible points compared to extant discriminative point trackers.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20951) | **Categories:** cs.CV

---

### [2] [Towards Physics-informed Spatial Intelligence with Human Priors: An Autonomous Driving Pilot Study](https://arxiv.org/abs/2510.21160)
*Guanlin Wu, Boyan Su, Yang Zhao, Pu Wang, Yichen Lin, Hao Frank Yang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: How to integrate and verify spatial intelligence in foundation models remains an open challenge. Current practice often proxies Visual-Spatial Intelligence (VSI) with purely textual prompts and VQA-style scoring, which obscures geometry, invites linguistic shortcuts, and weakens attribution to genuinely spatial skills. We introduce Spatial Intelligence Grid (SIG): a structured, grid-based schema that explicitly encodes object layouts, inter-object relations, and physically grounded priors. As a complementary channel to text, SIG provides a faithful, compositional representation of scene structure for foundation-model reasoning. Building on SIG, we derive SIG-informed evaluation metrics that quantify a model's intrinsic VSI, which separates spatial capability from language priors. In few-shot in-context learning with state-of-the-art multimodal LLMs (e.g. GPT- and Gemini-family models), SIG yields consistently larger, more stable, and more comprehensive gains across all VSI metrics compared to VQA-only representations, indicating its promise as a data-labeling and training schema for learning VSI. We also release SIGBench, a benchmark of 1.4K driving frames annotated with ground-truth SIG labels and human gaze traces, supporting both grid-based machine VSI tasks and attention-driven, human-like VSI tasks in autonomous-driving scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21160) | **Categories:** cs.CV

---

### [3] [Gaze-VLM:Bridging Gaze and VLMs through Attention Regularization for Egocentric Understanding](https://arxiv.org/abs/2510.21356)
*Anupam Pani, Yanchao Yang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Eye gaze offers valuable cues about attention, short-term intent, and future actions, making it a powerful signal for modeling egocentric behavior. In this work, we propose a gaze-regularized framework that enhances VLMs for two key egocentric understanding tasks: fine-grained future event prediction and current activity understanding. Unlike prior approaches that rely solely on visual inputs or use gaze as an auxiliary input signal , our method uses gaze only during training. We introduce a gaze-regularized attention mechanism that aligns model focus with human visual gaze. This design is flexible and modular, allowing it to generalize across multiple VLM architectures that utilize attention. Experimental results show that our approach improves semantic prediction scores by up to 11 for future event prediction and around 7 for current activity understanding, compared to the corresponding baseline models trained without gaze regularization. These results highlight the value of gaze-guided training in improving the accuracy and robustness of egocentric VLMs. Overall, this work establishes a foundation for using human gaze to enhance the predictive capabilities of VLMs in real-world scenarios like assistive robots and human-machine collaboration. Code and additional information is available at: https://github.com/anupampani/Gaze-VLM

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21356) | **Categories:** cs.CV, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [LLM-Integrated Bayesian State Space Models for Multimodal Time-Series Forecasting](https://arxiv.org/abs/2510.20952)
*Sungjun Cho, Changho Shin, Suenggwan Jo, Xinya Yan, Shourjo Aditya Chaudhuri, Frederic Sala*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Forecasting in the real world requires integrating structured time-series data with unstructured textual information, but existing methods are architecturally limited by fixed input/output horizons and are unable to model or quantify uncertainty. We address this challenge by introducing LLM-integrated Bayesian State space models (LBS), a novel probabilistic framework for multimodal temporal forecasting. At a high level, LBS consists of two components: (1) a state space model (SSM) backbone that captures the temporal dynamics of latent states from which both numerical and textual observations are generated and (2) a pretrained large language model (LLM) that is adapted to encode textual inputs for posterior state estimation and decode textual forecasts consistent with the latent trajectory. This design enables flexible lookback and forecast windows, principled uncertainty quantification, and improved temporal generalization thanks to the well-suited inductive bias of SSMs toward modeling dynamical systems. Experiments on the TextTimeCorpus benchmark demonstrate that LBS improves the previous state-of-the-art by 13.20% while providing human-readable summaries of each forecast. Our work is the first to unify LLMs and SSMs for joint numerical and textual prediction, offering a novel foundation for multimodal temporal reasoning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20952) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Aircraft Collision Avoidance Systems: Technological Challenges and Solutions on the Path to Regulatory Acceptance](https://arxiv.org/abs/2510.20916)
*Sydney M. Katz, Robert J. Moss, Dylan M. Asmar, Wesley A. Olson, James K. Kuchar, Mykel J. Kochenderfer*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Aircraft collision avoidance systems is critical to modern aviation. These systems are designed to predict potential collisions between aircraft and recommend appropriate avoidance actions. Creating effective collision avoidance systems requires solutions to a variety of technical challenges related to surveillance, decision making, and validation. These challenges have sparked significant research and development efforts over the past several decades that have resulted in a variety of proposed solutions. This article provides an overview of these challenges and solutions with an emphasis on those that have been put through a rigorous validation process and accepted by regulatory bodies. The challenges posed by the collision avoidance problem are often present in other domains, and aircraft collision avoidance systems can serve as case studies that provide valuable insights for a wide range of safety-critical systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20916) | **Categories:** cs.RO, cs.AI, cs.SY, eess.SY

---

### [2] [HRT1: One-Shot Human-to-Robot Trajectory Transfer for Mobile Manipulation](https://arxiv.org/abs/2510.21026)
*Sai Haneesh Allu, Jishnu Jaykumar P, Ninad Khargonkar, Tyler Summers, Jian Yao, Yu Xiang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce a novel system for human-to-robot trajectory transfer that enables robots to manipulate objects by learning from human demonstration videos. The system consists of four modules. The first module is a data collection module that is designed to collect human demonstration videos from the point of view of a robot using an AR headset. The second module is a video understanding module that detects objects and extracts 3D human-hand trajectories from demonstration videos. The third module transfers a human-hand trajectory into a reference trajectory of a robot end-effector in 3D space. The last module utilizes a trajectory optimization algorithm to solve a trajectory in the robot configuration space that can follow the end-effector trajectory transferred from the human demonstration. Consequently, these modules enable a robot to watch a human demonstration video once and then repeat the same mobile manipulation task in different environments, even when objects are placed differently from the demonstrations. Experiments of different manipulation tasks are conducted on a mobile manipulator to verify the effectiveness of our system

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21026) | **Categories:** cs.RO

---

### [3] [Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos](https://arxiv.org/abs/2510.21571)
*Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang, Baining Guo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a novel approach for pretraining robotic manipulation Vision-Language-Action (VLA) models using a large corpus of unscripted real-life video recordings of human hand activities. Treating human hand as dexterous robot end-effector, we show that "in-the-wild" egocentric human videos without any annotations can be transformed into data formats fully aligned with existing robotic V-L-A training data in terms of task granularity and labels. This is achieved by the development of a fully-automated holistic human activity analysis approach for arbitrary human hand videos. This approach can generate atomic-level hand activity segments and their language descriptions, each accompanied with framewise 3D hand motion and camera motion. We process a large volume of egocentric videos and create a hand-VLA training dataset containing 1M episodes and 26M frames. This training data covers a wide range of objects and concepts, dexterous manipulation tasks, and environment variations in real life, vastly exceeding the coverage of existing robot data. We design a dexterous hand VLA model architecture and pretrain the model on this dataset. The model exhibits strong zero-shot capabilities on completely unseen real-world observations. Additionally, fine-tuning it on a small amount of real robot action data significantly improves task success rates and generalization to novel objects in real robotic experiments. We also demonstrate the appealing scaling behavior of the model's task performance with respect to pretraining data scale. We believe this work lays a solid foundation for scalable VLA pretraining, advancing robots toward truly generalizable embodied intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21571) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [4] [Design and Structural Validation of a Micro-UAV with On-Board Dynamic Route Planning](https://arxiv.org/abs/2510.21648)
*Inbazhagan Ravikumar, Ram Sundhar, Narendhiran Vijayakumar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Micro aerial vehicles are becoming increasingly important in search and rescue operations due to their agility, speed, and ability to access confined spaces or hazardous areas. However, designing lightweight aerial systems presents significant structural, aerodynamic, and computational challenges. This work addresses two key limitations in many low-cost aerial systems under two kilograms: their lack of structural durability during flight through rough terrains and inability to replan paths dynamically when new victims or obstacles are detected. We present a fully customised drone built from scratch using only commonly available components and materials, emphasising modularity, low cost, and ease of assembly. The structural frame is reinforced with lightweight yet durable materials to withstand impact, while the onboard control system is powered entirely by free, open-source software solutions. The proposed system demonstrates real-time perception and adaptive navigation capabilities without relying on expensive hardware accelerators, offering an affordable and practical solution for real-world search and rescue missions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21648) | **Categories:** cs.RO, cs.CE

---
