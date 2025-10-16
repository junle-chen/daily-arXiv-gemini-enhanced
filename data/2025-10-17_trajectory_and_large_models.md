# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-17

## 目录

- [人工智能 (Artificial Intelligence) (6)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)
- [eess.SY (2)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [From Narratives to Probabilistic Reasoning: Predicting and Interpreting Drivers' Hazardous Actions in Crashes Using Large Language Model](https://arxiv.org/abs/2510.13002)
*Boyou Chen, Gerui Xu, Zifei Wang, Huizhong Guo, Ananna Ahmed, Zhaonan Sun, Zhen Hu, Kaihan Zhang, Shan Bao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vehicle crashes involve complex interactions between road users, split-second decisions, and challenging environmental conditions. Among these, two-vehicle crashes are the most prevalent, accounting for approximately 70% of roadway crashes and posing a significant challenge to traffic safety. Identifying Driver Hazardous Action (DHA) is essential for understanding crash causation, yet the reliability of DHA data in large-scale databases is limited by inconsistent and labor-intensive manual coding practices. Here, we present an innovative framework that leverages a fine-tuned large language model to automatically infer DHAs from textual crash narratives, thereby improving the validity and interpretability of DHA classifications. Using five years of two-vehicle crash data from MTCF, we fine-tuned the Llama 3.2 1B model on detailed crash narratives and benchmarked its performance against conventional machine learning classifiers, including Random Forest, XGBoost, CatBoost, and a neural network. The fine-tuned LLM achieved an overall accuracy of 80%, surpassing all baseline models and demonstrating pronounced improvements in scenarios with imbalanced data. To increase interpretability, we developed a probabilistic reasoning approach, analyzing model output shifts across original test sets and three targeted counterfactual scenarios: variations in driver distraction and age. Our analysis revealed that introducing distraction for one driver substantially increased the likelihood of "General Unsafe Driving"; distraction for both drivers maximized the probability of "Both Drivers Took Hazardous Actions"; and assigning a teen driver markedly elevated the probability of "Speed and Stopping Violations." Our framework and analytical methods provide a robust and interpretable solution for large-scale automated DHA detection, offering new opportunities for traffic safety analysis and intervention.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13002) | **Categories:** cs.AI, cs.LG

---

### [2] [DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping](https://arxiv.org/abs/2510.12979)
*Wei Fan, Wenlin Yao, Zheng Li, Feng Yao, Xin Liu, Liang Qiu, Qingyu Yin, Yangqiu Song, Bing Yin*

Main category: cs.AI

TL;DR: DeepPlanner通过优化规划阶段，显著提升了深度研究智能体的规划能力，并在多个基准测试中取得了最先进的结果。


<details>
  <summary>Details</summary>
Motivation: 现有方法在利用外部工具解决复杂任务时，要么依赖于推理阶段的隐式规划，要么引入显式规划器但未系统地优化规划阶段。

Method: 提出了DeepPlanner，一个端到端的强化学习框架，通过基于熵的token级别优势塑造和选择性地提升样本级别优势，来增强深度研究智能体的规划能力。

Result: 在七个深度研究基准测试中，DeepPlanner提高了规划质量，并在较低的训练预算下取得了最先进的结果。

Conclusion: DeepPlanner有效提升了深度研究智能体的规划能力，并在多个复杂任务中表现出色。

Abstract: 具有多步推理和行动生成能力的大型语言模型（LLM）在利用外部工具解决需要长期规划的复杂任务方面显示出潜力。然而，现有方法要么依赖于推理阶段的隐式规划，要么引入显式规划器但未系统地解决如何优化规划阶段的问题。作为证据，我们观察到在原始强化学习（RL）下，规划token表现出比其他行动token显著更高的熵，揭示了仍然未被充分优化的不确定决策点。为了解决这个问题，我们提出了DeepPlanner，一个端到端的RL框架，可以有效地增强深度研究智能体的规划能力。我们的方法使用基于熵的项来塑造token级别的优势，从而为高熵token分配更大的更新，并选择性地增加规划密集型rollout的样本级别优势。在七个深度研究基准测试中进行的大量实验表明，DeepPlanner提高了规划质量，并在大大降低的训练预算下实现了最先进的结果。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12979) | **Categories:** cs.AI, cs.CL

---

### [3] [SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents](https://arxiv.org/abs/2510.12985)
*Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li, Qi Zhu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present Sentinel, the first framework for formally evaluating the physical safety of Large Language Model(LLM-based) embodied agents across the semantic, plan, and trajectory levels. Unlike prior methods that rely on heuristic rules or subjective LLM judgments, Sentinel grounds practical safety requirements in formal temporal logic (TL) semantics that can precisely specify state invariants, temporal dependencies, and timing constraints. It then employs a multi-level verification pipeline where (i) at the semantic level, intuitive natural language safety requirements are formalized into TL formulas and the LLM agent's understanding of these requirements is probed for alignment with the TL formulas; (ii) at the plan level, high-level action plans and subgoals generated by the LLM agent are verified against the TL formulas to detect unsafe plans before execution; and (iii) at the trajectory level, multiple execution trajectories are merged into a computation tree and efficiently verified against physically-detailed TL specifications for a final safety check. We apply Sentinel in VirtualHome and ALFRED, and formally evaluate multiple LLM-based embodied agents against diverse safety requirements. Our experiments show that by grounding physical safety in temporal logic and applying verification methods across multiple levels, Sentinel provides a rigorous foundation for systematically evaluating LLM-based embodied agents in physical environments, exposing safety violations overlooked by previous methods and offering insights into their failure modes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12985) | **Categories:** cs.AI

---

### [4] [Toward Reasoning-Centric Time-Series Analysis](https://arxiv.org/abs/2510.13029)
*Xinlei Wang, Mingtian Tan, Jing Qiu, Junhua Zhao, Jinjin Gu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional time series analysis has long relied on pattern recognition, trained on static and well-established benchmarks. However, in real-world settings -- where policies shift, human behavior adapts, and unexpected events unfold -- effective analysis must go beyond surface-level trends to uncover the actual forces driving them. The recent rise of Large Language Models (LLMs) presents new opportunities for rethinking time series analysis by integrating multimodal inputs. However, as the use of LLMs becomes popular, we must remain cautious, asking why we use LLMs and how to exploit them effectively. Most existing LLM-based methods still employ their numerical regression ability and ignore their deeper reasoning potential. This paper argues for rethinking time series with LLMs as a reasoning task that prioritizes causal structure and explainability. This shift brings time series analysis closer to human-aligned understanding, enabling transparent and context-aware insights in complex real-world environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13029) | **Categories:** cs.AI

---

### [5] [Personalized Learning Path Planning with Goal-Driven Learner State Modeling](https://arxiv.org/abs/2510.13215)
*Joy Jia Yin Lim, Ye He, Jifan Yu, Xin Cong, Daniel Zhang-Li, Zhiyuan Liu, Huiqin Liu, Lei Hou, Juanzi Li, Bin Xu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Personalized Learning Path Planning (PLPP) aims to design adaptive learning paths that align with individual goals. While large language models (LLMs) show potential in personalizing learning experiences, existing approaches often lack mechanisms for goal-aligned planning. We introduce Pxplore, a novel framework for PLPP that integrates a reinforcement-based training paradigm and an LLM-driven educational architecture. We design a structured learner state model and an automated reward function that transforms abstract objectives into computable signals. We train the policy combining supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), and deploy it within a real-world learning platform. Extensive experiments validate Pxplore's effectiveness in producing coherent, personalized, and goal-driven learning paths. We release our code and dataset to facilitate future research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13215) | **Categories:** cs.AI, cs.CL

---

### [6] [From Refusal to Recovery: A Control-Theoretic Approach to Generative AI Guardrails](https://arxiv.org/abs/2510.13727)
*Ravi Pandya, Madison Bland, Duy P. Nguyen, Changliu Liu, Jaime Fernández Fisac, Andrea Bajcsy*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generative AI systems are increasingly assisting and acting on behalf of end users in practical settings, from digital shopping assistants to next-generation autonomous cars. In this context, safety is no longer about blocking harmful content, but about preempting downstream hazards like financial or physical harm. Yet, most AI guardrails continue to rely on output classification based on labeled datasets and human-specified criteria,making them brittle to new hazardous situations. Even when unsafe conditions are flagged, this detection offers no path to recovery: typically, the AI system simply refuses to act--which is not always a safe choice. In this work, we argue that agentic AI safety is fundamentally a sequential decision problem: harmful outcomes arise from the AI system's continually evolving interactions and their downstream consequences on the world. We formalize this through the lens of safety-critical control theory, but within the AI model's latent representation of the world. This enables us to build predictive guardrails that (i) monitor an AI system's outputs (actions) in real time and (ii) proactively correct risky outputs to safe ones, all in a model-agnostic manner so the same guardrail can be wrapped around any AI model. We also offer a practical training recipe for computing such guardrails at scale via safety-critical reinforcement learning. Our experiments in simulated driving and e-commerce settings demonstrate that control-theoretic guardrails can reliably steer LLM agents clear of catastrophic outcomes (from collisions to bankruptcy) while preserving task performance, offering a principled dynamic alternative to today's flag-and-block guardrails.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13727) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models](https://arxiv.org/abs/2510.13108)
*Jingyu Song, Zhenxin Li, Shiyi Lan, Xinglong Sun, Nadine Chang, Maying Shen, Joshua Chen, Katherine A. Skinner, Jose M. Alvarez*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13108) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [SceneAdapt: Scene-aware Adaptation of Human Motion Diffusion](https://arxiv.org/abs/2510.13044)
*Jungbin Cho, Minsu Kim, Jisoo Kim, Ce Zheng, Laszlo A. Jeni, Ming-Hsuan Yang, Youngjae Yu, Seonjoo Kim*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human motion is inherently diverse and semantically rich, while also shaped by the surrounding scene. However, existing motion generation approaches address either motion semantics or scene-awareness in isolation, since constructing large-scale datasets with both rich text--motion coverage and precise scene interactions is extremely challenging. In this work, we introduce SceneAdapt, a framework that injects scene awareness into text-conditioned motion models by leveraging disjoint scene--motion and text--motion datasets through two adaptation stages: inbetweening and scene-aware inbetweening. The key idea is to use motion inbetweening, learnable without text, as a proxy task to bridge two distinct datasets and thereby inject scene-awareness to text-to-motion models. In the first stage, we introduce keyframing layers that modulate motion latents for inbetweening while preserving the latent manifold. In the second stage, we add a scene-conditioning layer that injects scene geometry by adaptively querying local context through cross-attention. Experimental results show that SceneAdapt effectively injects scene awareness into text-to-motion models, and we further analyze the mechanisms through which this awareness emerges. Code and models will be released.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13044) | **Categories:** cs.CV, cs.AI

---

### [3] [EPIPTrack: Rethinking Prompt Modeling with Explicit and Implicit Prompts for Multi-Object Tracking](https://arxiv.org/abs/2510.13235)
*Yukuan Zhang, Jiarui Zhao, Shangqing Nie, Jin Kuang, Shengsheng Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal semantic cues, such as textual descriptions, have shown strong potential in enhancing target perception for tracking. However, existing methods rely on static textual descriptions from large language models, which lack adaptability to real-time target state changes and prone to hallucinations. To address these challenges, we propose a unified multimodal vision-language tracking framework, named EPIPTrack, which leverages explicit and implicit prompts for dynamic target modeling and semantic alignment. Specifically, explicit prompts transform spatial motion information into natural language descriptions to provide spatiotemporal guidance. Implicit prompts combine pseudo-words with learnable descriptors to construct individualized knowledge representations capturing appearance attributes. Both prompts undergo dynamic adjustment via the CLIP text encoder to respond to changes in target state. Furthermore, we design a Discriminative Feature Augmentor to enhance visual and cross-modal representations. Extensive experiments on MOT17, MOT20, and DanceTrack demonstrate that EPIPTrack outperforms existing trackers in diverse scenarios, exhibiting robust adaptability and superior performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13235) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Lifting Manifolds to Mitigate Pseudo-Alignment in LLM4TS](https://arxiv.org/abs/2510.12847)
*Liangwei Nathan Zheng, Wenhao Liang, Wei Emma Zhang, Miao Xu, Olaf Maennel, Weitong Chen*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Pseudo-Alignment is a pervasive challenge in many large language models for time series (LLM4TS) models, often causing them to underperform compared to linear models or randomly initialised backbones. However, there is limited discussion in the community for the reasons that pseudo-alignment occurs. In this work, we conduct a thorough investigation into the root causes of pseudo-alignment in LLM4TS and build a connection of pseudo-alignment to the cone effect in LLM. We demonstrate that pseudo-alignment arises from the interplay of cone effect within pretrained LLM components and the intrinsically low-dimensional manifold of time-series data. In addition, we also introduce \textit{\textbf{TimeSUP}}, a novel technique designed to mitigate this issue and improve forecast performance in existing LLM4TS approaches. TimeSUP addresses this by increasing the time series manifold to more closely match the intrinsic dimension of language embeddings, allowing the model to distinguish temporal signals clearly while still capturing shared structures across modalities. As a result, representations for time and language tokens remain distinct yet exhibit high cosine similarity, signifying that the model preserves each modality unique features while learning their commonalities in a unified embedding space. Empirically, TimeSUP consistently outperforms state-of-the-art LLM4TS methods and other lightweight baselines on long-term forecasting performance. Furthermore, it can be seamlessly integrated into four existing LLM4TS pipelines and delivers significant improvements in forecasting performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12847) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles](https://arxiv.org/abs/2510.12992)
*Neel P. Bhatt, Po-han Li, Kushagra Gupta, Rohan Siva, Daniel Milan, Alexander T. Hogue, Sandeep P. Chinchali, David Fridovich-Keil, Zhangyang Wang, Ufuk Topcu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: https://uncap-project.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12992) | **Categories:** cs.RO, cs.CL, cs.CV, cs.MA

---

### [2] [Geometric Model Predictive Path Integral for Agile UAV Control with Online Collision Avoidance](https://arxiv.org/abs/2510.12924)
*Pavel Pochobradský, Ondřej Procházka, Robert Pěnička, Vojtěch Vonásek, Martin Saska*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this letter, we introduce Geometric Model Predictive Path Integral (GMPPI), a sampling-based controller capable of tracking agile trajectories while avoiding obstacles. In each iteration, GMPPI generates a large number of candidate rollout trajectories and then averages them to create a nominal control to be followed by the Unmanned Aerial Vehicle (UAV). We propose using geometric SE(3) control to generate part of the rollout trajectories, significantly increasing precision in agile flight. Furthermore, we introduce varying rollout simulation time step length and dynamic cost and noise parameters, vastly improving tracking performance of smooth and low-speed trajectories over an existing Model Predictive Path Integral (MPPI) implementation. Finally, we propose an integration of GMPPI with a stereo depth camera, enabling online obstacle avoidance at high speeds, a crucial step towards autonomous UAV flights in complex environments. The proposed controller can track simulated agile reference trajectories with position error similar to the geometric SE(3) controller. However, the same configuration of the proposed controller can avoid obstacles in a simulated forest environment at speeds of up to 13m/s, surpassing the performance of a state-of-the-art obstacle-aware planner. In real-world experiments, GMPPI retains the capability to track agile trajectories and avoids obstacles at speeds of up to 10m/s.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12924) | **Categories:** cs.RO

---

### [3] [Enhancing Sampling-based Planning with a Library of Paths](https://arxiv.org/abs/2510.12962)
*Michal Minařík, Vojtěch Vonásek, Robert Pěnička*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Path planning for 3D solid objects is a challenging problem, requiring a search in a six-dimensional configuration space, which is, nevertheless, essential in many robotic applications such as bin-picking and assembly. The commonly used sampling-based planners, such as Rapidly-exploring Random Trees, struggle with narrow passages where the sampling probability is low, increasing the time needed to find a solution. In scenarios like robotic bin-picking, various objects must be transported through the same environment. However, traditional planners start from scratch each time, losing valuable information gained during the planning process. We address this by using a library of past solutions, allowing the reuse of previous experiences even when planning for a new, previously unseen object. Paths for a set of objects are stored, and when planning for a new object, we find the most similar one in the library and use its paths as approximate solutions, adjusting for possible mutual transformations. The configuration space is then sampled along the approximate paths. Our method is tested in various narrow passage scenarios and compared with state-of-the-art methods from the OMPL library. Results show significant speed improvements (up to 85% decrease in the required time) of our method, often finding a solution in cases where the other planners fail. Our implementation of the proposed method is released as an open-source package.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.12962) | **Categories:** cs.RO

---

### [4] [Real-Time Knee Angle Prediction Using EMG and Kinematic Data with an Attention-Based CNN-LSTM Network and Transfer Learning Across Multiple Datasets](https://arxiv.org/abs/2510.13443)
*Mojtaba Mollahossein, Gholamreza Vossoughi, Mohammad Hossein Rohban*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Electromyography (EMG) signals are widely used for predicting body joint angles through machine learning (ML) and deep learning (DL) methods. However, these approaches often face challenges such as limited real-time applicability, non-representative test conditions, and the need for large datasets to achieve optimal performance. This paper presents a transfer-learning framework for knee joint angle prediction that requires only a few gait cycles from new subjects. Three datasets - Georgia Tech, the University of California Irvine (UCI), and the Sharif Mechatronic Lab Exoskeleton (SMLE) - containing four EMG channels relevant to knee motion were utilized. A lightweight attention-based CNN-LSTM model was developed and pre-trained on the Georgia Tech dataset, then transferred to the UCI and SMLE datasets. The proposed model achieved Normalized Mean Absolute Errors (NMAE) of 6.8 percent and 13.7 percent for one-step and 50-step predictions on abnormal subjects using EMG inputs alone. Incorporating historical knee angles reduced the NMAE to 3.1 percent and 3.5 percent for normal subjects, and to 2.8 percent and 7.5 percent for abnormal subjects. When further adapted to the SMLE exoskeleton with EMG, kinematic, and interaction force inputs, the model achieved 1.09 percent and 3.1 percent NMAE for one- and 50-step predictions, respectively. These results demonstrate robust performance and strong generalization for both short- and long-term rehabilitation scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13443) | **Categories:** cs.RO

---

### [5] [InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy](https://arxiv.org/abs/2510.13778)
*Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, Yang Tian, Bin Wang, Bolun Wang, Fangjing Wang, Hanqing Wang, Tai Wang, Ziqin Wang, Xueyuan Wei, Chao Wu, Shuai Yang, Jinhui Ye, Junqiu Yu, Jia Zeng, Jingjing Zhang, Jinyu Zhang, Shi Zhang, Feng Zheng, Bowen Zhou, Yangkun Zhu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce InternVLA-M1, a unified framework for spatial grounding and robot control that advances instruction-following robots toward scalable, general-purpose intelligence. Its core idea is spatially guided vision-language-action training, where spatial grounding serves as the critical link between instructions and robot actions. InternVLA-M1 employs a two-stage pipeline: (i) spatial grounding pre-training on over 2.3M spatial reasoning data to determine ``where to act'' by aligning instructions with visual, embodiment-agnostic positions, and (ii) spatially guided action post-training to decide ``how to act'' by generating embodiment-aware actions through plug-and-play spatial prompting. This spatially guided training recipe yields consistent gains: InternVLA-M1 outperforms its variant without spatial guidance by +14.6% on SimplerEnv Google Robot, +17% on WidowX, and +4.3% on LIBERO Franka, while demonstrating stronger spatial reasoning capability in box, point, and trace prediction. To further scale instruction following, we built a simulation engine to collect 244K generalizable pick-and-place episodes, enabling a 6.2% average improvement across 200 tasks and 3K+ objects. In real-world clustered pick-and-place, InternVLA-M1 improved by 7.3%, and with synthetic co-training, achieved +20.6% on unseen objects and novel configurations. Moreover, in long-horizon reasoning-intensive scenarios, it surpassed existing works by over 10%. These results highlight spatially guided training as a unifying principle for scalable and resilient generalist robots. Code and models are available at https://github.com/InternRobotics/InternVLA-M1.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13778) | **Categories:** cs.RO, cs.AI, cs.CV

---


## eess.SY [eess.SY]
### [1] [Safe Driving in Occluded Environments](https://arxiv.org/abs/2510.13114)
*Zhuoyuan Wang, Tongyao Jia, Pharuj Rajborirug, Neeraj Ramesh, Hiroyuki Okuda, Tatsuya Suzuki, Soummya Kar, Yorie Nakahira*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring safe autonomous driving in the presence of occlusions poses a significant challenge in its policy design. While existing model-driven control techniques based on set invariance can handle visible risks, occlusions create latent risks in which safety-critical states are not observable. Data-driven techniques also struggle to handle latent risks because direct mappings from risk-critical objects in sensor inputs to safe actions cannot be learned without visible risk-critical objects. Motivated by these challenges, in this paper, we propose a probabilistic safety certificate for latent risk. Our key technical enabler is the application of probabilistic invariance: It relaxes the strict observability requirements imposed by set-invariance methods that demand the knowledge of risk-critical states. The proposed techniques provide linear action constraints that confine the latent risk probability within tolerance. Such constraints can be integrated into model predictive controllers or embedded in data-driven policies to mitigate latent risks. The proposed method is tested using the CARLA simulator and compared with a few existing techniques. The theoretical and empirical analysis jointly demonstrate that the proposed methods assure long-term safety in real-time control in occluded environments without being overly conservative and with transparency to exposed risks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13114) | **Categories:** eess.SY, cs.RO, cs.SY

---

### [2] [Physics-Informed Neural Network Modeling of Vehicle Collision Dynamics in Precision Immobilization Technique Maneuvers](https://arxiv.org/abs/2510.13461)
*Yangye Jiang, Jiachen Wang, Daofei Li*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate prediction of vehicle collision dynamics is crucial for advanced safety systems and post-impact control applications, yet existing methods face inherent trade-offs among computational efficiency, prediction accuracy, and data requirements. This paper proposes a dual Physics-Informed Neural Network framework addressing these challenges through two complementary networks. The first network integrates Gaussian Mixture Models with PINN architecture to learn impact force distributions from finite element analysis data while enforcing momentum conservation and energy consistency constraints. The second network employs an adaptive PINN with dynamic constraint weighting to predict post-collision vehicle dynamics, featuring an adaptive physics guard layer that prevents unrealistic predictions whil e preserving data-driven learning capabilities. The framework incorporates uncertainty quantification through time-varying parameters and enables rapid adaptation via fine-tuning strategies. Validation demonstrates significant improvements: the impact force model achieves relative errors below 15.0% for force prediction on finite element analysis (FEA) datasets, while the vehicle dynamics model reduces average trajectory prediction error by 63.6% compared to traditional four-degree-of-freedom models in scaled vehicle experiments. The integrated system maintains millisecond-level computational efficiency suitable for real-time applications while providing probabilistic confidence bounds essential for safety-critical control. Comprehensive validation through FEA simulation, dynamic modeling, and scaled vehicle experiments confirms the framework's effectiveness for Precision Immobilization Technique scenarios and general collision dynamics prediction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.13461) | **Categories:** eess.SY, cs.RO, cs.SY

---
