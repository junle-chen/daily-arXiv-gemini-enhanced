# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-09

## 目录

- [人工智能 (Artificial Intelligence) (5)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (13)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents](https://arxiv.org/abs/2507.03293)
*Anand Gokhale, Vaibhav Srivastava, Francesco Bullo*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03293) | **Categories:** cs.AI, cs.CL, cs.LG, cs.SY, eess.SY

---

### [2] [Memory Mosaics at scale](https://arxiv.org/abs/2507.03285)
*Jianyu Zhang, Léon Bottou*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Memory Mosaics [Zhang et al., 2025], networks of associative memories, have demonstrated appealing compositional and in-context learning capabilities on medium-scale networks (GPT-2 scale) and synthetic small datasets. This work shows that these favorable properties remain when we scale memory mosaics to large language model sizes (llama-8B scale) and real-world datasets.   To this end, we scale memory mosaics to 10B size, we train them on one trillion tokens, we introduce a couple architectural modifications ("Memory Mosaics v2"), we assess their capabilities across three evaluation dimensions: training-knowledge storage, new-knowledge storage, and in-context learning.   Throughout the evaluation, memory mosaics v2 match transformers on the learning of training knowledge (first dimension) and significantly outperforms transformers on carrying out new tasks at inference time (second and third dimensions). These improvements cannot be easily replicated by simply increasing the training data for transformers. A memory mosaics v2 trained on one trillion tokens still perform better on these tasks than a transformer trained on eight trillion tokens.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03285) | **Categories:** cs.AI

---

### [3] [A Technical Survey of Reinforcement Learning Techniques for Large Language Models](https://arxiv.org/abs/2507.04136)
*Saksham Sahai Srivastava, Vaneet Aggarwal*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement Learning (RL) has emerged as a transformative approach for aligning and enhancing Large Language Models (LLMs), addressing critical challenges in instruction following, ethical alignment, and reasoning capabilities. This survey offers a comprehensive foundation on the integration of RL with language models, highlighting prominent algorithms such as Proximal Policy Optimization (PPO), Q-Learning, and Actor-Critic methods. Additionally, it provides an extensive technical overview of RL techniques specifically tailored for LLMs, including foundational methods like Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), as well as advanced strategies such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO). We systematically analyze their applications across domains, i.e., from code generation to tool-augmented reasoning. We also present a comparative taxonomy based on reward modeling, feedback mechanisms, and optimization strategies. Our evaluation highlights key trends. RLHF remains dominant for alignment, and outcome-based RL such as RLVR significantly improves stepwise reasoning. However, persistent challenges such as reward hacking, computational costs, and scalable feedback collection underscore the need for continued innovation. We further discuss emerging directions, including hybrid RL algorithms, verifier-guided training, and multi-objective alignment frameworks. This survey serves as a roadmap for researchers advancing RL-driven LLM development, balancing capability enhancement with safety and scalability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04136) | **Categories:** cs.AI

---

### [4] [Towards Machine Theory of Mind with Large Language Model-Augmented Inverse Planning](https://arxiv.org/abs/2507.03682)
*Rebekah A. Gelpí, Eric Xue, William A. Cunningham*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose a hybrid approach to machine Theory of Mind (ToM) that uses large language models (LLMs) as a mechanism for generating hypotheses and likelihood functions with a Bayesian inverse planning model that computes posterior probabilities for an agent's likely mental states given its actions. Bayesian inverse planning models can accurately predict human reasoning on a variety of ToM tasks, but these models are constrained in their ability to scale these predictions to scenarios with a large number of possible hypotheses and actions. Conversely, LLM-based approaches have recently demonstrated promise in solving ToM benchmarks, but can exhibit brittleness and failures on reasoning tasks even when they pass otherwise structurally identical versions. By combining these two methods, this approach leverages the strengths of each component, closely matching optimal results on a task inspired by prior inverse planning models and improving performance relative to models that utilize LLMs alone or with chain-of-thought prompting, even with smaller LLMs that typically perform poorly on ToM tasks. We also exhibit the model's potential to predict mental states on open-ended tasks, offering a promising direction for future development of ToM models and the creation of socially intelligent generative agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03682) | **Categories:** cs.AI, cs.LG

---

### [5] [LLMs model how humans induce logically structured rules](https://arxiv.org/abs/2507.03876)
*Alyssa Loo, Ellie Pavlick, Roman Feiman*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03876) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DriveMRP: Enhancing Vision-Language Models with Synthetic Motion Data for Motion Risk Prediction](https://arxiv.org/abs/2507.02948)
*Zhiyi Hou, Enhui Ma, Fang Li, Zhiyi Lai, Kalok Ho, Zhanqian Wu, Lijun Zhou, Long Chen, Chitian Sun, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Kaicheng Yu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous driving has seen significant progress, driven by extensive real-world data. However, in long-tail scenarios, accurately predicting the safety of the ego vehicle's future motion remains a major challenge due to uncertainties in dynamic environments and limitations in data coverage. In this work, we aim to explore whether it is possible to enhance the motion risk prediction capabilities of Vision-Language Models (VLM) by synthesizing high-risk motion data. Specifically, we introduce a Bird's-Eye View (BEV) based motion simulation method to model risks from three aspects: the ego-vehicle, other vehicles, and the environment. This allows us to synthesize plug-and-play, high-risk motion data suitable for VLM training, which we call DriveMRP-10K. Furthermore, we design a VLM-agnostic motion risk estimation framework, named DriveMRP-Agent. This framework incorporates a novel information injection strategy for global context, ego-vehicle perspective, and trajectory projection, enabling VLMs to effectively reason about the spatial relationships between motion waypoints and the environment. Extensive experiments demonstrate that by fine-tuning with DriveMRP-10K, our DriveMRP-Agent framework can significantly improve the motion risk prediction performance of multiple VLM baselines, with the accident recognition accuracy soaring from 27.13% to 88.03%. Moreover, when tested via zero-shot evaluation on an in-house real-world high-risk motion dataset, DriveMRP-Agent achieves a significant performance leap, boosting the accuracy from base_model's 29.42% to 68.50%, which showcases the strong generalization capabilities of our method in real-world scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.02948) | **Categories:** cs.CV, cs.AI, cs.RO, I.4.8; I.2.7; I.2.10

---

### [2] [Text-Guided Multi-Instance Learning for Scoliosis Screening via Gait Video Analysis](https://arxiv.org/abs/2507.02996)
*Haiqing Li, Yuzhi Guo, Feng Jiang, Thao M. Dang, Hehuan Ma, Qifeng Zhou, Jean Gao, Junzhou Huang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Early-stage scoliosis is often difficult to detect, particularly in adolescents, where delayed diagnosis can lead to serious health issues. Traditional X-ray-based methods carry radiation risks and rely heavily on clinical expertise, limiting their use in large-scale screenings. To overcome these challenges, we propose a Text-Guided Multi-Instance Learning Network (TG-MILNet) for non-invasive scoliosis detection using gait videos. To handle temporal misalignment in gait sequences, we employ Dynamic Time Warping (DTW) clustering to segment videos into key gait phases. To focus on the most relevant diagnostic features, we introduce an Inter-Bag Temporal Attention (IBTA) mechanism that highlights critical gait phases. Recognizing the difficulty in identifying borderline cases, we design a Boundary-Aware Model (BAM) to improve sensitivity to subtle spinal deviations. Additionally, we incorporate textual guidance from domain experts and large language models (LLM) to enhance feature representation and improve model interpretability. Experiments on the large-scale Scoliosis1K gait dataset show that TG-MILNet achieves state-of-the-art performance, particularly excelling in handling class imbalance and accurately detecting challenging borderline cases. The code is available at https://github.com/lhqqq/TG-MILNet

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.02996) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [BERT4Traj: Transformer Based Trajectory Reconstruction for Sparse Mobility Data](https://arxiv.org/abs/2507.03062)
*Hao Yang, Angela Yao, Christopher Whalen, Gengchen Mai*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Understanding human mobility is essential for applications in public health, transportation, and urban planning. However, mobility data often suffers from sparsity due to limitations in data collection methods, such as infrequent GPS sampling or call detail record (CDR) data that only capture locations during communication events. To address this challenge, we propose BERT4Traj, a transformer based model that reconstructs complete mobility trajectories by predicting hidden visits in sparse movement sequences. Inspired by BERT's masked language modeling objective and self_attention mechanisms, BERT4Traj leverages spatial embeddings, temporal embeddings, and contextual background features such as demographics and anchor points. We evaluate BERT4Traj on real world CDR and GPS datasets collected in Kampala, Uganda, demonstrating that our approach significantly outperforms traditional models such as Markov Chains, KNN, RNNs, and LSTMs. Our results show that BERT4Traj effectively reconstructs detailed and continuous mobility trajectories, enhancing insights into human movement patterns.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03062) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [VLM-TDP: VLM-guided Trajectory-conditioned Diffusion Policy for Robust Long-Horizon Manipulation](https://arxiv.org/abs/2507.04524)
*Kefeng Huang, Tingguang Li, Yuzhen Liu, Zhe Zhang, Jiankun Wang, Lei Han*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion policy has demonstrated promising performance in the field of robotic manipulation. However, its effectiveness has been primarily limited in short-horizon tasks, and its performance significantly degrades in the presence of image noise. To address these limitations, we propose a VLM-guided trajectory-conditioned diffusion policy (VLM-TDP) for robust and long-horizon manipulation. Specifically, the proposed method leverages state-of-the-art vision-language models (VLMs) to decompose long-horizon tasks into concise, manageable sub-tasks, while also innovatively generating voxel-based trajectories for each sub-task. The generated trajectories serve as a crucial conditioning factor, effectively steering the diffusion policy and substantially enhancing its performance. The proposed Trajectory-conditioned Diffusion Policy (TDP) is trained on trajectories derived from demonstration data and validated using the trajectories generated by the VLM. Simulation experimental results indicate that our method significantly outperforms classical diffusion policies, achieving an average 44% increase in success rate, over 100% improvement in long-horizon tasks, and a 20% reduction in performance degradation in challenging conditions, such as noisy images or altered environments. These findings are further reinforced by our real-world experiments, where the performance gap becomes even more pronounced in long-horizon tasks. Videos are available on https://youtu.be/g0T6h32OSC8

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04524) | **Categories:** cs.RO

---

### [2] [Label-Free Long-Horizon 3D UAV Trajectory Prediction via Motion-Aligned RGB and Event Cues](https://arxiv.org/abs/2507.03365)
*Hanfang Liang, Shenghai Yuan, Fen Liu, Yizhuo Yang, Bing Wang, Zhuyu Huang, Chenyang Shi, Jing Jin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The widespread use of consumer drones has introduced serious challenges for airspace security and public safety. Their high agility and unpredictable motion make drones difficult to track and intercept. While existing methods focus on detecting current positions, many counter-drone strategies rely on forecasting future trajectories and thus require more than reactive detection to be effective. To address this critical gap, we propose an unsupervised vision-based method for predicting the three-dimensional trajectories of drones. Our approach first uses an unsupervised technique to extract drone trajectories from raw LiDAR point clouds, then aligns these trajectories with camera images through motion consistency to generate reliable pseudo-labels. We then combine kinematic estimation with a visual Mamba neural network in a self-supervised manner to predict future drone trajectories. We evaluate our method on the challenging MMAUD dataset, including the V2 sequences that feature wide-field-of-view multimodal sensors and dynamic UAV motion in urban scenes. Extensive experiments show that our framework outperforms supervised image-only and audio-visual baselines in long-horizon trajectory prediction, reducing 5-second 3D error by around 40 percent without using any manual 3D labels. The proposed system offers a cost-effective, scalable alternative for real-time counter-drone deployment. All code will be released upon acceptance to support reproducible research in the robotics community.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03365) | **Categories:** cs.RO

---

### [3] [SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](https://arxiv.org/abs/2507.04263)
*Liwen Xiao, Zhiyu Pan, Zhicheng Wang, Zhiguo Cao, Wei Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate prediction of multi-agent future trajectories is crucial for autonomous driving systems to make safe and efficient decisions. Trajectory refinement has emerged as a key strategy to enhance prediction accuracy. However, existing refinement methods often overlook the topological relationships between trajectories, which are vital for improving prediction precision. Inspired by braid theory, we propose a novel trajectory refinement approach, Soft-Braid Refiner (SRefiner), guided by the soft-braid topological structure of trajectories using Soft-Braid Attention. Soft-Braid Attention captures spatio-temporal topological relationships between trajectories by considering both spatial proximity and vehicle motion states at ``soft intersection points". Additionally, we extend this approach to model interactions between trajectories and lanes, further improving the prediction accuracy. SRefiner is a multi-iteration, multi-agent framework that iteratively refines trajectories, incorporating topological information to enhance interactions within traffic scenarios. SRefiner achieves significant performance improvements over four baseline methods across two datasets, establishing a new state-of-the-art in trajectory refinement. Code is here https://github.com/Liwen-Xiao/SRefiner.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04263) | **Categories:** cs.RO

---

### [4] ["Hi AirStar, Guide Me to the Badminton Court."](https://arxiv.org/abs/2507.04430)
*Ziqin Wang, Jinyu Chen, Xiangyi Zheng, Qinan Liao, Linjiang Huang, Si Liu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unmanned Aerial Vehicles, operating in environments with relatively few obstacles, offer high maneuverability and full three-dimensional mobility. This allows them to rapidly approach objects and perform a wide range of tasks often challenging for ground robots, making them ideal for exploration, inspection, aerial imaging, and everyday assistance. In this paper, we introduce AirStar, a UAV-centric embodied platform that turns a UAV into an intelligent aerial assistant: a large language model acts as the cognitive core for environmental understanding, contextual reasoning, and task planning. AirStar accepts natural interaction through voice commands and gestures, removing the need for a remote controller and significantly broadening its user base. It combines geospatial knowledge-driven long-distance navigation with contextual reasoning for fine-grained short-range control, resulting in an efficient and accurate vision-and-language navigation (VLN) capability.Furthermore, the system also offers built-in capabilities such as cross-modal question answering, intelligent filming, and target tracking. With a highly extensible framework, it supports seamless integration of new functionalities, paving the way toward a general-purpose, instruction-driven intelligent UAV agent. The supplementary PPT is available at \href{https://buaa-colalab.github.io/airstar.github.io}{https://buaa-colalab.github.io/airstar.github.io}.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04430) | **Categories:** cs.RO

---

### [5] [MOSU: Autonomous Long-range Robot Navigation with Multi-modal Scene Understanding](https://arxiv.org/abs/2507.04686)
*Jing Liang, Kasun Weerakoon, Daeun Song, Senthurbavan Kirubaharan, Xuesu Xiao, Dinesh Manocha*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present MOSU, a novel autonomous long-range navigation system that enhances global navigation for mobile robots through multimodal perception and on-road scene understanding. MOSU addresses the outdoor robot navigation challenge by integrating geometric, semantic, and contextual information to ensure comprehensive scene understanding. The system combines GPS and QGIS map-based routing for high-level global path planning and multi-modal trajectory generation for local navigation refinement. For trajectory generation, MOSU leverages multi-modalities: LiDAR-based geometric data for precise obstacle avoidance, image-based semantic segmentation for traversability assessment, and Vision-Language Models (VLMs) to capture social context and enable the robot to adhere to social norms in complex environments. This multi-modal integration improves scene understanding and enhances traversability, allowing the robot to adapt to diverse outdoor conditions. We evaluate our system in real-world on-road environments and benchmark it on the GND dataset, achieving a 10% improvement in traversability on navigable terrains while maintaining a comparable navigation distance to existing global navigation methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04686) | **Categories:** cs.RO

---

### [6] [Personalised Explanations in Long-term Human-Robot Interactions](https://arxiv.org/abs/2507.03049)
*Ferran Gebellí, Anaís Garrell, Jan-Gerrit Habekost, Séverin Lemaignan, Stefan Wermter, Raquel Ros*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In the field of Human-Robot Interaction (HRI), a fundamental challenge is to facilitate human understanding of robots. The emerging domain of eXplainable HRI (XHRI) investigates methods to generate explanations and evaluate their impact on human-robot interactions. Previous works have highlighted the need to personalise the level of detail of these explanations to enhance usability and comprehension. Our paper presents a framework designed to update and retrieve user knowledge-memory models, allowing for adapting the explanations' level of detail while referencing previously acquired concepts. Three architectures based on our proposed framework that use Large Language Models (LLMs) are evaluated in two distinct scenarios: a hospital patrolling robot and a kitchen assistant robot. Experimental results demonstrate that a two-stage architecture, which first generates an explanation and then personalises it, is the framework architecture that effectively reduces the level of detail only when there is related user knowledge.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03049) | **Categories:** cs.RO, cs.AI, cs.HC

---

### [7] [Image-driven Robot Drawing with Rapid Lognormal Movements](https://arxiv.org/abs/2507.03166)
*Daniel Berio, Guillaume Clivaz, Michael Stroh, Oliver Deussen, Réjean Plamondon, Sylvain Calinon, Frederic Fol Leymarie*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large image generation and vision models, combined with differentiable rendering technologies, have become powerful tools for generating paths that can be drawn or painted by a robot. However, these tools often overlook the intrinsic physicality of the human drawing/writing act, which is usually executed with skillful hand/arm gestures. Taking this into account is important for the visual aesthetics of the results and for the development of closer and more intuitive artist-robot collaboration scenarios. We present a method that bridges this gap by enabling gradient-based optimization of natural human-like motions guided by cost functions defined in image space. To this end, we use the sigma-lognormal model of human hand/arm movements, with an adaptation that enables its use in conjunction with a differentiable vector graphics (DiffVG) renderer. We demonstrate how this pipeline can be used to generate feasible trajectories for a robot by combining image-driven objectives with a minimum-time smoothing criterion. We demonstrate applications with generation and robotic reproduction of synthetic graffiti as well as image abstraction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.03166) | **Categories:** cs.RO, cs.GR

---

### [8] [Are Learning-Based Approaches Ready for Real-World Indoor Navigation? A Case for Imitation Learning](https://arxiv.org/abs/2507.04086)
*Nigitha Selvaraj, Alex Mitrevski, Sebastian Houben*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional indoor robot navigation methods provide a reliable solution when adapted to constrained scenarios, but lack flexibility or require manual re-tuning when deployed in more complex settings. In contrast, learning-based approaches learn directly from sensor data and environmental interactions, enabling easier adaptability. While significant work has been presented in the context of learning navigation policies, learning-based methods are rarely compared to traditional navigation methods directly, which is a problem for their ultimate acceptance in general navigation contexts. In this work, we explore the viability of imitation learning (IL) for indoor navigation, using expert (joystick) demonstrations to train various navigation policy networks based on RGB images, LiDAR, and a combination of both, and we compare our IL approach to a traditional potential field-based navigation method. We evaluate the approach on a physical mobile robot platform equipped with a 2D LiDAR and a camera in an indoor university environment. Our multimodal model demonstrates superior navigation capabilities in most scenarios, but faces challenges in dynamic environments, likely due to limited diversity in the demonstrations. Nevertheless, the ability to learn directly from data and generalise across layouts suggests that IL can be a practical navigation approach, and potentially a useful initialisation strategy for subsequent lifelong learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04086) | **Categories:** cs.RO

---

### [9] [An improved 2D time-to-collision for articulated vehicles: predicting sideswipe and rear-end collisions](https://arxiv.org/abs/2507.04184)
*Abhijeet Behera, Sogol Kharrazi, Erik Frisk, Maytheewat Aramrattana*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time-to-collision (TTC) is a widely used measure for estimating the time until a rear-end collision between two vehicles, assuming both maintain constant speeds and headings in the prediction horizon. To also capture sideswipe collisions, a two-dimensional extension, TTC$_{\text{2D}}$, was introduced. However, this formulation assumes both vehicles have the same heading and that their headings remain unchanged during the manoeuvre, in addition to the standard assumptions on the prediction horizon. Moreover, its use for articulated vehicles like a tractor-semitrailer remains unclear. This paper addresses these limitations by developing three enhanced versions of TTC$_{\text{2D}}$. The first incorporates vehicle heading information, which is missing in the original formulation. The standard assumption of constant speed and heading in the prediction horizon holds. The second adapts this to articulated vehicles while retaining the assumptions of the first version. The third version maintains the constant heading assumption but relaxes the constant speed assumption by allowing constant acceleration. The versions are tested in a cut-in scenario using the CARLA simulation environment. They detect rear-end collisions, similar to TTC, and moreover, they also identify sideswipe risks, something TTC could not predict.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04184) | **Categories:** cs.RO

---

### [10] [Implicit Dual-Control for Visibility-Aware Navigation in Unstructured Environments](https://arxiv.org/abs/2507.04371)
*Benjamin Johnson, Qilun Zhu, Robert Prucka, Morgan Barron, Miriam Figueroa-Santos, Matthew Castanier*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Navigating complex, cluttered, and unstructured environments that are a priori unknown presents significant challenges for autonomous ground vehicles, particularly when operating with a limited field of view(FOV) resulting in frequent occlusion and unobserved space. This paper introduces a novel visibility-aware model predictive path integral framework(VA-MPPI). Formulated as a dual control problem where perceptual uncertainties and control decisions are intertwined, it reasons over perception uncertainty evolution within a unified planning and control pipeline. Unlike traditional methods that rely on explicit uncertainty objectives, the VA-MPPI controller implicitly balances exploration and exploitation, reducing uncertainty only when system performance would be increased. The VA-MPPI framework is evaluated in simulation against deterministic and prescient controllers across multiple scenarios, including a cluttered urban alleyway and an occluded off-road environment. The results demonstrate that VA-MPPI significantly improves safety by reducing collision with unseen obstacles while maintaining competitive performance. For example, in the off-road scenario with 400 control samples, the VA-MPPI controller achieved a success rate of 84%, compared to only 8% for the deterministic controller, with all VA-MPPI failures arising from unmet stopping criteria rather than collisions. Furthermore, the controller implicitly avoids unobserved space, improving safety without explicit directives. The proposed framework highlights the potential for robust, visibility-aware navigation in unstructured and occluded environments, paving the way for future advancements in autonomous ground vehicle systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04371) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [11] [Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition](https://arxiv.org/abs/2507.04384)
*Wule Mao, Zhouheng Li, Yunhao Luo, Yilun Du, Lei Xie*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe trajectory planning remains a significant challenge in complex environments, where traditional methods often trade off computational efficiency for safety. Comprehensive obstacle modeling improves safety but is computationally expensive, while approximate methods are more efficient but may compromise safety. To address this issue, this paper introduces a rapid and safe trajectory planning framework based on state-based diffusion models. Leveraging only low-dimensional vehicle states, the diffusion models achieve notable inference efficiency while ensuring sufficient collision-free characteristics. By composing diffusion models, the proposed framework can safely generalize across diverse scenarios, planning collision-free trajectories even in unseen scenes. To further ensure the safety of the generated trajectories, an efficient, rule-based safety filter is proposed, which selects optimal trajectories that satisfy both sufficient safety and control feasibility from among candidate trajectories. Both in seen and unseen scenarios, the proposed method achieves efficient inference time while maintaining high safety and stability. Evaluations on the F1TENTH vehicle further demonstrate that the proposed method is practical in real-world applications. The project page is at: https://rstp-comp-diffuser.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04384) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [12] [DragonFly: Single mmWave Radar 3D Localization of Highly Dynamic Tags in GPS-Denied Environments](https://arxiv.org/abs/2507.04602)
*Skanda Harisha, Jimmy G. D. Hester, Aline Eid*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The accurate localization and tracking of dynamic targets, such as equipment, people, vehicles, drones, robots, and the assets that they interact with in GPS-denied indoor environments is critical to enabling safe and efficient operations in the next generation of spatially aware industrial facilities. This paper presents DragonFly , a 3D localization system of highly dynamic backscatter tags using a single MIMO mmWave radar. The system delivers the first demonstration of a mmWave backscatter system capable of exploiting the capabilities of MIMO radars for the 3D localization of mmID tags moving at high speeds and accelerations at long ranges by introducing a critical Doppler disambiguation algorithm and a fully integrated cross-polarized dielectric lens-based mmID tag consuming a mere 68 uW. DragonFly was extensively evaluated in static and dynamic configurations, including on a flying quadcopter, and benchmarked against multiple baselines, demonstrating its ability to track the positions of multiple tags with a median 3D accuracy of 12 cm at speeds and acceleration on the order of 10 m/s-1 and 4 m/s-2 and at ranges of up to 50 m.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04602) | **Categories:** cs.RO

---

### [13] [Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning](https://arxiv.org/abs/2507.04790)
*Giwon Lee, Wooseong Jeong, Daehee Park, Jaewoo Jeong, Kuk-Jin Yoon*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose Interaction-Merged Motion Planning (IMMP), a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.04790) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---
