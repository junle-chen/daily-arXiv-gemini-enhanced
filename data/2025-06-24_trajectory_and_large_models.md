# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-24

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (11)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Are Bias Evaluation Methods Biased ?](https://arxiv.org/abs/2506.17111)
*Lina Berrayana, Sean Rooney, Luis Garcés-Erice, Ioana Giurgiu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The creation of benchmarks to evaluate the safety of Large Language Models is one of the key activities within the trusted AI community. These benchmarks allow models to be compared for different aspects of safety such as toxicity, bias, harmful behavior etc. Independent benchmarks adopt different approaches with distinct data sets and evaluation methods. We investigate how robust such benchmarks are by using different approaches to rank a set of representative models for bias and compare how similar are the overall rankings. We show that different but widely used bias evaluations methods result in disparate model rankings. We conclude with recommendations for the community in the usage of such benchmarks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17111) | **Categories:** cs.AI, cs.CL

---

### [2] [Large Language Models are Near-Optimal Decision-Makers with a Non-Human Learning Behavior](https://arxiv.org/abs/2506.16163)
*Hao Li, Gengrui Zhang, Petter Holme, Shuyue Hu, Zhen Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human decision-making belongs to the foundation of our society and civilization, but we are on the verge of a future where much of it will be delegated to artificial intelligence. The arrival of Large Language Models (LLMs) has transformed the nature and scope of AI-supported decision-making; however, the process by which they learn to make decisions, compared to humans, remains poorly understood. In this study, we examined the decision-making behavior of five leading LLMs across three core dimensions of real-world decision-making: uncertainty, risk, and set-shifting. Using three well-established experimental psychology tasks designed to probe these dimensions, we benchmarked LLMs against 360 newly recruited human participants. Across all tasks, LLMs often outperformed humans, approaching near-optimal performance. Moreover, the processes underlying their decisions diverged fundamentally from those of humans. On the one hand, our finding demonstrates the ability of LLMs to manage uncertainty, calibrate risk, and adapt to changes. On the other hand, this disparity highlights the risks of relying on them as substitutes for human judgment, calling for further inquiry.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16163) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Privacy-Preserving in Connected and Autonomous Vehicles Through Vision to Text Transformation](https://arxiv.org/abs/2506.15854)
*Abdolazim Rezaei, Mehdi Sookhak, Ahmad Patooghy*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Connected and Autonomous Vehicles (CAVs) rely on a range of devices that often process privacy-sensitive data. Among these, roadside units play a critical role particularly through the use of AI-equipped (AIE) cameras for applications such as violation detection. However, the privacy risks associated with captured imagery remain a major concern, as such data can be misused for identity theft, profiling, or unauthorized commercial purposes. While traditional techniques such as face blurring and obfuscation have been applied to mitigate privacy risks, individual privacy remains at risk, as individuals can still be tracked using other features such as their clothing. This paper introduces a novel privacy-preserving framework that leverages feedback-based reinforcement learning (RL) and vision-language models (VLMs) to protect sensitive visual information captured by AIE cameras. The main idea is to convert images into semantically equivalent textual descriptions, ensuring that scene-relevant information is retained while visual privacy is preserved. A hierarchical RL strategy is employed to iteratively refine the generated text, enhancing both semantic accuracy and privacy. Evaluation results demonstrate significant improvements in both privacy protection and textual quality, with the Unique Word Count increasing by approximately 77\% and Detail Density by around 50\% compared to existing approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15854) | **Categories:** cs.CV, cs.LG

---

### [2] [VideoGAN-based Trajectory Proposal for Automated Vehicles](https://arxiv.org/abs/2506.16209)
*Annajoyce Mariani, Kira Maag, Hanno Gottschalk*

Main category: cs.CV

TL;DR: 该论文提出了一种使用生成对抗网络从鸟瞰图交通场景视频中生成逼真轨迹的方法。


<details>
  <summary>Details</summary>
Motivation: 现有的基于模型、规则和经典学习的方法难以有效地捕捉未来轨迹的复杂、多模态分布。

Method: 该论文提出了一种流水线，使用低分辨率鸟瞰图占用网格视频作为视频生成模型的训练数据，并从生成的交通场景视频中使用单帧对象检测和帧到帧对象匹配提取抽象轨迹数据。

Result: 该论文在100 GPU小时内获得了最佳结果，推理时间低于20毫秒，并证明了所提出的轨迹在空间和动态参数的分布上与Waymo Open Motion数据集中的真实视频对齐。

Conclusion: 该论文证明了使用生成对抗网络（GAN）从鸟瞰图交通场景视频中生成的轨迹在空间和动态参数的分布上与Waymo Open Motion数据集中的真实视频对齐，具有物理真实性。

Abstract: 能够生成逼真的轨迹选项是提高道路车辆自动化程度的核心。虽然目前基于模型、基于规则和基于经典学习的方法被广泛用于解决这些任务，但它们可能难以有效地捕捉未来轨迹的复杂、多模态分布。在本文中，我们研究了在鸟瞰图（BEV）交通场景视频上训练的生成对抗网络（GAN）是否可以生成统计上准确的轨迹，从而正确捕获智能体之间的空间关系。为此，我们提出了一种流水线，该流水线使用低分辨率BEV占用网格视频作为视频生成模型的训练数据。从生成的交通场景视频中，我们使用单帧对象检测和帧到帧对象匹配提取抽象轨迹数据。我们特别选择了一种GAN架构，以便相对于扩散模型实现快速的训练和推理时间。我们在100个GPU小时的训练时间内获得了最佳结果，推理时间低于20毫秒。我们证明了所提出的轨迹在空间和动态参数的分布上与Waymo Open Motion数据集中的真实视频对齐，具有物理真实性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16209) | **Categories:** cs.CV, cs.LG

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [RAST: Reasoning Activation in LLMs via Small-model Transfer](https://arxiv.org/abs/2506.15710)
*Siru Ouyang, Xinyu Zhu, Zilin Xiao, Minhao Jiang, Yu Meng, Jiawei Han*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) has become a powerful approach for improving the reasoning capabilities of large language models (LLMs), as evidenced by recent successes such as OpenAI's o1 and Deepseek-R1. However, applying RL at scale remains intimidatingly resource-intensive, requiring multiple model copies and extensive GPU workloads. On the other hand, while being powerful, recent studies suggest that RL does not fundamentally endow models with new knowledge; rather, it primarily reshapes the model's output distribution to activate reasoning capabilities latent in the base model. Building on this insight, we hypothesize that the changes in output probabilities induced by RL are largely model-size invariant, opening the door to a more efficient paradigm: training a small model with RL and transferring its induced probability shifts to larger base models. To verify our hypothesis, we conduct a token-level analysis of decoding trajectories and find high alignment in RL-induced output distributions across model scales, validating our hypothesis. Motivated by this, we propose RAST, a simple yet effective method that transfers reasoning behaviors by injecting RL-induced probability adjustments from a small RL-trained model into larger models. Experiments across multiple mathematical reasoning benchmarks show that RAST substantially and consistently enhances the reasoning capabilities of base models while requiring significantly lower GPU memory than direct RL training, sometimes even yielding better performance than the RL-trained counterparts. Our findings offer new insights into the nature of RL-driven reasoning and practical strategies for scaling its benefits without incurring its full computational cost. The project page of RAST is available at https://ozyyshr.github.io/RAST/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15710) | **Categories:** cs.LG, cs.AI

---

### [2] [SimuGen: Multi-modal Agentic Framework for Constructing Block Diagram-Based Simulation Models](https://arxiv.org/abs/2506.15695)
*Xinxing Ren, Qianbo Zang, Zekun Guo*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in large language models (LLMs) have shown impressive performance in mathematical reasoning and code generation. However, LLMs still struggle in the simulation domain, particularly in generating Simulink models, which are essential tools in engineering and scientific research. Our preliminary experiments indicate that LLM agents often fail to produce reliable and complete Simulink simulation code from text-only inputs, likely due to the lack of Simulink-specific data in their pretraining. To address this challenge, we propose SimuGen, a multimodal agent-based framework that automatically generates accurate Simulink simulation code by leveraging both the visual Simulink diagram and domain knowledge. SimuGen coordinates several specialized agents, including an investigator, unit test reviewer, code generator, executor, debug locator, and report writer, supported by a domain-specific knowledge base. This collaborative and modular design enables interpretable, robust, and reproducible Simulink simulation generation. Our source code is publicly available at https://github.com/renxinxing123/SimuGen_beta.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15695) | **Categories:** cs.LG

---

### [3] [Contraction Actor-Critic: Contraction Metric-Guided Reinforcement Learning for Robust Path Tracking](https://arxiv.org/abs/2506.15700)
*Minjae Cho, Hiroyasu Tsukamoto, Huy Trong Tran*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Control contraction metrics (CCMs) provide a framework to co-synthesize a controller and a corresponding contraction metric -- a positive-definite Riemannian metric under which a closed-loop system is guaranteed to be incrementally exponentially stable. However, the synthesized controller only ensures that all the trajectories of the system converge to one single trajectory and, as such, does not impose any notion of optimality across an entire trajectory. Furthermore, constructing CCMs requires a known dynamics model and non-trivial effort in solving an infinite-dimensional convex feasibility problem, which limits its scalability to complex systems featuring high dimensionality with uncertainty. To address these issues, we propose to integrate CCMs into reinforcement learning (RL), where CCMs provide dynamics-informed feedback for learning control policies that minimize cumulative tracking error under unknown dynamics. We show that our algorithm, called contraction actor-critic (CAC), formally enhances the capability of CCMs to provide a set of contracting policies with the long-term optimality of RL in a fully automated setting. Given a pre-trained dynamics model, CAC simultaneously learns a contraction metric generator (CMG) -- which generates a contraction metric -- and uses an actor-critic algorithm to learn an optimal tracking policy guided by that metric. We demonstrate the effectiveness of our algorithm relative to established baselines through extensive empirical studies, including simulated and real-world robot experiments, and provide a theoretical rationale for incorporating contraction theory into RL.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15700) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [CooperRisk: A Driving Risk Quantification Pipeline with Multi-Agent Cooperative Perception and Prediction](https://arxiv.org/abs/2506.15868)
*Mingyue Lei, Zewei Zhou, Hongchen Li, Jia Hu, Jiaqi Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Risk quantification is a critical component of safe autonomous driving, however, constrained by the limited perception range and occlusion of single-vehicle systems in complex and dense scenarios. Vehicle-to-everything (V2X) paradigm has been a promising solution to sharing complementary perception information, nevertheless, how to ensure the risk interpretability while understanding multi-agent interaction with V2X remains an open question. In this paper, we introduce the first V2X-enabled risk quantification pipeline, CooperRisk, to fuse perception information from multiple agents and quantify the scenario driving risk in future multiple timestamps. The risk is represented as a scenario risk map to ensure interpretability based on risk severity and exposure, and the multi-agent interaction is captured by the learning-based cooperative prediction model. We carefully design a risk-oriented transformer-based prediction model with multi-modality and multi-agent considerations. It aims to ensure scene-consistent future behaviors of multiple agents and avoid conflicting predictions that could lead to overly conservative risk quantification and cause the ego vehicle to become overly hesitant to drive. Then, the temporal risk maps could serve to guide a model predictive control planner. We evaluate the CooperRisk pipeline in a real-world V2X dataset V2XPnP, and the experiments demonstrate its superior performance in risk quantification, showing a 44.35% decrease in conflict rate between the ego vehicle and background traffic participants.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15868) | **Categories:** cs.RO

---

### [2] [Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning](https://arxiv.org/abs/2506.15828)
*Emanuele Musumeci, Michele Brienza, Francesco Argenziano, Vincenzo Suriani, Daniele Nardi, Domenico D. Bloisi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Classical planning in AI and Robotics addresses complex tasks by shifting from imperative to declarative approaches (e.g., PDDL). However, these methods often fail in real scenarios due to limited robot perception and the need to ground perceptions to planning predicates. This often results in heavily hard-coded behaviors that struggle to adapt, even with scenarios where goals can be achieved through relaxed planning. Meanwhile, Large Language Models (LLMs) lead to planning systems that leverage commonsense reasoning but often at the cost of generating unfeasible and/or unsafe plans. To address these limitations, we present an approach integrating classical planning with LLMs, leveraging their ability to extract commonsense knowledge and ground actions. We propose a hierarchical formulation that enables robots to make unfeasible tasks tractable by defining functionally equivalent goals through gradual relaxation. This mechanism supports partial achievement of the intended objective, suited to the agent's specific context. Our method demonstrates its ability to adapt and execute tasks effectively within environments modeled using 3D Scene Graphs through comprehensive qualitative and quantitative evaluations. We also show how this method succeeds in complex scenarios where other benchmark methods are more likely to fail. Code, dataset, and additional material are released to the community.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15828) | **Categories:** cs.RO, cs.AI

---

### [3] [KARL: Kalman-Filter Assisted Reinforcement Learner for Dynamic Object Tracking and Grasping](https://arxiv.org/abs/2506.15945)
*Kowndinya Boyalakuntla, Abdeslam Boularias, Jingjin Yu*

Main category: cs.RO

TL;DR: KARL通过结合卡尔曼滤波器和强化学习，提高了眼手系统在动态环境中跟踪和抓取物体的性能。


<details>
  <summary>Details</summary>
Motivation: 扩展眼手 (EoH) 系统在具有挑战性的现实环境中的能力。

Method: KARL 结合了一个新颖的六阶段 RL 课程，并在感知和强化学习 (RL) 控制模块之间集成了一个强大的卡尔曼滤波器层，并引入了允许重试的机制。

Result: 在模拟和真实世界实验中进行的大量评估定性和定量地证实了 KARL 相对于早期系统的优势，实现了更高的抓取成功率和更快的机器人执行速度。

Conclusion: KARL在动态物体跟踪和抓取方面优于早期系统，实现了更高的抓取成功率和更快的机器人执行速度。

Abstract: 我们提出了卡尔曼滤波器辅助强化学习器 (KARL)，用于眼手 (EoH) 系统上的动态物体跟踪和抓取，显着扩展了此类系统在具有挑战性的现实环境中的能力。与之前的最先进技术相比，KARL (1) 结合了一个新颖的六阶段 RL 课程，该课程使系统的运动范围增加了一倍，从而大大提高了系统的抓取性能，(2) 在感知和强化学习 (RL) 控制模块之间集成了一个强大的卡尔曼滤波器层，使系统即使在目标物体暂时退出相机视野或经历快速、不可预测的运动时，也能保持不确定但连续的 6D 姿势估计，以及 (3) 引入了允许重试的机制，以便从不可避免的策略执行失败中优雅地恢复。在模拟和真实世界实验中进行的大量评估定性和定量地证实了 KARL 相对于早期系统的优势，实现了更高的抓取成功率和更快的机器人执行速度。KARL 的源代码和补充材料将在以下网址提供：https://github.com/arc-l/karl。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15945) | **Categories:** cs.RO

---

### [4] [From Theory to Practice: Identifying the Optimal Approach for Offset Point Tracking in the Context of Agricultural Robotics](https://arxiv.org/abs/2506.16143)
*Stephane Ngnepiepaye Wembe, Vincent Rousseau, Johann Laconte, Roland Lenain*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modern agriculture faces escalating challenges: increasing demand for food, labor shortages, and the urgent need to reduce environmental impact. Agricultural robotics has emerged as a promising response to these pressures, enabling the automation of precise and suitable field operations. In particular, robots equipped with implements for tasks such as weeding or sowing must interact delicately and accurately with the crops and soil. Unlike robots in other domains, these agricultural platforms typically use rigidly mounted implements, where the implement's position is more critical than the robot's center in determining task success. Yet, most control strategies in the literature focus on the vehicle body, often neglecting the acctual working point of the system. This is particularly important when considering new agriculture practices where crops row are not necessary straights. This paper presents a predictive control strategy targeting the implement's reference point. The method improves tracking performance by anticipating the motion of the implement, which, due to its offset from the vehicle's center of rotation, is prone to overshooting during turns if not properly accounted for.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16143) | **Categories:** cs.RO

---

### [5] [ControlVLA: Few-shot Object-centric Adaptation for Pre-trained Vision-Language-Action Models](https://arxiv.org/abs/2506.16211)
*Puhao Li, Yingying Wu, Ziheng Xi, Wanlin Li, Yuzhe Huang, Zhiyuan Zhang, Yinghan Chen, Jianan Wang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning real-world robotic manipulation is challenging, particularly when limited demonstrations are available. Existing methods for few-shot manipulation often rely on simulation-augmented data or pre-built modules like grasping and pose estimation, which struggle with sim-to-real gaps and lack extensibility. While large-scale imitation pre-training shows promise, adapting these general-purpose policies to specific tasks in data-scarce settings remains unexplored. To achieve this, we propose ControlVLA, a novel framework that bridges pre-trained VLA models with object-centric representations via a ControlNet-style architecture for efficient fine-tuning. Specifically, to introduce object-centric conditions without overwriting prior knowledge, ControlVLA zero-initializes a set of projection layers, allowing them to gradually adapt the pre-trained manipulation policies. In real-world experiments across 6 diverse tasks, including pouring cubes and folding clothes, our method achieves a 76.7% success rate while requiring only 10-20 demonstrations -- a significant improvement over traditional approaches that require more than 100 demonstrations to achieve comparable success. Additional experiments highlight ControlVLA's extensibility to long-horizon tasks and robustness to unseen objects and backgrounds.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16211) | **Categories:** cs.RO

---

### [6] [Probabilistic Collision Risk Estimation for Pedestrian Navigation](https://arxiv.org/abs/2506.16219)
*Amine Tourki, Paul Prevel, Nils Einecke, Tim Puphal, Alexandre Alahi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Intelligent devices for supporting persons with vision impairment are becoming more widespread, but they are lacking behind the advancements in intelligent driver assistant system. To make a first step forward, this work discusses the integration of the risk model technology, previously used in autonomous driving and advanced driver assistance systems, into an assistance device for persons with vision impairment. The risk model computes a probabilistic collision risk given object trajectories which has previously been shown to give better indications of an object's collision potential compared to distance or time-to-contact measures in vehicle scenarios. In this work, we show that the risk model is also superior in warning persons with vision impairment about dangerous objects. Our experiments demonstrate that the warning accuracy of the risk model is 67% while both distance and time-to-contact measures reach only 51% accuracy for real-world data.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16219) | **Categories:** cs.RO

---

### [7] [M-Predictive Spliner: Enabling Spatiotemporal Multi-Opponent Overtaking for Autonomous Racing](https://arxiv.org/abs/2506.16301)
*Nadine Imholz, Maurice Brunner, Nicolas Baumann, Edoardo Ghignone, Michele Magno*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unrestricted multi-agent racing presents a significant research challenge, requiring decision-making at the limits of a robot's operational capabilities. While previous approaches have either ignored spatiotemporal information in the decision-making process or been restricted to single-opponent scenarios, this work enables arbitrary multi-opponent head-to-head racing while considering the opponents' future intent. The proposed method employs a KF-based multi-opponent tracker to effectively perform opponent ReID by associating them across observations. Simultaneously, spatial and velocity GPR is performed on all observed opponent trajectories, providing predictive information to compute the overtaking maneuvers. This approach has been experimentally validated on a physical 1:10 scale autonomous racing car, achieving an overtaking success rate of up to 91.65% and demonstrating an average 10.13%-point improvement in safety at the same speed as the previous SotA. These results highlight its potential for high-performance autonomous racing.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16301) | **Categories:** cs.RO

---

### [8] [Goal-conditioned Hierarchical Reinforcement Learning for Sample-efficient and Safe Autonomous Driving at Intersections](https://arxiv.org/abs/2506.16336)
*Yiou Huang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning (RL) exhibits remarkable potential in addressing autonomous driving tasks. However, it is difficult to train a sample-efficient and safe policy in complex scenarios. In this article, we propose a novel hierarchical reinforcement learning (HRL) framework with a goal-conditioned collision prediction (GCCP) module. In the hierarchical structure, the GCCP module predicts collision risks according to different potential subgoals of the ego vehicle. A high-level decision-maker choose the best safe subgoal. A low-level motion-planner interacts with the environment according to the subgoal. Compared to traditional RL methods, our algorithm is more sample-efficient, since its hierarchical structure allows reusing the policies of subgoals across similar tasks for various navigation scenarios. In additional, the GCCP module's ability to predict both the ego vehicle's and surrounding vehicles' future actions according to different subgoals, ensures the safety of the ego vehicle throughout the decision-making process. Experimental results demonstrate that the proposed method converges to an optimal policy faster and achieves higher safety than traditional RL methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16336) | **Categories:** cs.RO, cs.MA

---

### [9] [CSC-MPPI: A Novel Constrained MPPI Framework with DBSCAN for Reliable Obstacle Avoidance](https://arxiv.org/abs/2506.16386)
*Leesai Park, Keunwoo Jang, Sanghyun Kim*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes Constrained Sampling Cluster Model Predictive Path Integral (CSC-MPPI), a novel constrained formulation of MPPI designed to enhance trajectory optimization while enforcing strict constraints on system states and control inputs. Traditional MPPI, which relies on a probabilistic sampling process, often struggles with constraint satisfaction and generates suboptimal trajectories due to the weighted averaging of sampled trajectories. To address these limitations, the proposed framework integrates a primal-dual gradient-based approach and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to steer sampled input trajectories into feasible regions while mitigating risks associated with weighted averaging. First, to ensure that sampled trajectories remain within the feasible region, the primal-dual gradient method is applied to iteratively shift sampled inputs while enforcing state and control constraints. Then, DBSCAN groups the sampled trajectories, enabling the selection of representative control inputs within each cluster. Finally, among the representative control inputs, the one with the lowest cost is chosen as the optimal action. As a result, CSC-MPPI guarantees constraint satisfaction, improves trajectory selection, and enhances robustness in complex environments. Simulation and real-world experiments demonstrate that CSC-MPPI outperforms traditional MPPI in obstacle avoidance, achieving improved reliability and efficiency. The experimental videos are available at https://cscmppi.github.io

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16386) | **Categories:** cs.RO

---

### [10] [Grounding Language Models with Semantic Digital Twins for Robotic Planning](https://arxiv.org/abs/2506.16493)
*Mehreen Naeem, Andrew Melnik, Michael Beetz*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce a novel framework that integrates Semantic Digital Twins (SDTs) with Large Language Models (LLMs) to enable adaptive and goal-driven robotic task execution in dynamic environments. The system decomposes natural language instructions into structured action triplets, which are grounded in contextual environmental data provided by the SDT. This semantic grounding allows the robot to interpret object affordances and interaction rules, enabling action planning and real-time adaptability. In case of execution failures, the LLM utilizes error feedback and SDT insights to generate recovery strategies and iteratively revise the action plan. We evaluate our approach using tasks from the ALFRED benchmark, demonstrating robust performance across various household scenarios. The proposed framework effectively combines high-level reasoning with semantic environment understanding, achieving reliable task completion in the face of uncertainty and failure.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16493) | **Categories:** cs.RO, cs.AI

---

### [11] [BIDA: A Bi-level Interaction Decision-making Algorithm for Autonomous Vehicles in Dynamic Traffic Scenarios](https://arxiv.org/abs/2506.16546)
*Liyang Yu, Tianyi Wang, Junfeng Jiao, Fengwu Shan, Hongqing Chu, Bingzhao Gao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In complex real-world traffic environments, autonomous vehicles (AVs) need to interact with other traffic participants while making real-time and safety-critical decisions accordingly. The unpredictability of human behaviors poses significant challenges, particularly in dynamic scenarios, such as multi-lane highways and unsignalized T-intersections. To address this gap, we design a bi-level interaction decision-making algorithm (BIDA) that integrates interactive Monte Carlo tree search (MCTS) with deep reinforcement learning (DRL), aiming to enhance interaction rationality, efficiency and safety of AVs in dynamic key traffic scenarios. Specifically, we adopt three types of DRL algorithms to construct a reliable value network and policy network, which guide the online deduction process of interactive MCTS by assisting in value update and node selection. Then, a dynamic trajectory planner and a trajectory tracking controller are designed and implemented in CARLA to ensure smooth execution of planned maneuvers. Experimental evaluations demonstrate that our BIDA not only enhances interactive deduction and reduces computational costs, but also outperforms other latest benchmarks, which exhibits superior safety, efficiency and interaction rationality under varying traffic conditions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.16546) | **Categories:** cs.RO, cs.AI, cs.ET, cs.LG, cs.SY, eess.SY

---
