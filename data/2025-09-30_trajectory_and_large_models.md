# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-30

## 目录

- [人工智能 (Artificial Intelligence) (6)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器人学 (Robotics) (18)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [D-Artemis: A Deliberative Cognitive Framework for Mobile GUI Multi-Agents](https://arxiv.org/abs/2509.21799)
*Hongze Mi, Yibo Feng, Wenjie Lu, Yuqi Wang, Jinyuan Li, Song Cao, He Cui, Tengfei Tian, Xuelin Zhang, Haotian Luo, Di Sun, Naiqiang Tan, Gang Pan*

Main category: cs.AI

TL;DR: D-Artemis是一个模仿人类认知循环的GUI代理框架，它通过细粒度的提示检索、预执行对齐和状态反思来提高通用多模态大语言模型在GUI任务中的性能，而无需在复杂轨迹数据集上进行训练。


<details>
  <summary>Details</summary>
Motivation: 现有GUI代理方法存在端到端训练中的数据瓶颈、延迟错误检测的高成本以及矛盾指导的风险。

Method: D-Artemis框架包含三个关键模块：细粒度的应用特定提示检索机制、Thought-Action Consistency (TAC)检查模块和Action Correction Agent (ACA)组成的预执行对齐阶段，以及执行后的Status Reflection Agent (SRA)。

Result: D-Artemis在AndroidWorld和ScreenSpot-V2两个基准测试中均取得了新的state-of-the-art (SOTA) 结果，成功率分别为75.8%和96.8%。

Conclusion: D-Artemis通过模仿人类认知循环，显著提高了通用多模态大语言模型在GUI任务中的性能，且无需在复杂轨迹数据集上进行训练，并在两个基准测试中取得了SOTA结果。

Abstract: 图形用户界面（GUI）代理旨在通过模拟用户交互来自动化各种人类任务。尽管发展迅速，但当前的方法受到几个关键挑战的阻碍：端到端训练中的数据瓶颈、延迟错误检测的高成本以及矛盾指导的风险。受到人类认知循环——思考、对齐和反思的启发，我们在本文中提出了D-Artemis——一种新颖的审议框架。D-Artemis利用细粒度的、特定于应用程序的提示检索机制来告知其决策过程。它还采用了一个主动的预执行对齐阶段，其中思想-行动一致性（TAC）检查模块和行动校正代理（ACA）协同工作，以降低执行失败的风险。执行后的状态反思代理（SRA）完成了认知循环，从而能够从经验中进行战略学习。至关重要的是，D-Artemis增强了通用多模态大型语言模型（MLLM）在GUI任务中的能力，而无需在复杂的轨迹数据集上进行训练，从而表现出强大的泛化能力。D-Artemis在两个主要基准测试中都建立了新的最先进（SOTA）结果，在AndroidWorld上实现了75.8%的成功率，在ScreenSpot-V2上实现了96.8%的成功率。广泛的消融研究进一步证明了每个组件对框架的重大贡献。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21799) | **Categories:** cs.AI

---

### [2] [DeepTravel: An End-to-End Agentic Reinforcement Learning Framework for Autonomous Travel Planning Agents](https://arxiv.org/abs/2509.21842)
*Yansong Ning, Rui Liu, Jun Wang, Kai Chen, Wei Li, Jun Fang, Kan Zheng, Naiqiang Tan, Hao Liu*

Main category: cs.AI

TL;DR: DeepTravel提出了一种端到端的agent强化学习框架，用于构建自主旅行规划agent，能够自主规划、执行工具以及反思工具响应。


<details>
  <summary>Details</summary>
Motivation: 现有旅行规划agent依赖于手工prompt和固定工作流程，缺乏灵活性和自主性。

Method: 提出了DeepTravel框架，包含一个鲁棒的sandbox环境、一个分层奖励建模系统和一个reply增强的强化学习方法。

Result: DeepTravel使小尺寸LLM（例如Qwen3 32B）在旅行规划任务中显著优于现有的前沿LLM，例如OpenAI o1、o3和DeepSeek R1。

Conclusion: DeepTravel框架能够有效提升旅行规划agent的自主性和性能。

Abstract: 旅行规划 (TP) 代理最近已成为一个新兴的构建块，用于与外部工具和资源交互以生成旅行行程，从而确保愉快的用户体验。 尽管它具有优势，但现有研究依赖于手工制作的提示和固定的代理工作流程，阻碍了更灵活和自主的TP代理。 本文提出了一种端到端agent强化学习框架DeepTravel，用于构建自主旅行规划agent，该agent能够自主规划、执行工具以及反思工具响应，从而在多步推理中探索、验证和完善中间操作。 为了实现这一目标，我们首先通过缓存交通、住宿和POI数据来构建一个强大的sandbox环境，从而促进TP代理训练，而不受现实世界API限制（例如，不一致的输出）的约束。 此外，我们开发了一个分层奖励建模系统，其中轨迹级别的验证器首先检查时空可行性并过滤不满足的旅行行程，然后turn级别的验证器进一步验证行程细节与工具响应的一致性，从而实现高效而精确的奖励服务。 最后，我们提出了reply增强的强化学习方法，该方法使TP代理能够定期从失败经验缓冲区中进行replay，从而产生显着的agent能力。 我们在DiDi Enterprise Solutions App上部署了经过训练的TP代理，并进行了全面的线上和线下评估，结果表明DeepTravel使小尺寸LLM（例如Qwen3 32B）在旅行计划任务中显着优于现有的前沿LLM，例如OpenAI o1、o3和DeepSeek R1。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21842) | **Categories:** cs.AI

---

### [3] [GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models](https://arxiv.org/abs/2509.21593)
*Peng Luo, Xiayin Lou, Yu Zheng, Zhuo Zheng, Stefano Ermon*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Geospatial modeling provides critical solutions for pressing global challenges such as sustainability and climate change. Existing large language model (LLM)-based algorithm discovery frameworks, such as AlphaEvolve, excel at evolving generic code but lack the domain knowledge and multi-step reasoning required for complex geospatial problems. We introduce GeoEvolve, a multi-agent LLM framework that couples evolutionary search with geospatial domain knowledge to automatically design and refine geospatial algorithms. GeoEvolve operates in two nested loops: an inner loop leverages a code evolver to generate and mutate candidate solutions, while an outer agentic controller evaluates global elites and queries a GeoKnowRAG module -- a structured geospatial knowledge base that injects theoretical priors from geography. This knowledge-guided evolution steers the search toward theoretically meaningful and computationally efficient algorithms. We evaluate GeoEvolve on two fundamental and classical tasks: spatial interpolation (kriging) and spatial uncertainty quantification (geospatial conformal prediction). Across these benchmarks, GeoEvolve automatically improves and discovers new algorithms, incorporating geospatial theory on top of classical models. It reduces spatial interpolation error (RMSE) by 13-21% and enhances uncertainty estimation performance by 17\%. Ablation studies confirm that domain-guided retrieval is essential for stable, high-quality evolution. These results demonstrate that GeoEvolve provides a scalable path toward automated, knowledge-driven geospatial modeling, opening new opportunities for trustworthy and efficient AI-for-Science discovery.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21593) | **Categories:** cs.AI, physics.soc-ph

---

### [4] [Can AI Perceive Physical Danger and Intervene?](https://arxiv.org/abs/2509.21651)
*Abhishek Jindal, Dmitry Kalashnikov, Oscar Chang, Divya Garikapati, Anirudha Majumdar, Pierre Sermanet, Vikas Sindhwani*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: When AI interacts with the physical world -- as a robot or an assistive agent -- new safety challenges emerge beyond those of purely ``digital AI". In such interactions, the potential for physical harm is direct and immediate. How well do state-of-the-art foundation models understand common-sense facts about physical safety, e.g. that a box may be too heavy to lift, or that a hot cup of coffee should not be handed to a child? In this paper, our contributions are three-fold: first, we develop a highly scalable approach to continuous physical safety benchmarking of Embodied AI systems, grounded in real-world injury narratives and operational safety constraints. To probe multi-modal safety understanding, we turn these narratives and constraints into photorealistic images and videos capturing transitions from safe to unsafe states, using advanced generative models. Secondly, we comprehensively analyze the ability of major foundation models to perceive risks, reason about safety, and trigger interventions; this yields multi-faceted insights into their deployment readiness for safety-critical agentic applications. Finally, we develop a post-training paradigm to teach models to explicitly reason about embodiment-specific safety constraints provided through system instructions. The resulting models generate thinking traces that make safety reasoning interpretable and transparent, achieving state of the art performance in constraint satisfaction evaluations. The benchmark will be released at https://asimov-benchmark.github.io/v2

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21651) | **Categories:** cs.AI

---

### [5] [CoBel-World: Harnessing LLM Reasoning to Build a Collaborative Belief World for Optimizing Embodied Multi-Agent Collaboration](https://arxiv.org/abs/2509.21981)
*Zhimin Wang, Shaokang He, Duo Wu, Jinghe Wang, Linjia Kang, Jing Yu, Zhi Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents -- a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a collaborative belief world -- an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse open-world task knowledge into structured beliefs via a symbolic belief language, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 22-60% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21981) | **Categories:** cs.AI, cs.MA

---

### [6] [GeoSketch: A Neural-Symbolic Approach to Geometric Multimodal Reasoning with Auxiliary Line Construction and Affine Transformation](https://arxiv.org/abs/2509.22460)
*Shichao Weng, Zhiqiang Wang, Yuhua Zhou, Rui Lu, Ting Liu, Zhiyang Teng, Xiaozhang Liu, Hanmeng Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Geometric Problem Solving (GPS) poses a unique challenge for Multimodal Large Language Models (MLLMs), requiring not only the joint interpretation of text and diagrams but also iterative visuospatial reasoning. While existing approaches process diagrams as static images, they lack the capacity for dynamic manipulation - a core aspect of human geometric reasoning involving auxiliary line construction and affine transformations. We present GeoSketch, a neural-symbolic framework that recasts geometric reasoning as an interactive perception-reasoning-action loop. GeoSketch integrates: (1) a Perception module that abstracts diagrams into structured logic forms, (2) a Symbolic Reasoning module that applies geometric theorems to decide the next deductive step, and (3) a Sketch Action module that executes operations such as drawing auxiliary lines or applying transformations, thereby updating the diagram in a closed loop. To train this agent, we develop a two-stage pipeline: supervised fine-tuning on 2,000 symbolic-curated trajectories followed by reinforcement learning with dense, symbolic rewards to enhance robustness and strategic exploration. To evaluate this paradigm, we introduce the GeoSketch Benchmark, a high-quality set of 390 geometry problems requiring auxiliary construction or affine transformations. Experiments on strong MLLM baselines demonstrate that GeoSketch significantly improves stepwise reasoning accuracy and problem-solving success over static perception methods. By unifying hierarchical decision-making, executable visual actions, and symbolic verification, GeoSketch advances multimodal reasoning from static interpretation to dynamic, verifiable interaction, establishing a new foundation for solving complex visuospatial problems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22460) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [What Happens Next? Anticipating Future Motion by Generating Point Trajectories](https://arxiv.org/abs/2509.21592)
*Gabrijel Boduljak, Laurynas Karazija, Iro Laina, Christian Rupprecht, Andrea Vedaldi*

Main category: cs.CV

TL;DR: 本文提出了一种从单张图像预测物体运动轨迹的方法，通过条件生成密集轨迹网格，实现了比现有方法更准确和多样的预测。


<details>
  <summary>Details</summary>
Motivation: 解决仅通过单张图像预测物体运动轨迹的问题，无需物体速度或作用力等其他参数。

Method: 提出了一种条件生成密集轨迹网格的模型，该模型架构与现代视频生成器类似，但输出的是运动轨迹而不是像素。

Result: 在模拟数据上进行了广泛评估，证明了其在机器人等下游应用中的有效性，并在真实世界的直观物理数据集上显示出良好的准确性。

Conclusion: 结果表明，直接建模运动比生成像素更能有效预测单张图像中的物体运动。

Abstract: 本文研究了从单张图像预测运动的问题，即预测世界中的物体可能如何移动，而无需观察物体速度或施加在物体上的力等其他参数。 我们将此任务定义为密集轨迹网格的条件生成，该模型与现代视频生成器的架构非常相似，但输出的是运动轨迹而不是像素。 这种方法捕捉了场景范围内的动态和不确定性，从而产生比先前的回归器和生成器更准确和多样的预测。 我们在模拟数据上广泛评估了我们的方法，证明了其在机器人等下游应用中的有效性，并在真实世界的直观物理数据集上显示出良好的准确性。 尽管最近最先进的视频生成器通常被认为是世界模型，但我们表明，即使在简单的物理场景（例如掉落的块或机械物体交互）中，它们也很难从单个图像预测运动，尽管对此类数据进行了微调。 我们表明，这种限制是由于生成像素而不是直接建模运动的开销造成的。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21592) | **Categories:** cs.CV, cs.AI, cs.LG

---

### [2] [Learning GUI Grounding with Spatial Reasoning from Visual Feedback](https://arxiv.org/abs/2509.21552)
*Yu Zhao, Wei-Ning Chen, Huseyin Atahan Inan, Samuel Kessler, Lu Wang, Lukas Wutschitz, Fangkai Yang, Chaoyun Zhang, Pasquale Minervini, Saravan Rajmohan, Robert Sim*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Graphical User Interface (GUI) grounding is commonly framed as a coordinate prediction task -- given a natural language instruction, generate on-screen coordinates for actions such as clicks and keystrokes. However, recent Vision Language Models (VLMs) often fail to predict accurate numeric coordinates when processing high-resolution GUI images with complex layouts. To address this issue, we reframe GUI grounding as an \emph{interactive search task}, where the VLM generates actions to move a cursor in the GUI to locate UI elements. At each step, the model determines the target object, evaluates the spatial relations between the cursor and the target, and moves the cursor closer to the target conditioned on the movement history. In this interactive process, the rendered cursor provides visual feedback to help the model align its predictions with the corresponding on-screen locations. We train our GUI grounding model, GUI-Cursor, using multi-step online reinforcement learning with a dense trajectory-based reward function. Our experimental results show that GUI-Cursor, based on Qwen2.5-VL-7B, improves the GUI grounding accuracy and achieves state-of-the-art results on ScreenSpot-v2 ($88.8\% \rightarrow 93.9\%$) and ScreenSpot-Pro ($26.8\% \rightarrow 56.5\%$). Moreover, we observe that GUI-Cursor learns to solve the problem within two steps for 95\% of instances and can adaptively conduct more steps on more difficult examples.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21552) | **Categories:** cs.CV, cs.CL

---

### [3] [Motion-Aware Transformer for Multi-Object Tracking](https://arxiv.org/abs/2509.21715)
*Xu Yang, Gady Agam*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-object tracking (MOT) in videos remains challenging due to complex object motions and crowded scenes. Recent DETR-based frameworks offer end-to-end solutions but typically process detection and tracking queries jointly within a single Transformer Decoder layer, leading to conflicts and degraded association accuracy. We introduce the Motion-Aware Transformer (MATR), which explicitly predicts object movements across frames to update track queries in advance. By reducing query collisions, MATR enables more consistent training and improves both detection and association. Extensive experiments on DanceTrack, SportsMOT, and BDD100k show that MATR delivers significant gains across standard metrics. On DanceTrack, MATR improves HOTA by more than 9 points over MOTR without additional data and reaches a new state-of-the-art score of 71.3 with supplementary data. MATR also achieves state-of-the-art results on SportsMOT (72.2 HOTA) and BDD100k (54.7 mTETA, 41.6 mHOTA) without relying on external datasets. These results demonstrate that explicitly modeling motion within end-to-end Transformers offers a simple yet highly effective approach to advancing multi-object tracking.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21715) | **Categories:** cs.CV

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Improved Vehicle Maneuver Prediction using Game Theoretic Priors](https://arxiv.org/abs/2509.21873)
*Nishant Doshi*

Main category: cs.RO

TL;DR: 本文提出了一种结合博弈论和传统运动模型的车辆行为预测方法，以提高预测精度。


<details>
  <summary>Details</summary>
Motivation: 传统行为预测方法在预测变道行为时，需要场景信息，并且没有考虑智能体之间的交互。

Method: 利用Level-k博弈论对车辆间的交互进行建模，并将博弈论的输出作为先验知识与传统运动模型结合。

Result: 该方法可以更准确地预测车辆的合理行为。

Conclusion: 该预测结果可用于自适应巡航控制等决策系统，从而提高燃油效率。

Abstract: 传统的机动预测方法使用某种分类模型对时间轨迹数据进行分析，以预测智能体在一定时间范围内的行为。尽管这些模型具有最佳的精确度和召回率，但除非它们包含有关整个场景的信息，否则无法准确预测变道行为。Level-k博弈论可以利用类人分层推理来得出每个智能体在一组中可以做出的最合理的决策。这可以用于模拟不同车辆在彼此存在的情况下的交互，从而计算出每个智能体将做出的最合理的决策。博弈论评估的结果可以用作“先验”或与传统的基于运动的分类模型结合使用，以实现更准确的预测。所提出的方法假设目标引导车辆周围的车辆状态是已知的。该模块将基于在线优化解决方案输出目标车辆的最合理的机动预测。这些预测有助于诸如自适应巡航控制（ACC）或Traxen的iQ-Cruise等决策系统，从而进一步提高燃油效率。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21873) | **Categories:** cs.RO

---

### [2] [WoW: Towards a World omniscient World model Through Embodied Interaction](https://arxiv.org/abs/2509.22642)
*Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi-min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, Jian Tang*

Main category: cs.RO

TL;DR: 该论文提出了一种名为WoW的生成世界模型，通过在大量机器人交互轨迹上训练，并结合SOPHIA和逆动力学模型，提升了AI在物理因果关系方面的理解能力。


<details>
  <summary>Details</summary>
Motivation: 当前视频模型（如Sora）依赖被动观察，难以理解物理因果关系；人类通过与世界的积极互动发展直观物理理解。论文旨在验证：世界模型的真实物理直觉必须基于与真实世界的广泛、因果丰富的交互。

Method: 论文提出了一个140亿参数的生成世界模型WoW，该模型在200万个机器人交互轨迹上进行训练。同时，使用SOPHIA（视觉-语言模型智能体）评估DiT生成的输出，并通过迭代演化语言指令来指导改进，从而约束模型向物理真实性靠拢。此外，还训练了一个逆动力学模型，将改进后的计划转化为可执行的机器人动作。

Result: 实验结果表明，WoW模型对物理的理解是合理结果的概率分布，导致随机不稳定性和物理幻觉。通过SOPHIA的约束，可以主动地将模型引导向物理真实性。WoW在WoWBench基准测试中取得了最先进的性能，证明了其在物理因果关系、碰撞动力学和物体持久性方面的强大能力。

Conclusion: 大规模的真实世界交互是AI发展物理直觉的基石。论文的模型、数据和基准测试将开源。

Abstract: 人类通过与世界的积极互动来发展对直观物理的理解。这与当前的视频模型（如Sora）形成鲜明对比，后者依赖于被动观察，因此难以掌握物理因果关系。这一观察结果引出了我们的中心假设：世界模型的真实物理直觉必须建立在与现实世界的大量、因果丰富的互动之上。为了验证这一假设，我们提出了WoW，一个在200万个机器人互动轨迹上训练的140亿参数的生成世界模型。我们的研究结果表明，该模型对物理的理解是合理结果的概率分布，导致随机不稳定性和物理幻觉。此外，我们证明了这种新兴能力可以通过SOPHIA主动地约束到物理真实性，其中视觉-语言模型智能体评估DiT生成的输出，并通过迭代演化语言指令来指导其改进。此外，一个共同训练的逆动力学模型将这些改进的计划转化为可执行的机器人动作，从而闭合了从想象到行动的循环。我们建立了WoWBench，这是一个新的基准，专注于视频中的物理一致性和因果推理，WoW在人类和自主评估中都取得了最先进的性能，证明了其在物理因果关系、碰撞动力学和物体持久性方面的强大能力。我们的工作提供了系统的证据，表明大规模的真实世界互动是人工智能发展物理直觉的基石。模型、数据和基准测试将开源。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22642) | **Categories:** cs.RO, cs.CV, cs.MM

---

### [3] [VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search](https://arxiv.org/abs/2509.22643)
*Wenkai Guo, Guanxing Lu, Haoyuan Deng, Zhenyu Wu, Yansong Tang, Ziwei Wang*

Main category: cs.RO

TL;DR: VLA-Reasoner通过测试时扩展，使现成的VLA具备了预测未来状态的能力，从而显著提升了机器人操作的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的视觉-语言-动作模型(VLA)在处理长时程轨迹任务时，由于预测的短视性，会产生累积偏差，导致性能下降。

Method: 提出了一个名为VLA-Reasoner的插件框架，该框架通过采样和推演可能的动作轨迹来预测未来状态，并利用蒙特卡洛树搜索(MCTS)提高搜索效率，同时引入基于核密度估计(KDE)的置信度抽样机制，以实现高效探索。此外，还采用了离线奖励塑造策略来评估中间状态，并纠正偏差。

Result: 在模拟器和真实世界的实验中，VLA-Reasoner相比最先进的VLA模型取得了显著的改进。

Conclusion: 该方法展示了机器人操作可扩展测试时计算的潜在途径。

Abstract: 视觉-语言-动作模型(VLA)通过扩展模仿学习在通用机器人操作任务中取得了强大的性能。然而，现有的VLA模型仅限于预测短视的下一步动作，由于增量偏差，难以处理长时程轨迹任务。为了解决这个问题，我们提出了一个名为VLA-Reasoner的插件框架，该框架通过测试时扩展，有效地使现成的VLA具备了预测未来状态的能力。具体来说，VLA-Reasoner采样并推演可能的动作轨迹，其中涉及的动作是使用世界模型生成未来状态的理由，这使得VLA-Reasoner能够预见和推理潜在的结果，并搜索最佳动作。我们进一步利用蒙特卡洛树搜索(MCTS)来提高大型动作空间中的搜索效率，其中逐步的VLA预测为根节点提供种子。同时，我们引入了一种基于核密度估计(KDE)的置信度抽样机制，以在MCTS中实现高效探索，而无需冗余的VLA查询。我们通过离线奖励塑造策略评估MCTS中的中间状态，以评估预测的未来并使用长期反馈纠正偏差。我们在模拟器和真实世界中进行了广泛的实验，证明了我们提出的VLA-Reasoner相比最先进的VLA模型取得了显著的改进。我们的方法突出了机器人操作可扩展测试时计算的潜在途径。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22643) | **Categories:** cs.RO

---

### [4] [DroneFL: Federated Learning for Multi-UAV Visual Target Tracking](https://arxiv.org/abs/2509.21523)
*Xiaofan Yu, Yuwei Wu, Katherine Mao, Ye Tian, Vijay Kumar, Tajana Rosing*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-robot target tracking is a fundamental problem that requires coordinated monitoring of dynamic entities in applications such as precision agriculture, environmental monitoring, disaster response, and security surveillance. While Federated Learning (FL) has the potential to enhance learning across multiple robots without centralized data aggregation, its use in multi-Unmanned Aerial Vehicle (UAV) target tracking remains largely underexplored. Key challenges include limited onboard computational resources, significant data heterogeneity in FL due to varying targets and the fields of view, and the need for tight coupling between trajectory prediction and multi-robot planning. In this paper, we introduce DroneFL, the first federated learning framework specifically designed for efficient multi-UAV target tracking. We design a lightweight local model to predict target trajectories from sensor inputs, using a frozen YOLO backbone and a shallow transformer for efficient onboard training. The updated models are periodically aggregated in the cloud for global knowledge sharing. To alleviate the data heterogeneity that hinders FL convergence, DroneFL introduces a position-invariant model architecture with altitude-based adaptive instance normalization. Finally, we fuse predictions from multiple UAVs in the cloud and generate optimal trajectories that balance target prediction accuracy and overall tracking performance. Our results show that DroneFL reduces prediction error by 6%-83% and tracking distance by 0.4%-4.6% compared to a distributed non-FL framework. In terms of efficiency, DroneFL runs in real time on a Raspberry Pi 5 and has on average just 1.56 KBps data rate to the cloud.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21523) | **Categories:** cs.RO

---

### [5] [Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation](https://arxiv.org/abs/2509.21690)
*Muqun Hu, Wenxi Chen, Wenjing Li, Falak Mandali, Zijian He, Renhong Zhang, Praveen Krisna, Katherine Christian, Leo Benaharon, Dizhi Ma, Karthik Ramani, Yan Gu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing -- capabilities that remain difficult for unified controllers. We propose a reinforcement learning framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate $\geq$ 96% and success rate $\geq$ 92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21690) | **Categories:** cs.RO

---

### [6] [SAGE: Scene Graph-Aware Guidance and Execution for Long-Horizon Manipulation Tasks](https://arxiv.org/abs/2509.21928)
*Jialiang Li, Wenzheng Wu, Gaojing Zhang, Yifan Han, Wenzhao Lian*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Successfully solving long-horizon manipulation tasks remains a fundamental challenge. These tasks involve extended action sequences and complex object interactions, presenting a critical gap between high-level symbolic planning and low-level continuous control. To bridge this gap, two essential capabilities are required: robust long-horizon task planning and effective goal-conditioned manipulation. Existing task planning methods, including traditional and LLM-based approaches, often exhibit limited generalization or sparse semantic reasoning. Meanwhile, image-conditioned control methods struggle to adapt to unseen tasks. To tackle these problems, we propose SAGE, a novel framework for Scene Graph-Aware Guidance and Execution in Long-Horizon Manipulation Tasks. SAGE utilizes semantic scene graphs as a structural representation for scene states. A structural scene graph enables bridging task-level semantic reasoning and pixel-level visuo-motor control. This also facilitates the controllable synthesis of accurate, novel sub-goal images. SAGE consists of two key components: (1) a scene graph-based task planner that uses VLMs and LLMs to parse the environment and reason about physically-grounded scene state transition sequences, and (2) a decoupled structural image editing pipeline that controllably converts each target sub-goal graph into a corresponding image through image inpainting and composition. Extensive experiments have demonstrated that SAGE achieves state-of-the-art performance on distinct long-horizon tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21928) | **Categories:** cs.RO, cs.AI

---

### [7] [FlowDrive: moderated flow matching with data balancing for trajectory planning](https://arxiv.org/abs/2509.21961)
*Lingguang Wang, Ömer Şahin Taş, Marlon Steiner, Christoph Stiller*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning-based planners are sensitive to the long-tailed distribution of driving data. Common maneuvers dominate datasets, while dangerous or rare scenarios are sparse. This imbalance can bias models toward the frequent cases and degrade performance on critical scenarios. To tackle this problem, we compare balancing strategies for sampling training data and find reweighting by trajectory pattern an effective approach. We then present FlowDrive, a flow-matching trajectory planner that learns a conditional rectified flow to map noise directly to trajectory distributions with few flow-matching steps. We further introduce moderated, in-the-loop guidance that injects small perturbation between flow steps to systematically increase trajectory diversity while remaining scene-consistent. On nuPlan and the interaction-focused interPlan benchmarks, FlowDrive achieves state-of-the-art results among learning-based planners and approaches methods with rule-based refinements. After adding moderated guidance and light post-processing (FlowDrive*), it achieves overall state-of-the-art performance across nearly all benchmark splits.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21961) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [8] [Hybrid Diffusion for Simultaneous Symbolic and Continuous Planning](https://arxiv.org/abs/2509.21983)
*Sigmund Hennum Høeg, Aksel Vaaler, Chaoqi Liu, Olav Egeland, Yilun Du*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Constructing robots to accomplish long-horizon tasks is a long-standing challenge within artificial intelligence. Approaches using generative methods, particularly Diffusion Models, have gained attention due to their ability to model continuous robotic trajectories for planning and control. However, we show that these models struggle with long-horizon tasks that involve complex decision-making and, in general, are prone to confusing different modes of behavior, leading to failure. To remedy this, we propose to augment continuous trajectory generation by simultaneously generating a high-level symbolic plan. We show that this requires a novel mix of discrete variable diffusion and continuous diffusion, which dramatically outperforms the baselines. In addition, we illustrate how this hybrid diffusion process enables flexible trajectory synthesis, allowing us to condition synthesized actions on partial and complete symbolic conditions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21983) | **Categories:** cs.RO, cs.AI

---

### [9] [Developing Vision-Language-Action Model from Egocentric Videos](https://arxiv.org/abs/2509.21986)
*Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, Shinsuke Mori*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Egocentric videos capture how humans manipulate objects and tools, providing diverse motion cues for learning object manipulation. Unlike the costly, expert-driven manual teleoperation commonly used in training Vision-Language-Action models (VLAs), egocentric videos offer a scalable alternative. However, prior studies that leverage such videos for training robot policies typically rely on auxiliary annotations, such as detailed hand-pose recordings. Consequently, it remains unclear whether VLAs can be trained directly from raw egocentric videos. In this work, we address this challenge by leveraging EgoScaler, a framework that extracts 6DoF object manipulation trajectories from egocentric videos without requiring auxiliary recordings. We apply EgoScaler to four large-scale egocentric video datasets and automatically refine noisy or incomplete trajectories, thereby constructing a new large-scale dataset for VLA pre-training. Our experiments with a state-of-the-art $\pi_0$ architecture in both simulated and real-robot environments yield three key findings: (i) pre-training on our dataset improves task success rates by over 20\% compared to training from scratch, (ii) the performance is competitive with that achieved using real-robot datasets, and (iii) combining our dataset with real-robot data yields further improvements. These results demonstrate that egocentric videos constitute a promising and scalable resource for advancing VLA research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.21986) | **Categories:** cs.RO, cs.AI

---

### [10] [Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation](https://arxiv.org/abs/2509.22093)
*Xiaohuan Pei, Yuxing Chen, Siyu Xu, Yunke Wang, Yuheng Shi, Chang Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic manipulation with Vision-Language-Action models requires efficient inference over long-horizon multi-modal context, where attention to dense visual tokens dominates computational cost. Existing methods optimize inference speed by reducing visual redundancy within VLA models, but they overlook the varying redundancy across robotic manipulation stages. We observe that the visual token redundancy is higher in coarse manipulation phase than in fine-grained operations, and is strongly correlated with the action dynamic. Motivated by this observation, we propose \textbf{A}ction-aware \textbf{D}ynamic \textbf{P}runing (\textbf{ADP}), a multi-modal pruning framework that integrates text-driven token selection with action-aware trajectory gating. Our method introduces a gating mechanism that conditions the pruning signal on recent action trajectories, using past motion windows to adaptively adjust token retention ratios in accordance with dynamics, thereby balancing computational efficiency and perceptual precision across different manipulation stages. Extensive experiments on the LIBERO suites and diverse real-world scenarios demonstrate that our method significantly reduces FLOPs and action inference latency (\textit{e.g.} $1.35 \times$ speed up on OpenVLA-OFT) while maintaining competitive success rates (\textit{e.g.} 25.8\% improvements with OpenVLA) compared to baselines, thereby providing a simple plug-in path to efficient robot policies that advances the efficiency and performance frontier of robotic manipulation. Our project website is: \href{https://vla-adp.github.io/}{ADP.com}.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22093) | **Categories:** cs.RO, cs.AI

---

### [11] [Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting](https://arxiv.org/abs/2509.22195)
*Asher J. Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, Anirudha Majumdar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Fine-tuning vision-language models (VLMs) on robot teleoperation data to create vision-language-action (VLA) models is a promising paradigm for training generalist policies, but it suffers from a fundamental tradeoff: learning to produce actions often diminishes the VLM's foundational reasoning and multimodal understanding, hindering generalization to novel scenarios, instruction following, and semantic understanding. We argue that this catastrophic forgetting is due to a distribution mismatch between the VLM's internet-scale pretraining corpus and the robotics fine-tuning data. Inspired by this observation, we introduce VLM2VLA: a VLA training paradigm that first resolves this mismatch at the data level by representing low-level actions with natural language. This alignment makes it possible to train VLAs solely with Low-Rank Adaptation (LoRA), thereby minimally modifying the VLM backbone and averting catastrophic forgetting. As a result, the VLM can be fine-tuned on robot teleoperation data without fundamentally altering the underlying architecture and without expensive co-training on internet-scale VLM datasets. Through extensive Visual Question Answering (VQA) studies and over 800 real-world robotics experiments, we demonstrate that VLM2VLA preserves the VLM's core capabilities, enabling zero-shot generalization to novel tasks that require open-world semantic reasoning and multilingual instruction following.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22195) | **Categories:** cs.RO

---

### [12] [MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training](https://arxiv.org/abs/2509.22199)
*Haoyun Li, Ivan Zhang, Runqi Ouyang, Xiaofeng Wang, Zheng Zhu, Zhiqin Yang, Zhentao Zhang, Boyuan Wang, Chaojun Ni, Wenkang Qin, Xinze Chen, Yun Ye, Guan Huang, Zhenbo Song, Xingang Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision Language Action (VLA) models derive their generalization capability from diverse training data, yet collecting embodied robot interaction data remains prohibitively expensive. In contrast, human demonstration videos are far more scalable and cost-efficient to collect, and recent studies confirm their effectiveness in training VLA models. However, a significant domain gap persists between human videos and robot-executed videos, including unstable camera viewpoints, visual discrepancies between human hands and robotic arms, and differences in motion dynamics. To bridge this gap, we propose MimicDreamer, a framework that turns fast, low-cost human demonstrations into robot-usable supervision by jointly aligning vision, viewpoint, and actions to directly support policy training. For visual alignment, we propose H2R Aligner, a video diffusion model that generates high-fidelity robot demonstration videos by transferring motion from human manipulation footage. For viewpoint stabilization, EgoStabilizer is proposed, which canonicalizes egocentric videos via homography and inpaints occlusions and distortions caused by warping. For action alignment, we map human hand trajectories to the robot frame and apply a constrained inverse kinematics solver to produce feasible, low-jitter joint commands with accurate pose tracking. Empirically, VLA models trained purely on our synthesized human-to-robot videos achieve few-shot execution on real robots. Moreover, scaling training with human data significantly boosts performance compared to models trained solely on real robot data; our approach improves the average success rate by 14.7\% across six representative manipulation tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22199) | **Categories:** cs.RO, cs.AI

---

### [13] [From Watch to Imagine: Steering Long-horizon Manipulation via Human Demonstration and Future Envisionment](https://arxiv.org/abs/2509.22205)
*Ke Ye, Jiaming Zhou, Yuanfeng Qiu, Jiayi Liu, Shihui Zhou, Kun-Yu Lin, Junwei Liang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generalizing to long-horizon manipulation tasks in a zero-shot setting remains a central challenge in robotics. Current multimodal foundation based approaches, despite their capabilities, typically fail to decompose high-level commands into executable action sequences from static visual input alone. To address this challenge, we introduce Super-Mimic, a hierarchical framework that enables zero-shot robotic imitation by directly inferring procedural intent from unscripted human demonstration videos. Our framework is composed of two sequential modules. First, a Human Intent Translator (HIT) parses the input video using multimodal reasoning to produce a sequence of language-grounded subtasks. These subtasks then condition a Future Dynamics Predictor (FDP), which employs a generative model that synthesizes a physically plausible video rollout for each step. The resulting visual trajectories are dynamics-aware, explicitly modeling crucial object interactions and contact points to guide the low-level controller. We validate this approach through extensive experiments on a suite of long-horizon manipulation tasks, where Super-Mimic significantly outperforms state-of-the-art zero-shot methods by over 20\%. These results establish that coupling video-driven intent parsing with prospective dynamics modeling is a highly effective strategy for developing general-purpose robotic systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22205) | **Categories:** cs.RO

---

### [14] [Beyond Detection -- Orchestrating Human-Robot-Robot Assistance via an Internet of Robotic Things Paradigm](https://arxiv.org/abs/2509.22296)
*Joseph Hunt, Koyo Fujii, Aly Magassouba, Praminda Caleb-Solly*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Hospital patient falls remain a critical and costly challenge worldwide. While conventional fall prevention systems typically rely on post-fall detection or reactive alerts, they also often suffer from high false positive rates and fail to address the underlying patient needs that lead to bed-exit attempts. This paper presents a novel system architecture that leverages the Internet of Robotic Things (IoRT) to orchestrate human-robot-robot interaction for proactive and personalized patient assistance. The system integrates a privacy-preserving thermal sensing model capable of real-time bed-exit prediction, with two coordinated robotic agents that respond dynamically based on predicted intent and patient input. This orchestrated response could not only reduce fall risk but also attend to the patient's underlying motivations for movement, such as thirst, discomfort, or the need for assistance, before a hazardous situation arises. Our contributions with this pilot study are three-fold: (1) a modular IoRT-based framework enabling distributed sensing, prediction, and multi-robot coordination; (2) a demonstration of low-resolution thermal sensing for accurate, privacy-preserving preemptive bed-exit detection; and (3) results from a user study and systematic error analysis that inform the design of situationally aware, multi-agent interactions in hospital settings. The findings highlight how interactive and connected robotic systems can move beyond passive monitoring to deliver timely, meaningful assistance, empowering safer, more responsive care environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22296) | **Categories:** cs.RO

---

### [15] [UnderwaterVLA: Dual-brain Vision-Language-Action architecture for Autonomous Underwater Navigation](https://arxiv.org/abs/2509.22441)
*Zhangyuan Wang, Yunpeng Zhu, Yuqi Yan, Xiaoyuan Tian, Xinhao Shao, Meixuan Li, Weikun Li, Guangsheng Su, Weicheng Cui, Dixia Fan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents UnderwaterVLA, a novel framework for autonomous underwater navigation that integrates multimodal foundation models with embodied intelligence systems. Underwater operations remain difficult due to hydrodynamic disturbances, limited communication bandwidth, and degraded sensing in turbid waters. To address these challenges, we introduce three innovations. First, a dual-brain architecture decouples high-level mission reasoning from low-level reactive control, enabling robust operation under communication and computational constraints. Second, we apply Vision-Language-Action(VLA) models to underwater robotics for the first time, incorporating structured chain-of-thought reasoning for interpretable decision-making. Third, a hydrodynamics-informed Model Predictive Control(MPC) scheme compensates for fluid effects in real time without costly task-specific training. Experimental results in field tests show that UnderwaterVLA reduces navigation errors in degraded visual conditions while maintaining higher task completion by 19% to 27% over baseline. By minimizing reliance on underwater-specific training data and improving adaptability across environments, UnderwaterVLA provides a scalable and cost-effective path toward the next generation of intelligent AUVs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22441) | **Categories:** cs.RO

---

### [16] [An Intention-driven Lane Change Framework Considering Heterogeneous Dynamic Cooperation in Mixed-traffic Environment](https://arxiv.org/abs/2509.22550)
*Xiaoyun Qiu, Haichao Liu, Yue Pan, Jun Ma, Xinhu Zheng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In mixed-traffic environments, where autonomous vehicles (AVs) interact with diverse human-driven vehicles (HVs), unpredictable intentions and heterogeneous behaviors make safe and efficient lane change maneuvers highly challenging. Existing methods often oversimplify these interactions by assuming uniform patterns. We propose an intention-driven lane change framework that integrates driving-style recognition, cooperation-aware decision-making, and coordinated motion planning. A deep learning classifier trained on the NGSIM dataset identifies human driving styles in real time. A cooperation score with intrinsic and interactive components estimates surrounding drivers' intentions and quantifies their willingness to cooperate with the ego vehicle. Decision-making combines behavior cloning with inverse reinforcement learning to determine whether a lane change should be initiated. For trajectory generation, model predictive control is integrated with IRL-based intention inference to produce collision-free and socially compliant maneuvers. Experiments show that the proposed model achieves 94.2\% accuracy and 94.3\% F1-score, outperforming rule-based and learning-based baselines by 4-15\% in lane change recognition. These results highlight the benefit of modeling inter-driver heterogeneity and demonstrate the potential of the framework to advance context-aware and human-like autonomous driving in complex traffic environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22550) | **Categories:** cs.RO

---

### [17] [MINT-RVAE: Multi-Cues Intention Prediction of Human-Robot Interaction using Human Pose and Emotion Information from RGB-only Camera Data](https://arxiv.org/abs/2509.22573)
*Farida Mohsen, Ali Safa*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Efficiently detecting human intent to interact with ubiquitous robots is crucial for effective human-robot interaction (HRI) and collaboration. Over the past decade, deep learning has gained traction in this field, with most existing approaches relying on multimodal inputs, such as RGB combined with depth (RGB-D), to classify time-sequence windows of sensory data as interactive or non-interactive. In contrast, we propose a novel RGB-only pipeline for predicting human interaction intent with frame-level precision, enabling faster robot responses and improved service quality. A key challenge in intent prediction is the class imbalance inherent in real-world HRI datasets, which can hinder the model's training and generalization. To address this, we introduce MINT-RVAE, a synthetic sequence generation method, along with new loss functions and training strategies that enhance generalization on out-of-sample data. Our approach achieves state-of-the-art performance (AUROC: 0.95) outperforming prior works (AUROC: 0.90-0.912), while requiring only RGB input and supporting precise frame onset prediction. Finally, to support future research, we openly release our new dataset with frame-level labeling of human interaction intent.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22573) | **Categories:** cs.RO, cs.CV

---

### [18] [Pixel Motion Diffusion is What We Need for Robot Control](https://arxiv.org/abs/2509.22652)
*E-Ro Nguyen, Yichi Zhang, Kanchana Ranasinghe, Xiang Li, Michael S. Ryoo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: https://nero1342.github.io/DAWN/

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.22652) | **Categories:** cs.RO, cs.CV

---
