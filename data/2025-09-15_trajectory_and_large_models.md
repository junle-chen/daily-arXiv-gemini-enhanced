# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-15

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器人学 (Robotics) (4)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting](https://arxiv.org/abs/2509.09210)
*Xing Gao, Zherui Huang, Weiyao Lin, Xiao Sun*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09210) | **Categories:** cs.AI, cs.RO

---

### [2] [Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning](https://arxiv.org/abs/2509.09284)
*Bingning Huang, Tu Nguyen, Matthieu Zimmer*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09284) | **Categories:** cs.AI, cs.CL, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [MGTraj: Multi-Granularity Goal-Guided Human Trajectory Prediction with Recursive Refinement Network](https://arxiv.org/abs/2509.09200)
*Ge Sun, Jun Ma*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate human trajectory prediction is crucial for robotics navigation and autonomous driving. Recent research has demonstrated that incorporating goal guidance significantly enhances prediction accuracy by reducing uncertainty and leveraging prior knowledge. Most goal-guided approaches decouple the prediction task into two stages: goal prediction and subsequent trajectory completion based on the predicted goal, which operate at extreme granularities: coarse-grained goal prediction forecasts the overall intention, while fine-grained trajectory completion needs to generate the positions for all future timesteps. The potential utility of intermediate temporal granularity remains largely unexplored, which motivates multi-granularity trajectory modeling. While prior work has shown that multi-granularity representations capture diverse scales of human dynamics and motion patterns, effectively integrating this concept into goal-guided frameworks remains challenging. In this paper, we propose MGTraj, a novel Multi-Granularity goal-guided model for human Trajectory prediction. MGTraj recursively encodes trajectory proposals from coarse to fine granularity levels. At each level, a transformer-based recursive refinement network (RRN) captures features and predicts progressive refinements. Features across different granularities are integrated using a weight-sharing strategy, and velocity prediction is employed as an auxiliary task to further enhance performance. Comprehensive experimental results in EHT/UCY and Stanford Drone Dataset indicate that MGTraj outperforms baseline methods and achieves state-of-the-art performance among goal-guided methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09200) | **Categories:** cs.CV

---

### [2] [Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles](https://arxiv.org/abs/2509.09349)
*Ian Nell, Shane Gilroy*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09349) | **Categories:** cs.CV, cs.AI, cs.ET, cs.RO, eess.IV

---

### [3] [Improving Human Motion Plausibility with Body Momentum](https://arxiv.org/abs/2509.09496)
*Ha Linh Nguyen, Tze Ho Elden Tse, Angela Yao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Many studies decompose human motion into local motion in a frame attached to the root joint and global motion of the root joint in the world frame, treating them separately. However, these two components are not independent. Global movement arises from interactions with the environment, which are, in turn, driven by changes in the body configuration. Motion models often fail to precisely capture this physical coupling between local and global dynamics, while deriving global trajectories from joint torques and external forces is computationally expensive and complex. To address these challenges, we propose using whole-body linear and angular momentum as a constraint to link local motion with global movement. Since momentum reflects the aggregate effect of joint-level dynamics on the body's movement through space, it provides a physically grounded way to relate local joint behavior to global displacement. Building on this insight, we introduce a new loss term that enforces consistency between the generated momentum profiles and those observed in ground-truth data. Incorporating our loss reduces foot sliding and jitter, improves balance, and preserves the accuracy of the recovered motion. Code and data are available at the project page https://hlinhn.github.io/momentum_bmvc.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09496) | **Categories:** cs.CV

---

### [4] [DualTrack: Sensorless 3D Ultrasound needs Local and Global Context](https://arxiv.org/abs/2509.09530)
*Paul F. R. Wilson, Matteo Ronchetti, Rüdiger Göbl, Viktoria Markova, Sebastian Rosenzweig, Raphael Prevost, Parvin Mousavi, Oliver Zettinig*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Three-dimensional ultrasound (US) offers many clinical advantages over conventional 2D imaging, yet its widespread adoption is limited by the cost and complexity of traditional 3D systems. Sensorless 3D US, which uses deep learning to estimate a 3D probe trajectory from a sequence of 2D US images, is a promising alternative. Local features, such as speckle patterns, can help predict frame-to-frame motion, while global features, such as coarse shapes and anatomical structures, can situate the scan relative to anatomy and help predict its general shape. In prior approaches, global features are either ignored or tightly coupled with local feature extraction, restricting the ability to robustly model these two complementary aspects. We propose DualTrack, a novel dual-encoder architecture that leverages decoupled local and global encoders specialized for their respective scales of feature extraction. The local encoder uses dense spatiotemporal convolutions to capture fine-grained features, while the global encoder utilizes an image backbone (e.g., a 2D CNN or foundation model) and temporal attention layers to embed high-level anatomical features and long-range dependencies. A lightweight fusion module then combines these features to estimate the trajectory. Experimental results on a large public benchmark show that DualTrack achieves state-of-the-art accuracy and globally consistent 3D reconstructions, outperforming previous methods and yielding an average reconstruction error below 5 mm.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09530) | **Categories:** cs.CV

---


## 机器人学 (Robotics) [cs.RO]
### [1] [KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning](https://arxiv.org/abs/2509.09074)
*Alice Kate Li, Thales C Silva, Victoria Edwards, Vijay Kumar, M. Ani Hsieh*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals -- a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09074) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [2] [Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments](https://arxiv.org/abs/2509.09206)
*Farhad Nawaz, Faizan M. Tariq, Sangjae Bae, David Isele, Avinash Singh, Nadia Figueroa, Nikolai Matni, Jovin D'sa*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurately reasoning about future parking spot availability and integrated planning is critical for enabling safe and efficient autonomous valet parking in dynamic, uncertain environments. Unlike existing methods that rely solely on instantaneous observations or static assumptions, we present an approach that predicts future parking spot occupancy by explicitly distinguishing between initially vacant and occupied spots, and by leveraging the predicted motion of dynamic agents. We introduce a probabilistic spot occupancy estimator that incorporates partial and noisy observations within a limited Field-of-View (FoV) model and accounts for the evolving uncertainty of unobserved regions. Coupled with this, we design a strategy planner that adaptively balances goal-directed parking maneuvers with exploratory navigation based on information gain, and intelligently incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency, safety margins, and trajectory smoothness compared to existing approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09206) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [3] [OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](https://arxiv.org/abs/2509.09332)
*Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible.To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09332) | **Categories:** cs.RO, cs.AI, cs.CL, cs.CV

---

### [4] [VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model](https://arxiv.org/abs/2509.09372)
*Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, Donglin Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: https://vla-adapter.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09372) | **Categories:** cs.RO

---
