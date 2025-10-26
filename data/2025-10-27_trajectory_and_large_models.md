# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-27

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (5)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (6)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Classical Feature Embeddings Help in BERT-Based Human Mobility Prediction](https://arxiv.org/abs/2510.20275)
*Yunzhi Liu, Haokai Tan, Rushi Kanjaria, Lihuan Li, Flora D. Salim*

Main category: cs.AI

TL;DR: 本文提出了一种名为STaBERT的语义时序感知BERT模型，通过融合POI和时间信息，显著提升了人类移动预测的准确性。


<details>
  <summary>Details</summary>
Motivation: 现有模型未能充分利用兴趣点(POI)提供的丰富语义信息，仅将时间信息作为辅助输入，限制了人类移动预测的准确性。

Method: 提出STaBERT模型，通过在每个位置整合POI和时间信息，构建统一的、语义丰富的移动表示。

Result: 实验结果表明，STaBERT显著提高了预测准确性：单城市预测的GEO-BLEU得分从0.34提高到0.75；多城市预测的GEO-BLEU得分从0.34提高到0.56。

Conclusion: STaBERT模型能够有效融合POI和时间信息，从而显著提升人类移动预测的准确性。

Abstract: 人类移动预测对于灾害救援、城市规划和公共卫生至关重要。然而，现有的模型要么只对位置序列进行建模，要么仅将时间信息作为辅助输入，未能充分利用兴趣点（POI）提供的丰富语义信息。为了解决这个问题，我们利用导出的时间描述符和POI嵌入来丰富基于BERT的移动模型，从而更好地捕捉人类移动的底层语义。我们提出了STaBERT（语义-时间感知BERT），它在每个位置整合POI和时间信息，以构建统一的、语义丰富的移动表示。实验结果表明，STaBERT显著提高了预测准确性：对于单城市预测，GEO-BLEU得分从0.34提高到0.75；对于多城市预测，GEO-BLEU得分从0.34提高到0.56。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20275) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Evaluating Video Models as Simulators of Multi-Person Pedestrian Trajectories](https://arxiv.org/abs/2510.20182)
*Aaron Appelle, Jerome P. Lynch*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large-scale video generation models have demonstrated high visual realism in diverse contexts, spurring interest in their potential as general-purpose world simulators. Existing benchmarks focus on individual subjects rather than scenes with multiple interacting people. However, the plausibility of multi-agent dynamics in generated videos remains unverified. We propose a rigorous evaluation protocol to benchmark text-to-video (T2V) and image-to-video (I2V) models as implicit simulators of pedestrian dynamics. For I2V, we leverage start frames from established datasets to enable comparison with a ground truth video dataset. For T2V, we develop a prompt suite to explore diverse pedestrian densities and interactions. A key component is a method to reconstruct 2D bird's-eye view trajectories from pixel-space without known camera parameters. Our analysis reveals that leading models have learned surprisingly effective priors for plausible multi-agent behavior. However, failure modes like merging and disappearing people highlight areas for future improvement.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20182) | **Categories:** cs.CV

---

### [2] [FutrTrack: A Camera-LiDAR Fusion Transformer for 3D Multiple Object Tracking](https://arxiv.org/abs/2510.19981)
*Martha Teiko Teye, Ori Maoz, Matthias Rottmann*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose FutrTrack, a modular camera-LiDAR multi-object tracking framework that builds on existing 3D detectors by introducing a transformer-based smoother and a fusion-driven tracker. Inspired by query-based tracking frameworks, FutrTrack employs a multimodal two-stage transformer refinement and tracking pipeline. Our fusion tracker integrates bounding boxes with multimodal bird's-eye-view (BEV) fusion features from multiple cameras and LiDAR without the need for an explicit motion model. The tracker assigns and propagates identities across frames, leveraging both geometric and semantic cues for robust re-identification under occlusion and viewpoint changes. Prior to tracking, we refine sequences of bounding boxes with a temporal smoother over a moving window to refine trajectories, reduce jitter, and improve spatial consistency. Evaluated on nuScenes and KITTI, FutrTrack demonstrates that query-based transformer tracking methods benefit significantly from multimodal sensor features compared with previous single-sensor approaches. With an aMOTA of 74.7 on the nuScenes test set, FutrTrack achieves strong performance on 3D MOT benchmarks, reducing identity switches while maintaining competitive accuracy. Our approach provides an efficient framework for improving transformer-based trackers to compete with other neural-network-based methods even with limited data and without pretraining.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19981) | **Categories:** cs.CV

---

### [3] [Physics-Guided Fusion for Robust 3D Tracking of Fast Moving Small Objects](https://arxiv.org/abs/2510.20126)
*Prithvi Raj Singh, Raju Gottumukkala, Anthony S. Maida, Alan B. Barhorst, Vijaya Gopu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While computer vision has advanced considerably for general object detection and tracking, the specific problem of fast-moving tiny objects remains underexplored. This paper addresses the significant challenge of detecting and tracking rapidly moving small objects using an RGB-D camera. Our novel system combines deep learning-based detection with physics-based tracking to overcome the limitations of existing approaches. Our contributions include: (1) a comprehensive system design for object detection and tracking of fast-moving small objects in 3D space, (2) an innovative physics-based tracking algorithm that integrates kinematics motion equations to handle outliers and missed detections, and (3) an outlier detection and correction module that significantly improves tracking performance in challenging scenarios such as occlusions and rapid direction changes. We evaluated our proposed system on a custom racquetball dataset. Our evaluation shows our system surpassing kalman filter based trackers with up to 70\% less Average Displacement Error. Our system has significant applications for improving robot perception on autonomous platforms and demonstrates the effectiveness of combining physics-based models with deep learning approaches for real-time 3D detection and tracking of challenging small objects.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20126) | **Categories:** cs.CV

---

### [4] [EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence](https://arxiv.org/abs/2510.20578)
*Ding Zou, Feifan Wang, Mengyu Ge, Siyuan Fan, Zongbing Zhang, Wei Chen, Lingfeng Wang, Zhongyou Hu, Wenrui Yan, Zhengwei Gao, Hao Wang, Weizhao Jin, Yu Zhang, Hainan Zhao, Mingliang Zhang, Xianxian Xi, Yaru Zhang, Wenyuan Li, Zhengguang Gao, Yurui Zhu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The realization of Artificial General Intelligence (AGI) necessitates Embodied AI agents capable of robust spatial perception, effective task planning, and adaptive execution in physical environments. However, current large language models (LLMs) and multimodal LLMs (MLLMs) for embodied tasks suffer from key limitations, including a significant gap between model design and agent requirements, an unavoidable trade-off between real-time latency and performance, and the use of unauthentic, offline evaluation metrics. To address these challenges, we propose EmbodiedBrain, a novel vision-language foundation model available in both 7B and 32B parameter sizes. Our framework features an agent-aligned data structure and employs a powerful training methodology that integrates large-scale Supervised Fine-Tuning (SFT) with Step-Augumented Group Relative Policy Optimization (Step-GRPO), which boosts long-horizon task success by integrating preceding steps as Guided Precursors. Furthermore, we incorporate a comprehensive reward system, including a Generative Reward Model (GRM) accelerated at the infrastructure level, to improve training efficiency. For enable thorough validation, we establish a three-part evaluation system encompassing General, Planning, and End-to-End Simulation Benchmarks, highlighted by the proposal and open-sourcing of a novel, challenging simulation environment. Experimental results demonstrate that EmbodiedBrain achieves superior performance across all metrics, establishing a new state-of-the-art for embodied foundation models. Towards paving the way for the next generation of generalist embodied agents, we open-source all of our data, model weight, and evaluating methods, which are available at https://zterobot.github.io/EmbodiedBrain.github.io.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20578) | **Categories:** cs.CV, cs.RO

---

### [5] [Blur2seq: Blind Deblurring and Camera Trajectory Estimation from a Single Camera Motion-blurred Image](https://arxiv.org/abs/2510.20539)
*Guillermo Carbajal, Andrés Almansa, Pablo Musé*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Motion blur caused by camera shake, particularly under large or rotational movements, remains a major challenge in image restoration. We propose a deep learning framework that jointly estimates the latent sharp image and the underlying camera motion trajectory from a single blurry image. Our method leverages the Projective Motion Blur Model (PMBM), implemented efficiently using a differentiable blur creation module compatible with modern networks. A neural network predicts a full 3D rotation trajectory, which guides a model-based restoration network trained end-to-end. This modular architecture provides interpretability by revealing the camera motion that produced the blur. Moreover, this trajectory enables the reconstruction of the sequence of sharp images that generated the observed blurry image. To further refine results, we optimize the trajectory post-inference via a reblur loss, improving consistency between the blurry input and the restored output. Extensive experiments show that our method achieves state-of-the-art performance on both synthetic and real datasets, particularly in cases with severe or spatially variant blur, where end-to-end deblurring networks struggle.   Code and trained models are available at https://github.com/GuillermoCarbajal/Blur2Seq/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20539) | **Categories:** cs.CV, cs.LG

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [From Optimization to Prediction: Transformer-Based Path-Flow Estimation to the Traffic Assignment Problem](https://arxiv.org/abs/2510.19889)
*Mostafa Ameli, Van Anh Le, Sulthana Shams, Alexander Skabardonis*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The traffic assignment problem is essential for traffic flow analysis, traditionally solved using mathematical programs under the Equilibrium principle. These methods become computationally prohibitive for large-scale networks due to non-linear growth in complexity with the number of OD pairs. This study introduces a novel data-driven approach using deep neural networks, specifically leveraging the Transformer architecture, to predict equilibrium path flows directly. By focusing on path-level traffic distribution, the proposed model captures intricate correlations between OD pairs, offering a more detailed and flexible analysis compared to traditional link-level approaches. The Transformer-based model drastically reduces computation time, while adapting to changes in demand and network structure without the need for recalculation. Numerical experiments are conducted on the Manhattan-like synthetic network, the Sioux Falls network, and the Eastern-Massachusetts network. The results demonstrate that the proposed model is orders of magnitude faster than conventional optimization. It efficiently estimates path-level traffic flows in multi-class networks, reducing computational costs and improving prediction accuracy by capturing detailed trip and flow information. The model also adapts flexibly to varying demand and network conditions, supporting traffic management and enabling rapid `what-if' analyses for enhanced transportation planning and policy-making.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19889) | **Categories:** cs.LG, cs.AI, math.OC

---

### [2] [SALT: Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph](https://arxiv.org/abs/2510.20022)
*Jiazheng Li, Yawei Wang, David Yan, Yijun Tian, Zhichao Xu, Huan Song, Panpan Xu, Lin Lee Cheong*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have demonstrated remarkable capabilities, enabling language agents to excel at single-turn tasks. However, their application to complex, multi-step, and long-horizon tasks remains challenging. While reinforcement learning (RL) offers a promising avenue for addressing these challenges, mainstream approaches typically rely solely on sparse, outcome-based rewards, a limitation that becomes especially problematic for group-based RL algorithms lacking critic models, such as Group Relative Policy Optimization (GRPO). In such methods, uniformly rewarding or penalizing all actions within a trajectory can lead to training instability and suboptimal policies, because beneficial and detrimental actions are often entangled across multi-step interactions. To address this challenge, we propose SALT, a novel and lightweight framework that provides a finer-grained advantage assignment, derived solely from outcome rewards. We achieve this by constructing a graph from trajectories of the same prompt, which allows us to quantify the quality of each step and assign advantages accordingly. Crucially, SALT is designed as a plug-and-play module that seamlessly integrates with existing group-based RL algorithms, requiring no modifications to the rollout procedure and introducing negligible computational overhead. Extensive experiments on the WebShop, ALFWorld, and AppWorld benchmarks with various model sizes demonstrate that SALT consistently improves performance. We also conduct a thorough analysis to validate the design choices behind SALT and offer actionable insights.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20022) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Simultaneous learning of state-to-state minimum-time planning and control](https://arxiv.org/abs/2510.20008)
*Swati Dantu, Robert Pěnička, Martin Saska*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper tackles the challenge of learning a generalizable minimum-time flight policy for UAVs, capable of navigating between arbitrary start and goal states while balancing agile flight and stable hovering. Traditional approaches, particularly in autonomous drone racing, achieve impressive speeds and agility but are constrained to predefined track layouts, limiting real-world applicability. To address this, we propose a reinforcement learning-based framework that simultaneously learns state-to-state minimum-time planning and control and generalizes to arbitrary state-to-state flights. Our approach leverages Point Mass Model (PMM) trajectories as proxy rewards to approximate the true optimal flight objective and employs curriculum learning to scale the training process efficiently and to achieve generalization. We validate our method through simulation experiments, comparing it against Nonlinear Model Predictive Control (NMPC) tracking PMM-generated trajectories and conducting ablation studies to assess the impact of curriculum learning. Finally, real-world experiments confirm the robustness of our learned policy in outdoor environments, demonstrating its ability to generalize and operate on a small ARM-based single-board computer.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20008) | **Categories:** cs.RO

---

### [2] [PathFormer: A Transformer with 3D Grid Constraints for Digital Twin Robot-Arm Trajectory Generation](https://arxiv.org/abs/2510.20161)
*Ahmed Alanazi, Duy Ho, Yugyung Lee*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic arms require precise, task-aware trajectory planning, yet sequence models that ignore motion structure often yield invalid or inefficient executions. We present a Path-based Transformer that encodes robot motion with a 3-grid (where/what/when) representation and constraint-masked decoding, enforcing lattice-adjacent moves and workspace bounds while reasoning over task graphs and action order. Trained on 53,755 trajectories (80% train / 20% validation), the model aligns closely with ground truth -- 89.44% stepwise accuracy, 93.32% precision, 89.44% recall, and 90.40% F1 -- with 99.99% of paths legal by construction. Compiled to motor primitives on an xArm Lite 6 with a depth-camera digital twin, it attains up to 97.5% reach and 92.5% pick success in controlled tests, and 86.7% end-to-end success across 60 language-specified tasks in cluttered scenes, absorbing slips and occlusions via local re-grounding without global re-planning. These results show that path-structured representations enable Transformers to generate accurate, reliable, and interpretable robot trajectories, bridging graph-based planning and sequence-based learning and providing a practical foundation for general-purpose manipulation and sim-to-real transfer.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20161) | **Categories:** cs.RO, 68T07, 68T40, I.2.9; I.2.10; I.2.11

---

### [3] [MemER: Scaling Up Memory for Robot Control via Experience Retrieval](https://arxiv.org/abs/2510.20328)
*Ajay Sridhar, Jennifer Pan, Satvik Sharma, Chelsea Finn*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humans routinely rely on memory to perform tasks, yet most robot policies lack this capability; our goal is to endow robot policies with the same ability. Naively conditioning on long observation histories is computationally expensive and brittle under covariate shift, while indiscriminate subsampling of history leads to irrelevant or redundant information. We propose a hierarchical policy framework, where the high-level policy is trained to select and track previous relevant keyframes from its experience. The high-level policy uses selected keyframes and the most recent frames when generating text instructions for a low-level policy to execute. This design is compatible with existing vision-language-action (VLA) models and enables the system to efficiently reason over long-horizon dependencies. In our experiments, we finetune Qwen2.5-VL-7B-Instruct and $\pi_{0.5}$ as the high-level and low-level policies respectively, using demonstrations supplemented with minimal language annotations. Our approach, MemER, outperforms prior methods on three real-world long-horizon robotic manipulation tasks that require minutes of memory. Videos and code can be found at https://jen-pan.github.io/memer/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20328) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [4] [Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking](https://arxiv.org/abs/2510.20335)
*Zixuan Wu, Hengyuan Zhang, Ting-Hsuan Chen, Yuliang Guo, David Paz, Xinyu Huang, Liu Ren*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Parking is a critical pillar of driving safety. While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge. Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts. We train our pipeline in CARLA at regular setting and transfer it to more adversarial settings in a zero-shot fashion. Our model consistently achieves a parking success rate above 90% across all tested out-of-distribution (OOD) scenarios, with ablation studies confirming that both the network architecture and algorithmic design significantly enhance cross-domain performance over existing baselines. Furthermore, testing in a 3D Gaussian splatting (3DGS) environment reconstructed from a real-world parking lot demonstrates promising sim-to-real transfer.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20335) | **Categories:** cs.RO, cs.CV

---

### [5] [Robot Path and Trajectory Planning Considering a Spatially Fixed TCP](https://arxiv.org/abs/2510.20473)
*Bernhard Rameder, Hubert Gattringer, Andreas Mueller, Ronald Naderer*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a method for planning a trajectory in workspace coordinates using a spatially fixed tool center point (TCP), while taking into account the processing path on a part. This approach is beneficial if it is easier to move the part rather than moving the tool. Whether a mathematical description that defines the shape to be processed or single points from a design program are used, the robot path is finally represented using B-splines. The use of splines enables the path to be continuous with a desired degree, which finally leads to a smooth robot trajectory. While calculating the robot trajectory through prescribed orientation, additionally a given velocity at the TCP has to be considered. The procedure was validated on a real system using an industrial robot moving an arbitrary defined part.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20473) | **Categories:** cs.RO

---

### [6] [VAMOS: A Hierarchical Vision-Language-Action Model for Capability-Modulated and Steerable Navigation](https://arxiv.org/abs/2510.20818)
*Mateo Guaman Castro, Sidharth Rajagopal, Daniel Gorbatov, Matt Schmittle, Rohan Baijal, Octi Zhang, Rosario Scalise, Sidharth Talia, Emma Romig, Celso de Melo, Byron Boots, Abhishek Gupta*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A fundamental challenge in robot navigation lies in learning policies that generalize across diverse environments while conforming to the unique physical constraints and capabilities of a specific embodiment (e.g., quadrupeds can walk up stairs, but rovers cannot). We propose VAMOS, a hierarchical VLA that decouples semantic planning from embodiment grounding: a generalist planner learns from diverse, open-world data, while a specialist affordance model learns the robot's physical constraints and capabilities in safe, low-cost simulation. We enabled this separation by carefully designing an interface that lets a high-level planner propose candidate paths directly in image space that the affordance model then evaluates and re-ranks. Our real-world experiments show that VAMOS achieves higher success rates in both indoor and complex outdoor navigation than state-of-the-art model-based and end-to-end learning methods. We also show that our hierarchical design enables cross-embodied navigation across legged and wheeled robots and is easily steerable using natural language. Real-world ablations confirm that the specialist model is key to embodiment grounding, enabling a single high-level planner to be deployed across physically distinct wheeled and legged robots. Finally, this model significantly enhances single-robot reliability, achieving 3X higher success rates by rejecting physically infeasible plans. Website: https://vamos-vla.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20818) | **Categories:** cs.RO, cs.AI, cs.LG

---


## eess.SY [eess.SY]
### [1] [Behavior-Aware Online Prediction of Obstacle Occupancy using Zonotopes](https://arxiv.org/abs/2510.20437)
*Alvaro Carrizosa-Rendon, Jian Zhou, Erik Frisk, Vicenc Puig, Fatiha Nejjari*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Predicting the motion of surrounding vehicles is key to safe autonomous driving, especially in unstructured environments without prior information. This paper proposes a novel online method to accurately predict the occupancy sets of surrounding vehicles based solely on motion observations. The approach is divided into two stages: first, an Extended Kalman Filter and a Linear Programming (LP) problem are used to estimate a compact zonotopic set of control actions; then, a reachability analysis propagates this set to predict future occupancy. The effectiveness of the method has been validated through simulations in an urban environment, showing accurate and compact predictions without relying on prior assumptions or prior training data.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.20437) | **Categories:** eess.SY, cs.RO, cs.SY

---
