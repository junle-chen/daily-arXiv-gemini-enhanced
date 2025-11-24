# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-25

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (9)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Hybrid Differential Reward: Combining Temporal Difference and Action Gradients for Efficient Multi-Agent Reinforcement Learning in Cooperative Driving](https://arxiv.org/abs/2511.16916)
*Ye Han, Lijun Zhang, Dejian Meng, Zhuang Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In multi-vehicle cooperative driving tasks involving high-frequency continuous control, traditional state-based reward functions suffer from the issue of vanishing reward differences. This phenomenon results in a low signal-to-noise ratio (SNR) for policy gradients, significantly hindering algorithm convergence and performance improvement. To address this challenge, this paper proposes a novel Hybrid Differential Reward (HDR) mechanism. We first theoretically elucidate how the temporal quasi-steady nature of traffic states and the physical proximity of actions lead to the failure of traditional reward signals. Building on this analysis, the HDR framework innovatively integrates two complementary components: (1) a Temporal Difference Reward (TRD) based on a global potential function, which utilizes the evolutionary trend of potential energy to ensure optimal policy invariance and consistency with long-term objectives; and (2) an Action Gradient Reward (ARG), which directly measures the marginal utility of actions to provide a local guidance signal with a high SNR. Furthermore, we formulate the cooperative driving problem as a Multi-Agent Partially Observable Markov Game (POMDPG) with a time-varying agent set and provide a complete instantiation scheme for HDR within this framework. Extensive experiments conducted using both online planning (MCTS) and Multi-Agent Reinforcement Learning (QMIX, MAPPO, MADDPG) algorithms demonstrate that the HDR mechanism significantly improves convergence speed and policy stability. The results confirm that HDR guides agents to learn high-quality cooperative policies that effectively balance traffic efficiency and safety.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16916) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding](https://arxiv.org/abs/2511.17053)
*Teng Fu, Mengyang Zhao, Ke Niu, Kaixin Peng, Bin Li*

Main category: cs.CV

TL;DR: 本文提出了一种新的统一行人跟踪框架OmniPT，该框架能够交互式地跟踪、基于参考进行跟踪并生成被跟踪对象的语义理解。


<details>
  <summary>Details</summary>
Motivation: 现有LVLM在实例级任务（如视觉定位和目标检测）中表现不如专家模型；同时，行人跟踪领域涌现了结合对象跟踪和自然语言的新课题，强调模型对被跟踪对象的高级语义理解。本文旨在解决这些问题。

Method: 本文提出一个包含RL-Mid Training-SFT-RL的训练阶段。首先，执行一个简单的RL阶段，使模型能够输出固定的、可监督的边界框格式。其次，使用大量的行人相关数据集进行中间训练阶段。最后，在几个行人跟踪数据集上进行监督微调，然后进行另一个RL阶段，以提高模型的跟踪性能并增强其遵循指令的能力。

Result: 在跟踪基准上进行的实验结果表明，所提出的方法可以比以前的方法表现更好。

Conclusion: 本文提出的OmniPT框架在行人跟踪任务上表现出色，能够有效地结合视觉信息和语义理解。

Abstract: 大型视觉语言模型（LVLM）在图像级别的任务（如视觉问答和图像描述）中表现出色。然而，在许多实例级别的任务中，例如视觉定位和目标检测，与之前的专家模型相比，LVLM仍然存在性能差距。同时，虽然行人跟踪是一个经典的任务，但在结合对象跟踪和自然语言方面出现了一些新的课题，例如Referring MOT、Cross-view Referring MOT和Semantic MOT。这些任务强调模型应该在高级语义层面理解被跟踪的对象，而这正是LVLM所擅长的。在本文中，我们提出了一个新的统一的行人跟踪框架，即OmniPT，它可以交互式地跟踪、基于参考进行跟踪，并生成被跟踪对象的语义理解。我们解决了两个问题：如何将跟踪任务建模成基础模型可以执行的任务，以及如何使模型输出格式化的答案。为此，我们实现了一个包含RL-Mid Training-SFT-RL的训练阶段。基于LVLM的预训练权重，我们首先执行一个简单的RL阶段，使模型能够输出固定且可监督的边界框格式。随后，我们使用大量的行人相关数据集进行中间训练阶段。最后，我们在几个行人跟踪数据集上进行监督微调，然后进行另一个RL阶段，以提高模型的跟踪性能并增强其遵循指令的能力。我们在跟踪基准上进行了实验，实验结果表明，所提出的方法可以比以前的方法表现更好。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17053) | **Categories:** cs.CV, cs.AI

---

### [2] [DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](https://arxiv.org/abs/2511.17150)
*Liuhan Yin, Runkun Ju, Guodong Guo, Erkang Cheng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unlike discriminative approaches in autonomous driving that predict a fixed set of candidate trajectories of the ego vehicle, generative methods, such as diffusion models, learn the underlying distribution of future motion, enabling more flexible trajectory prediction. However, since these methods typically rely on denoising human-crafted trajectory anchors or random noise, there remains significant room for improvement. In this paper, we propose DiffRefiner, a novel two-stage trajectory prediction framework. The first stage uses a transformer-based Proposal Decoder to generate coarse trajectory predictions by regressing from sensor inputs using predefined trajectory anchors. The second stage applies a Diffusion Refiner that iteratively denoises and refines these initial predictions. In this way, we enhance the performance of diffusion-based planning by incorporating a discriminative trajectory proposal module, which provides strong guidance for the generative refinement process. Furthermore, we design a fine-grained denoising decoder to enhance scene compliance, enabling more accurate trajectory prediction through enhanced alignment with the surrounding environment. Experimental results demonstrate that DiffRefiner achieves state-of-the-art performance, attaining 87.4 EPDMS on NAVSIM v2, and 87.1 DS along with 71.4 SR on Bench2Drive, thereby setting new records on both public benchmarks. The effectiveness of each component is validated via ablation studies as well.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17150) | **Categories:** cs.CV

---

### [3] [RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis](https://arxiv.org/abs/2511.17045)
*Linfeng Dong, Yuchen Yang, Hao Wu, Wei Wang, Yuenan HouZhihang Zhong, Xiao Sun*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at https://github.com/OrcustD/RacketVision

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17045) | **Categories:** cs.CV, cs.AI, cs.MM

---

### [4] [WorldGen: From Text to Traversable and Interactive 3D Worlds](https://arxiv.org/abs/2511.16825)
*Dilin Wang, Hyunyoung Jung, Tom Monnier, Kihyuk Sohn, Chuhang Zou, Xiaoyu Xiang, Yu-Ying Yeh, Di Liu, Zixuan Huang, Thu Nguyen-Phuoc, Yuchen Fan, Sergiu Oprea, Ziyan Wang, Roman Shapovalov, Nikolaos Sarafianos, Thibault Groueix, Antoine Toisoul, Prithviraj Dhar, Xiao Chu, Minghao Chen, Geon Yeong Park, Mahima Gupta, Yassir Azziz, Rakesh Ranjan, Andrea Vedaldi*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce WorldGen, a system that enables the automatic creation of large-scale, interactive 3D worlds directly from text prompts. Our approach transforms natural language descriptions into traversable, fully textured environments that can be immediately explored or edited within standard game engines. By combining LLM-driven scene layout reasoning, procedural generation, diffusion-based 3D generation, and object-aware scene decomposition, WorldGen bridges the gap between creative intent and functional virtual spaces, allowing creators to design coherent, navigable worlds without manual modeling or specialized 3D expertise. The system is fully modular and supports fine-grained control over layout, scale, and style, producing worlds that are geometrically consistent, visually rich, and efficient to render in real time. This work represents a step towards accessible, generative world-building at scale, advancing the frontier of 3D generative AI for applications in gaming, simulation, and immersive social environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16825) | **Categories:** cs.CV, cs.AI

---

### [5] [BOP-ASK: Object-Interaction Reasoning for Vision-Language Models](https://arxiv.org/abs/2511.16857)
*Vineet Bhat, Sungsu Kim, Valts Blukis, Greg Heinrich, Prashanth Krishnamurthy, Ramesh Karri, Stan Birchfield, Farshad Khorrami, Jonathan Tremblay*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision Language Models (VLMs) have achieved impressive performance on spatial reasoning benchmarks, yet these evaluations mask critical weaknesses in understanding object interactions. Current benchmarks test high level relationships ('left of,' 'behind', etc.) but ignore fine-grained spatial understanding needed for real world applications: precise 3D localization, physical compatibility between objects, object affordances and multi step spatial planning. In this work, we present BOP-ASK, a novel large scale dataset for object interaction reasoning for both training and benchmarking. Our data generation pipeline leverages 6D object poses from the Benchmark for Object Pose Estimation (BOP) datasets from which we derive fine grained annotations such as grasp poses, referred object poses, path planning trajectories, relative spatial and depth relationships, and object-to-object relationships. BOP-ASK comprises over 150k images and 33M question answer pairs spanning six tasks (four novel), providing a rich resource for training and evaluating VLMs. We evaluate proprietary and open sourced VLMs, and conduct human evaluations on BOP-ASK-core, a contributed test benchmark. We also release BOP-ASK-lab, an out-of-distribution benchmark with images not sourced from BOP, enabling testing of generalization. Our experiments demonstrate that models trained on BOP-ASK outperform baselines and exhibit emergent capabilities such as precise object and grasp pose estimation, trajectory planning, and fine-grained object-centric spatial reasoning in cluttered environments. We will publicly release our datasets and dataset generation pipeline.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16857) | **Categories:** cs.CV, cs.RO

---

### [6] [R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](https://arxiv.org/abs/2511.16901)
*Lu Zhu, Tiantian Geng, Yangye Chen, Teng Wang, Ping Lu, Feng Zheng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16901) | **Categories:** cs.CV

---

### [7] [OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios](https://arxiv.org/abs/2511.16937)
*Hong Gao, Jingyu Wu, Xiangkai Xu, Kangni Xie, Yunchen Zhang, Bin Zhong, Xurui Gao, Min-Ling Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Spatio-Temporal Video Grounding (STVG) aims to localize target objects in videos based on natural language descriptions. Despite recent advances in Multimodal Large Language Models, a significant gap remains between current models and real-world demands involving diverse objects and complex queries. We attribute this to limited benchmark scope, causing models to exhibit category bias, oversimplified reasoning, and poor linguistic robustness. To address these limitations, we introduce OmniGround, a comprehensive benchmark with 3,475 videos spanning 81 categories and complex real-world queries. We propose the Forward-Backward-Refinement annotation pipeline that combines multi-directional tracking with intelligent error correction for high-quality labels. We further introduce DeepSTG, a systematic evaluation framework quantifying dataset quality across four complementary dimensions beyond superficial statistics. Evaluations reveal performance average drop of 10.4% on complex real-world scenes, particularly with small/occluded objects and intricate spatial relations. Motivated by these, we propose PG-TAF, a training-free two-stage framework decomposing STVG into high-level temporal grounding and fine-grained spatio-temporal propagation. Experiments demonstrate PG-TAF achieves 25.6% and 35.6% improvements in m\_tIoU and m\_vIoU on OmniGround with consistent gains across four benchmarks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16937) | **Categories:** cs.CV, cs.AI

---

### [8] [FingerCap: Fine-grained Finger-level Hand Motion Captioning](https://arxiv.org/abs/2511.16951)
*Xin Shen, Rui Zhu, Lei Shen, Xinyu Wang, Kaihao Zhang, Tianqing Zhu, Shuchen Wu, Chenxi Miao, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang, Xin Yu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Understanding fine-grained human hand motion is fundamental to visual perception, embodied intelligence, and multimodal communication. In this work, we propose Fine-grained Finger-level Hand Motion Captioning (FingerCap), which aims to generate textual descriptions that capture detailed finger-level semantics of hand actions. To support this task, we curate FingerCap-40K, a large-scale corpus of 40K paired hand-motion videos and captions spanning two complementary sources: concise instruction-style finger motions and diverse, naturalistic hand-object interactions. To enable effective evaluation, we employ HandJudge, a LLM-based rubric that measures finger-level correctness and motion completeness. Temporal sparsity remains a fundamental bottleneck for current Video-MLLMs, since sparse RGB sampling is insufficient to capture the subtle, high-frequency dynamics underlying fine finger motions. As a simple and compute-friendly remedy, we introduce FiGOP (Finger Group-of-Pictures), which pairs each RGB keyframe with subsequent hand keypoints until the next keyframe. A lightweight temporal encoder converts the keypoints into motion embeddings and integrates them with RGB features. FiGOP adapts the classic GOP concept to finger motion, recovering fine temporal cues without increasing RGB density. Experiments on FingerCap-40K show that strong open- and closed-source Video-MLLMs still struggle with finger-level reasoning, while our FiGOP-augmented model yield consistent gains under HandJudge and human studies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16951) | **Categories:** cs.CV

---

### [9] [Sparse Reasoning is Enough: Biological-Inspired Framework for Video Anomaly Detection with Large Pre-trained Models](https://arxiv.org/abs/2511.17094)
*He Huang, Zixuan Hu, Dongxiao Li, Yao Xiao, Ling-Yu Duan*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video anomaly detection (VAD) plays a vital role in real-world applications such as security surveillance, autonomous driving, and industrial monitoring. Recent advances in large pre-trained models have opened new opportunities for training-free VAD by leveraging rich prior knowledge and general reasoning capabilities. However, existing studies typically rely on dense frame-level inference, incurring high computational costs and latency. This raises a fundamental question: Is dense reasoning truly necessary when using powerful pre-trained models in VAD systems? To answer this, we propose ReCoVAD, a novel framework inspired by the dual reflex and conscious pathways of the human nervous system, enabling selective frame processing to reduce redundant computation. ReCoVAD consists of two core pathways: (i) a Reflex pathway that uses a lightweight CLIP-based module to fuse visual features with prototype prompts and produce decision vectors, which query a dynamic memory of past frames and anomaly scores for fast response; and (ii) a Conscious pathway that employs a medium-scale vision-language model to generate textual event descriptions and refined anomaly scores for novel frames. It continuously updates the memory and prototype prompts, while an integrated large language model periodically reviews accumulated descriptions to identify unseen anomalies, correct errors, and refine prototypes. Extensive experiments show that ReCoVAD achieves state-of-the-art training-free performance while processing only 28.55\% and 16.04\% of the frames used by previous methods on the UCF-Crime and XD-Violence datasets, demonstrating that sparse reasoning is sufficient for effective large-model-based VAD.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17094) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [CroTad: A Contrastive Reinforcement Learning Framework for Online Trajectory Anomaly Detection](https://arxiv.org/abs/2511.16929)
*Rui Xue, Dan He, Fengmei Jin, Chen Zhang, Xiaofang Zhou*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Detecting trajectory anomalies is a vital task in modern Intelligent Transportation Systems (ITS), enabling the identification of unsafe, inefficient, or irregular travel behaviours. While deep learning has emerged as the dominant approach, several key challenges remain unresolved. First, sub-trajectory anomaly detection, capable of pinpointing the precise segments where anomalies occur, remains underexplored compared to whole-trajectory analysis. Second, many existing methods depend on carefully tuned thresholds, limiting their adaptability in real-world applications. Moreover, the irregular sampling of trajectory data and the presence of noise in training sets further degrade model performance, making it difficult to learn reliable representations of normal routes. To address these challenges, we propose a contrastive reinforcement learning framework for online trajectory anomaly detection, CroTad. Our method is threshold-free and robust to noisy, irregularly sampled data. By incorporating contrastive learning, CroTad learns to extract diverse normal travel patterns for different itineraries and effectively distinguish anomalous behaviours at both sub-trajectory and point levels. The detection module leverages deep reinforcement learning to perform online, real-time anomaly scoring, enabling timely and fine-grained identification of abnormal segments. Extensive experiments on two real-world datasets demonstrate the effectiveness and robustness of our framework across various evaluation scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16929) | **Categories:** cs.LG, cs.DB

---

### [2] [Multi-Agent Pointer Transformer: Seq-to-Seq Reinforcement Learning for Multi-Vehicle Dynamic Pickup-Delivery Problems](https://arxiv.org/abs/2511.17435)
*Zengyu Zou, Jingyuan Wang, Yixuan Huang, Junjie Wu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper addresses the cooperative Multi-Vehicle Dynamic Pickup and Delivery Problem with Stochastic Requests (MVDPDPSR) and proposes an end-to-end centralized decision-making framework based on sequence-to-sequence, named Multi-Agent Pointer Transformer (MAPT). MVDPDPSR is an extension of the vehicle routing problem and a spatio-temporal system optimization problem, widely applied in scenarios such as on-demand delivery. Classical operations research methods face bottlenecks in computational complexity and time efficiency when handling large-scale dynamic problems. Although existing reinforcement learning methods have achieved some progress, they still encounter several challenges: 1) Independent decoding across multiple vehicles fails to model joint action distributions; 2) The feature extraction network struggles to capture inter-entity relationships; 3) The joint action space is exponentially large. To address these issues, we designed the MAPT framework, which employs a Transformer Encoder to extract entity representations, combines a Transformer Decoder with a Pointer Network to generate joint action sequences in an AutoRegressive manner, and introduces a Relation-Aware Attention module to capture inter-entity relationships. Additionally, we guide the model's decision-making using informative priors to facilitate effective exploration. Experiments on 8 datasets demonstrate that MAPT significantly outperforms existing baseline methods in terms of performance and exhibits substantial computational time advantages compared to classical operations research methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17435) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [MfNeuPAN: Proactive End-to-End Navigation in Dynamic Environments via Direct Multi-Frame Point Constraints](https://arxiv.org/abs/2511.17013)
*Yiwen Ying, Hanjing Ye, Senzi Luo, Luyao Liu, Yu Zhan, Li He, Hong Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Obstacle avoidance in complex and dynamic environments is a critical challenge for real-time robot navigation. Model-based and learning-based methods often fail in highly dynamic scenarios because traditional methods assume a static environment and cannot adapt to real-time changes, while learning-based methods rely on single-frame observations for motion constraint estimation, limiting their adaptability. To overcome these limitations, this paper proposes a novel framework that leverages multi-frame point constraints, including current and future frames predicted by a dedicated module, to enable proactive end-to-end navigation. By incorporating a prediction module that forecasts the future path of moving obstacles based on multi-frame observations, our method allows the robot to proactively anticipate and avoid potential dangers. This proactive planning capability significantly enhances navigation robustness and efficiency in unknown dynamic environments. Simulations and real-world experiments validate the effectiveness of our approach.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17013) | **Categories:** cs.RO

---

### [2] [TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making](https://arxiv.org/abs/2511.17225)
*Shanshan Li, Da Huang, Yu He, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In daily life, people often move through spaces to find objects that meet their needs, posing a key challenge in embodied AI. Traditional Demand-Driven Navigation (DDN) handles one need at a time but does not reflect the complexity of real-world tasks involving multiple needs and personal choices. To bridge this gap, we introduce Task-Preferenced Multi-Demand-Driven Navigation (TP-MDDN), a new benchmark for long-horizon navigation involving multiple sub-demands with explicit task preferences. To solve TP-MDDN, we propose AWMSystem, an autonomous decision-making system composed of three key modules: BreakLLM (instruction decomposition), LocateLLM (goal selection), and StatusMLLM (task monitoring). For spatial memory, we design MASMap, which combines 3D point cloud accumulation with 2D semantic mapping for accurate and efficient environmental understanding. Our Dual-Tempo action generation framework integrates zero-shot planning with policy-based fine control, and is further supported by an Adaptive Error Corrector that handles failure cases in real time. Experiments demonstrate that our approach outperforms state-of-the-art baselines in both perception accuracy and navigation robustness.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17225) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [3] [IndustryNav: Exploring Spatial Reasoning of Embodied Agents in Dynamic Industrial Navigation](https://arxiv.org/abs/2511.17384)
*Yifan Li, Lichi Li, Anh Dao, Xinyu Zhou, Yicheng Qiao, Zheda Mai, Daeun Lee, Zichen Chen, Zhen Tan, Mohit Bansal, Yu Kong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While Visual Large Language Models (VLLMs) show great promise as embodied agents, they continue to face substantial challenges in spatial reasoning. Existing embodied benchmarks largely focus on passive, static household environments and evaluate only isolated capabilities, failing to capture holistic performance in dynamic, real-world complexity. To fill this gap, we present IndustryNav, the first dynamic industrial navigation benchmark for active spatial reasoning. IndustryNav leverages 12 manually created, high-fidelity Unity warehouse scenarios featuring dynamic objects and human movement. Our evaluation employs a PointGoal navigation pipeline that effectively combines egocentric vision with global odometry to assess holistic local-global planning. Crucially, we introduce the "collision rate" and "warning rate" metrics to measure safety-oriented behaviors and distance estimation. A comprehensive study of nine state-of-the-art VLLMs (including models such as GPT-5-mini, Claude-4.5, and Gemini-2.5) reveals that closed-source models maintain a consistent advantage; however, all agents exhibit notable deficiencies in robust path planning, collision avoidance and active exploration. This highlights a critical need for embodied research to move beyond passive perception and toward tasks that demand stable planning, active exploration, and safe behavior in dynamic, real-world environment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17384) | **Categories:** cs.RO, cs.CV

---

### [4] [MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments](https://arxiv.org/abs/2511.17496)
*Zhiyu Huang, Zewei Zhou, Tianhui Cai, Yun Zhang, Jiaqi Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modeling realistic and interactive multi-agent behavior is critical to autonomous driving and traffic simulation. However, existing diffusion and autoregressive approaches are limited by iterative sampling, sequential decoding, or task-specific designs, which hinder efficiency and reuse. We propose Masked Denoising Generation (MDG), a unified generative framework that reformulates multi-agent behavior modeling as the reconstruction of independently noised spatiotemporal tensors. Instead of relying on diffusion time steps or discrete tokenization, MDG applies continuous, per-agent and per-timestep noise masks that enable localized denoising and controllable trajectory generation in a single or few forward passes. This mask-driven formulation generalizes across open-loop prediction, closed-loop simulation, motion planning, and conditional generation within one model. Trained on large-scale real-world driving datasets, MDG achieves competitive closed-loop performance on the Waymo Sim Agents and nuPlan Planning benchmarks, while providing efficient, consistent, and controllable open-loop multi-agent trajectory generation. These results position MDG as a simple yet versatile paradigm for multi-agent behavior modeling.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17496) | **Categories:** cs.RO, cs.MA

---

### [5] [MobileOcc: A Human-Aware Semantic Occupancy Dataset for Mobile Robots](https://arxiv.org/abs/2511.16949)
*Junseo Kim, Guido Dumont, Xinyu Gao, Gang Chen, Holger Caesar, Javier Alonso-Mora*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Dense 3D semantic occupancy perception is critical for mobile robots operating in pedestrian-rich environments, yet it remains underexplored compared to its application in autonomous driving. To address this gap, we present MobileOcc, a semantic occupancy dataset for mobile robots operating in crowded human environments. Our dataset is built using an annotation pipeline that incorporates static object occupancy annotations and a novel mesh optimization framework explicitly designed for human occupancy modeling. It reconstructs deformable human geometry from 2D images and subsequently refines and optimizes it using associated LiDAR point data. Using MobileOcc, we establish benchmarks for two tasks, i) Occupancy prediction and ii) Pedestrian velocity prediction, using different methods including monocular, stereo, and panoptic occupancy, with metrics and baseline implementations for reproducible comparison. Beyond occupancy prediction, we further assess our annotation method on 3D human pose estimation datasets. Results demonstrate that our method exhibits robust performance across different datasets.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16949) | **Categories:** cs.RO, cs.CV

---

### [6] [Efficient Robot Design with Multi-Objective Black-Box Optimization and Large Language Models](https://arxiv.org/abs/2511.17178)
*Kento Kawaharazuka, Yoshiki Obinata, Naoaki Kanazawa, Haoyu Jia, Kei Okada*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Various methods for robot design optimization have been developed so far. These methods are diverse, ranging from numerical optimization to black-box optimization. While numerical optimization is fast, it is not suitable for cases involving complex structures or discrete values, leading to frequent use of black-box optimization instead. However, black-box optimization suffers from low sampling efficiency and takes considerable sampling iterations to obtain good solutions. In this study, we propose a method to enhance the efficiency of robot body design based on black-box optimization by utilizing large language models (LLMs). In parallel with the sampling process based on black-box optimization, sampling is performed using LLMs, which are provided with problem settings and extensive feedback. We demonstrate that this method enables more efficient exploration of design solutions and discuss its characteristics and limitations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17178) | **Categories:** cs.RO

---

### [7] [Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM](https://arxiv.org/abs/2511.17335)
*Chiori Hori, Yoshiki Masuyama, Siddarth Jain, Radu Corcodel, Devesh Jha, Diego Romeres, Jonathan Le Roux*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17335) | **Categories:** cs.RO, cs.CL, cs.CV, cs.SD, eess.AS

---

### [8] [HALO: High-Altitude Language-Conditioned Monocular Aerial Exploration and Navigation](https://arxiv.org/abs/2511.17497)
*Yuezhan Tao, Dexter Ong, Fernando Cladera, Jason Hughes, Camillo J. Taylor, Pratik Chaudhari, Vijay Kumar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We demonstrate real-time high-altitude aerial metric-semantic mapping and exploration using a monocular camera paired with a global positioning system (GPS) and an inertial measurement unit (IMU). Our system, named HALO, addresses two key challenges: (i) real-time dense 3D reconstruction using vision at large distances, and (ii) mapping and exploration of large-scale outdoor environments with accurate scene geometry and semantics. We demonstrate that HALO can plan informative paths that exploit this information to complete missions with multiple tasks specified in natural language. In simulation-based evaluation across large-scale environments of size up to 78,000 sq. m., HALO consistently completes tasks with less exploration time and achieves up to 68% higher competitive ratio in terms of the distance traveled compared to the state-of-the-art semantic exploration baseline. We use real-world experiments on a custom quadrotor platform to demonstrate that (i) all modules can run onboard the robot, and that (ii) in diverse environments HALO can support effective autonomous execution of missions covering up to 24,600 sq. m. area at an altitude of 40 m. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/halo/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17497) | **Categories:** cs.RO

---

### [9] [RynnVLA-002: A Unified Vision-Language-Action and World Model](https://arxiv.org/abs/2511.17502)
*Jun Cen, Siteng Huang, Yuqian Yuan, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Kehan Li, Hao Luo, Fan Wang, Xin Li, Deli Zhao, Hao Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce RynnVLA-002, a unified Vision-Language-Action (VLA) and world model. The world model leverages action and visual inputs to predict future image states, learning the underlying physics of the environment to refine action generation. Conversely, the VLA model produces subsequent actions from image observations, enhancing visual understanding and supporting the world model's image generation. The unified framework of RynnVLA-002 enables joint learning of environmental dynamics and action planning. Our experiments show that RynnVLA-002 surpasses individual VLA and world models, demonstrating their mutual enhancement. We evaluate RynnVLA-002 in both simulation and real-world robot tasks. RynnVLA-002 achieves 97.4% success rate on the LIBERO simulation benchmark without pretraining, while in real-world LeRobot experiments, its integrated world model boosts the overall success rate by 50%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.17502) | **Categories:** cs.RO

---
