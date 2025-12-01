# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-12-02

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (6)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (14)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Co-Evolving Agents: Learning from Failures as Hard Negatives](https://arxiv.org/abs/2511.22254)
*Yeonsung Jung, Trilok Padhi, Sina Shaham, Dipika Khullar, Joonhyun Jeong, Ninareh Mehrabi, Eunho Yang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid progress of large foundation models has accelerated the development of task-specialized agents across diverse domains. However, the effectiveness of agents remains tightly coupled with the quality of training data, while curating task-specific datasets remains costly and often infeasible in real-world scenarios. Recent work has explored self-improving agents that autonomously generate, refine, and re-train on their own trajectories. A prominent line of approaches further leverages preference optimization by pairing predicted trajectories with scarce ground-truth trajectories, enabling agents to learn directly from their own failures. While these methods outperform supervised fine-tuning, their heavy reliance on predicted trajectories under limited ground-truth supervision leaves them prone to overfitting. To address this, we propose a co-evolving agents framework in which a target agent improves jointly with an auxiliary failure agent. The failure agent learns through preference optimization over failure trajectories from both the target and itself, thereby generating hard negatives that are close to success yet remain failures. Incorporating these informative hard negatives into the target agent's optimization sharpens decision boundaries and enhances generalization. Our comprehensive analysis and experiments across benchmark datasets show that our method not only shows improved performance but also demonstrates that failures, instead of being used as-is, can be systematically transformed into structured and valuable learning signals in self-improving agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22254) | **Categories:** cs.AI

---

### [2] [MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](https://arxiv.org/abs/2511.23055)
*Ruoxuan Zhang, Qiyun Zheng, Zhiyu Zhou, Ziqi Liao, Siyu Wu, Jian-Yu Jiang-Lin, Bin Wen, Hongxia Xie, Jianlong Fu, Wen-Huang Cheng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Theory of Mind (ToM) refers to the ability to infer others' mental states, such as beliefs, desires, and intentions. Current vision-language embodied agents lack ToM-based decision-making, and existing benchmarks focus solely on human mental states while ignoring the agent's own perspective, hindering coherent decision and action generation. To address this, we propose MindPower, a Robot-Centric framework integrating Perception, Mental Reasoning, Decision Making and Action. Given multimodal inputs, MindPower first perceives the environment and human states, then performs ToM Reasoning to model both self and others, and finally generates decisions and actions guided by inferred mental states. Furthermore, we introduce Mind-Reward, a novel optimization objective that encourages VLMs to produce consistent ToM Reasoning and behavior. Our model outperforms GPT-4o by 12.77% in decision making and 12.49% in action generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23055) | **Categories:** cs.AI

---

### [3] [Evolutionary Discovery of Heuristic Policies for Traffic Signal Control](https://arxiv.org/abs/2511.23122)
*Ruibing Wang, Shuhan Guo, Zeen Li, Zhen Wang, Quanming Yao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traffic Signal Control (TSC) involves a challenging trade-off: classic heuristics are efficient but oversimplified, while Deep Reinforcement Learning (DRL) achieves high performance yet suffers from poor generalization and opaque policies. Online Large Language Models (LLMs) provide general reasoning but incur high latency and lack environment-specific optimization. To address these issues, we propose Temporal Policy Evolution for Traffic (\textbf{\method{}}), which uses LLMs as an evolution engine to derive specialized heuristic policies. The framework introduces two key modules: (1) Structured State Abstraction (SSA), converting high-dimensional traffic data into temporal-logical facts for reasoning; and (2) Credit Assignment Feedback (CAF), tracing flawed micro-decisions to poor macro-outcomes for targeted critique. Operating entirely at the prompt level without training, \method{} yields lightweight, robust policies optimized for specific traffic environments, outperforming both heuristics and online LLM actors.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23122) | **Categories:** cs.AI

---

### [4] [Thinking by Doing: Building Efficient World Model Reasoning in LLMs via Multi-turn Interaction](https://arxiv.org/abs/2511.23476)
*Bao Shu, Yan Cai, Jianjian Sun, Chunrui Han, En Yu, Liang Zhao, Jingcheng Hu, Yinmin Zhang, Haoran Lv, Yuang Peng, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Xiangyu Yue*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Developing robust world model reasoning is crucial for large language model (LLM) agents to plan and interact in complex environments. While multi-turn interaction offers a superior understanding of environmental dynamics via authentic feedback, current approaches often impose a rigid reasoning process, which constrains the model's active learning, ultimately hindering efficient world model reasoning. To address these issues, we explore world-model internalization through efficient interaction and active reasoning (WMAct), which liberates the model from structured reasoning, allowing the model to shape thinking directly through its doing, and achieves effective and efficient world model reasoning with two key mechanisms: (1) a reward rescaling mechanism adjusting outcome reward based on action efficacy to incentivize redundancy reduction and purposeful interaction; (2) an interaction frequency annealing strategy to progressively reduce the maximum allowed interaction turns, which compels the model to condense its learning and internalize environmental dynamics rather than over-relying on environmental cues. Our experiments on Sokoban, Maze, and Taxi show that WMAct yields effective world model reasoning capable of resolving tasks in a single turn that previously required multiple interactions and fosters strong transferability to complex environments, improving performance on a suite of reasoning benchmarks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23476) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven Image Encoding and Multiple Trajectory Prediction](https://arxiv.org/abs/2511.22181)
*Maitrayee Keskar, Mohan Trivedi, Ross Greer*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a method for trajectory planning for autonomous driving, learning image-based context embeddings that align with motion prediction frameworks and planning-based intention input. Within our method, a ViT encoder takes raw images and past kinematic state as input and is trained to produce context embeddings, drawing inspiration from those generated by the recent MTR (Motion Transformer) encoder, effectively substituting map-based features with learned visual representations. MTR provides a strong foundation for multimodal trajectory prediction by localizing agent intent and refining motion iteratively via motion query pairs; we name our approach MTR-VP (Motion Transformer for Vision-based Planning), and instead of the learnable intention queries used in the MTR decoder, we use cross attention on the intent and the context embeddings, which reflect a combination of information encoded from the driving scene and past vehicle states. We evaluate our methods on the Waymo End-to-End Driving Dataset, which requires predicting the agent's future 5-second trajectory in bird's-eye-view coordinates using prior camera images, agent pose history, and routing goals. We analyze our architecture using ablation studies, removing input images and multiple trajectory output. Our results suggest that transformer-based methods that are used to combine the visual features along with the kinetic features such as the past trajectory features are not effective at combining both modes to produce useful scene context embeddings, even when intention embeddings are augmented with foundation-model representations of scene context from CLIP and DINOv2, but that predicting a distribution over multiple futures instead of a single future trajectory boosts planning performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22181) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving](https://arxiv.org/abs/2511.19221)
*Jianhua Han, Meng Tian, Jiangtong Zhu, Fan He, Huixin Zhang, Sitong Guo, Dechang Zhu, Hao Tang, Pei Xu, Yuze Guo, Minzhe Niu, Haojie Zhu, Qichao Dong, Xuechao Yan, Siyuan Dong, Lu Hou, Qingqiu Huang, Xiaosong Jia, Hang Xu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision-language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.19221) | **Categories:** cs.CV, cs.RO

---

### [3] [SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model](https://arxiv.org/abs/2511.22039)
*Jiayuan Du, Yiming Zhao, Zhenglong Guo, Yong Pan, Wenbo Hou, Zhihui Hao, Kun Zhan, Qijun Chen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces a novel architecture for trajectory-conditioned forecasting of future 3D scene occupancy. In contrast to methods that rely on variational autoencoders (VAEs) to generate discrete occupancy tokens, which inherently limit representational capacity, our approach predicts multi-frame future occupancy in an end-to-end manner directly from raw image features. Inspired by the success of attention-based transformer architectures in foundational vision and language models such as GPT and VGGT, we employ a sparse occupancy representation that bypasses the intermediate bird's eye view (BEV) projection and its explicit geometric priors. This design allows the transformer to capture spatiotemporal dependencies more effectively. By avoiding both the finite-capacity constraint of discrete tokenization and the structural limitations of BEV representations, our method achieves state-of-the-art performance on the nuScenes benchmark for 1-3 second occupancy forecasting, outperforming existing approaches by a significant margin. Furthermore, it demonstrates robust scene dynamics understanding, consistently delivering high accuracy under arbitrary future trajectory conditioning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22039) | **Categories:** cs.CV

---

### [4] [TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video](https://arxiv.org/abs/2511.21946)
*Finlay G. C. Hudson, James A. D. Gardner, William A. P. Smith*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humans excel at constructing panoramic mental models of their surroundings, maintaining object permanence and inferring scene structure beyond visible regions. In contrast, current artificial vision systems struggle with persistent, panoramic understanding, often processing scenes egocentrically on a frame-by-frame basis. This limitation is pronounced in the Track Any Point (TAP) task, where existing methods fail to track 2D points outside the field of view. To address this, we introduce TAPVid-360, a novel task that requires predicting the 3D direction to queried scene points across a video sequence, even when far outside the narrow field of view of the observed video. This task fosters learning allocentric scene representations without needing dynamic 4D ground truth scene models for training. Instead, we exploit 360 videos as a source of supervision, resampling them into narrow field-of-view perspectives while computing ground truth directions by tracking points across the full panorama using a 2D pipeline. We introduce a new dataset and benchmark, TAPVid360-10k comprising 10k perspective videos with ground truth directional point tracking. Our baseline adapts CoTracker v3 to predict per-point rotations for direction updates, outperforming existing TAP and TAPVid 3D methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21946) | **Categories:** cs.CV

---

### [5] [Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](https://arxiv.org/abs/2511.21998)
*Apratim Bhattacharyya, Bicheng Xu, Sanjay Haresh, Reza Pourreza, Litian Liu, Sunny Panchal, Pulkit Madan, Leonid Sigal, Roland Memisevic*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-modal Large Language Models (LLM) have advanced conversational abilities but struggle with providing live, interactive step-by-step guidance, a key capability for future AI assistants. Effective guidance requires not only delivering instructions but also detecting their successful execution, as well as identifying and alerting users to mistakes, all of which has to happen in real-time. This requires models that are not turn-based, but that can react asynchronously to a video stream, as well as video data showing users performing tasks including mistakes and their corrections. To this end, we introduce Qualcomm Interactive Cooking, a new benchmark and dataset built upon CaptainCook4D, which contains user mistakes during task execution. Our dataset and benchmark features densely annotated, timed instructions and feedback messages, specifically including mistake alerts precisely timestamped to their visual occurrence in the video. We evaluate state-of-the-art multi-modal LLMs on the Qualcomm Interactive Cooking benchmark and introduce LiveMamba, a streaming multi-modal LLM designed for interactive instructional guidance. This work provides the first dedicated benchmark and a strong baseline for developing and evaluating on live, situated coaching.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21998) | **Categories:** cs.CV

---

### [6] [HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving](https://arxiv.org/abs/2511.22187)
*Qiang Li, Yingwenqi Jiang, Tuoxi Li, Duyu Chen, Xiang Feng, Yucheng Ao, Shangyue Liu, Xingchen Yu, Youcheng Cai, Yumeng Liu, Yuexin Ma, Xin Hu, Li Liu, Yu Zhang, Linkun Xu, Bingtao Gao, Xueyuan Wang, Shuchang Zhou, Xianming Liu, Ligang Liu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22187) | **Categories:** cs.CV, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [PULSE-ICU: A Pretrained Unified Long-Sequence Encoder for Multi-task Prediction in Intensive Care Units](https://arxiv.org/abs/2511.22199)
*Sejeong Jang, Joo Heung Yoon, Hyo Kyung Lee*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Intensive care unit (ICU) data are highly irregular, heterogeneous, and temporally fragmented, posing challenges for generalizable clinical prediction. We present PULSE-ICU, a self-supervised foundation model that learns event-level ICU representations from large-scale EHR sequences without resampling or manual feature engineering. A unified embedding module encodes event identity, continuous values, units, and temporal attributes, while a Longformer-based encoder enables efficient modeling of long trajectories. PULSE-ICU was fine-tuned across 18 prediction tasks, including mortality, intervention forecasting, and phenotype identification, achieving strong performance across task types. External validation on eICU, HiRID, and P12 showed substantial improvements with minimal fine-tuning, demonstrating robustness to domain shift and variable constraints. These findings suggest that foundation-style modeling can improve data efficiency and adaptability, providing a scalable framework for ICU decision support across diverse clinical environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22199) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Threat-Aware UAV Dodging of Human-Thrown Projectiles with an RGB-D Camera](https://arxiv.org/abs/2511.22847)
*Yuying Zhang, Na Fan, Haowen Zheng, Junning Liang, Zongliang Pan, Qifeng Chen, Ximin Lyu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Uncrewed aerial vehicles (UAVs) performing tasks such as transportation and aerial photography are vulnerable to intentional projectile attacks from humans. Dodging such a sudden and fast projectile poses a significant challenge for UAVs, requiring ultra-low latency responses and agile maneuvers. Drawing inspiration from baseball, in which pitchers' body movements are analyzed to predict the ball's trajectory, we propose a novel real-time dodging system that leverages an RGB-D camera. Our approach integrates human pose estimation with depth information to predict the attacker's motion trajectory and the subsequent projectile trajectory. Additionally, we introduce an uncertainty-aware dodging strategy to enable the UAV to dodge incoming projectiles efficiently. Our perception system achieves high prediction accuracy and outperforms the baseline in effective distance and latency. The dodging strategy addresses temporal and spatial uncertainties to ensure UAV safety. Extensive real-world experiments demonstrate the framework's reliable dodging capabilities against sudden attacks and its outstanding robustness across diverse scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22847) | **Categories:** cs.RO

---

### [2] [Seeing before Observable: Potential Risk Reasoning in Autonomous Driving via Vision Language Models](https://arxiv.org/abs/2511.22928)
*Jiaxin Liu, Xiangyu Yan, Liang Peng, Lei Yang, Lingjun Zhang, Yuechen Luo, Yueming Tao, Ashton Yu Xuan Tan, Mu Li, Lei Zhang, Ziqi Zhan, Sai Guo, Hong Wang, Jun Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring safety remains a key challenge for autonomous vehicles (AVs), especially in rare and complex scenarios. One critical but understudied aspect is the \textbf{potential risk} situations, where the risk is \textbf{not yet observable} but can be inferred from subtle precursors, such as anomalous behaviors or commonsense violations. Recognizing these precursors requires strong semantic understanding and reasoning capabilities, which are often absent in current AV systems due to the scarcity of such cases in existing driving or risk-centric datasets. Moreover, current autonomous driving accident datasets often lack annotations of the causal reasoning chains behind incidents, which are essential for identifying potential risks before they become observable. To address these gaps, we introduce PotentialRiskQA, a novel vision-language dataset designed for reasoning about potential risks prior to observation. Each sample is annotated with structured scene descriptions, semantic precursors, and inferred risk outcomes. Based on this dataset, we further propose PR-Reasoner, a vision-language-model-based framework tailored for onboard potential risk reasoning. Experimental results show that fine-tuning on PotentialRiskQA enables PR-Reasoner to significantly enhance its performance on the potential risk reasoning task compared to baseline VLMs. Together, our dataset and model provide a foundation for developing autonomous systems with improved foresight and proactive safety capabilities, moving toward more intelligent and resilient AVs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22928) | **Categories:** cs.RO

---

### [3] [Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary](https://arxiv.org/abs/2511.22963)
*Zhirui Liu, Kaiyang Ji, Ke Yang, Jingyi Yu, Ye Shi, Jingya Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Enabling humanoid robots to follow free-form language commands is critical for seamless human-robot interaction, collaborative task execution, and general-purpose embodied intelligence. While recent advances have improved low-level humanoid locomotion and robot manipulation, language-conditioned whole-body control remains a significant challenge. Existing methods are often limited to simple instructions and sacrifice either motion diversity or physical plausibility. To address this, we introduce Humanoid-LLA, a Large Language Action Model that maps expressive language commands to physically executable whole-body actions for humanoid robots. Our approach integrates three core components: a unified motion vocabulary that aligns human and humanoid motion primitives into a shared discrete space; a vocabulary-directed controller distilled from a privileged policy to ensure physical feasibility; and a physics-informed fine-tuning stage using reinforcement learning with dynamics-aware rewards to enhance robustness and stability. Extensive evaluations in simulation and on a real-world Unitree G1 humanoid show that Humanoid-LLA delivers strong language generalization while maintaining high physical fidelity, outperforming existing language-conditioned controllers in motion naturalness, stability, and execution success rate.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22963) | **Categories:** cs.RO, cs.AI

---

### [4] [LatBot: Distilling Universal Latent Actions for Vision-Language-Action Models](https://arxiv.org/abs/2511.23034)
*Zuolei Li, Xingyu Gao, Xiaofan Wang, Jianlong Fu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning transferable latent actions from large-scale object manipulation videos can significantly enhance generalization in downstream robotics tasks, as such representations are agnostic to different robot embodiments. Existing approaches primarily rely on visual reconstruction objectives while neglecting physical priors, leading to sub-optimal performance in learning universal representations. To address these challenges, we propose a Universal Latent Action Learning framework that takes task instructions and multiple frames as inputs, and optimizes both future frame reconstruction and action sequence prediction. Unlike prior works, incorporating action predictions (e.g., gripper or hand trajectories and orientations) allows the model to capture richer physical priors such as real-world distances and orientations, thereby enabling seamless transferability to downstream tasks. We further decompose the latent actions into learnable motion and scene tokens to distinguish the robot's active movements from environmental changes, thus filtering out irrelevant dynamics. By distilling the learned latent actions into the latest VLA models, we achieve strong performance across both simulated (SIMPLER and LIBERO) and real-world robot settings. Notably, with only 10 real-world trajectories per task collected on a Franka robot, our approach successfully completes all five challenging tasks, demonstrating strong few-shot transferability in robotic manipulation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23034) | **Categories:** cs.RO

---

### [5] [SwordRiding: A Unified Navigation Framework for Quadrotors in Unknown Complex Environments via Online Guiding Vector Fields](https://arxiv.org/abs/2511.22043)
*Xuchen Liu, Ruocheng Li, Bin Xin, Weijia Yao, Qigeng Duan, Jinqiang Cui, Ben M. Chen, Jie Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Although quadrotor navigation has achieved high performance in trajectory planning and control, real-time adaptability in unknown complex environments remains a core challenge. This difficulty mainly arises because most existing planning frameworks operate in an open-loop manner, making it hard to cope with environmental uncertainties such as wind disturbances or external perturbations. This paper presents a unified real-time navigation framework for quadrotors in unknown complex environments, based on the online construction of guiding vector fields (GVFs) from discrete reference path points. In the framework, onboard perception modules build a Euclidean Signed Distance Field (ESDF) representation of the environment, which enables obstacle awareness and path distance evaluation. The system first generates discrete, collision-free path points using a global planner, and then parameterizes them via uniform B-splines to produce a smooth and physically feasible reference trajectory. An adaptive GVF is then synthesized from the ESDF and the optimized B-spline trajectory. Unlike conventional approaches, the method adopts a closed-loop navigation paradigm, which significantly enhances robustness under external disturbances. Compared with conventional GVF methods, the proposed approach directly accommodates discretized paths and maintains compatibility with standard planning algorithms. Extensive simulations and real-world experiments demonstrate improved robustness against external disturbances and superior real-time performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22043) | **Categories:** cs.RO, eess.SY

---

### [6] [LLM-Based Generalizable Hierarchical Task Planning and Execution for Heterogeneous Robot Teams with Event-Driven Replanning](https://arxiv.org/abs/2511.22354)
*Suraj Borate, Bhavish Rai B, Vipul Pardeshi, Madhu Vadali*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces CoMuRoS (Collaborative Multi-Robot System), a generalizable hierarchical architecture for heterogeneous robot teams that unifies centralized deliberation with decentralized execution, and supports event-driven replanning. A Task Manager LLM interprets natural-language goals, classifies tasks, and allocates subtasks using static rules plus dynamic contexts (task, history, robot and task status, and events).Each robot runs a local LLM that composes executable Python code from primitive skills (ROS2 nodes, policies), while onboard perception (VLMs/image processing) continuously monitors events and classifies them into relevant or irrelevant to the task. Task failures or user intent changes trigger replanning, allowing robots to assist teammates, resume tasks, or request human help. Hardware studies demonstrate autonomous recovery from disruptive events, filtering of irrelevant distractions, and tightly coordinated transport with emergent human-robot cooperation (e.g., multirobot collaborative object recovery success rate: 9/10, coordinated transport: 8/8, human-assisted recovery: 5/5).Simulation studies show intention-aware replanning. A curated textual benchmark spanning 22 scenarios (3 tasks each, around 20 robots) evaluates task allocation, classification, IoU, executability, and correctness, with high average scores (e.g., correctness up to 0.91) across multiple LLMs, a separate replanning set (5 scenarios) achieves 1.0 correctness. Compared with prior LLM-based systems, CoMuRoS uniquely demonstrates runtime, event-driven replanning on physical robots, delivering robust, flexible multi-robot and human-robot collaboration.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22354) | **Categories:** cs.RO

---

### [7] [BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands](https://arxiv.org/abs/2511.22364)
*Seongwon Cho, Daechul Ahn, Donghyun Shin, Hyeonbeom Choi, San Kim, Jonghyun Choi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Open-vocabulary mobile manipulation (OVMM) requires robots to follow language instructions, navigate, and manipulate while updating their world representation under dynamic environmental changes. However, most prior approaches update their world representation only at discrete update points such as navigation targets, waypoints, or the end of an action step, leaving robots blind between updates and causing cascading failures: overlooked objects, late error detection, and delayed replanning. To address this limitation, we propose BINDER (Bridging INstant and DEliberative Reasoning), a dual process framework that decouples strategic planning from continuous environment monitoring. Specifically, BINDER integrates a Deliberative Response Module (DRM, a multimodal LLM for task planning) with an Instant Response Module (IRM, a VideoLLM for continuous monitoring). The two modules play complementary roles: the DRM performs strategic planning with structured 3D scene updates and guides what the IRM attends to, while the IRM analyzes video streams to update memory, correct ongoing actions, and trigger replanning when necessary. Through this bidirectional coordination, the modules address the trade off between maintaining awareness and avoiding costly updates, enabling robust adaptation under dynamic conditions. Evaluated in three real world environments with dynamic object placement, BINDER achieves substantially higher success and efficiency than SoTA baselines, demonstrating its effectiveness for real world deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22364) | **Categories:** cs.RO, cs.AI

---

### [8] [CAPE: Context-Aware Diffusion Policy Via Proximal Mode Expansion for Collision Avoidance](https://arxiv.org/abs/2511.22773)
*Rui Heng Yang, Xuan Zhao, Leo Maxime Brunswic, Montgomery Alban, Mateo Clemente, Tongtong Cao, Jun Jin, Amir Rasouli*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In robotics, diffusion models can capture multi-modal trajectories from demonstrations, making them a transformative approach in imitation learning. However, achieving optimal performance following this regiment requires a large-scale dataset, which is costly to obtain, especially for challenging tasks, such as collision avoidance. In those tasks, generalization at test time demands coverage of many obstacles types and their spatial configurations, which are impractical to acquire purely via data. To remedy this problem, we propose Context-Aware diffusion policy via Proximal mode Expansion (CAPE), a framework that expands trajectory distribution modes with context-aware prior and guidance at inference via a novel prior-seeded iterative guided refinement procedure. The framework generates an initial trajectory plan and executes a short prefix trajectory, and then the remaining trajectory segment is perturbed to an intermediate noise level, forming a trajectory prior. Such a prior is context-aware and preserves task intent. Repeating the process with context-aware guided denoising iteratively expands mode support to allow finding smoother, less collision-prone trajectories. For collision avoidance, CAPE expands trajectory distribution modes with collision-aware context, enabling the sampling of collision-free trajectories in previously unseen environments while maintaining goal consistency. We evaluate CAPE on diverse manipulation tasks in cluttered unseen simulated and real-world settings and show up to 26% and 80% higher success rates respectively compared to SOTA methods, demonstrating better generalization to unseen environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22773) | **Categories:** cs.RO, cs.AI

---

### [9] [Safe Autonomous Lane Changing: Planning with Dynamic Risk Fields and Time-Varying Convex Space Generation](https://arxiv.org/abs/2511.22829)
*Zhen Tian, Zhihao Lin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a novel trajectory planning pipeline for complex driving scenarios like autonomous lane changing, by integrating risk-aware planning with guaranteed collision avoidance into a unified optimization framework. We first construct a dynamic risk fields (DRF) that captures both the static and dynamic collision risks from surrounding vehicles. Then, we develop a rigorous strategy for generating time-varying convex feasible spaces that ensure kinematic feasibility and safety requirements. The trajectory planning problem is formulated as a finite-horizon optimal control problem and solved using a constrained iterative Linear Quadratic Regulator (iLQR) algorithm that jointly optimizes trajectory smoothness, control effort, and risk exposure while maintaining strict feasibility. Extensive simulations demonstrate that our method outperforms traditional approaches in terms of safety and efficiency, achieving collision-free trajectories with shorter lane-changing distances (28.59 m) and times (2.84 s) while maintaining smooth and comfortable acceleration patterns. In dense roundabout environments the planner further demonstrates robust adaptability, producing larger safety margins, lower jerk, and superior curvature smoothness compared with APF, MPC, and RRT based baselines. These results confirm that the integrated DRF with convex feasible space and constrained iLQR solver provides a balanced solution for safe, efficient, and comfortable trajectory generation in dynamic and interactive traffic scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22829) | **Categories:** cs.RO, cs.HC

---

### [10] [SUPER-AD: Semantic Uncertainty-aware Planning for End-to-End Robust Autonomous Driving](https://arxiv.org/abs/2511.22865)
*Wonjeong Ryu, Seungjun Yu, Seokha Moon, Hojun Choi, Junsung Park, Jinkyu Kim, Hyunjung Shim*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-End (E2E) planning has become a powerful paradigm for autonomous driving, yet current systems remain fundamentally uncertainty-blind. They assume perception outputs are fully reliable, even in ambiguous or poorly observed scenes, leaving the planner without an explicit measure of uncertainty. To address this limitation, we propose a camera-only E2E framework that estimates aleatoric uncertainty directly in BEV space and incorporates it into planning. Our method produces a dense, uncertainty-aware drivability map that captures both semantic structure and geometric layout at pixel-level resolution. To further promote safe and rule-compliant behavior, we introduce a lane-following regularization that encodes lane structure and traffic norms. This prior stabilizes trajectory planning under normal conditions while preserving the flexibility needed for maneuvers such as overtaking or lane changes. Together, these components enable robust and interpretable trajectory planning, even under challenging uncertainty conditions. Evaluated on the NAVSIM benchmark, our method achieves state-of-the-art performance, delivering substantial gains on both the challenging NAVHARD and NAVSAFE subsets. These results demonstrate that our principled aleatoric uncertainty modeling combined with driving priors significantly advances the safety and reliability of camera-only E2E autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.22865) | **Categories:** cs.RO, cs.CV

---

### [11] [Automated Generation of MDPs Using Logic Programming and LLMs for Robotic Applications](https://arxiv.org/abs/2511.23143)
*Enrico Saccon, Davide De Martini, Matteo Saveriano, Edoardo Lamon, Luigi Palopoli, Marco Roveri*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a novel framework that integrates Large Language Models (LLMs) with automated planning and formal verification to streamline the creation and use of Markov Decision Processes (MDP). Our system leverages LLMs to extract structured knowledge in the form of a Prolog knowledge base from natural language (NL) descriptions. It then automatically constructs an MDP through reachability analysis, and synthesises optimal policies using the Storm model checker. The resulting policy is exported as a state-action table for execution. We validate the framework in three human-robot interaction scenarios, demonstrating its ability to produce executable policies with minimal manual effort. This work highlights the potential of combining language models with formal methods to enable more accessible and scalable probabilistic planning in robotics.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23143) | **Categories:** cs.RO, cs.AI

---

### [12] [Fault-Tolerant MARL for CAVs under Observation Perturbations for Highway On-Ramp Merging](https://arxiv.org/abs/2511.23193)
*Yuchen Shi, Huaxin Pei, Yi Zhang, Danya Yao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-Agent Reinforcement Learning (MARL) holds significant promise for enabling cooperative driving among Connected and Automated Vehicles (CAVs). However, its practical application is hindered by a critical limitation, i.e., insufficient fault tolerance against observational faults. Such faults, which appear as perturbations in the vehicles' perceived data, can substantially compromise the performance of MARL-based driving systems. Addressing this problem presents two primary challenges. One is to generate adversarial perturbations that effectively stress the policy during training, and the other is to equip vehicles with the capability to mitigate the impact of corrupted observations. To overcome the challenges, we propose a fault-tolerant MARL method for cooperative on-ramp vehicles incorporating two key agents. First, an adversarial fault injection agent is co-trained to generate perturbations that actively challenge and harden the vehicle policies. Second, we design a novel fault-tolerant vehicle agent equipped with a self-diagnosis capability, which leverages the inherent spatio-temporal correlations in vehicle state sequences to detect faults and reconstruct credible observations, thereby shielding the policy from misleading inputs. Experiments in a simulated highway merging scenario demonstrate that our method significantly outperforms baseline MARL approaches, achieving near-fault-free levels of safety and efficiency under various observation fault patterns.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23193) | **Categories:** cs.RO, cs.LG, eess.SY

---

### [13] [SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot](https://arxiv.org/abs/2511.23300)
*Yara Mahmoud, Jeffrin Sam, Nguyen Khang, Marcelino Fernando, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Muhammad Haris Khan, Artem Lykov, Dzmitry Tsetserukou*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe and trustworthy Human Robot Interaction (HRI) requires robots not only to complete tasks but also to regulate impedance and speed according to scene context and human proximity. We present SafeHumanoid, an egocentric vision pipeline that links Vision Language Models (VLMs) with Retrieval-Augmented Generation (RAG) to schedule impedance and velocity parameters for a humanoid robot. Egocentric frames are processed by a structured VLM prompt, embedded and matched against a curated database of validated scenarios, and mapped to joint-level impedance commands via inverse kinematics. We evaluate the system on tabletop manipulation tasks with and without human presence, including wiping, object handovers, and liquid pouring. The results show that the pipeline adapts stiffness, damping, and speed profiles in a context-aware manner, maintaining task success while improving safety. Although current inference latency (up to 1.4 s) limits responsiveness in highly dynamic settings, SafeHumanoid demonstrates that semantic grounding of impedance control is a viable path toward safer, standard-compliant humanoid collaboration.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.23300) | **Categories:** cs.RO

---

### [14] [$\mathcal{E}_0$: Enhancing Generalization and Fine-Grained Control in VLA Models via Continuized Discrete Diffusion](https://arxiv.org/abs/2511.21542)
*Zhihao Zhan, Jiaying Zhou, Likui Zhang, Qinhan Lv, Hao Liu, Jusheng Zhang, Weizheng Li, Ziliang Chen, Tianshui Chen, Keze Wang, Liang Lin, Guangrun Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. Yet existing VLA models still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We introduce E0, a continuized discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. Compared with continuous diffusion policies, E0 offers two key advantages: (1) discrete action tokens align naturally with the symbolic structure of pretrained VLM/VLA backbones, enabling stronger semantic conditioning; and 2. discrete diffusion matches the true quantized nature of real-world robot control-whose hardware constraints (e.g., encoder resolution, control frequency, actuation latency) inherently discretize continuous signals-and therefore benefits from a Bayes-optimal denoiser that models the correct discrete action distribution, leading to stronger generalization. Compared with discrete autoregressive and mask-based discrete diffusion models, E0 supports a significantly larger and finer-grained action vocabulary and avoids the distributional mismatch introduced by masking-based corruptions-yielding more accurate fine-grained action control. We further introduce a spherical viewpoint perturbation augmentation method to improve robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, and ManiSkill show that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average. Real-world evaluation on a Franka arm confirms that E0 delivers precise, robust, and transferable manipulation, establishing discrete diffusion as a promising direction for generalizable VLA policy learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.21542) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---
