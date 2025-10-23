# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-24

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (7)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [cs.MA (1)](#cs-ma)
- [机器人学 (Robotics) (3)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Rectifying Shortcut Behaviors in Preference-based Reward Learning](https://arxiv.org/abs/2510.19050)
*Wenqian Ye, Guangtao Zheng, Aidong Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In reinforcement learning from human feedback, preference-based reward models play a central role in aligning large language models to human-aligned behavior. However, recent studies show that these models are prone to reward hacking and often fail to generalize well due to over-optimization. They achieve high reward scores by exploiting shortcuts, that is, exploiting spurious features (e.g., response verbosity, agreeable tone, or sycophancy) that correlate with human preference labels in the training data rather than genuinely reflecting the intended objectives. In this paper, instead of probing these issues one at a time, we take a broader view of the reward hacking problem as shortcut behaviors and introduce a principled yet flexible approach to mitigate shortcut behaviors in preference-based reward learning. Inspired by the invariant theory in the kernel perspective, we propose Preference-based Reward Invariance for Shortcut Mitigation (PRISM), which learns group-invariant kernels with feature maps in a closed-form learning objective. Experimental results in several benchmarks show that our method consistently improves the accuracy of the reward model on diverse out-of-distribution tasks and reduces the dependency on shortcuts in downstream policy models, establishing a robust framework for preference-based alignment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19050) | **Categories:** cs.AI, cs.LG

---

### [2] [WebGraphEval: Multi-Turn Trajectory Evaluation for Web Agents using Graph Representation](https://arxiv.org/abs/2510.19205)
*Yaoyao Qian, Yuanli Wang, Jinda Zhang, Yun Zong, Meixu Chen, Hanhan Zhou, Jindan Huang, Yifan Zeng, Xinyu Hu, Chan Hee Song, Danqing Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Current evaluation of web agents largely reduces to binary success metrics or conformity to a single reference trajectory, ignoring the structural diversity present in benchmark datasets. We present WebGraphEval, a framework that abstracts trajectories from multiple agents into a unified, weighted action graph. This representation is directly compatible with benchmarks such as WebArena, leveraging leaderboard runs and newly collected trajectories without modifying environments. The framework canonically encodes actions, merges recurring behaviors, and applies structural analyses including reward propagation and success-weighted edge statistics. Evaluations across thousands of trajectories from six web agents show that the graph abstraction captures cross-model regularities, highlights redundancy and inefficiency, and identifies critical decision points overlooked by outcome-based metrics. By framing web interaction as graph-structured data, WebGraphEval establishes a general methodology for multi-path, cross-agent, and efficiency-aware evaluation of web agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19205) | **Categories:** cs.AI

---

### [3] [AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing](https://arxiv.org/abs/2510.19661)
*Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19661) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Robust Driving QA through Metadata-Grounded Context and Task-Specific Prompts](https://arxiv.org/abs/2510.19001)
*Seungjun Yu, Junsung Park, Youngsun Lim, Hyunjung Shim*

Main category: cs.CV

TL;DR: 该论文提出了一种两阶段视觉语言问答系统，用于自动驾驶，能够回答高层次的感知、预测和规划问题。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在解决自动驾驶场景中，如何利用视觉语言模型进行高层次的理解和推理问答的问题。

Method: 该方法包括两个阶段：第一阶段利用多模态大语言模型（Qwen2.5-VL-32B）结合多相机输入、历史信息和思维链提示进行问答；第二阶段通过增加nuScenes场景元数据和特定类别的问题指令来增强提示。

Result: 实验结果表明，该方法在驾驶问答基准测试中显著优于基线模型，例如，在第一阶段使用5帧历史和10-shot提示可以达到65.1%的总体准确率，使用自洽性可以将准确率提高到66.85%，第二阶段可以达到67.37%的总体准确率。此外，该系统在严重的视觉损坏下仍能保持96%的准确率。

Conclusion: 研究结果表明，精心设计的提示和上下文 grounding 可以极大地提高使用预训练视觉语言模型进行高层次驾驶问答的能力。

Abstract: 本文提出了一种用于自动驾驶的两阶段视觉语言问答系统，该系统能够回答高层次的感知、预测和规划问题。在第一阶段，一个大型多模态语言模型（Qwen2.5-VL-32B）以六个摄像头输入、一段历史时间窗口以及带有少量示例的思维链提示为条件。自洽性集成（多个采样的推理链）进一步提高了答案的可靠性。在第二阶段，我们使用nuScenes场景元数据（对象注释、自车状态等）和类别特定的问题指令（针对感知、预测、规划任务的单独提示）来扩充提示。在驾驶问答基准测试中的实验表明，我们的方法明显优于基线 Qwen2.5 模型。例如，在第一阶段使用 5 个历史帧和 10-shot 提示可产生 65.1% 的总体准确率（而零样本为 62.61%）；应用自洽性将其提高到 66.85%。第二阶段达到 67.37% 的总体准确率。值得注意的是，该系统在严重的视觉损坏下仍保持 96% 的准确率。这些结果表明，精心设计的提示和上下文基础可以极大地增强使用预训练视觉语言模型进行高层次驾驶问答的能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19001) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction](https://arxiv.org/abs/2510.19654)
*Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, Huchuan Lu*

Main category: cs.CV

TL;DR: 本文提出了一种新的驾驶范式，Policy World Model (PWM)，它通过无动作的未来状态预测来统一世界建模和轨迹规划，并利用学习到的世界知识来改进规划。


<details>
  <summary>Details</summary>
Motivation: 现有的世界模型主要用于世界模拟，与轨迹规划分离；统一世界建模和规划的协同促进机制仍需探索。

Method: 提出 Policy World Model (PWM)，通过协作状态-动作预测模仿类人预测感知，并引入动态增强的并行 token 生成机制。

Result: 仅使用前置摄像头输入，该方法就能达到或超过依赖多视角和多模态输入的现有最佳方法。

Conclusion: Policy World Model (PWM) 能够有效整合世界建模和轨迹规划，并通过学习到的世界知识提升规划性能。

Abstract: 尽管驾驶世界模型取得了显著进展，但它们在自主系统中的潜力仍未得到充分利用：世界模型主要用于世界模拟，与轨迹规划分离。虽然最近的研究试图在单一框架内统一世界建模和规划，但世界建模对规划的协同促进机制仍有待进一步探索。在这项工作中，我们引入了一种新的驾驶范式，名为 Policy World Model (PWM)，它不仅在统一架构内集成了世界建模和轨迹规划，而且还能够通过提出的无动作的未来状态预测方案，利用学习到的世界知识来改进规划。通过协作状态-动作预测，PWM 可以模仿类人预测感知，从而产生更可靠的规划性能。为了提高视频预测的效率，我们进一步引入了一种动态增强的并行 token 生成机制，该机制配备了上下文引导的 tokenizer 和自适应动态 focal loss。尽管仅使用前置摄像头输入，我们的方法就能达到或超过依赖多视角和多模态输入的现有最佳方法。代码和模型权重将在 https://github.com/6550Zhao/Policy-World-Model 上发布。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19654) | **Categories:** cs.CV, cs.AI, cs.CL, cs.RO

---

### [3] [MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models](https://arxiv.org/abs/2510.19022)
*Aritra Bhowmik, Denis Korzhenkov, Cees G. M. Snoek, Amirhossein Habibian, Mohsen Ghafoorian*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Text-to-video diffusion models have enabled high-quality video synthesis, yet often fail to generate temporally coherent and physically plausible motion. A key reason is the models' insufficient understanding of complex motions that natural videos often entail. Recent works tackle this problem by aligning diffusion model features with those from pretrained video encoders. However, these encoders mix video appearance and dynamics into entangled features, limiting the benefit of such alignment. In this paper, we propose a motion-centric alignment framework that learns a disentangled motion subspace from a pretrained video encoder. This subspace is optimized to predict ground-truth optical flow, ensuring it captures true motion dynamics. We then align the latent features of a text-to-video diffusion model to this new subspace, enabling the generative model to internalize motion knowledge and generate more plausible videos. Our method improves the physical commonsense in a state-of-the-art video diffusion model, while preserving adherence to textual prompts, as evidenced by empirical evaluations on VideoPhy, VideoPhy2, VBench, and VBench-2.0, along with a user study.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19022) | **Categories:** cs.CV

---

### [4] [X-Ego: Acquiring Team-Level Tactical Situational Awareness via Cross-Egocentric Contrastive Video Representation Learning](https://arxiv.org/abs/2510.19150)
*Yunzhe Wang, Soham Hans, Volkan Ustun*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human team tactics emerge from each player's individual perspective and their ability to anticipate, interpret, and adapt to teammates' intentions. While advances in video understanding have improved the modeling of team interactions in sports, most existing work relies on third-person broadcast views and overlooks the synchronous, egocentric nature of multi-agent learning. We introduce X-Ego-CS, a benchmark dataset consisting of 124 hours of gameplay footage from 45 professional-level matches of the popular e-sports game Counter-Strike 2, designed to facilitate research on multi-agent decision-making in complex 3D environments. X-Ego-CS provides cross-egocentric video streams that synchronously capture all players' first-person perspectives along with state-action trajectories. Building on this resource, we propose Cross-Ego Contrastive Learning (CECL), which aligns teammates' egocentric visual streams to foster team-level tactical situational awareness from an individual's perspective. We evaluate CECL on a teammate-opponent location prediction task, demonstrating its effectiveness in enhancing an agent's ability to infer both teammate and opponent positions from a single first-person view using state-of-the-art video encoders. Together, X-Ego-CS and CECL establish a foundation for cross-egocentric multi-agent benchmarking in esports. More broadly, our work positions gameplay understanding as a testbed for multi-agent modeling and tactical learning, with implications for spatiotemporal reasoning and human-AI teaming in both virtual and real-world domains. Code and dataset are available at https://github.com/HATS-ICT/x-ego.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19150) | **Categories:** cs.CV, cs.AI, cs.LG

---

### [5] [Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks](https://arxiv.org/abs/2510.19195)
*Kai Zeng, Zhanqian Wu, Kaixin Xiong, Xiaobao Wei, Xiangyu Guo, Zhenxin Zhu, Kalok Ho, Lijun Zhou, Bohan Zeng, Ming Lu, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wentao Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in driving world models enable controllable generation of high-quality RGB videos or multimodal videos. Existing methods primarily focus on metrics related to generation quality and controllability. However, they often overlook the evaluation of downstream perception tasks, which are $\mathbf{really\ crucial}$ for the performance of autonomous driving. Existing methods usually leverage a training strategy that first pretrains on synthetic data and finetunes on real data, resulting in twice the epochs compared to the baseline (real data only). When we double the epochs in the baseline, the benefit of synthetic data becomes negligible. To thoroughly demonstrate the benefit of synthetic data, we introduce Dream4Drive, a novel synthetic data generation framework designed for enhancing the downstream perception tasks. Dream4Drive first decomposes the input video into several 3D-aware guidance maps and subsequently renders the 3D assets onto these guidance maps. Finally, the driving world model is fine-tuned to produce the edited, multi-view photorealistic videos, which can be used to train the downstream perception models. Dream4Drive enables unprecedented flexibility in generating multi-view corner cases at scale, significantly boosting corner case perception in autonomous driving. To facilitate future research, we also contribute a large-scale 3D asset dataset named DriveObj3D, covering the typical categories in driving scenarios and enabling diverse 3D-aware video editing. We conduct comprehensive experiments to show that Dream4Drive can effectively boost the performance of downstream perception models under various training epochs. Project: $\href{https://wm-research.github.io/Dream4Drive/}{this\ https\ URL}$

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19195) | **Categories:** cs.CV, cs.AI

---

### [6] [Space Object Detection using Multi-frame Temporal Trajectory Completion Method](https://arxiv.org/abs/2510.19220)
*Xiaoqing Lan, Biqiao Xin, Bingshu Wang, Han Zhang, Laixian Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Space objects in Geostationary Earth Orbit (GEO) present significant detection challenges in optical imaging due to weak signals, complex stellar backgrounds, and environmental interference. In this paper, we enhance high-frequency features of GEO targets while suppressing background noise at the single-frame level through wavelet transform. Building on this, we propose a multi-frame temporal trajectory completion scheme centered on the Hungarian algorithm for globally optimal cross-frame matching. To effectively mitigate missing and false detections, a series of key steps including temporal matching and interpolation completion, temporal-consistency-based noise filtering, and progressive trajectory refinement are designed in the post-processing pipeline. Experimental results on the public SpotGEO dataset demonstrate the effectiveness of the proposed method, achieving an F_1 score of 90.14%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19220) | **Categories:** cs.CV

---

### [7] [Advances in 4D Representation: Geometry, Motion, and Interaction](https://arxiv.org/abs/2510.19255)
*Mingrui Zhao, Sauradip Nag, Kai Wang, Aditya Vora, Guangda Ji, Peter Chun, Ali Mahdavi-Amiri, Hao Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:https://mingrui-zhao.github.io/4DRep-GMI/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19255) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Semantic World Models](https://arxiv.org/abs/2510.19818)
*Jacob Berg, Chuning Zhu, Yanda Bao, Ishan Durugkar, Abhishek Gupta*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planning with world models offers a powerful paradigm for robotic control. Conventional approaches train a model to predict future frames conditioned on current frames and actions, which can then be used for planning. However, the objective of predicting future pixels is often at odds with the actual planning objective; strong pixel reconstruction does not always correlate with good planning decisions. This paper posits that instead of reconstructing future frames as pixels, world models only need to predict task-relevant semantic information about the future. For such prediction the paper poses world modeling as a visual question answering problem about semantic information in future frames. This perspective allows world modeling to be approached with the same tools underlying vision language models. Thus vision language models can be trained as "semantic" world models through a supervised finetuning process on image-action-text data, enabling planning for decision-making while inheriting many of the generalization and robustness properties from the pretrained vision-language models. The paper demonstrates how such a semantic world model can be used for policy improvement on open-ended robotics tasks, leading to significant generalization improvements over typical paradigms of reconstruction-based action-conditional world modeling. Website available at https://weirdlabuw.github.io/swm.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19818) | **Categories:** cs.LG, cs.AI, cs.RO

---

### [2] [Foundation Model Forecasts: Form and Function](https://arxiv.org/abs/2510.19345)
*Alvaro Perez-Diaz, James C. Loach, Danielle E. Toutoungi, Lee Middleton*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time-series foundation models (TSFMs) achieve strong forecast accuracy, yet accuracy alone does not determine practical value. The form of a forecast -- point, quantile, parametric, or trajectory ensemble -- fundamentally constrains which operational tasks it can support. We survey recent TSFMs and find that two-thirds produce only point or parametric forecasts, while many operational tasks require trajectory ensembles that preserve temporal dependence. We establish when forecast types can be converted and when they cannot: trajectory ensembles convert to simpler forms via marginalization without additional assumptions, but the reverse requires imposing temporal dependence through copulas or conformal methods. We prove that marginals cannot determine path-dependent event probabilities -- infinitely many joint distributions share identical marginals but yield different answers to operational questions. We map six fundamental forecasting tasks to minimal sufficient forecast types and provide a task-aligned evaluation framework. Our analysis clarifies when forecast type, not accuracy, differentiates practical utility.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19345) | **Categories:** cs.LG, cs.AI

---

### [3] [Optimization Benchmark for Diffusion Models on Dynamical Systems](https://arxiv.org/abs/2510.19376)
*Fabian Schaipp*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The training of diffusion models is often absent in the evaluation of new optimization techniques. In this work, we benchmark recent optimization algorithms for training a diffusion model for denoising flow trajectories. We observe that Muon and SOAP are highly efficient alternatives to AdamW (18% lower final loss). We also revisit several recent phenomena related to the training of models for text or image applications in the context of diffusion model training. This includes the impact of the learning-rate schedule on the training dynamics, and the performance gap between Adam and SGD.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19376) | **Categories:** cs.LG, math.OC

---


## cs.MA [cs.MA]
### [1] [Local Guidance for Configuration-Based Multi-Agent Pathfinding](https://arxiv.org/abs/2510.19072)
*Tomoki Arita, Keisuke Okumura*

Main category: cs.MA

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19072) | **Categories:** cs.MA, cs.AI, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [A Cross-Environment and Cross-Embodiment Path Planning Framework via a Conditional Diffusion Model](https://arxiv.org/abs/2510.19128)
*Mehran Ghafarian Tamizi, Homayoun Honari, Amir Mehdi Soufi Enayati, Aleksey Nozdryn-Plotnicki, Homayoun Najjaran*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Path planning for a robotic system in high-dimensional cluttered environments needs to be efficient, safe, and adaptable for different environments and hardware. Conventional methods face high computation time and require extensive parameter tuning, while prior learning-based methods still fail to generalize effectively. The primary goal of this research is to develop a path planning framework capable of generalizing to unseen environments and new robotic manipulators without the need for retraining. We present GADGET (Generalizable and Adaptive Diffusion-Guided Environment-aware Trajectory generation), a diffusion-based planning model that generates joint-space trajectories conditioned on voxelized scene representations as well as start and goal configurations. A key innovation is GADGET's hybrid dual-conditioning mechanism that combines classifier-free guidance via learned scene encoding with classifier-guided Control Barrier Function (CBF) safety shaping, integrating environment awareness with real-time collision avoidance directly in the denoising process. This design supports zero-shot transfer to new environments and robotic embodiments without retraining. Experimental results show that GADGET achieves high success rates with low collision intensity in spherical-obstacle, bin-picking, and shelf environments, with CBF guidance further improving safety. Moreover, comparative evaluations indicate strong performance relative to both sampling-based and learning-based baselines. Furthermore, GADGET provides transferability across Franka Panda, Kinova Gen3 (6/7-DoF), and UR5 robots, and physical execution on a Kinova Gen3 demonstrates its ability to generate safe, collision-free trajectories in real-world settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19128) | **Categories:** cs.RO, cs.AI, 68T40 (Primary), 70Q05 (Secondary)

---

### [2] [LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments](https://arxiv.org/abs/2510.19655)
*Hongyu Ding, Ziming Xu, Yudong Fang, You Wu, Zixuan Chen, Jieqi Shi, Jing Huo, Yifan Zhang, Yang Gao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires an agent to navigate unseen environments based on natural language instructions without any prior training. Current methods face a critical trade-off: either rely on environment-specific waypoint predictors that limit scene generalization, or underutilize the reasoning capabilities of large models during navigation. We introduce LaViRA, a simple yet effective zero-shot framework that addresses this dilemma by decomposing action into a coarse-to-fine hierarchy: Language Action for high-level planning, Vision Action for perceptual grounding, and Robot Action for robust navigation. This modular decomposition allows us to leverage the distinct strengths of different scales of Multimodal Large Language Models (MLLMs) at each stage, creating a system that is powerful in its reasoning, grounding and practical control. LaViRA significantly outperforms existing state-of-the-art methods on the VLN-CE benchmark, demonstrating superior generalization capabilities in unseen environments, while maintaining transparency and efficiency for real-world deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19655) | **Categories:** cs.RO

---

### [3] [ProTerrain: Probabilistic Physics-Informed Rough Terrain World Modeling](https://arxiv.org/abs/2510.19364)
*Golnaz Raja, Ruslan Agishev, Miloš Prágr, Joni Pajarinen, Karel Zimmermann, Arun Kumar Singh, Reza Ghabcheloo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Uncertainty-aware robot motion prediction is crucial for downstream traversability estimation and safe autonomous navigation in unstructured, off-road environments, where terrain is heterogeneous and perceptual uncertainty is high. Most existing methods assume deterministic or spatially independent terrain uncertainties, ignoring the inherent local correlations of 3D spatial data and often producing unreliable predictions. In this work, we introduce an efficient probabilistic framework that explicitly models spatially correlated aleatoric uncertainty over terrain parameters as a probabilistic world model and propagates this uncertainty through a differentiable physics engine for probabilistic trajectory forecasting. By leveraging structured convolutional operators, our approach provides high-resolution multivariate predictions at manageable computational cost. Experimental evaluation on a publicly available dataset shows significantly improved uncertainty estimation and trajectory prediction accuracy over aleatoric uncertainty estimation baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.19364) | **Categories:** cs.RO

---
