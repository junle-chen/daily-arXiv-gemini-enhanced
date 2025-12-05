# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-12-06

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (11)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [BiTAgent: A Task-Aware Modular Framework for Bidirectional Coupling between Multimodal Large Language Models and World Models](https://arxiv.org/abs/2512.04513)
*Yu-Wei Zhan, Xin Wang, Pengzhe Mao, Tongtong Feng, Ren Wang, Wenwu Zhu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Building generalist embodied agents requires a unified system that can interpret multimodal goals, model environment dynamics, and execute reliable actions across diverse real-world tasks. Multimodal large language models (MLLMs) offer strong semantic priors and cross-modal generalization, while world models (WMs) provide actionable latent dynamics for prediction and control. Their combination holds promise for open-ended embodied intelligence, yet introduces two key challenges: (1) establishing a tight coupling between the semantic intent from MLLMs and the dynamic state representations within the WM's latent space, and (2) achieving task-aware adaptability that supports multi-task learning and cross-environment generalization. To address these limitations, we propose BiTAgent, a task-aware dynamic joint framework that enables bidirectional coupling between MLLMs and WMs. BiTAgent establishes two complementary pathways: a forward path that injects MLLM representations into the WM's latent space for semantically guided imagination, and a backward path where WM-generated feedback refines the MLLM's semantic space via dense text-conditioned rewards. This bidirectional interaction is realized through three synergistic components: Task-Aware Dynamic Joint Learning, Task-Aware Behavior Learning, and MLLM-WM Joint Optimization, which together harmonize semantic reasoning and dynamic prediction. Extensive experiments across multi-task and cross-environment settings demonstrate superior stability and generalization over state-of-the-art baselines, marking a step toward open-ended embodied learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04513) | **Categories:** cs.AI

---

### [2] [SIMA 2: A Generalist Embodied Agent for Virtual Worlds](https://arxiv.org/abs/2512.04797)
*SIMA team, Adrian Bolton, Alexander Lerchner, Alexandra Cordell, Alexandre Moufarek, Andrew Bolt, Andrew Lampinen, Anna Mitenkova, Arne Olav Hallingstad, Bojan Vujatovic, Bonnie Li, Cong Lu, Daan Wierstra, Daniel P. Sawyer, Daniel Slater, David Reichert, Davide Vercelli, Demis Hassabis, Drew A. Hudson, Duncan Williams, Ed Hirst, Fabio Pardo, Felix Hill, Frederic Besse, Hannah Openshaw, Harris Chan, Hubert Soyer, Jane X. Wang, Jeff Clune, John Agapiou, John Reid, Joseph Marino, Junkyung Kim, Karol Gregor, Kaustubh Sridhar, Kay McKinney, Laura Kampis, Lei M. Zhang, Loic Matthey, Luyu Wang, Maria Abi Raad, Maria Loks-Thompson, Martin Engelcke, Matija Kecman, Matthew Jackson, Maxime Gazeau, Ollie Purkiss, Oscar Knagg, Peter Stys, Piermaria Mendolicchio, Raia Hadsell, Rosemary Ke, Ryan Faulkner, Sarah Chakera, Satinder Singh Baveja, Shane Legg, Sheleem Kashem, Tayfun Terzi, Thomas Keck, Tim Harley, Tim Scholtes, Tyson Roberts, Volodymyr Mnih, Yulan Liu, Zhengdong Wang, Zoubin Ghahramani*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce SIMA 2, a generalist embodied agent that understands and acts in a wide variety of 3D virtual worlds. Built upon a Gemini foundation model, SIMA 2 represents a significant step toward active, goal-directed interaction within an embodied environment. Unlike prior work (e.g., SIMA 1) limited to simple language commands, SIMA 2 acts as an interactive partner, capable of reasoning about high-level goals, conversing with the user, and handling complex instructions given through language and images. Across a diverse portfolio of games, SIMA 2 substantially closes the gap with human performance and demonstrates robust generalization to previously unseen environments, all while retaining the base model's core reasoning capabilities. Furthermore, we demonstrate a capacity for open-ended self-improvement: by leveraging Gemini to generate tasks and provide rewards, SIMA 2 can autonomously learn new skills from scratch in a new environment. This work validates a path toward creating versatile and continuously learning agents for both virtual and, eventually, physical worlds.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04797) | **Categories:** cs.AI, cs.RO

---

### [3] [STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions](https://arxiv.org/abs/2512.04871)
*Junjie Fan, Hongye Zhao, Linduo Wei, Jiayu Rao, Guijia Li, Jiaxin Yuan, Wenqi Xu, Yong Qi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent adaptations of Large Language Models (LLMs) for time series forecasting often fail to effectively enhance information for raw series, leaving LLM reasoning capabilities underutilized. Existing prompting strategies rely on static correlations rather than generative interpretations of dynamic behavior, lacking critical global and instance-specific context. To address this, we propose STELLA (Semantic-Temporal Alignment with Language Abstractions), a framework that systematically mines and injects structured supplementary and complementary information. STELLA employs a dynamic semantic abstraction mechanism that decouples input series into trend, seasonality, and residual components. It then translates intrinsic behavioral features of these components into Hierarchical Semantic Anchors: a Corpus-level Semantic Prior (CSP) for global context and a Fine-grained Behavioral Prompt (FBP) for instance-level patterns. Using these anchors as prefix-prompts, STELLA guides the LLM to model intrinsic dynamics. Experiments on eight benchmark datasets demonstrate that STELLA outperforms state-of-the-art methods in long- and short-term forecasting, showing superior generalization in zero-shot and few-shot settings. Ablation studies further validate the effectiveness of our dynamically generated semantic anchors.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04871) | **Categories:** cs.AI, cs.CL, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [MindDrive: An All-in-One Framework Bridging World Models and Vision-Language Model for End-to-End Autonomous Driving](https://arxiv.org/abs/2512.04441)
*Bin Suna, Yaoguang Caob, Yan Wanga, Rui Wanga, Jiachen Shanga, Xiejie Fenga, Jiayi Lu, Jia Shi, Shichun Yang, Xiaoyu Yane, Ziying Song*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-End autonomous driving (E2E-AD) has emerged as a new paradigm, where trajectory planning plays a crucial role. Existing studies mainly follow two directions: trajectory generation oriented, which focuses on producing high-quality trajectories with simple decision mechanisms, and trajectory selection oriented, which performs multi-dimensional evaluation to select the best trajectory yet lacks sufficient generative capability. In this work, we propose MindDrive, a harmonized framework that integrates high-quality trajectory generation with comprehensive decision reasoning. It establishes a structured reasoning paradigm of "context simulation - candidate generation - multi-objective trade-off". In particular, the proposed Future-aware Trajectory Generator (FaTG), based on a World Action Model (WaM), performs ego-conditioned "what-if" simulations to predict potential future scenes and generate foresighted trajectory candidates. Building upon this, the VLM-oriented Evaluator (VLoE) leverages the reasoning capability of a large vision-language model to conduct multi-objective evaluations across safety, comfort, and efficiency dimensions, leading to reasoned and human-aligned decision making. Extensive experiments on the NAVSIM-v1 and NAVSIM-v2 benchmarks demonstrate that MindDrive achieves state-of-the-art performance across multi-dimensional driving metrics, significantly enhancing safety, compliance, and generalization. This work provides a promising path toward interpretable and cognitively guided autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04441) | **Categories:** cs.CV

---

### [2] [MoReGen: Multi-Agent Motion-Reasoning Engine for Code-based Text-to-Video Synthesis](https://arxiv.org/abs/2512.04221)
*Xiangyu Bai, He Liang, Bishoy Galoaa, Utsav Nandi, Shayda Moezzi, Yuhang He, Sarah Ostadabbas*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While text-to-video (T2V) generation has achieved remarkable progress in photorealism, generating intent-aligned videos that faithfully obey physics principles remains a core challenge. In this work, we systematically study Newtonian motion-controlled text-to-video generation and evaluation, emphasizing physical precision and motion coherence. We introduce MoReGen, a motion-aware, physics-grounded T2V framework that integrates multi-agent LLMs, physics simulators, and renderers to generate reproducible, physically accurate videos from text prompts in the code domain. To quantitatively assess physical validity, we propose object-trajectory correspondence as a direct evaluation metric and present MoReSet, a benchmark of 1,275 human-annotated videos spanning nine classes of Newtonian phenomena with scene descriptions, spatiotemporal relations, and ground-truth trajectories. Using MoReSet, we conduct experiments on existing T2V models, evaluating their physical validity through both our MoRe metrics and existing physics-based evaluators. Our results reveal that state-of-the-art models struggle to maintain physical validity, while MoReGen establishes a principled direction toward physically coherent video synthesis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04221) | **Categories:** cs.CV

---

### [3] [dVLM-AD: Enhance Diffusion Vision-Language-Model for Driving via Controllable Reasoning](https://arxiv.org/abs/2512.04459)
*Yingzi Ma, Yulong Cao, Wenhao Ding, Shuibai Zhang, Yan Wang, Boris Ivanovic, Ming Jiang, Marco Pavone, Chaowei Xiao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The autonomous driving community is increasingly focused on addressing the challenges posed by out-of-distribution (OOD) driving scenarios. A dominant research trend seeks to enhance end-to-end (E2E) driving systems by integrating vision-language models (VLMs), leveraging their rich world knowledge and reasoning abilities to improve generalization across diverse environments. However, most existing VLMs or vision-language agents (VLAs) are built upon autoregressive (AR) models. In this paper, we observe that existing AR-based VLMs -- limited by causal attention and sequential token generation -- often fail to maintain consistency and controllability between high-level reasoning and low-level planning. In contrast, recent discrete diffusion VLMs equipped with bidirectional attention exhibit superior controllability and reliability through iterative denoising. Building on these observations, we introduce dVLM-AD, a diffusion-based vision-language model that unifies perception, structured reasoning, and low-level planning for end-to-end driving. Evaluated on nuScenes and WOD-E2E, dVLM-AD yields more consistent reasoning-action pairs and achieves planning performance comparable to existing driving VLM/VLA systems despite a modest backbone, outperforming AR-based baselines with a 9 percent improvement in behavior-trajectory consistency and a 6 percent increase in RFS on long-tail WOD-E2E scenarios. These results suggest a controllable and reliable pathway for scalable end-to-end driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04459) | **Categories:** cs.CV

---

### [4] [PhyVLLM: Physics-Guided Video Language Model with Motion-Appearance Disentanglement](https://arxiv.org/abs/2512.04532)
*Yu-Wei Zhan, Xin Wang, Hong Chen, Tongtong Feng, Wei Feng, Ren Wang, Guangyao Li, Qing Li, Wenwu Zhu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video Large Language Models (Video LLMs) have shown impressive performance across a wide range of video-language tasks. However, they often fail in scenarios requiring a deeper understanding of physical dynamics. This limitation primarily arises from their reliance on appearance-based matching. Incorporating physical motion modeling is crucial for deeper video understanding, but presents three key challenges: (1) motion signals are often entangled with appearance variations, making it difficult to extract clean physical cues; (2) effective motion modeling requires not only continuous-time motion representations but also capturing physical dynamics; and (3) collecting accurate annotations for physical attributes is costly and often impractical. To address these issues, we propose PhyVLLM, a physical-guided video-language framework that explicitly incorporates physical motion into Video LLMs. Specifically, PhyVLLM disentangles visual appearance and object motion through a dual-branch encoder. To model physical dynamics over time, we incorporate a Neural Ordinary Differential Equation (Neural ODE) module, which generates differentiable physical dynamic representations. The resulting motion-aware representations are projected into the token space of a pretrained LLM, enabling physics reasoning without compromising the model's original multimodal capabilities. To circumvent the need for explicit physical labels, PhyVLLM employs a self-supervised manner to model the continuous evolution of object motion. Experimental results demonstrate that PhyVLLM significantly outperforms state-of-the-art Video LLMs on both physical reasoning and general video understanding tasks, highlighting the advantages of incorporating explicit physical modeling.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04532) | **Categories:** cs.CV, cs.AI

---

### [5] [Look Around and Pay Attention: Multi-camera Point Tracking Reimagined with Transformers](https://arxiv.org/abs/2512.04213)
*Bishoy Galoaa, Xiangyu Bai, Shayda Moezzi, Utsav Nandi, Sai Siddhartha Vivek Dhir Rangoju, Somaieh Amraee, Sarah Ostadabbas*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents LAPA (Look Around and Pay Attention), a novel end-to-end transformer-based architecture for multi-camera point tracking that integrates appearance-based matching with geometric constraints. Traditional pipelines decouple detection, association, and tracking, leading to error propagation and temporal inconsistency in challenging scenarios. LAPA addresses these limitations by leveraging attention mechanisms to jointly reason across views and time, establishing soft correspondences through a cross-view attention mechanism enhanced with geometric priors. Instead of relying on classical triangulation, we construct 3D point representations via attention-weighted aggregation, inherently accommodating uncertainty and partial observations. Temporal consistency is further maintained through a transformer decoder that models long-range dependencies, preserving identities through extended occlusions. Extensive experiments on challenging datasets, including our newly created multi-camera (MC) versions of TAPVid-3D panoptic and PointOdyssey, demonstrate that our unified approach significantly outperforms existing methods, achieving 37.5% APD on TAPVid-3D-MC and 90.3% APD on PointOdyssey-MC, particularly excelling in scenarios with complex motions and occlusions. Code is available at https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04213) | **Categories:** cs.CV

---

### [6] [Inference-time Stochastic Refinement of GRU-Normalizing Flow for Real-time Video Motion Transfer](https://arxiv.org/abs/2512.04282)
*Tasmiah Haque, Srinjoy Das*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Real-time video motion transfer applications such as immersive gaming and vision-based anomaly detection require accurate yet diverse future predictions to support realistic synthesis and robust downstream decision making under uncertainty. To improve the diversity of such sequential forecasts we propose a novel inference-time refinement technique that combines Gated Recurrent Unit-Normalizing Flows (GRU-NF) with stochastic sampling methods. While GRU-NF can capture multimodal distributions through its integration of normalizing flows within a temporal forecasting framework, its deterministic transformation structure can limit expressivity. To address this, inspired by Stochastic Normalizing Flows (SNF), we introduce Markov Chain Monte Carlo (MCMC) steps during GRU-NF inference, enabling the model to explore a richer output space and better approximate the true data distribution without retraining. We validate our approach in a keypoint-based video motion transfer pipeline, where capturing temporally coherent and perceptually diverse future trajectories is essential for realistic samples and low bandwidth communication. Experiments show that our inference framework, Gated Recurrent Unit- Stochastic Normalizing Flows (GRU-SNF) outperforms GRU-NF in generating diverse outputs without sacrificing accuracy, even under longer prediction horizons. By injecting stochasticity during inference, our approach captures multimodal behavior more effectively. These results highlight the potential of integrating stochastic dynamics with flow-based sequence models for generative time series forecasting.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04282) | **Categories:** cs.CV, cs.LG

---

### [7] [SyncTrack4D: Cross-Video Motion Alignment and Video Synchronization for Multi-Video 4D Gaussian Splatting](https://arxiv.org/abs/2512.04315)
*Yonghan Lee, Tsung-Wei Huang, Shiv Gehlot, Jaehoon Choi, Guan-Ming Su, Dinesh Manocha*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modeling dynamic 3D scenes is challenging due to their high-dimensional nature, which requires aggregating information from multiple views to reconstruct time-evolving 3D geometry and motion. We present a novel multi-video 4D Gaussian Splatting (4DGS) approach designed to handle real-world, unsynchronized video sets. Our approach, SyncTrack4D, directly leverages dense 4D track representation of dynamic scene parts as cues for simultaneous cross-video synchronization and 4DGS reconstruction. We first compute dense per-video 4D feature tracks and cross-video track correspondences by Fused Gromov-Wasserstein optimal transport approach. Next, we perform global frame-level temporal alignment to maximize overlapping motion of matched 4D tracks. Finally, we achieve sub-frame synchronization through our multi-video 4D Gaussian splatting built upon a motion-spline scaffold representation. The final output is a synchronized 4DGS representation with dense, explicit 3D trajectories, and temporal offsets for each video. We evaluate our approach on the Panoptic Studio and SyncNeRF Blender, demonstrating sub-frame synchronization accuracy with an average temporal error below 0.26 frames, and high-fidelity 4D reconstruction reaching 26.3 PSNR scores on the Panoptic Studio dataset. To the best of our knowledge, our work is the first general 4D Gaussian Splatting approach for unsynchronized video sets, without assuming the existence of predefined scene objects or prior models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04315) | **Categories:** cs.CV

---

### [8] [Explainable Parkinsons Disease Gait Recognition Using Multimodal RGB-D Fusion and Large Language Models](https://arxiv.org/abs/2512.04425)
*Manar Alnaasan, Md Selim Sarowar, Sungho Kim*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate and interpretable gait analysis plays a crucial role in the early detection of Parkinsons disease (PD),yet most existing approaches remain limited by single-modality inputs, low robustness, and a lack of clinical transparency. This paper presents an explainable multimodal framework that integrates RGB and Depth (RGB-D) data to recognize Parkinsonian gait patterns under realistic conditions. The proposed system employs dual YOLOv11-based encoders for modality-specific feature extraction, followed by a Multi-Scale Local-Global Extraction (MLGE) module and a Cross-Spatial Neck Fusion mechanism to enhance spatial-temporal representation. This design captures both fine-grained limb motion (e.g., reduced arm swing) and overall gait dynamics (e.g., short stride or turning difficulty), even in challenging scenarios such as low lighting or occlusion caused by clothing. To ensure interpretability, a frozen Large Language Model (LLM) is incorporated to translate fused visual embeddings and structured metadata into clinically meaningful textual explanations. Experimental evaluations on multimodal gait datasets demonstrate that the proposed RGB-D fusion framework achieves higher recognition accuracy, improved robustness to environmental variations, and clear visual-linguistic reasoning compared with single-input baselines. By combining multimodal feature learning with language-based interpretability, this study bridges the gap between visual recognition and clinical understanding, offering a novel vision-language paradigm for reliable and explainable Parkinsons disease gait analysis. Code:https://github.com/manaralnaasan/RGB-D_parkinson-LLM

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04425) | **Categories:** cs.CV, cs.AI

---

### [9] [FASTer: Toward Efficient Autoregressive Vision Language Action Modeling via neural Action Tokenization](https://arxiv.org/abs/2512.04952)
*Yicheng Liu, Shiduo Zhang, Zibin Dong, Baijun Ye, Tianyuan Yuan, Xiaopeng Yu, Linqi Yin, Chenhao Lu, Junhao Shi, Luca Jiang-Tao Yu, Liangtao Zheng, Tao Jiang, Jingjing Gong, Xipeng Qiu, Hang Zhao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autoregressive vision-language-action (VLA) models have recently demonstrated strong capabilities in robotic manipulation. However, their core process of action tokenization often involves a trade-off between reconstruction fidelity and inference efficiency. We introduce FASTer, a unified framework for efficient and generalizable robot learning that integrates a learnable tokenizer with an autoregressive policy built upon it. FASTerVQ encodes action chunks as single-channel images, capturing global spatio-temporal dependencies while maintaining a high compression ratio. FASTerVLA builds on this tokenizer with block-wise autoregressive decoding and a lightweight action expert, achieving both faster inference and higher task performance. Extensive experiments across simulated and real-world benchmarks show that FASTerVQ delivers superior reconstruction quality, high token utilization, and strong cross-task and cross-embodiment generalization, while FASTerVLA further improves overall capability, surpassing previous state-of-the-art VLA models in both inference speed and task performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04952) | **Categories:** cs.CV, cs.RO

---

### [10] [Controllable Long-term Motion Generation with Extended Joint Targets](https://arxiv.org/abs/2512.04487)
*Eunjong Lee, Eunhee Kim, Sanghoon Hong, Eunho Jung, Jihoon Kim*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generating stable and controllable character motion in real-time is a key challenge in computer animation. Existing methods often fail to provide fine-grained control or suffer from motion degradation over long sequences, limiting their use in interactive applications. We propose COMET, an autoregressive framework that runs in real time, enabling versatile character control and robust long-horizon synthesis. Our efficient Transformer-based conditional VAE allows for precise, interactive control over arbitrary user-specified joints for tasks like goal-reaching and in-betweening from a single model. To ensure long-term temporal stability, we introduce a novel reference-guided feedback mechanism that prevents error accumulation. This mechanism also serves as a plug-and-play stylization module, enabling real-time style transfer. Extensive evaluations demonstrate that COMET robustly generates high-quality motion at real-time speeds, significantly outperforming state-of-the-art approaches in complex motion control tasks and confirming its readiness for demanding interactive applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04487) | **Categories:** cs.CV

---

### [11] [Back to Basics: Motion Representation Matters for Human Motion Generation Using Diffusion Model](https://arxiv.org/abs/2512.04499)
*Yuduo Jin, Brandon Haworth*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion models have emerged as a widely utilized and successful methodology in human motion synthesis. Task-oriented diffusion models have significantly advanced action-to-motion, text-to-motion, and audio-to-motion applications. In this paper, we investigate fundamental questions regarding motion representations and loss functions in a controlled study, and we enumerate the impacts of various decisions in the workflow of the generative motion diffusion model. To answer these questions, we conduct empirical studies based on a proxy motion diffusion model (MDM). We apply v loss as the prediction objective on MDM (vMDM), where v is the weighted sum of motion data and noise. We aim to enhance the understanding of latent data distributions and provide a foundation for improving the state of conditional motion diffusion models. First, we evaluate the six common motion representations in the literature and compare their performance in terms of quality and diversity metrics. Second, we compare the training time under various configurations to shed light on how to speed up the training process of motion diffusion models. Finally, we also conduct evaluation analysis on a large motion dataset. The results of our experiments indicate clear performance differences across motion representations in diverse datasets. Our results also demonstrate the impacts of distinct configurations on model training and suggest the importance and effectiveness of these decisions on the outcomes of motion diffusion models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04499) | **Categories:** cs.CV, cs.GR

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space](https://arxiv.org/abs/2512.04601)
*Joey Hong, Kang Liu, Zhan Ling, Jiecao Chen, Sergey Levine*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language model (LLM) agents -- LLMs that dynamically interact with an environment over long horizons -- have become an increasingly important area of research, enabling automation in complex tasks involving tool-use, web browsing, and dialogue with people. In the absence of expert demonstrations, training LLM agents has relied on policy gradient methods that optimize LLM policies with respect to an (often sparse) reward function. However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity. Furthermore, policy improvement hinges on discovering better actions through exploration, which can be difficult when actions lie in natural language space. In this paper, we propose Natural Language Actor-Critic (NLAC), a novel actor-critic algorithm that trains LLM policies using a generative LLM critic that produces natural language rather than scalar values. This approach leverages the inherent strengths of LLMs to provide a richer and more actionable training signal; particularly, in tasks with large, open-ended action spaces, natural language explanations for why an action is suboptimal can be immensely useful for LLM policies to reason how to improve their actions, without relying on random exploration. Furthermore, our approach can be trained off-policy without policy gradients, offering a more data-efficient and stable alternative to existing on-policy methods. We present results on a mixture of reasoning, web browsing, and tool-use with dialogue tasks, demonstrating that NLAC shows promise in outperforming existing training approaches and offers a more scalable and stable training paradigm for LLM agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04601) | **Categories:** cs.LG, cs.CL

---


## 机器人学 (Robotics) [cs.RO]
### [1] [FALCON: Actively Decoupled Visuomotor Policies for Loco-Manipulation with Foundation-Model-Based Coordination](https://arxiv.org/abs/2512.04381)
*Chengyang He, Ge Sun, Yue Bai, Junkai Lu, Jiadong Zhao, Guillaume Sartoretti*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present FoundAtion-model-guided decoupled LoCO-maNipulation visuomotor policies (FALCON), a framework for loco-manipulation that combines modular diffusion policies with a vision-language foundation model as the coordinator. Our approach explicitly decouples locomotion and manipulation into two specialized visuomotor policies, allowing each subsystem to rely on its own observations. This mitigates the performance degradation that arise when a single policy is forced to fuse heterogeneous, potentially mismatched observations from locomotion and manipulation. Our key innovation lies in restoring coordination between these two independent policies through a vision-language foundation model, which encodes global observations and language instructions into a shared latent embedding conditioning both diffusion policies. On top of this backbone, we introduce a phase-progress head that uses textual descriptions of task stages to infer discrete phase and continuous progress estimates without manual phase labels. To further structure the latent space, we incorporate a coordination-aware contrastive loss that explicitly encodes cross-subsystem compatibility between arm and base actions. We evaluate FALCON on two challenging loco-manipulation tasks requiring navigation, precise end-effector placement, and tight base-arm coordination. Results show that it surpasses centralized and decentralized baselines while exhibiting improved robustness and generalization to out-of-distribution scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04381) | **Categories:** cs.RO

---

### [2] [From Generated Human Videos to Physically Plausible Robot Trajectories](https://arxiv.org/abs/2512.05094)
*James Ni, Zekai Wang, Wei Lin, Amir Bar, Yann LeCun, Trevor Darrell, Jitendra Malik, Roei Herzig*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video generation models are rapidly improving in their ability to synthesize human actions in novel contexts, holding the potential to serve as high-level planners for contextual robot control. To realize this potential, a key research question remains open: how can a humanoid execute the human actions from generated videos in a zero-shot manner? This challenge arises because generated videos are often noisy and exhibit morphological distortions that make direct imitation difficult compared to real video. To address this, we introduce a two-stage pipeline. First, we lift video pixels into a 4D human representation and then retarget to the humanoid morphology. Second, we propose GenMimic-a physics-aware reinforcement learning policy conditioned on 3D keypoints, and trained with symmetry regularization and keypoint-weighted tracking rewards. As a result, GenMimic can mimic human actions from noisy, generated videos. We curate GenMimicBench, a synthetic human-motion dataset generated using two video generation models across a spectrum of actions and contexts, establishing a benchmark for assessing zero-shot generalization and policy robustness. Extensive experiments demonstrate improvements over strong baselines in simulation and confirm coherent, physically stable motion tracking on a Unitree G1 humanoid robot without fine-tuning. This work offers a promising path to realizing the potential of video generation models as high-level policies for robot control.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.05094) | **Categories:** cs.RO, cs.CV

---

### [3] [Driving Beyond Privilege: Distilling Dense-Reward Knowledge into Sparse-Reward Policies](https://arxiv.org/abs/2512.04279)
*Feeza Khan Khanzada, Jaerock Kwon*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We study how to exploit dense simulator-defined rewards in vision-based autonomous driving without inheriting their misalignment with deployment metrics. In realistic simulators such as CARLA, privileged state (e.g., lane geometry, infractions, time-to-collision) can be converted into dense rewards that stabilize and accelerate model-based reinforcement learning, but policies trained directly on these signals often overfit and fail to generalize when evaluated on sparse objectives such as route completion and collision-free overtaking. We propose reward-privileged world model distillation, a two-stage framework in which a teacher DreamerV3-style agent is first trained with a dense privileged reward, and only its latent dynamics are distilled into a student trained solely on sparse task rewards. Teacher and student share the same observation space (semantic bird's-eye-view images); privileged information enters only through the teacher's reward, and the student does not imitate the teacher's actions or value estimates. Instead, the student's world model is regularized to match the teacher's latent dynamics while its policy is learned from scratch on sparse success/failure signals. In CARLA lane-following and overtaking benchmarks, sparse-reward students outperform both dense-reward teachers and sparse-from-scratch baselines. On unseen lane-following routes, reward-privileged distillation improves success by about 23 percent relative to the dense teacher while maintaining comparable or better safety. On overtaking, students retain near-perfect performance on training routes and achieve up to a 27x improvement in success on unseen routes, with improved lane keeping. These results show that dense rewards can be leveraged to learn richer dynamics models while keeping the deployed policy optimized strictly for sparse, deployment-aligned objectives.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04279) | **Categories:** cs.RO

---

### [4] [On Disturbance-Aware Minimum-Time Trajectory Planning: Evidence from Tests on a Dynamic Driving Simulator](https://arxiv.org/abs/2512.04917)
*Matteo Masoni, Vincenzo Palermo, Marco Gabiccini, Martino Gulisano, Giorgio Previati, Massimiliano Gobbi, Francesco Comolli, Gianpiero Mastinu, Massimo Guiggiani*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This work investigates how disturbance-aware, robustness-embedded reference trajectories translate into driving performance when executed by professional drivers in a dynamic simulator. Three planned reference trajectories are compared against a free-driving baseline (NOREF) to assess trade-offs between lap time (LT) and steering effort (SE): NOM, the nominal time-optimal trajectory; TLC, a track-limit-robust trajectory obtained by tightening margins to the track edges; and FLC, a friction-limit-robust trajectory obtained by tightening against axle and tire saturation. All trajectories share the same minimum lap-time objective with a small steering-smoothness regularizer and are evaluated by two professional drivers using a high-performance car on a virtual track. The trajectories derive from a disturbance-aware minimum-lap-time framework recently proposed by the authors, where worst-case disturbance growth is propagated over a finite horizon and used to tighten tire-friction and track-limit constraints, preserving performance while providing probabilistic safety margins. LT and SE are used as performance indicators, while RMS lateral deviation, speed error, and drift angle characterize driving style. Results show a Pareto-like LT-SE trade-off: NOM yields the shortest LT but highest SE; TLC minimizes SE at the cost of longer LT; FLC lies near the efficient frontier, substantially reducing SE relative to NOM with only a small LT increase. Removing trajectory guidance (NOREF) increases both LT and SE, confirming that reference trajectories improve pace and control efficiency. Overall, the findings highlight reference-based and disturbance-aware planning, especially FLC, as effective tools for training and for achieving fast yet stable trajectories.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.04917) | **Categories:** cs.RO, eess.SY

---

### [5] [STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models](https://arxiv.org/abs/2512.05107)
*Feng Xu, Guangyao Zhai, Xin Kong, Tingzhong Fu, Daniel F. N. Gordon, Xueli An, Benjamin Busam*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in Vision-Language-Action (VLA) models, powered by large language models and reinforcement learning-based fine-tuning, have shown remarkable progress in robotic manipulation. Existing methods often treat long-horizon actions as linguistic sequences and apply trajectory-level optimization methods such as Trajectory-wise Preference Optimization (TPO) or Proximal Policy Optimization (PPO), leading to coarse credit assignment and unstable training. However, unlike language, where a unified semantic meaning is preserved despite flexible sentence order, action trajectories progress through causally chained stages with different learning difficulties. This motivates progressive stage optimization. Thereby, we present Stage-Aware Reinforcement (STARE), a module that decomposes a long-horizon action trajectory into semantically meaningful stages and provides dense, interpretable, and stage-aligned reinforcement signals. Integrating STARE into TPO and PPO, we yield Stage-Aware TPO (STA-TPO) and Stage-Aware PPO (STA-PPO) for offline stage-wise preference and online intra-stage interaction, respectively. Further building on supervised fine-tuning as initialization, we propose the Imitation -> Preference -> Interaction (IPI), a serial fine-tuning pipeline for improving action accuracy in VLA models. Experiments on SimplerEnv and ManiSkill3 demonstrate substantial gains, achieving state-of-the-art success rates of 98.0 percent on SimplerEnv and 96.4 percent on ManiSkill3 tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2512.05107) | **Categories:** cs.RO

---
