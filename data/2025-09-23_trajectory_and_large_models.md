# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-23

## 目录

- [计算机视觉 (Computer Vision) (5)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [CoPAD : Multi-source Trajectory Fusion and Cooperative Trajectory Prediction with Anchor-oriented Decoder in V2X Scenarios](https://arxiv.org/abs/2509.15984)
*Kangyu Wu, Jiaqi Qiao, Ya Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently, data-driven trajectory prediction methods have achieved remarkable results, significantly advancing the development of autonomous driving. However, the instability of single-vehicle perception introduces certain limitations to trajectory prediction. In this paper, a novel lightweight framework for cooperative trajectory prediction, CoPAD, is proposed. This framework incorporates a fusion module based on the Hungarian algorithm and Kalman filtering, along with the Past Time Attention (PTA) module, mode attention module and anchor-oriented decoder (AoD). It effectively performs early fusion on multi-source trajectory data from vehicles and road infrastructure, enabling the trajectories with high completeness and accuracy. The PTA module can efficiently capture potential interaction information among historical trajectories, and the mode attention module is proposed to enrich the diversity of predictions. Additionally, the decoder based on sparse anchors is designed to generate the final complete trajectories. Extensive experiments show that CoPAD achieves the state-of-the-art performance on the DAIR-V2X-Seq dataset, validating the effectiveness of the model in cooperative trajectory prediction in V2X scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15984) | **Categories:** cs.CV, cs.MA, cs.RO

---

### [2] [SAMPO:Scale-wise Autoregression with Motion PrOmpt for generative world models](https://arxiv.org/abs/2509.15536)
*Sen Wang, Jingyi Tian, Le Wang, Zhimin Liao, Jiayi Li, Huaiyi Dong, Kun Xia, Sanping Zhou, Wei Tang, Hua Gang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: World models allow agents to simulate the consequences of actions in imagined environments for planning, control, and long-horizon decision-making. However, existing autoregressive world models struggle with visually coherent predictions due to disrupted spatial structure, inefficient decoding, and inadequate motion modeling. In response, we propose \textbf{S}cale-wise \textbf{A}utoregression with \textbf{M}otion \textbf{P}r\textbf{O}mpt (\textbf{SAMPO}), a hybrid framework that combines visual autoregressive modeling for intra-frame generation with causal modeling for next-frame generation. Specifically, SAMPO integrates temporal causal decoding with bidirectional spatial attention, which preserves spatial locality and supports parallel decoding within each scale. This design significantly enhances both temporal consistency and rollout efficiency. To further improve dynamic scene understanding, we devise an asymmetric multi-scale tokenizer that preserves spatial details in observed frames and extracts compact dynamic representations for future frames, optimizing both memory usage and model performance. Additionally, we introduce a trajectory-aware motion prompt module that injects spatiotemporal cues about object and robot trajectories, focusing attention on dynamic regions and improving temporal consistency and physical realism. Extensive experiments show that SAMPO achieves competitive performance in action-conditioned video prediction and model-based control, improving generation quality with 4.4$\times$ faster inference. We also evaluate SAMPO's zero-shot generalization and scaling behavior, demonstrating its ability to generalize to unseen tasks and benefit from larger model sizes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15536) | **Categories:** cs.CV, cs.RO

---

### [3] [Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning](https://arxiv.org/abs/2509.15250)
*Wenda Qin, Andrea Burns, Bryan A. Plummer, Margrit Betke*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large models achieve strong performance on Vision-and-Language Navigation (VLN) tasks, but are costly to run in resource-limited environments. Token pruning offers appealing tradeoffs for efficiency with minimal performance loss by reducing model input size, but prior work overlooks VLN-specific challenges. For example, information loss from pruning can effectively increase computational cost due to longer walks. Thus, the inability to identify uninformative tokens undermines the supposed efficiency gains from pruning. To address this, we propose Navigation-Aware Pruning (NAP), which uses navigation-specific traits to simplify the pruning process by pre-filtering tokens into foreground and background. For example, image views are filtered based on whether the agent can navigate in that direction. We also extract navigation-relevant instructions using a Large Language Model. After filtering, we focus pruning on background tokens, minimizing information loss. To further help avoid increases in navigation length, we discourage backtracking by removing low-importance navigation nodes. Experiments on standard VLN benchmarks show NAP significantly outperforms prior work, preserving higher success rates while saving more than 50% FLOPS.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15250) | **Categories:** cs.CV, cs.AI

---

### [4] [How Good are Foundation Models in Step-by-Step Embodied Reasoning?](https://arxiv.org/abs/2509.15293)
*Dinura Dissanayake, Ahmed Heakl, Omkar Thawakar, Noor Ahsan, Ritesh Thawkar, Ketan More, Jean Lahoud, Rao Anwer, Hisham Cholakkal, Ivan Laptev, Fahad Shahbaz Khan, Salman Khan*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied agents operating in the physical world must make decisions that are not only effective but also safe, spatially coherent, and grounded in context. While recent advances in large multimodal models (LMMs) have shown promising capabilities in visual understanding and language generation, their ability to perform structured reasoning for real-world embodied tasks remains underexplored. In this work, we aim to understand how well foundation models can perform step-by-step reasoning in embodied environments. To this end, we propose the Foundation Model Embodied Reasoning (FoMER) benchmark, designed to evaluate the reasoning capabilities of LMMs in complex embodied decision-making scenarios. Our benchmark spans a diverse set of tasks that require agents to interpret multimodal observations, reason about physical constraints and safety, and generate valid next actions in natural language. We present (i) a large-scale, curated suite of embodied reasoning tasks, (ii) a novel evaluation framework that disentangles perceptual grounding from action reasoning, and (iii) empirical analysis of several leading LMMs under this setting. Our benchmark includes over 1.1k samples with detailed step-by-step reasoning across 10 tasks and 8 embodiments, covering three different robot types. Our results highlight both the potential and current limitations of LMMs in embodied reasoning, pointing towards key challenges and opportunities for future research in robot intelligence. Our data and code will be made publicly available.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15293) | **Categories:** cs.CV, cs.RO

---

### [5] [OpenViGA: Video Generation for Automotive Driving Scenes by Streamlining and Fine-Tuning Open Source Models with Public Data](https://arxiv.org/abs/2509.15479)
*Björn Möller, Zhengyang Li, Malte Stelzer, Thomas Graave, Fabian Bettels, Muaaz Ataya, Tim Fingscheidt*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent successful video generation systems that predict and create realistic automotive driving scenes from short video inputs assign tokenization, future state prediction (world model), and video decoding to dedicated models. These approaches often utilize large models that require significant training resources, offer limited insight into design choices, and lack publicly available code and datasets. In this work, we address these deficiencies and present OpenViGA, an open video generation system for automotive driving scenes. Our contributions are: Unlike several earlier works for video generation, such as GAIA-1, we provide a deep analysis of the three components of our system by separate quantitative and qualitative evaluation: Image tokenizer, world model, video decoder. Second, we purely build upon powerful pre-trained open source models from various domains, which we fine-tune by publicly available automotive data (BDD100K) on GPU hardware at academic scale. Third, we build a coherent video generation system by streamlining interfaces of our components. Fourth, due to public availability of the underlying models and data, we allow full reproducibility. Finally, we also publish our code and models on Github. For an image size of 256x256 at 4 fps we are able to predict realistic driving scene videos frame-by-frame with only one frame of algorithmic latency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15479) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning](https://arxiv.org/abs/2509.15561)
*Om Naphade, Saksham Bansal, Parikshit Pareek*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15561) | **Categories:** cs.LG, cs.CL

---

### [2] [KoopCast: Trajectory Forecasting via Koopman Operators](https://arxiv.org/abs/2509.15513)
*Jungjin Lee, Jaeuk Shin, Gihwan Kim, Joonho Han, Insoon Yang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present KoopCast, a lightweight yet efficient model for trajectory forecasting in general dynamic environments. Our approach leverages Koopman operator theory, which enables a linear representation of nonlinear dynamics by lifting trajectories into a higher-dimensional space. The framework follows a two-stage design: first, a probabilistic neural goal estimator predicts plausible long-term targets, specifying where to go; second, a Koopman operator-based refinement module incorporates intention and history into a nonlinear feature space, enabling linear prediction that dictates how to go. This dual structure not only ensures strong predictive accuracy but also inherits the favorable properties of linear operators while faithfully capturing nonlinear dynamics. As a result, our model offers three key advantages: (i) competitive accuracy, (ii) interpretability grounded in Koopman spectral theory, and (iii) low-latency deployment. We validate these benefits on ETH/UCY, the Waymo Open Motion Dataset, and nuScenes, which feature rich multi-agent interactions and map-constrained nonlinear motion. Across benchmarks, KoopCast consistently delivers high predictive accuracy together with mode-level interpretability and practical efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15513) | **Categories:** cs.LG, cs.RO, cs.SY, eess.SY

---

### [3] [GUI-ReWalk: Massive Data Generation for GUI Agent via Stochastic Exploration and Intent-Aware Reasoning](https://arxiv.org/abs/2509.15738)
*Musen Lin, Minghao Liu, Taoran Lu, Lichen Yuan, Yiwei Liu, Haonan Xu, Yu Miao, Yuhao Chao, Zhaojian Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Graphical User Interface (GUI) Agents, powered by large language and vision-language models, hold promise for enabling end-to-end automation in digital environments. However, their progress is fundamentally constrained by the scarcity of scalable, high-quality trajectory data. Existing data collection strategies either rely on costly and inconsistent manual annotations or on synthetic generation methods that trade off between diversity and meaningful task coverage. To bridge this gap, we present GUI-ReWalk: a reasoning-enhanced, multi-stage framework for synthesizing realistic and diverse GUI trajectories. GUI-ReWalk begins with a stochastic exploration phase that emulates human trial-and-error behaviors, and progressively transitions into a reasoning-guided phase where inferred goals drive coherent and purposeful interactions. Moreover, it supports multi-stride task generation, enabling the construction of long-horizon workflows across multiple applications. By combining randomness for diversity with goal-aware reasoning for structure, GUI-ReWalk produces data that better reflects the intent-aware, adaptive nature of human-computer interaction. We further train Qwen2.5-VL-7B on the GUI-ReWalk dataset and evaluate it across multiple benchmarks, including Screenspot-Pro, OSWorld-G, UI-Vision, AndroidControl, and GUI-Odyssey. Results demonstrate that GUI-ReWalk enables superior coverage of diverse interaction flows, higher trajectory entropy, and more realistic user intent. These findings establish GUI-ReWalk as a scalable and data-efficient framework for advancing GUI agent research and enabling robust real-world automation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15738) | **Categories:** cs.LG

---

### [4] [Exploring multimodal implicit behavior learning for vehicle navigation in simulated cities](https://arxiv.org/abs/2509.15400)
*Eric Aislan Antonelo, Gustavo Claudio Karl Couto, Christian Möller*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Standard Behavior Cloning (BC) fails to learn multimodal driving decisions, where multiple valid actions exist for the same scenario. We explore Implicit Behavioral Cloning (IBC) with Energy-Based Models (EBMs) to better capture this multimodality. We propose Data-Augmented IBC (DA-IBC), which improves learning by perturbing expert actions to form the counterexamples of IBC training and using better initialization for derivative-free inference. Experiments in the CARLA simulator with Bird's-Eye View inputs demonstrate that DA-IBC outperforms standard IBC in urban driving tasks designed to evaluate multimodal behavior learning in a test environment. The learned energy landscapes are able to represent multimodal action distributions, which BC fails to achieve.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15400) | **Categories:** cs.LG, cs.AI, cs.RO

---

### [5] [KITE: Kernelized and Information Theoretic Exemplars for In-Context Learning](https://arxiv.org/abs/2509.15676)
*Vaibhav Singh, Soumya Suvra Ghosal, Kapu Nirmal Joshua, Soumyabrata Pal, Sayak Ray Chowdhury*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In-context learning (ICL) has emerged as a powerful paradigm for adapting large language models (LLMs) to new and data-scarce tasks using only a few carefully selected task-specific examples presented in the prompt. However, given the limited context size of LLMs, a fundamental question arises: Which examples should be selected to maximize performance on a given user query? While nearest-neighbor-based methods like KATE have been widely adopted for this purpose, they suffer from well-known drawbacks in high-dimensional embedding spaces, including poor generalization and a lack of diversity. In this work, we study this problem of example selection in ICL from a principled, information theory-driven perspective. We first model an LLM as a linear function over input embeddings and frame the example selection task as a query-specific optimization problem: selecting a subset of exemplars from a larger example bank that minimizes the prediction error on a specific query. This formulation departs from traditional generalization-focused learning theoretic approaches by targeting accurate prediction for a specific query instance. We derive a principled surrogate objective that is approximately submodular, enabling the use of a greedy algorithm with an approximation guarantee. We further enhance our method by (i) incorporating the kernel trick to operate in high-dimensional feature spaces without explicit mappings, and (ii) introducing an optimal design-based regularizer to encourage diversity in the selected examples. Empirically, we demonstrate significant improvements over standard retrieval methods across a suite of classification tasks, highlighting the benefits of structure-aware, diverse example selection for ICL in real-world, label-scarce scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15676) | **Categories:** cs.LG, cs.AI, cs.CL

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories](https://arxiv.org/abs/2509.16176)
*Yifan Lin, Sophie Ziyu Liu, Ran Qi, George Z. Xue, Xinping Song, Chao Qin, Hugh H. -T. Liu*

Main category: cs.RO

TL;DR: ACDC提出了一种自主无人机电影摄影系统，该系统通过人类导演和无人机之间的自然语言交流驱动，将自然语言提示转换为可执行的室内无人机视频拍摄。


<details>
  <summary>Details</summary>
Motivation: 先前无人机电影摄影工作流程的主要限制是需要手动选择航点和视角，这既费力又导致性能不一致。

Method: 该方法包括用于初始航点选择的视觉-语言检索管道，使用审美反馈细化姿势的基于偏好的贝叶斯优化框架，以及生成安全四旋翼飞行器轨迹的运动规划器。

Result: 通过模拟和硬件在环实验验证了ACDC，证明了它可以在各种室内场景中稳健地生成专业质量的镜头，而无需机器人或电影摄影方面的专业知识。

Conclusion: 这些结果突出了具身AI代理在从开放词汇对话到现实世界自主空中电影摄影闭环方面的潜力。

Abstract: 我们提出了Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories (ACDC)，这是一个由人类导演和无人机之间的自然语言交流驱动的自主无人机电影摄影系统。先前无人机电影摄影工作流程的主要限制是，它们需要基于预定义的人类意图手动选择航点和视角，这既费力又导致性能不一致。在本文中，我们提出采用大型语言模型（LLM）和视觉基础模型（VFM）将自由形式的自然语言提示直接转换为可执行的室内无人机视频拍摄。具体来说，我们的方法包括用于初始航点选择的视觉-语言检索管道，使用审美反馈细化姿势的基于偏好的贝叶斯优化框架，以及生成安全四旋翼飞行器轨迹的运动规划器。我们通过模拟和硬件在环实验验证了ACDC，证明了它可以在各种室内场景中稳健地生成专业质量的镜头，而无需机器人或电影摄影方面的专业知识。这些结果突出了具身AI代理在从开放词汇对话到现实世界自主空中电影摄影闭环方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.16176) | **Categories:** cs.RO

---

### [2] [PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models](https://arxiv.org/abs/2509.15607)
*Ruiqi Wang, Dezhong Zhao, Ziqin Yuan, Tianyu Shao, Guohua Chen, Dominic Kao, Sungeun Hong, Byung-Cheol Min*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for teaching robots complex behaviors without reward engineering. However, its effectiveness is often limited by two critical challenges: the reliance on extensive human input and the inherent difficulties in resolving query ambiguity and credit assignment during reward learning. In this paper, we introduce PRIMT, a PbRL framework designed to overcome these challenges by leveraging foundation models (FMs) for multimodal synthetic feedback and trajectory synthesis. Unlike prior approaches that rely on single-modality FM evaluations, PRIMT employs a hierarchical neuro-symbolic fusion strategy, integrating the complementary strengths of large language models and vision-language models in evaluating robot behaviors for more reliable and comprehensive feedback. PRIMT also incorporates foresight trajectory generation, which reduces early-stage query ambiguity by warm-starting the trajectory buffer with bootstrapped samples, and hindsight trajectory augmentation, which enables counterfactual reasoning with a causal auxiliary loss to improve credit assignment. We evaluate PRIMT on 2 locomotion and 6 manipulation tasks on various benchmarks, demonstrating superior performance over FM-based and scripted baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15607) | **Categories:** cs.RO

---

### [3] [DIPP: Discriminative Impact Point Predictor for Catching Diverse In-Flight Objects](https://arxiv.org/abs/2509.15254)
*Ngoc Huy Nguyen, Kazuki Shibata, Takamitsu Matsubara*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this study, we address the problem of in-flight object catching using a quadruped robot with a basket. Our objective is to accurately predict the impact point, defined as the object's landing position. This task poses two key challenges: the absence of public datasets capturing diverse objects under unsteady aerodynamics, which are essential for training reliable predictors; and the difficulty of accurate early-stage impact point prediction when trajectories appear similar across objects. To overcome these issues, we construct a real-world dataset of 8,000 trajectories from 20 objects, providing a foundation for advancing in-flight object catching under complex aerodynamics. We then propose the Discriminative Impact Point Predictor (DIPP), consisting of two modules: (i) a Discriminative Feature Embedding (DFE) that separates trajectories by dynamics to enable early-stage discrimination and generalization, and (ii) an Impact Point Predictor (IPP) that estimates the impact point from these features. Two IPP variants are implemented: an Neural Acceleration Estimator (NAE)-based method that predicts trajectories and derives the impact point, and a Direct Point Estimator (DPE)-based method that directly outputs it. Experimental results show that our dataset is more diverse and complex than existing dataset, and that our method outperforms baselines on both 15 seen and 5 unseen objects. Furthermore, we show that improved early-stage prediction enhances catching success in simulation and demonstrate the effectiveness of our approach through real-world experiments. The demonstration is available at https://sites.google.com/view/robot-catching-2025.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15254) | **Categories:** cs.RO

---

### [4] [CoReVLA: A Dual-Stage End-to-End Autonomous Driving Framework for Long-Tail Scenarios via Collect-and-Refine](https://arxiv.org/abs/2509.15968)
*Shiyu Fang, Yiming Cui, Haoyang Liang, Chen Lv, Peng Hang, Jian Sun*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous Driving (AD) systems have made notable progress, but their performance in long-tail, safety-critical scenarios remains limited. These rare cases contribute a disproportionate number of accidents. Vision-Language Action (VLA) models have strong reasoning abilities and offer a potential solution, but their effectiveness is limited by the lack of high-quality data and inefficient learning in such conditions. To address these challenges, we propose CoReVLA, a continual learning end-to-end autonomous driving framework that improves the performance in long-tail scenarios through a dual-stage process of data Collection and behavior Refinement. First, the model is jointly fine-tuned on a mixture of open-source driving QA datasets, allowing it to acquire a foundational understanding of driving scenarios. Next, CoReVLA is deployed within the Cave Automatic Virtual Environment (CAVE) simulation platform, where driver takeover data is collected from real-time interactions. Each takeover indicates a long-tail scenario that CoReVLA fails to handle reliably. Finally, the model is refined via Direct Preference Optimization (DPO), allowing it to learn directly from human preferences and thereby avoid reward hacking caused by manually designed rewards. Extensive open-loop and closed-loop experiments demonstrate that the proposed CoReVLA model can accurately perceive driving scenarios and make appropriate decisions. On the Bench2Drive benchmark, CoReVLA achieves a Driving Score (DS) of 72.18 and a Success Rate (SR) of 50%, outperforming state-of-the-art methods by 7.96 DS and 15% SR under long-tail, safety-critical scenarios. Furthermore, case studies demonstrate the model's ability to continually improve its performance in similar failure-prone scenarios by leveraging past takeover experiences. All codea and preprocessed datasets are available at: https://github.com/FanGShiYuu/CoReVLA

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15968) | **Categories:** cs.RO, cs.CV

---

### [5] [Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning](https://arxiv.org/abs/2509.15443)
*Xingyu Chen, Hanyu Wu, Sikai Wu, Mingliang Zhou, Diyun Xiang, Haodong Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human-to-humanoid imitation learning aims to learn a humanoid whole-body controller from human motion. Motion retargeting is a crucial step in enabling robots to acquire reference trajectories when exploring locomotion skills. However, current methods focus on motion retargeting frame by frame, which lacks scalability. Could we directly convert large-scale human motion into robot-executable motion through a more efficient approach? To address this issue, we propose Implicit Kinodynamic Motion Retargeting (IKMR), a novel efficient and scalable retargeting framework that considers both kinematics and dynamics. In kinematics, IKMR pretrains motion topology feature representation and a dual encoder-decoder architecture to learn a motion domain mapping. In dynamics, IKMR integrates imitation learning with the motion retargeting network to refine motion into physically feasible trajectories. After fine-tuning using the tracking results, IKMR can achieve large-scale physically feasible motion retargeting in real time, and a whole-body controller could be directly trained and deployed for tracking its retargeted trajectories. We conduct our experiments both in the simulator and the real robot on a full-size humanoid robot. Extensive experiments and evaluation results verify the effectiveness of our proposed framework.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15443) | **Categories:** cs.RO, cs.AI

---

### [6] [Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios](https://arxiv.org/abs/2509.15582)
*Yuting Zeng, Zhiwen Zheng, You Zhou, JiaLing Xiao, Yongbin Yu, Manping Fan, Bo Gong, Liyong Ren*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes a momentum-constrained hybrid heuristic trajectory optimization framework (MHHTOF) tailored for assistive navigation in visually impaired scenarios, integrating trajectory sampling generation, optimization and evaluation with residual-enhanced deep reinforcement learning (DRL). In the first stage, heuristic trajectory sampling cluster (HTSC) is generated in the Frenet coordinate system using third-order interpolation with fifth-order polynomials and momentum-constrained trajectory optimization (MTO) constraints to ensure smoothness and feasibility. After first stage cost evaluation, the second stage leverages a residual-enhanced actor-critic network with LSTM-based temporal feature modeling to adaptively refine trajectory selection in the Cartesian coordinate system. A dual-stage cost modeling mechanism (DCMM) with weight transfer aligns semantic priorities across stages, supporting human-centered optimization. Experimental results demonstrate that the proposed LSTM-ResB-PPO achieves significantly faster convergence, attaining stable policy performance in approximately half the training iterations required by the PPO baseline, while simultaneously enhancing both reward outcomes and training stability. Compared to baseline method, the selected model reduces average cost and cost variance by 30.3% and 53.3%, and lowers ego and obstacle risks by over 77%. These findings validate the framework's effectiveness in enhancing robustness, safety, and real-time feasibility in complex assistive planning tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15582) | **Categories:** cs.RO, cs.AI

---

### [7] [SMART: Scalable Multi-Agent Reasoning and Trajectory Planning in Dense Environments](https://arxiv.org/abs/2509.15737)
*Heye Huang, Yibin Yang, Wang Chen, Tiantian Chen, Xiaopeng Li, Sikai Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-vehicle trajectory planning is a non-convex problem that becomes increasingly difficult in dense environments due to the rapid growth of collision constraints. Efficient exploration of feasible behaviors and resolution of tight interactions are essential for real-time, large-scale coordination. This paper introduces SMART, Scalable Multi-Agent Reasoning and Trajectory Planning, a hierarchical framework that combines priority-based search with distributed optimization to achieve efficient and feasible multi-vehicle planning. The upper layer explores diverse interaction modes using reinforcement learning-based priority estimation and large-step hybrid A* search, while the lower layer refines solutions via parallelizable convex optimization. By partitioning space among neighboring vehicles and constructing robust feasible corridors, the method decouples the joint non-convex problem into convex subproblems solved efficiently in parallel. This design alleviates the step-size trade-off while ensuring kinematic feasibility and collision avoidance. Experiments show that SMART consistently outperforms baselines. On 50 m x 50 m maps, it sustains over 90% success within 1 s up to 25 vehicles, while baselines often drop below 50%. On 100 m x 100 m maps, SMART achieves above 95% success up to 50 vehicles and remains feasible up to 90 vehicles, with runtimes more than an order of magnitude faster than optimization-only approaches. Built on vehicle-to-everything communication, SMART incorporates vehicle-infrastructure cooperation through roadside sensing and agent coordination, improving scalability and safety. Real-world experiments further validate this design, achieving planning times as low as 0.014 s while preserving cooperative behaviors.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15737) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [8] [An MPC framework for efficient navigation of mobile robots in cluttered environments](https://arxiv.org/abs/2509.15917)
*Johannes Köhler, Daniel Zhang, Raffaele Soloperto, Andrea Carron, Melanie Zeilinger*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a model predictive control (MPC) framework for efficient navigation of mobile robots in cluttered environments. The proposed approach integrates a finite-segment shortest path planner into the finite-horizon trajectory optimization of the MPC. This formulation ensures convergence to dynamically selected targets and guarantees collision avoidance, even under general nonlinear dynamics and cluttered environments. The approach is validated through hardware experiments on a small ground robot, where a human operator dynamically assigns target locations. The robot successfully navigated through complex environments and reached new targets within 2-3 seconds.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.15917) | **Categories:** cs.RO, cs.SY, eess.SY, math.OC

---

### [9] [Defining and Monitoring Complex Robot Activities via LLMs and Symbolic Reasoning](https://arxiv.org/abs/2509.16006)
*Francesco Argenziano, Elena Umili, Francesco Leotta, Daniele Nardi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent years have witnessed a growing interest in automating labor-intensive and complex activities, i.e., those consisting of multiple atomic tasks, by deploying robots in dynamic and unpredictable environments such as industrial and agricultural settings. A key characteristic of these contexts is that activities are not predefined: while they involve a limited set of possible tasks, their combinations may vary depending on the situation. Moreover, despite recent advances in robotics, the ability for humans to monitor the progress of high-level activities - in terms of past, present, and future actions - remains fundamental to ensure the correct execution of safety-critical processes. In this paper, we introduce a general architecture that integrates Large Language Models (LLMs) with automated planning, enabling humans to specify high-level activities (also referred to as processes) using natural language, and to monitor their execution by querying a robot. We also present an implementation of this architecture using state-of-the-art components and quantitatively evaluate the approach in a real-world precision agriculture scenario.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.16006) | **Categories:** cs.RO, cs.HC

---
