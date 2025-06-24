# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-25

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (5)](#cs-cv)
- [cs.DL (1)](#cs-dl)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Beyond Syntax: Action Semantics Learning for App Agents](https://arxiv.org/abs/2506.17697)
*Bohan Tang, Dezhao Luo, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The advent of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with closed LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. With this insight, ASL employs a novel SEmantic Estimator (SEE) to compute a semantic reward to train the App agents in generating actions aligned with the semantics of ground truth actions, even when the syntactic forms differ. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments on offline and online smartphone App operation benchmarks show that ASL significantly improves the accuracy and generalisation of App agents over existing methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17697) | **Categories:** cs.AI

---

### [2] [AnyMAC: Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction](https://arxiv.org/abs/2506.17784)
*Song Wang, Zhen Tan, Zihan Chen, Shuang Zhou, Tianlong Chen, Jundong Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent progress in large language model (LLM)-based multi-agent collaboration highlights the power of structured communication in enabling collective intelligence. However, existing methods largely rely on static or graph-based inter-agent topologies, lacking the potential adaptability and flexibility in communication. In this work, we propose a new framework that rethinks multi-agent coordination through a sequential structure rather than a graph structure, offering a significantly larger topology space for multi-agent communication. Our method focuses on two key directions: (1) Next-Agent Prediction, which selects the most suitable agent role at each step, and (2) Next-Context Selection (NCS), which enables each agent to selectively access relevant information from any previous step. Together, these components construct task-adaptive communication pipelines that support both role flexibility and global information flow. Extensive evaluations across multiple benchmarks demonstrate that our approach achieves superior performance while substantially reducing communication overhead.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17784) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark For Driving](https://arxiv.org/abs/2506.17590)
*Mihir Godbole, Xiangbo Gao, Zhengzhong Tu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Understanding the short-term motion of vulnerable road users (VRUs) like pedestrians and cyclists is critical for safe autonomous driving, especially in urban scenarios with ambiguous or high-risk behaviors. While vision-language models (VLMs) have enabled open-vocabulary perception, their utility for fine-grained intent reasoning remains underexplored. Notably, no existing benchmark evaluates multi-class intent prediction in safety-critical situations, To address this gap, we introduce DRAMA-X, a fine-grained benchmark constructed from the DRAMA dataset via an automated annotation pipeline. DRAMA-X contains 5,686 accident-prone frames labeled with object bounding boxes, a nine-class directional intent taxonomy, binary risk scores, expert-generated action suggestions for the ego vehicle, and descriptive motion summaries. These annotations enable a structured evaluation of four interrelated tasks central to autonomous decision-making: object detection, intent prediction, risk assessment, and action suggestion. As a reference baseline, we propose SGG-Intent, a lightweight, training-free framework that mirrors the ego vehicle's reasoning pipeline. It sequentially generates a scene graph from visual input using VLM-backed detectors, infers intent, assesses risk, and recommends an action using a compositional reasoning stage powered by a large language model. We evaluate a range of recent VLMs, comparing performance across all four DRAMA-X tasks. Our experiments demonstrate that scene-graph-based reasoning enhances intent prediction and risk assessment, especially when contextual cues are explicitly modeled.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17590) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [DreamJourney: Perpetual View Generation with Video Diffusion Models](https://arxiv.org/abs/2506.17705)
*Bo Pan, Yang Chen, Yingwei Pan, Ting Yao, Wei Chen, Tao Mei*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Perpetual view generation aims to synthesize a long-term video corresponding to an arbitrary camera trajectory solely from a single input image. Recent methods commonly utilize a pre-trained text-to-image diffusion model to synthesize new content of previously unseen regions along camera movement. However, the underlying 2D diffusion model lacks 3D awareness and results in distorted artifacts. Moreover, they are limited to generating views of static 3D scenes, neglecting to capture object movements within the dynamic 4D world. To alleviate these issues, we present DreamJourney, a two-stage framework that leverages the world simulation capacity of video diffusion models to trigger a new perpetual scene view generation task with both camera movements and object dynamics. Specifically, in stage I, DreamJourney first lifts the input image to 3D point cloud and renders a sequence of partial images from a specific camera trajectory. A video diffusion model is then utilized as generative prior to complete the missing regions and enhance visual coherence across the sequence, producing a cross-view consistent video adheres to the 3D scene and camera trajectory. Meanwhile, we introduce two simple yet effective strategies (early stopping and view padding) to further stabilize the generation process and improve visual quality. Next, in stage II, DreamJourney leverages a multimodal large language model to produce a text prompt describing object movements in current view, and uses video diffusion model to animate current view with object movements. Stage I and II are repeated recurrently, enabling perpetual dynamic scene view generation. Extensive experiments demonstrate the superiority of our DreamJourney over state-of-the-art methods both quantitatively and qualitatively. Our project page: https://dream-journey.vercel.app.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17705) | **Categories:** cs.CV

---

### [3] [Scene-R1: Video-Grounded Large Language Models for 3D Scene Reasoning without 3D Annotations](https://arxiv.org/abs/2506.17545)
*Zhihao Yuan, Shuyi Jiang, Chun-Mei Feng, Yaolun Zhang, Shuguang Cui, Zhen Li, Na Zhao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Currently, utilizing large language models to understand the 3D world is becoming popular. Yet existing 3D-aware LLMs act as black boxes: they output bounding boxes or textual answers without revealing how those decisions are made, and they still rely on pre-trained 3D detectors to supply object proposals. We introduce Scene-R1, a video-grounded framework that learns to reason about 3D scenes without any point-wise 3D instance supervision by pairing reinforcement-learning-driven reasoning with a two-stage grounding pipeline. In the temporal grounding stage, we explicitly reason about the video and select the video snippets most relevant to an open-ended query. In the subsequent image grounding stage, we analyze the image and predict the 2D bounding box. After that, we track the object using SAM2 to produce pixel-accurate masks in RGB frames, and project them back into 3D, thereby eliminating the need for 3D detector-based proposals while capturing fine geometry and material cues. Scene-R1 can also adapt to the 3D visual question answering task to answer free-form questions directly from video. Our training pipeline only needs task-level 2D boxes or textual labels without dense 3D point-wise labels. Scene-R1 surpasses existing open-vocabulary baselines on multiple datasets, while delivering transparent, step-by-step rationales. These results show that reinforcement-learning-based reasoning combined with RGB-D video alone offers a practical, annotation-efficient route to trustworthy 3D scene understanding.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17545) | **Categories:** cs.CV

---

### [4] [CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning](https://arxiv.org/abs/2506.17629)
*Kailing Li, Qi'ao Xu, Tianwen Qian, Yuqian Fu, Yang Jiao, Xiaoling Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at https://github.com/Teacher-Tom/CLiViS.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17629) | **Categories:** cs.CV, cs.AI, cs.CL

---

### [5] [PhysID: Physics-based Interactive Dynamics from a Single-view Image](https://arxiv.org/abs/2506.17746)
*Sourabh Vasant Gothe, Ayon Chattopadhyay, Gunturi Venkata Sai Phani Kiran, Pratik, Vibhav Agarwal, Jayesh Rajkumar Vachhani, Sourav Ghosh, Parameswaranath VM, Barath Raj KR*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Transforming static images into interactive experiences remains a challenging task in computer vision. Tackling this challenge holds the potential to elevate mobile user experiences, notably through interactive and AR/VR applications. Current approaches aim to achieve this either using pre-recorded video responses or requiring multi-view images as input. In this paper, we present PhysID, that streamlines the creation of physics-based interactive dynamics from a single-view image by leveraging large generative models for 3D mesh generation and physical property prediction. This significantly reduces the expertise required for engineering-intensive tasks like 3D modeling and intrinsic property calibration, enabling the process to be scaled with minimal manual intervention. We integrate an on-device physics-based engine for physically plausible real-time rendering with user interactions. PhysID represents a leap forward in mobile-based interactive dynamics, offering real-time, non-deterministic interactions and user-personalization with efficient on-device memory consumption. Experiments evaluate the zero-shot capabilities of various Multimodal Large Language Models (MLLMs) on diverse tasks and the performance of 3D reconstruction models. These results demonstrate the cohesive functioning of all modules within the end-to-end framework, contributing to its effectiveness.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17746) | **Categories:** cs.CV

---


## cs.DL [cs.DL]
### [1] [Mapping the Evolution of Research Contributions using KnoVo](https://arxiv.org/abs/2506.17508)
*Sajratul Y. Rubaiat, Syed N. Sakib, Hasan M. Jamil*

Main category: cs.DL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents KnoVo (Knowledge Evolution), an intelligent framework designed for quantifying and analyzing the evolution of research novelty in the scientific literature. Moving beyond traditional citation analysis, which primarily measures impact, KnoVo determines a paper's novelty relative to both prior and subsequent work within its multilayered citation network. Given a target paper's abstract, KnoVo utilizes Large Language Models (LLMs) to dynamically extract dimensions of comparison (e.g., methodology, application, dataset). The target paper is then compared to related publications along these same extracted dimensions. This comparative analysis, inspired by tournament selection, yields quantitative novelty scores reflecting the relative improvement, equivalence, or inferiority of the target paper in specific aspects. By aggregating these scores and visualizing their progression, for instance, through dynamic evolution graphs and comparative radar charts, KnoVo facilitates researchers not only to assess originality and identify similar work, but also to track knowledge evolution along specific research dimensions, uncover research gaps, and explore cross-disciplinary connections. We demonstrate these capabilities through a detailed analysis of 20 diverse papers from multiple scientific fields and report on the performance of various open-source LLMs within the KnoVo framework.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17508) | **Categories:** cs.DL, cs.AI, cs.DB, cs.ET, cs.IR

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [AutomataGPT: Forecasting and Ruleset Inference for Two-Dimensional Cellular Automata](https://arxiv.org/abs/2506.17333)
*Jaime A. Berkovich, Noah S. David, Markus J. Buehler*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Cellular automata (CA) provide a minimal formalism for investigating how simple local interactions generate rich spatiotemporal behavior in domains as diverse as traffic flow, ecology, tissue morphogenesis and crystal growth. However, automatically discovering the local update rules for a given phenomenon and using them for quantitative prediction remains challenging. Here we present AutomataGPT, a decoder-only transformer pretrained on around 1 million simulated trajectories that span 100 distinct two-dimensional binary deterministic CA rules on toroidal grids. When evaluated on previously unseen rules drawn from the same CA family, AutomataGPT attains 98.5% perfect one-step forecasts and reconstructs the governing update rule with up to 96% functional (application) accuracy and 82% exact rule-matrix match. These results demonstrate that large-scale pretraining over wider regions of rule space yields substantial generalization in both the forward (state forecasting) and inverse (rule inference) problems, without hand-crafted priors. By showing that transformer models can faithfully infer and execute CA dynamics from data alone, our work lays the groundwork for abstracting real-world dynamical phenomena into data-efficient CA surrogates, opening avenues in biology, tissue engineering, physics and AI-driven scientific discovery.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17333) | **Categories:** cs.LG, cond-mat.dis-nn, cond-mat.mtrl-sci, q-bio.QM

---

### [2] [LLM-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting](https://arxiv.org/abs/2506.17631)
*Zesen Wang, Yonggang Li, Lijuan Lan*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17631) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Reflective VLM Planning for Dual-Arm Desktop Cleaning: Bridging Open-Vocabulary Perception and Precise Manipulation](https://arxiv.org/abs/2506.17328)
*Yufan Liu, Yi Wu, Gweneth Ge, Haoliang Cheng, Rui Liu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Desktop cleaning demands open-vocabulary recognition and precise manipulation for heterogeneous debris. We propose a hierarchical framework integrating reflective Vision-Language Model (VLM) planning with dual-arm execution via structured scene representation. Grounded-SAM2 facilitates open-vocabulary detection, while a memory-augmented VLM generates, critiques, and revises manipulation sequences. These sequences are converted into parametric trajectories for five primitives executed by coordinated Franka arms. Evaluated in simulated scenarios, our system achieving 87.2% task completion, a 28.8% improvement over static VLM and 36.2% over single-arm baselines. Structured memory integration proves crucial for robust, generalizable manipulation while maintaining real-time control performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17328) | **Categories:** cs.RO

---

### [2] [General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting](https://arxiv.org/abs/2506.17462)
*Bernard Lange, Anil Yildiz, Mansur Arief, Shehryar Khattak, Mykel Kochenderfer, Georgios Georgakis*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Developing general-purpose navigation policies for unknown environments remains a core challenge in robotics. Most existing systems rely on task-specific neural networks and fixed data flows, limiting generalizability. Large Vision-Language Models (LVLMs) offer a promising alternative by embedding human-like knowledge suitable for reasoning and planning. Yet, prior LVLM-robot integrations typically depend on pre-mapped spaces, hard-coded representations, and myopic exploration. We introduce the Agentic Robotic Navigation Architecture (ARNA), a general-purpose navigation framework that equips an LVLM-based agent with a library of perception, reasoning, and navigation tools available within modern robotic stacks. At runtime, the agent autonomously defines and executes task-specific workflows that iteratively query the robotic modules, reason over multimodal inputs, and select appropriate navigation actions. This approach enables robust navigation and reasoning in previously unmapped environments, providing a new perspective on robotic stack design. Evaluated in Habitat Lab on the HM-EQA benchmark, ARNA achieves state-of-the-art performance, demonstrating effective exploration, navigation, and embodied question answering without relying on handcrafted plans, fixed input representations, or pre-existing maps.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17462) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [3] [Distilling On-device Language Models for Robot Planning with Minimal Human Intervention](https://arxiv.org/abs/2506.17486)
*Zachary Ravichandran, Ignacio Hounie, Fernando Cladera, Alejandro Ribeiro, George J. Pappas, Vijay Kumar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) provide robots with powerful contextual reasoning abilities and a natural human interface. Yet, current LLM-enabled robots typically depend on cloud-hosted models, limiting their usability in environments with unreliable communication infrastructure, such as outdoor or industrial settings. We present PRISM, a framework for distilling small language model (SLM)-enabled robot planners that run on-device with minimal human supervision. Starting from an existing LLM-enabled planner, PRISM automatically synthesizes diverse tasks and environments, elicits plans from the LLM, and uses this synthetic dataset to distill a compact SLM as a drop-in replacement of the source model. We apply PRISM to three LLM-enabled planners for mapping and exploration, manipulation, and household assistance, and we demonstrate that PRISM improves the performance of Llama-3.2-3B from 10-20% of GPT-4o's performance to over 93% - using only synthetic data. We further demonstrate that the distilled planners generalize across heterogeneous robotic platforms (ground and aerial) and diverse environments (indoor and outdoor). We release all software, trained models, and datasets at https://zacravichandran.github.io/PRISM.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17486) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [4] [RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models](https://arxiv.org/abs/2506.17811)
*Jacky Kwok, Christopher Agia, Rohan Sinha, Matt Foutter, Shulu Li, Ion Stoica, Azalia Mirhoseini, Marco Pavone*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in visuomotor control, yet ensuring their robustness in unstructured real-world environments remains a persistent challenge. In this paper, we investigate test-time scaling through the lens of sampling and verification as means to enhance the robustness and generalization of VLAs. We first demonstrate that the relationship between action error and the number of generated samples follows an exponentiated power law across a range of VLAs, indicating the existence of inference-time scaling laws. Building on these insights, we introduce RoboMonkey, a test-time scaling framework for VLAs. At deployment, RoboMonkey samples a small set of actions from a VLA, applies Gaussian perturbation and majority voting to construct an action proposal distribution, and then uses a Vision Language Model (VLM)-based verifier to select the optimal action. We propose a synthetic data generation pipeline for training such VLM-based action verifiers, and demonstrate that scaling the synthetic dataset consistently improves verification and downstream accuracy. Through extensive simulated and hardware experiments, we show that pairing existing VLAs with RoboMonkey yields significant performance gains, achieving a 25% absolute improvement on out-of-distribution tasks and 8% on in-distribution tasks. Additionally, when adapting to new robot setups, we show that fine-tuning both VLAs and action verifiers yields a 7% performance increase compared to fine-tuning VLAs alone.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17811) | **Categories:** cs.RO, cs.AI, cs.SY, eess.SY

---

### [5] [GeNIE: A Generalizable Navigation System for In-the-Wild Environments](https://arxiv.org/abs/2506.17960)
*Jiaming Wang, Diwen Liu, Jizhuo Chen, Jiaxuan Da, Nuowen Qian, Tram Minh Man, Harold Soh*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reliable navigation in unstructured, real-world environments remains a significant challenge for embodied agents, especially when operating across diverse terrains, weather conditions, and sensor configurations. In this paper, we introduce GeNIE (Generalizable Navigation System for In-the-Wild Environments), a robust navigation framework designed for global deployment. GeNIE integrates a generalizable traversability prediction model built on SAM2 with a novel path fusion strategy that enhances planning stability in noisy and ambiguous settings. We deployed GeNIE in the Earth Rover Challenge (ERC) at ICRA 2025, where it was evaluated across six countries spanning three continents. GeNIE took first place and achieved 79% of the maximum possible score, outperforming the second-best team by 17%, and completed the entire competition without a single human intervention. These results set a new benchmark for robust, generalizable outdoor robot navigation. We will release the codebase, pretrained model weights, and newly curated datasets to support future research in real-world navigation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.17960) | **Categories:** cs.RO, cs.AI

---

### [6] [RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation](https://arxiv.org/abs/2506.18088)
*Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, Yao Mu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.18088) | **Categories:** cs.RO, cs.AI, cs.CL, cs.CV, cs.MA

---

### [7] [Integrating LLMs and Digital Twins for Adaptive Multi-Robot Task Allocation in Construction](https://arxiv.org/abs/2506.18178)
*Min Deng, Bo Fu, Lingyao Li, Xi Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-robot systems are emerging as a promising solution to the growing demand for productivity, safety, and adaptability across industrial sectors. However, effectively coordinating multiple robots in dynamic and uncertain environments, such as construction sites, remains a challenge, particularly due to unpredictable factors like material delays, unexpected site conditions, and weather-induced disruptions. To address these challenges, this study proposes an adaptive task allocation framework that strategically leverages the synergistic potential of Digital Twins, Integer Programming (IP), and Large Language Models (LLMs). The multi-robot task allocation problem is formally defined and solved using an IP model that accounts for task dependencies, robot heterogeneity, scheduling constraints, and re-planning requirements. A mechanism for narrative-driven schedule adaptation is introduced, in which unstructured natural language inputs are interpreted by an LLM, and optimization constraints are autonomously updated, enabling human-in-the-loop flexibility without manual coding. A digital twin-based system has been developed to enable real-time synchronization between physical operations and their digital representations. This closed-loop feedback framework ensures that the system remains dynamic and responsive to ongoing changes on site. A case study demonstrates both the computational efficiency of the optimization algorithm and the reasoning performance of several LLMs, with top-performing models achieving over 97% accuracy in constraint and parameter extraction. The results confirm the practicality, adaptability, and cross-domain applicability of the proposed methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.18178) | **Categories:** cs.RO

---

### [8] [Learning Approach to Efficient Vision-based Active Tracking of a Flying Target by an Unmanned Aerial Vehicle](https://arxiv.org/abs/2506.18264)
*Jagadeswara PKV Pothuri, Aditya Bhatt, Prajit KrisshnaKumar, Manaswin Oddiraju, Souma Chowdhury*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous tracking of flying aerial objects has important civilian and defense applications, ranging from search and rescue to counter-unmanned aerial systems (counter-UAS). Ground based tracking requires setting up infrastructure, could be range limited, and may not be feasible in remote areas, crowded cities or in dense vegetation areas. Vision based active tracking of aerial objects from another airborne vehicle, e.g., a chaser unmanned aerial vehicle (UAV), promises to fill this important gap, along with serving aerial coordination use cases. Vision-based active tracking by a UAV entails solving two coupled problems: 1) compute-efficient and accurate (target) object detection and target state estimation; and 2) maneuver decisions to ensure that the target remains in the field of view in the future time-steps and favorably positioned for continued detection. As a solution to the first problem, this paper presents a novel integration of standard deep learning based architectures with Kernelized Correlation Filter (KCF) to achieve compute-efficient object detection without compromising accuracy, unlike standalone learning or filtering approaches. The proposed perception framework is validated using a lab-scale setup. For the second problem, to obviate the linearity assumptions and background variations limiting effectiveness of the traditional controllers, we present the use of reinforcement learning to train a neuro-controller for fast computation of velocity maneuvers. New state space, action space and reward formulations are developed for this purpose, and training is performed in simulation using AirSim. The trained model is also tested in AirSim with respect to complex target maneuvers, and is found to outperform a baseline PID control in terms of tracking up-time and average distance maintained (from the target) during tracking.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.18264) | **Categories:** cs.RO

---

### [9] [NOVA: Navigation via Object-Centric Visual Autonomy for High-Speed Target Tracking in Unstructured GPS-Denied Environments](https://arxiv.org/abs/2506.18689)
*Alessandro Saviolo, Giuseppe Loianno*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous aerial target tracking in unstructured and GPS-denied environments remains a fundamental challenge in robotics. Many existing methods rely on motion capture systems, pre-mapped scenes, or feature-based localization to ensure safety and control, limiting their deployment in real-world conditions. We introduce NOVA, a fully onboard, object-centric framework that enables robust target tracking and collision-aware navigation using only a stereo camera and an IMU. Rather than constructing a global map or relying on absolute localization, NOVA formulates perception, estimation, and control entirely in the target's reference frame. A tightly integrated stack combines a lightweight object detector with stereo depth completion, followed by histogram-based filtering to infer robust target distances under occlusion and noise. These measurements feed a visual-inertial state estimator that recovers the full 6-DoF pose of the robot relative to the target. A nonlinear model predictive controller (NMPC) plans dynamically feasible trajectories in the target frame. To ensure safety, high-order control barrier functions are constructed online from a compact set of high-risk collision points extracted from depth, enabling real-time obstacle avoidance without maps or dense representations. We validate NOVA across challenging real-world scenarios, including urban mazes, forest trails, and repeated transitions through buildings with intermittent GPS loss and severe lighting changes that disrupt feature-based localization. Each experiment is repeated multiple times under similar conditions to assess resilience, showing consistent and reliable performance. NOVA achieves agile target following at speeds exceeding 50 km/h. These results show that high-speed vision-based tracking is possible in the wild using only onboard sensing, with no reliance on external localization or environment assumptions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.18689) | **Categories:** cs.RO, cs.AI

---
