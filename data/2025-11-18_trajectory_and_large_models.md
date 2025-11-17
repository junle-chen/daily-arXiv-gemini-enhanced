# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-18

## 目录

- [人工智能 (Artificial Intelligence) (5)](#cs-ai)
- [计算语言学 (Computation and Language) (4)](#cs-cl)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [HARNESS: Human-Agent Risk Navigation and Event Safety System for Proactive Hazard Forecasting in High-Risk DOE Environments](https://arxiv.org/abs/2511.10810)
*Ran Elgedawy, Sanjay Das, Ethan Seefried, Gavin Wiggins, Ryan Burchfield, Dana Hewit, Sudarshan Srinivasan, Todd Thomas, Prasanna Balaprakash, Tirthankar Ghosal*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Operational safety at mission-critical work sites is a top priority given the complex and hazardous nature of daily tasks. This paper presents the Human-Agent Risk Navigation and Event Safety System (HARNESS), a modular AI framework designed to forecast hazardous events and analyze operational risks in U.S. Department of Energy (DOE) environments. HARNESS integrates Large Language Models (LLMs) with structured work data, historical event retrieval, and risk analysis to proactively identify potential hazards. A human-in-the-loop mechanism allows subject matter experts (SMEs) to refine predictions, creating an adaptive learning loop that enhances performance over time. By combining SME collaboration with iterative agentic reasoning, HARNESS improves the reliability and efficiency of predictive safety systems. Preliminary deployment shows promising results, with future work focusing on quantitative evaluation of accuracy, SME agreement, and decision latency reduction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10810) | **Categories:** cs.AI

---

### [2] [UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios](https://arxiv.org/abs/2511.11252)
*Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at https://github.com/maferrag/UAVBench

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11252) | **Categories:** cs.AI

---

### [3] [Advanced Tool for Traffic Crash Analysis: An AI-Driven Multi-Agent Approach to Pre-Crash Reconstruction](https://arxiv.org/abs/2511.10853)
*Gerui Xu, Boyou Chen, Huizhong Guo, Dave LeBlanc, Ananna Ahmed, Zhaonan Sun, Shan Bao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traffic collision reconstruction traditionally relies on human expertise, often yielding inconsistent results when analyzing incomplete multimodal data. This study develops a multi-agent AI framework that reconstructs pre-crash scenarios and infers vehicle behaviors from fragmented collision data. We present a two-phase collaborative framework combining reconstruction and reasoning phases. The system processes 277 rear-end lead vehicle deceleration (LVD) collisions from the Crash Investigation Sampling System, integrating textual crash reports, structured tabular data, and visual scene diagrams. Phase I generates natural-language crash reconstructions from multimodal inputs. Phase II performs in-depth crash reasoning by combining these reconstructions with temporal Event Data Recorder (EDR).For validation, we applied it to all LVD cases, focusing on a subset of 39 complex crashes where multiple EDR records per collision introduced ambiguity (e.g., due to missing or conflicting data).The evaluation of the 39 LVD crash cases revealed our framework achieved perfect accuracy across all test cases, successfully identifying both the most relevant EDR event and correctly distinguishing striking versus struck vehicles, surpassing the 92% accuracy achieved by human researchers on the same challenging dataset. The system maintained robust performance even when processing incomplete data, including missing or erroneous EDR records and ambiguous scene diagrams. This study demonstrates superior AI capabilities in processing heterogeneous collision data, providing unprecedented precision in reconstructing impact dynamics and characterizing pre-crash behaviors.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10853) | **Categories:** cs.AI, cs.HC

---

### [4] [Autonomous Vehicle Path Planning by Searching With Differentiable Simulation](https://arxiv.org/abs/2511.11043)
*Asen Nachkov, Jan-Nico Zaech, Danda Pani Paudel, Xi Wang, Luc Van Gool*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planning allows an agent to safely refine its actions before executing them in the real world. In autonomous driving, this is crucial to avoid collisions and navigate in complex, dense traffic scenarios. One way to plan is to search for the best action sequence. However, this is challenging when all necessary components - policy, next-state predictor, and critic - have to be learned. Here we propose Differentiable Simulation for Search (DSS), a framework that leverages the differentiable simulator Waymax as both a next state predictor and a critic. It relies on the simulator's hardcoded dynamics, making state predictions highly accurate, while utilizing the simulator's differentiability to effectively search across action sequences. Our DSS agent optimizes its actions using gradient descent over imagined future trajectories. We show experimentally that DSS - the combination of planning gradients and stochastic search - significantly improves tracking and path planning accuracy compared to sequence prediction, imitation learning, model-free RL, and other planning methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11043) | **Categories:** cs.AI, cs.RO

---

### [5] [RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](https://arxiv.org/abs/2511.11323)
*Yitian Kou, Yihe Gu, Chen Zhou, DanDan Zhu, Shuguang Kuai*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Navigating human-populated environments without causing discomfort is a critical capability for socially-aware agents. While rule-based approaches offer interpretability through predefined psychological principles, they often lack generalizability and flexibility. Conversely, data-driven methods can learn complex behaviors from large-scale datasets, but are typically inefficient, opaque, and difficult to align with human intuitions. To bridge this gap, we propose RLSLM, a hybrid Reinforcement Learning framework that integrates a rule-based Social Locomotion Model, grounded in empirical behavioral experiments, into the reward function of a reinforcement learning framework. The social locomotion model generates an orientation-sensitive social comfort field that quantifies human comfort across space, enabling socially aligned navigation policies with minimal training. RLSLM then jointly optimizes mechanical energy and social comfort, allowing agents to avoid intrusions into personal or group space. A human-agent interaction experiment using an immersive VR-based setup demonstrates that RLSLM outperforms state-of-the-art rule-based models in user experience. Ablation and sensitivity analyses further show the model's significantly improved interpretability over conventional data-driven methods. This work presents a scalable, human-centered methodology that effectively integrates cognitive science and machine learning for real-world social navigation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11323) | **Categories:** cs.AI

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2511.10676)
*Shien Zhu, Samuel Bohl, Robin Oester, Gustavo Alonso*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Mixture-of-Experts (MoE) Large Language Models (LLMs) efficiently scale-up the model while keeping relatively low inference cost. As MoE models only activate part of the experts, related work has proposed expert prediction and caching methods to prefetch the experts for faster inference. However, existing approaches utilize the activations from the previous layer for prediction, incurring low accuracy and leave the first layer unoptimized. Applying complex layers or even training standalone networks for better prediction introduces high computation overhead. In this paper, we propose pre-attention expert prediction to achieve accurate and lightweight expert prefetching. The key insight is that some functions in LLMs are ranking-preserving, indicating that matching the ranking of selected experts using simple linear functions is possible. Therefore, we utilize the activations before the attention block in the same layer with 2 linear functions and ranking-aware loss to achieve accurate prediction, which also supports prefetching in the first layer. Our lightweight, pre-attention expert routers achieve 93.03% accuracy on DeepSeek V2 Lite, 94.69% on Qwen3-30B, and 97.62% on Phi-mini-MoE, showing about 15% improvement on absolute accuracy over the state-of-the-art methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10676) | **Categories:** cs.CL, cs.AI

---

### [2] [Unsupervised Cycle Detection in Agentic Applications](https://arxiv.org/abs/2511.10650)
*Felix George, Harshit Kumar, Divya Pathak, Kaustabha Ray, Mudit Verma, Pratibha Moogi*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Agentic applications powered by Large Language Models exhibit non-deterministic behaviors that can form hidden execution cycles, silently consuming resources without triggering explicit errors. Traditional observability platforms fail to detect these costly inefficiencies. We present an unsupervised cycle detection framework that combines structural and semantic analysis. Our approach first applies computationally efficient temporal call stack analysis to identify explicit loops and then leverages semantic similarity analysis to uncover subtle cycles characterized by redundant content generation. Evaluated on 1575 trajectories from a LangGraph-based stock market application, our hybrid approach achieves an F1 score of 0.72 (precision: 0.62, recall: 0.86), significantly outperforming individual structural (F1: 0.08) and semantic methods (F1: 0.28). While these results are encouraging, there remains substantial scope for improvement, and future work is needed to refine the approach and address its current limitations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10650) | **Categories:** cs.CL, cs.AI, cs.MA

---

### [3] [Data Analysis and Performance Evaluation of Simulation Deduction Based on LLMs](https://arxiv.org/abs/2511.10651)
*Shansi Zhang, Min Li*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Data analysis and performance evaluation of simulation deduction plays a pivotal role in modern warfare, which enables military personnel to gain invaluable insights into the potential effectiveness of different strategies, tactics, and operational plans. Traditional manual analysis approach is time-consuming and limited by human errors. To enhance efficiency and accuracy, large language models (LLMs) with strong analytical and inferencing capabilities can be employed. However, high-quality analysis reports with well-structured formatting cannot be obtained through a single instruction input to the LLM. To tackle this issue, we propose a method that first decomposes the complex task into several sub-tasks and designs effective system prompts and user prompts for each sub-task. Multi-round interactions with the LLM incorporating self-check and reflection are then conducted to enable structured data extraction as well as multi-step analysis and evaluation. Furthermore, custom tools are defined and invoked to generate figures and compute metrics. We also design multiple report templates, each tailored to a specific application and input data type, ensuring their adaptability across a variety of scenarios. Extensive evaluation results demonstrate that the reports generated by our method exhibit higher quality, therefore obtaining higher scores than the baseline method.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10651) | **Categories:** cs.CL, cs.AI

---

### [4] [Hybrid Quantum Transformer for Language Generation](https://arxiv.org/abs/2511.10653)
*Desheng Kong, Xiangshuo Cui, Jiaying Jin, Jing Xu, Donglin Wang*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Although quantum computing has been increasingly applied to replace classical computation, most existing quantum or hybrid models remain confined to simple tasks, with no successful application to large-scale natural language generation to date. In this work, we present the first hybrid quantum-classical large language model (LLM) for natural language generation, HyQuT, capable of performing coherent and context-aware dialogue. The proposed architecture integrates variational quantum circuits (VQCs) into the Transformer framework at both 8M and 150M parameter scales. Experimental results show that a minimal number of qubits (10 qubits with 80 quantum gates) can replace about 10% of the classical parameters in the 150M-parameter model, while achieving comparable convergence stability and generation quality. This study provides an early demonstration of the feasibility of integrating quantum computing to large-scale generative language models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10653) | **Categories:** cs.CL, cs.AI, quant-ph

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling](https://arxiv.org/abs/2511.10866)
*Seoik Jung, Taekyung Song, Yangro Lee, Sungjun Lee*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10866) | **Categories:** cs.CV, cs.AI

---

### [2] [AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning](https://arxiv.org/abs/2511.11025)
*Jirong Zha, Yuxuan Fan, Tianyu Zhang, Geng Chen, Yingfeng Chen, Chen Gao, Xinlei Chen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal Large Language Models (MLLMs) have shown promise in single-agent vision tasks, yet benchmarks for evaluating multi-agent collaborative perception remain scarce. This gap is critical, as multi-drone systems provide enhanced coverage, robustness, and collaboration compared to single-sensor setups. Existing multi-image benchmarks mainly target basic perception tasks using high-quality single-agent images, thus failing to evaluate MLLMs in more complex, egocentric collaborative scenarios, especially under real-world degraded perception conditions.To address these challenges, we introduce AirCopBench, the first comprehensive benchmark designed to evaluate MLLMs in embodied aerial collaborative perception under challenging perceptual conditions. AirCopBench includes 14.6k+ questions derived from both simulator and real-world data, spanning four key task dimensions: Scene Understanding, Object Understanding, Perception Assessment, and Collaborative Decision, across 14 task types. We construct the benchmark using data from challenging degraded-perception scenarios with annotated collaborative events, generating large-scale questions through model-, rule-, and human-based methods under rigorous quality control. Evaluations on 40 MLLMs show significant performance gaps in collaborative perception tasks, with the best model trailing humans by 24.38% on average and exhibiting inconsistent results across tasks. Fine-tuning experiments further confirm the feasibility of sim-to-real transfer in aerial collaborative perception and reasoning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11025) | **Categories:** cs.CV, cs.AI

---

### [3] [Stroke Modeling Enables Vectorized Character Generation with Large Vectorized Glyph Model](https://arxiv.org/abs/2511.11119)
*Xinyue Zhang, Haolong Li, Jiawei Ma, Chen Ye*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vectorized glyphs are widely used in poster design, network animation, art display, and various other fields due to their scalability and flexibility. In typography, they are often seen as special sequences composed of ordered strokes. This concept extends to the token sequence prediction abilities of large language models (LLMs), enabling vectorized character generation through stroke modeling. In this paper, we propose a novel Large Vectorized Glyph Model (LVGM) designed to generate vectorized Chinese glyphs by predicting the next stroke. Initially, we encode strokes into discrete latent variables called stroke embeddings. Subsequently, we train our LVGM via fine-tuning DeepSeek LLM by predicting the next stroke embedding. With limited strokes given, it can generate complete characters, semantically elegant words, and even unseen verses in vectorized form. Moreover, we release a new large-scale Chinese SVG dataset containing 907,267 samples based on strokes for dynamically vectorized glyph generation. Experimental results show that our model has scaling behaviors on data scales. Our generated vectorized glyphs have been validated by experts and relevant individuals.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11119) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Movement-Specific Analysis for FIM Score Classification Using Spatio-Temporal Deep Learning](https://arxiv.org/abs/2511.10713)
*Jun Masaki, Ariaki Higashi, Naoko Shinagawa, Kazuhiko Hirata, Yuichi Kurita, Akira Furui*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The functional independence measure (FIM) is widely used to evaluate patients' physical independence in activities of daily living. However, traditional FIM assessment imposes a significant burden on both patients and healthcare professionals. To address this challenge, we propose an automated FIM score estimation method that utilizes simple exercises different from the designated FIM assessment actions. Our approach employs a deep neural network architecture integrating a spatial-temporal graph convolutional network (ST-GCN), bidirectional long short-term memory (BiLSTM), and an attention mechanism to estimate FIM motor item scores. The model effectively captures long-term temporal dependencies and identifies key body-joint contributions through learned attention weights. We evaluated our method in a study of 277 rehabilitation patients, focusing on FIM transfer and locomotion items. Our approach successfully distinguishes between completely independent patients and those requiring assistance, achieving balanced accuracies of 70.09-78.79 % across different FIM items. Additionally, our analysis reveals specific movement patterns that serve as reliable predictors for particular FIM evaluation items.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10713) | **Categories:** cs.LG

---

### [2] [Toward Scalable Early Cancer Detection: Evaluating EHR-Based Predictive Models Against Traditional Screening Criteria](https://arxiv.org/abs/2511.11293)
*Jiheum Park, Chao Pang, Tristan Y. Lee, Jeong Yun Yang, Jacob Berkowitz, Alexander Z. Wei, Nicholas Tatonetti*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Current cancer screening guidelines cover only a few cancer types and rely on narrowly defined criteria such as age or a single risk factor like smoking history, to identify high-risk individuals. Predictive models using electronic health records (EHRs), which capture large-scale longitudinal patient-level health information, may provide a more effective tool for identifying high-risk groups by detecting subtle prediagnostic signals of cancer. Recent advances in large language and foundation models have further expanded this potential, yet evidence remains limited on how useful HER-based models are compared with traditional risk factors currently used in screening guidelines. We systematically evaluated the clinical utility of EHR-based predictive models against traditional risk factors, including gene mutations and family history of cancer, for identifying high-risk individuals across eight major cancers (breast, lung, colorectal, prostate, ovarian, liver, pancreatic, and stomach), using data from the All of Us Research Program, which integrates EHR, genomic, and survey data from over 865,000 participants. Even with a baseline modeling approach, EHR-based models achieved a 3- to 6-fold higher enrichment of true cancer cases among individuals identified as high risk compared with traditional risk factors alone, whether used as a standalone or complementary tool. The EHR foundation model, a state-of-the-art approach trained on comprehensive patient trajectories, further improved predictive performance across 26 cancer types, demonstrating the clinical potential of EHR-based predictive modeling to support more precise and scalable early detection strategies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11293) | **Categories:** cs.LG, q-bio.QM

---


## 机器人学 (Robotics) [cs.RO]
### [1] [MIGHTY: Hermite Spline-based Efficient Trajectory Planning](https://arxiv.org/abs/2511.10822)
*Kota Kondo, Yuwei Wu, Vijay Kumar, Jonathan P. How*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Hard-constraint trajectory planners often rely on commercial solvers and demand substantial computational resources. Existing soft-constraint methods achieve faster computation, but either (1) decouple spatial and temporal optimization or (2) restrict the search space. To overcome these limitations, we introduce MIGHTY, a Hermite spline-based planner that performs spatiotemporal optimization while fully leveraging the continuous search space of a spline. In simulation, MIGHTY achieves a 9.3% reduction in computation time and a 13.1% reduction in travel time over state-of-the-art baselines, with a 100% success rate. In hardware, MIGHTY completes multiple high-speed flights up to 6.7 m/s in a cluttered static environment and long-duration flights with dynamically added obstacles.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10822) | **Categories:** cs.RO

---

### [2] [Decentralized Swarm Control via SO(3) Embeddings for 3D Trajectories](https://arxiv.org/abs/2511.10858)
*Dimitria Silveria, Kleber Cabral, Peter Jardine, Sidney Givigi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a novel decentralized approach for achieving emergent behavior in multi-agent systems with minimal information sharing. Based on prior work in simple orbits, our method produces a broad class of stable, periodic trajectories by stabilizing the system around a Lie group-based geometric embedding. Employing the Lie group SO(3), we generate a wider range of periodic curves than existing quaternion-based methods. Furthermore, we exploit SO(3) properties to eliminate the need for velocity inputs, allowing agents to receive only position inputs. We also propose a novel phase controller that ensures uniform agent separation, along with a formal stability proof. Validation through simulations and experiments showcases the method's adaptability to complex low-level dynamics and disturbances.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.10858) | **Categories:** cs.RO, cs.MA

---

### [3] [Latent-Space Autoregressive World Model for Efficient and Robust Image-Goal Navigation](https://arxiv.org/abs/2511.11011)
*Zhiwei Zhang, Hui Zhang, Xieyuanli Chen, Kaihong Huang, Chenghao Shi, Huimin Lu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional navigation methods rely heavily on accurate localization and mapping. In contrast, world models that capture environmental dynamics in latent space have opened up new perspectives for navigation tasks, enabling systems to move beyond traditional multi-module pipelines. However, world model often suffers from high computational costs in both training and inference. To address this, we propose LS-NWM - a lightweight latent space navigation world model that is trained and operates entirely in latent space, compared to the state-of-the-art baseline, our method reduces training time by approximately 3.2x and planning time by about 447x,while further improving navigation performance with a 35% higher SR and an 11% higher SPL. The key idea is that accurate pixel-wise environmental prediction is unnecessary for navigation. Instead, the model predicts future latent states based on current observational features and action inputs, then performs path planning and decision-making within this compact representation, significantly improving computational efficiency. By incorporating an autoregressive multi-frame prediction strategy during training, the model effectively captures long-term spatiotemporal dependencies, thereby enhancing navigation performance in complex scenarios. Experimental results demonstrate that our method achieves state-of-the-art navigation performance while maintaining a substantial efficiency advantage over existing approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11011) | **Categories:** cs.RO

---

### [4] [Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning](https://arxiv.org/abs/2511.11218)
*Chenhao Liu, Leyun Jiang, Yibo Wang, Kairan Yao, Jinchen Fu, Xiaoyu Ren*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humanoid robots have demonstrated strong capability for interacting with deterministic scenes across locomotion, manipulation, and more challenging loco-manipulation tasks. Yet the real world is dynamic, quasi-static interactions are insufficient to cope with the various environmental conditions. As a step toward more dynamic interaction scenario, we present a reinforcement-learning-based training pipeline that produces a unified whole-body controller for humanoid badminton, enabling coordinated lower-body footwork and upper-body striking without any motion priors or expert demonstrations. Training follows a three-stage curriculum: first footwork acquisition, then precision-guided racket swing generation, and finally task-focused refinement, yielding motions in which both legs and arms serve the hitting objective. For deployment, we incorporate an Extended Kalman Filter (EKF) to estimate and predict shuttlecock trajectories for target striking. We also introduce a prediction-free variant that dispenses with EKF and explicit trajectory prediction. To validate the framework, we conduct five sets of experiment in both simulation and the real world. In simulation, two robots sustain a rally of 21 consecutive hits. Moreover, the prediction-free variant achieves successful hits with comparable performance relative to the target-known policy. In real-world tests, both the prediction and controller module exhibit high accuracy, and on-court hitting achieves an outgoing shuttle speed up to 10 m/s with a mean return landing distance of 3.5 m. These experiment results show that our humanoid robot can deliver highly dynamic while precise goal striking in badminton, and can be adapted to more dynamism critical domains.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11218) | **Categories:** cs.RO

---

### [5] [Simulating an Autonomous System in CARLA using ROS 2](https://arxiv.org/abs/2511.11310)
*Joseph Abdo, Aditya Shibu, Moaiz Saeed, Abdul Maajid Aga, Apsara Sivaprazad, Mohamed Al-Musleh*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous racing offers a rigorous setting to stress test perception, planning, and control under high speed and uncertainty. This paper proposes an approach to design and evaluate a software stack for an autonomous race car in CARLA: Car Learning to Act simulator, targeting competitive driving performance in the Formula Student UK Driverless (FS-AI) 2025 competition. By utilizing a 360° light detection and ranging (LiDAR), stereo camera, global navigation satellite system (GNSS), and inertial measurement unit (IMU) sensor via ROS 2 (Robot Operating System), the system reliably detects the cones marking the track boundaries at distances of up to 35 m. Optimized trajectories are computed considering vehicle dynamics and simulated environmental factors such as visibility and lighting to navigate the track efficiently. The complete autonomous stack is implemented in ROS 2 and validated extensively in CARLA on a dedicated vehicle (ADS-DV) before being ported to the actual hardware, which includes the Jetson AGX Orin 64GB, ZED2i Stereo Camera, Robosense Helios 16P LiDAR, and CHCNAV Inertial Navigation System (INS).

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11310) | **Categories:** cs.RO, eess.SY

---
