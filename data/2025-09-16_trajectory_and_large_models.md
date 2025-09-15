# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-16

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算语言学 (Computation and Language) (3)](#cs-cl)
- [计算机视觉 (Computer Vision) (6)](#cs-cv)
- [cs.IR (1)](#cs-ir)
- [cs.MA (1)](#cs-ma)
- [机器人学 (Robotics) (8)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [The Morality of Probability: How Implicit Moral Biases in LLMs May Shape the Future of Human-AI Symbiosis](https://arxiv.org/abs/2509.10297)
*Eoin O'Doherty, Nicole Weinrauch, Andrew Talone, Uri Klempner, Xiaoyuan Yi, Xing Xie, Yi Zeng*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Artificial intelligence (AI) is advancing at a pace that raises urgent questions about how to align machine decision-making with human moral values. This working paper investigates how leading AI systems prioritize moral outcomes and what this reveals about the prospects for human-AI symbiosis. We address two central questions: (1) What moral values do state-of-the-art large language models (LLMs) implicitly favour when confronted with dilemmas? (2) How do differences in model architecture, cultural origin, and explainability affect these moral preferences? To explore these questions, we conduct a quantitative experiment with six LLMs, ranking and scoring outcomes across 18 dilemmas representing five moral frameworks. Our findings uncover strikingly consistent value biases. Across all models, Care and Virtue values outcomes were rated most moral, while libertarian choices were consistently penalized. Reasoning-enabled models exhibited greater sensitivity to context and provided richer explanations, whereas non-reasoning models produced more uniform but opaque judgments. This research makes three contributions: (i) Empirically, it delivers a large-scale comparison of moral reasoning across culturally distinct LLMs; (ii) Theoretically, it links probabilistic model behaviour with underlying value encodings; (iii) Practically, it highlights the need for explainability and cultural awareness as critical design principles to guide AI toward a transparent, aligned, and symbiotic future.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10297) | **Categories:** cs.AI

---

### [2] [Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems](https://arxiv.org/abs/2509.10401)
*Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Yue Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this counterfactual inference gap, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10401) | **Categories:** cs.AI, cs.CL

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [CTCC: A Robust and Stealthy Fingerprinting Framework for Large Language Models via Cross-Turn Contextual Correlation Backdoor](https://arxiv.org/abs/2509.09703)
*Zhenhua Xu, Xixiang Zhao, Xubin Yue, Shengwei Tian, Changting Lin, Meng Han*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The widespread deployment of large language models (LLMs) has intensified concerns around intellectual property (IP) protection, as model theft and unauthorized redistribution become increasingly feasible. To address this, model fingerprinting aims to embed verifiable ownership traces into LLMs. However, existing methods face inherent trade-offs between stealthness, robustness, and generalizability, being either detectable via distributional shifts, vulnerable to adversarial modifications, or easily invalidated once the fingerprint is revealed. In this work, we introduce CTCC, a novel rule-driven fingerprinting framework that encodes contextual correlations across multiple dialogue turns, such as counterfactual, rather than relying on token-level or single-turn triggers. CTCC enables fingerprint verification under black-box access while mitigating false positives and fingerprint leakage, supporting continuous construction under a shared semantic rule even if partial triggers are exposed. Extensive experiments across multiple LLM architectures demonstrate that CTCC consistently achieves stronger stealth and robustness than prior work. Our findings position CTCC as a reliable and practical solution for ownership verification in real-world LLM deployment scenarios. Our code and data are publicly available at <https://github.com/Xuzhenhua55/CTCC>.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09703) | **Categories:** cs.CL, cs.AI

---

### [2] [Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data](https://arxiv.org/abs/2509.09710)
*Sepehr Golrokh Amin, Devin Rhoads, Fatemeh Fakhrmoosavi, Nicholas E. Lownes, John N. Ivan*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This study introduces a Large Language Model (LLM) scheme for generating individual travel diaries in agent-based transportation models. While traditional approaches rely on large quantities of proprietary household travel surveys, the method presented in this study generates personas stochastically from open-source American Community Survey (ACS) and Smart Location Database (SLD) data, then synthesizes diaries through direct prompting. This study features a novel one-to-cohort realism score: a composite of four metrics (Trip Count Score, Interval Score, Purpose Score, and Mode Score) validated against the Connecticut Statewide Transportation Study (CSTS) diaries, matched across demographic variables. The validation utilizes Jensen-Shannon Divergence to measure distributional similarities between generated and real diaries. When compared to diaries generated with classical methods (Negative Binomial for trip generation; Multinomial Logit for mode/purpose) calibrated on the validation set, LLM-generated diaries achieve comparable overall realism (LLM mean: 0.485 vs. 0.455). The LLM excels in determining trip purpose and demonstrates greater consistency (narrower realism score distribution), while classical models lead in numerical estimates of trip count and activity duration. Aggregate validation confirms the LLM's statistical representativeness (LLM mean: 0.612 vs. 0.435), demonstrating LLM's zero-shot viability and establishing a quantifiable metric of diary realism for future synthetic diary evaluation systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09710) | **Categories:** cs.CL, cs.AI, cs.LG

---

### [3] [HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation for Multi-hop Question Answering](https://arxiv.org/abs/2509.09713)
*Duolin Sun, Dan Yang, Yue Shen, Yihan Jiao, Zhehao Tan, Jie Feng, Lianzhen Zhong, Jian Wang, Peng Wei, Jinjie Gu*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The Retrieval-Augmented Generation (RAG) approach enhances question-answering systems and dialogue generation tasks by integrating information retrieval (IR) technologies with large language models (LLMs). This strategy, which retrieves information from external knowledge bases to bolster the response capabilities of generative models, has achieved certain successes. However, current RAG methods still face numerous challenges when dealing with multi-hop queries. For instance, some approaches overly rely on iterative retrieval, wasting too many retrieval steps on compound queries. Additionally, using the original complex query for retrieval may fail to capture content relevant to specific sub-queries, resulting in noisy retrieved content. If the noise is not managed, it can lead to the problem of noise accumulation. To address these issues, we introduce HANRAG, a novel heuristic-based framework designed to efficiently tackle problems of varying complexity. Driven by a powerful revelator, HANRAG routes queries, decomposes them into sub-queries, and filters noise from retrieved documents. This enhances the system's adaptability and noise resistance, making it highly capable of handling diverse queries. We compare the proposed framework against other leading industry methods across various benchmarks. The results demonstrate that our framework obtains superior performance in both single-hop and multi-hop question-answering tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09713) | **Categories:** cs.CL, cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals](https://arxiv.org/abs/2509.10080)
*Minsang Kong, Myeongjun Kim, Sang Gu Kang, Sang Hun Lee*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In autonomous driving, trajectory prediction is essential for ensuring safe and efficient navigation. To improve prediction accuracy, recent approaches often rely on pre-built high-definition (HD) maps or real-time local map construction modules to incorporate static environmental information. However, pre-built HD maps are limited to specific regions and cannot adapt to transient changes. In addition, local map construction modules, which recognize only predefined elements, may fail to capture critical scene details or introduce errors that degrade prediction performance. To overcome these limitations, we propose Bird's-Eye View Trajectory Prediction (BEVTraj), a novel trajectory prediction framework that operates directly in the bird's-eye view (BEV) space utilizing real-time sensor data without relying on any pre-built maps. The BEVTraj leverages deformable attention to efficiently extract relevant context from dense BEV features. Furthermore, we introduce a Sparse Goal Candidate Proposal (SGCP) module, which enables full end-to-end prediction without requiring any post-processing steps. Extensive experiments demonstrate that the BEVTraj achieves performance comparable to state-of-the-art HD map-based models while offering greater flexibility by eliminating the dependency on pre-built maps. The source code is available at https://github.com/Kongminsang/bevtraj.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10080) | **Categories:** cs.CV, I.2.9; I.4.8

---

### [2] [MITS: A Large-Scale Multimodal Benchmark Dataset for Intelligent Traffic Surveillance](https://arxiv.org/abs/2509.09730)
*Kaikai Zhao, Zhaoxiang Liu, Peng Wang, Xin Wang, Zhicheng Ma, Yajun Xu, Wenjing Zhang, Yibing Nan, Kai Wang, Shiguo Lian*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: General-domain large multimodal models (LMMs) have achieved significant advances in various image-text tasks. However, their performance in the Intelligent Traffic Surveillance (ITS) domain remains limited due to the absence of dedicated multimodal datasets. To address this gap, we introduce MITS (Multimodal Intelligent Traffic Surveillance), the first large-scale multimodal benchmark dataset specifically designed for ITS. MITS includes 170,400 independently collected real-world ITS images sourced from traffic surveillance cameras, annotated with eight main categories and 24 subcategories of ITS-specific objects and events under diverse environmental conditions. Additionally, through a systematic data generation pipeline, we generate high-quality image captions and 5 million instruction-following visual question-answer pairs, addressing five critical ITS tasks: object and event recognition, object counting, object localization, background analysis, and event reasoning. To demonstrate MITS's effectiveness, we fine-tune mainstream LMMs on this dataset, enabling the development of ITS-specific applications. Experimental results show that MITS significantly improves LMM performance in ITS applications, increasing LLaVA-1.5's performance from 0.494 to 0.905 (+83.2%), LLaVA-1.6's from 0.678 to 0.921 (+35.8%), Qwen2-VL's from 0.584 to 0.926 (+58.6%), and Qwen2.5-VL's from 0.732 to 0.930 (+27.0%). We release the dataset, code, and models as open-source, providing high-value resources to advance both ITS and LMM research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09730) | **Categories:** cs.CV, cs.AI

---

### [3] [World Modeling with Probabilistic Structure Integration](https://arxiv.org/abs/2509.09737)
*Klemen Kotar, Wanhee Lee, Rahul Venkatesh, Honglin Chen, Daniel Bear, Jared Watrous, Simon Kim, Khai Loong Aw, Lilian Naing Chen, Stefan Stojanov, Kevin Feigelis, Imran Thobani, Alex Durango, Khaled Jedoui, Atlas Kazemian, Dan Yamins*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present Probabilistic Structure Integration (PSI), a system for learning richly controllable and flexibly promptable world models from data. PSI consists of a three-step cycle. The first step, Probabilistic prediction, involves building a probabilistic graphical model Psi of the data, in the form of a random-access autoregressive sequence model. Psi supports a complete set of learned conditional distributions describing the dependence of any variables in the data on any other set of variables. In step 2, Structure extraction, we show how to extract underlying low-dimensional properties in the data, corresponding to a diverse set of meaningful "intermediate structures", in a zero-shot fashion via causal inference on Psi. Step 3, Integration, completes the cycle by converting these structures into new token types that are then continually mixed back into the training diet as conditioning signals and prediction targets. Each such cycle augments the capabilities of Psi, both allowing it to model the underlying data better, and creating new control handles -- akin to an LLM-like universal prompting language. We train an instance of Psi on 1.4 trillion tokens of internet video data; we use it to perform a variety of useful video prediction and understanding inferences; we extract state-of-the-art optical flow, self-supervised depth and object segmentation; and we use these structures to support a full cycle of predictive improvements.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09737) | **Categories:** cs.CV, cs.AI, cs.LG

---

### [4] [Segment Anything for Cell Tracking](https://arxiv.org/abs/2509.09943)
*Zhu Chen, Mert Edgü, Er Jin, Johannes Stegmaier*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Tracking cells and detecting mitotic events in time-lapse microscopy image sequences is a crucial task in biomedical research. However, it remains highly challenging due to dividing objects, low signal-tonoise ratios, indistinct boundaries, dense clusters, and the visually similar appearance of individual cells. Existing deep learning-based methods rely on manually labeled datasets for training, which is both costly and time-consuming. Moreover, their generalizability to unseen datasets remains limited due to the vast diversity of microscopy data. To overcome these limitations, we propose a zero-shot cell tracking framework by integrating Segment Anything 2 (SAM2), a large foundation model designed for general image and video segmentation, into the tracking pipeline. As a fully-unsupervised approach, our method does not depend on or inherit biases from any specific training dataset, allowing it to generalize across diverse microscopy datasets without finetuning. Our approach achieves competitive accuracy in both 2D and large-scale 3D time-lapse microscopy videos while eliminating the need for dataset-specific adaptation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09943) | **Categories:** cs.CV

---

### [5] [Multimodal Mathematical Reasoning Embedded in Aerial Vehicle Imagery: Benchmarking, Analysis, and Exploration](https://arxiv.org/abs/2509.10059)
*Yue Zhou, Litong Feng, Mengcheng Lan, Xue Yang, Qingyun Li, Yiping Ke, Xue Jiang, Wayne Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Mathematical reasoning is critical for tasks such as precise distance and area computations, trajectory estimations, and spatial analysis in unmanned aerial vehicle (UAV) based remote sensing, yet current vision-language models (VLMs) have not been adequately tested in this domain. To address this gap, we introduce AVI-Math, the first benchmark to rigorously evaluate multimodal mathematical reasoning in aerial vehicle imagery, moving beyond simple counting tasks to include domain-specific knowledge in areas such as geometry, logic, and algebra. The dataset comprises 3,773 high-quality vehicle-related questions captured from UAV views, covering 6 mathematical subjects and 20 topics. The data, collected at varying altitudes and from multiple UAV angles, reflects real-world UAV scenarios, ensuring the diversity and complexity of the constructed mathematical problems. In this paper, we benchmark 14 prominent VLMs through a comprehensive evaluation and demonstrate that, despite their success on previous multimodal benchmarks, these models struggle with the reasoning tasks in AVI-Math. Our detailed analysis highlights significant limitations in the mathematical reasoning capabilities of current VLMs and suggests avenues for future research. Furthermore, we explore the use of Chain-of-Thought prompting and fine-tuning techniques, which show promise in addressing the reasoning challenges in AVI-Math. Our findings not only expose the limitations of VLMs in mathematical reasoning but also offer valuable insights for advancing UAV-based trustworthy VLMs in real-world applications. The code, and datasets will be released at https://github.com/VisionXLab/avi-math

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10059) | **Categories:** cs.CV, cs.AI

---

### [6] [VARCO-VISION-2.0 Technical Report](https://arxiv.org/abs/2509.10105)
*Young-rok Cha, Jeongho Ju, SunYoung Park, Jong-Hyeon Lee, Younghyun Yu, Youngjune Kim*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10105) | **Categories:** cs.CV, cs.CL

---


## cs.IR [cs.IR]
### [1] [Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors](https://arxiv.org/abs/2509.09689)
*Himanshu Thakur, Eshani Agrawal, Smruthi Mukund*

Main category: cs.IR

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: A long-standing challenge in developing accurate recommendation models is simulating user behavior, mainly due to the complex and stochastic nature of user interactions. Towards this, one promising line of work has been the use of Large Language Models (LLMs) for simulating user behavior. However, aligning these general-purpose large pre-trained models with user preferences necessitates: (i) effectively and continously parsing large-scale tabular user-item interaction data, (ii) overcoming pre-training-induced inductive biases to accurately learn user specific knowledge, and (iii) achieving the former two at scale for millions of users. While most previous works have focused on complex methods to prompt an LLM or fine-tune it on tabular interaction datasets, our approach shifts the focus to extracting robust textual user representations using a frozen LLM and simulating cost-effective, resource-efficient user agents powered by fine-tuned Small Language Models (SLMs). Further, we showcase a method for training multiple low-rank adapters for groups of users or \textit{persona}, striking an optimal balance between scalability and performance of user behavior agents. Our experiments provide compelling empirical evidence of the efficacy of our methods, demonstrating that user agents developed using our approach have the potential to bridge the gap between offline metrics and real-world performance of recommender systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09689) | **Categories:** cs.IR, cs.AI, cs.CL, cs.LG

---


## cs.MA [cs.MA]
### [1] [A Holistic Architecture for Monitoring and Optimization of Robust Multi-Agent Path Finding Plan Execution](https://arxiv.org/abs/2509.10284)
*David Zahrádka, Denisa Mužíková, David Woller, Miroslav Kulich, Jiří Švancara, Roman Barták*

Main category: cs.MA

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The goal of Multi-Agent Path Finding (MAPF) is to find a set of paths for a fleet of agents moving in a shared environment such that the agents reach their goals without colliding with each other. In practice, some of the robots executing the plan may get delayed, which can introduce collision risk. Although robust execution methods are used to ensure safety even in the presence of delays, the delays may still have a significant impact on the duration of the execution. At some point, the accumulated delays may become significant enough that instead of continuing with the execution of the original plan, even if it was optimal, there may now exist an alternate plan which will lead to a shorter execution. However, the problem is how to decide when to search for the alternate plan, since it is a costly procedure. In this paper, we propose a holistic architecture for robust execution of MAPF plans, its monitoring and optimization. We exploit a robust execution method called Action Dependency Graph to maintain an estimate of the expected execution duration during the plan's execution. This estimate is used to predict the potential that finding an alternate plan would lead to shorter execution. We empirically evaluate the architecture in experiments in a real-time simulator which we designed to mimic our real-life demonstrator of an autonomous warehouse robotic fleet.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10284) | **Categories:** cs.MA, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [DECAMP: Towards Scene-Consistent Multi-Agent Motion Prediction with Disentangled Context-Aware Pre-Training](https://arxiv.org/abs/2509.10426)
*Jianxin Shi, Zengqi Peng, Xiaolong Chen, Tianyu Wo, Jun Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction is a critical component of autonomous driving, essential for ensuring both safety and efficiency on the road. However, traditional approaches often struggle with the scarcity of labeled data and exhibit suboptimal performance in multi-agent prediction scenarios. To address these challenges, we introduce a disentangled context-aware pre-training framework for multi-agent motion prediction, named DECAMP. Unlike existing methods that entangle representation learning with pretext tasks, our framework decouples behavior pattern learning from latent feature reconstruction, prioritizing interpretable dynamics and thereby enhancing scene representation for downstream prediction. Additionally, our framework incorporates context-aware representation learning alongside collaborative spatial-motion pretext tasks, which enables joint optimization of structural and intentional reasoning while capturing the underlying dynamic intentions. Our experiments on the Argoverse 2 benchmark showcase the superior performance of our method, and the results attained underscore its effectiveness in multi-agent motion forecasting. To the best of our knowledge, this is the first context autoencoder framework for multi-agent motion forecasting in autonomous driving. The code and models will be made publicly available.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10426) | **Categories:** cs.RO, cs.MA

---

### [2] [HHI-Assist: A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario](https://arxiv.org/abs/2509.10096)
*Saeed Saadatnejad, Reyhaneh Hosseininejad, Jose Barreiros, Katherine M. Tsui, Alexandre Alahi*

Main category: cs.RO

TL;DR: 本文提出了一种基于条件Transformer的去噪扩散模型，用于预测辅助任务中交互主体的姿势，并为此发布了一个新的人人交互辅助数据集（HHI-Assist）。


<details>
  <summary>Details</summary>
Motivation: 劳动力短缺和人口老龄化凸显了辅助机器人支持人类护理接受者的需求。为了实现安全和响应迅速的辅助，机器人需要在物理交互场景中进行准确的人类运动预测。

Method: 提出了一个基于条件Transformer的去噪扩散模型，用于预测交互主体的姿势。

Result: 该模型有效地捕捉了护理人员和护理接受者之间的耦合动力学，并在基线上有所改进，并能很好地推广到未见过的场景。

Conclusion: 通过推进交互感知运动预测并引入新的数据集，这项工作有潜力显着增强机器人辅助策略。

Abstract: 劳动力日益短缺和人口老龄化，凸显了需要辅助机器人来支持人类护理对象。为了使机器人能够提供安全和响应迅速的辅助，它们需要在物理交互场景中准确预测人类运动。然而，由于辅助环境的可变性以及物理交互中耦合动力学的复杂性，这仍然是一项具有挑战性的任务。在这项工作中，我们通过两个关键贡献来应对这些挑战：（1）HHI-Assist，一个包含辅助任务中人与人交互的运动捕捉片段的数据集；（2）一个基于条件Transformer的去噪扩散模型，用于预测交互主体的姿势。我们的模型有效地捕捉了护理人员和护理接受者之间的耦合动力学，与基线相比有所改进，并且能够很好地推广到未见过的场景。通过推进交互感知运动预测并引入新的数据集，我们的工作有潜力显着增强机器人辅助策略。数据集和代码可在以下网址获得：https://sites.google.com/view/hhi-assist/home

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10096) | **Categories:** cs.RO, cs.CV

---

### [3] [GundamQ: Multi-Scale Spatio-Temporal Representation Learning for Robust Robot Path Planning](https://arxiv.org/abs/2509.10305)
*Yutong Shen, Ruizhe Xia, Bokai Yan, Shunqi zhang, Pengrui Xiang, Sicheng He, Yixin Xu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In dynamic and uncertain environments, robotic path planning demands accurate spatiotemporal environment understanding combined with robust decision-making under partial observability. However, current deep reinforcement learning-based path planning methods face two fundamental limitations: (1) insufficient modeling of multi-scale temporal dependencies, resulting in suboptimal adaptability in dynamic scenarios, and (2) inefficient exploration-exploitation balance, leading to degraded path quality. To address these challenges, we propose GundamQ: A Multi-Scale Spatiotemporal Q-Network for Robotic Path Planning. The framework comprises two key modules: (i) the Spatiotemporal Perception module, which hierarchically extracts multi-granularity spatial features and multi-scale temporal dependencies ranging from instantaneous to extended time horizons, thereby improving perception accuracy in dynamic environments; and (ii) the Adaptive Policy Optimization module, which balances exploration and exploitation during training while optimizing for smoothness and collision probability through constrained policy updates. Experiments in dynamic environments demonstrate that GundamQ achieves a 15.3\% improvement in success rate and a 21.7\% increase in overall path quality, significantly outperforming existing state-of-the-art methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10305) | **Categories:** cs.RO

---

### [4] [MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos](https://arxiv.org/abs/2509.09769)
*Rutav Shah, Shuijing Liu, Qi Wang, Zhenyu Jiang, Sateesh Kumar, Mingyo Seo, Roberto Martín-Martín, Yuke Zhu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We aim to enable humanoid robots to efficiently solve new manipulation tasks from a few video examples. In-context learning (ICL) is a promising framework for achieving this goal due to its test-time data efficiency and rapid adaptability. However, current ICL methods rely on labor-intensive teleoperated data for training, which restricts scalability. We propose using human play videos -- continuous, unlabeled videos of people interacting freely with their environment -- as a scalable and diverse training data source. We introduce MimicDroid, which enables humanoids to perform ICL using human play videos as the only training data. MimicDroid extracts trajectory pairs with similar manipulation behaviors and trains the policy to predict the actions of one trajectory conditioned on the other. Through this process, the model acquired ICL capabilities for adapting to novel objects and environments at test time. To bridge the embodiment gap, MimicDroid first retargets human wrist poses estimated from RGB videos to the humanoid, leveraging kinematic similarity. It also applies random patch masking during training to reduce overfitting to human-specific cues and improve robustness to visual differences. To evaluate few-shot learning for humanoids, we introduce an open-source simulation benchmark with increasing levels of generalization difficulty. MimicDroid outperformed state-of-the-art methods and achieved nearly twofold higher success rates in the real world. Additional materials can be found on: ut-austin-rpl.github.io/MimicDroid

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09769) | **Categories:** cs.RO

---

### [5] [Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision](https://arxiv.org/abs/2509.09893)
*Hanbit Oh, Masaki Murooka, Tomohiro Motoda, Ryoichi Nakajo, Yukiyasu Domae*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Imitation learning is a promising paradigm for training robot agents; however, standard approaches typically require substantial data acquisition -- via numerous demonstrations or random exploration -- to ensure reliable performance. Although exploration reduces human effort, it lacks safety guarantees and often results in frequent collisions -- particularly in clearance-limited tasks (e.g., peg-in-hole) -- thereby, necessitating manual environmental resets and imposing additional human burden. This study proposes Self-Augmented Robot Trajectory (SART), a framework that enables policy learning from a single human demonstration, while safely expanding the dataset through autonomous augmentation. SART consists of two stages: (1) human teaching only once, where a single demonstration is provided and precision boundaries -- represented as spheres around key waypoints -- are annotated, followed by one environment reset; (2) robot self-augmentation, where the robot generates diverse, collision-free trajectories within these boundaries and reconnects to the original demonstration. This design improves the data collection efficiency by minimizing human effort while ensuring safety. Extensive evaluations in simulation and real-world manipulation tasks show that SART achieves substantially higher success rates than policies trained solely on human-collected demonstrations. Video results available at https://sites.google.com/view/sart-il .

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.09893) | **Categories:** cs.RO, cs.AI

---

### [6] [Robot guide with multi-agent control and automatic scenario generation with LLM](https://arxiv.org/abs/2509.10317)
*Elizaveta D. Moskovskaya, Anton D. Moscowsky*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The work describes the development of a hybrid control architecture for an anthropomorphic tour guide robot, combining a multi-agent resource management system with automatic behavior scenario generation based on large language models. The proposed approach aims to overcome the limitations of traditional systems, which rely on manual tuning of behavior scenarios. These limitations include manual configuration, low flexibility, and lack of naturalness in robot behavior. The process of preparing tour scenarios is implemented through a two-stage generation: first, a stylized narrative is created, then non-verbal action tags are integrated into the text. The multi-agent system ensures coordination and conflict resolution during the execution of parallel actions, as well as maintaining default behavior after the completion of main operations, contributing to more natural robot behavior. The results obtained from the trial demonstrate the potential of the proposed approach for automating and scaling social robot control systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10317) | **Categories:** cs.RO, cs.LG, 93C85, I.2.9; I.2.7; I.2.11

---

### [7] [Acetrans: An Autonomous Corridor-Based and Efficient UAV Suspended Transport System](https://arxiv.org/abs/2509.10349)
*Weiyan Lu, Huizhe Li, Yuhao Fang, Zhexuan Zhou, Junda Wu, Yude Li, Youmin Gong, Jie Mei*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unmanned aerial vehicles (UAVs) with suspended payloads offer significant advantages for aerial transportation in complex and cluttered environments. However, existing systems face critical limitations, including unreliable perception of the cable-payload dynamics, inefficient planning in large-scale environments, and the inability to guarantee whole-body safety under cable bending and external disturbances. This paper presents Acetrans, an Autonomous, Corridor-based, and Efficient UAV suspended transport system that addresses these challenges through a unified perception, planning, and control framework. A LiDAR-IMU fusion module is proposed to jointly estimate both payload pose and cable shape under taut and bent modes, enabling robust whole-body state estimation and real-time filtering of cable point clouds. To enhance planning scalability, we introduce the Multi-size-Aware Configuration-space Iterative Regional Inflation (MACIRI) algorithm, which generates safe flight corridors while accounting for varying UAV and payload geometries. A spatio-temporal, corridor-constrained trajectory optimization scheme is then developed to ensure dynamically feasible and collision-free trajectories. Finally, a nonlinear model predictive controller (NMPC) augmented with cable-bending constraints provides robust whole-body safety during execution. Simulation and experimental results validate the effectiveness of Acetrans, demonstrating substantial improvements in perception accuracy, planning efficiency, and control safety compared to state-of-the-art methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10349) | **Categories:** cs.RO

---

### [8] [GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation](https://arxiv.org/abs/2509.10454)
*Hang Yin, Haoyu Wei, Xiuwei Xu, Wenxuan Guo, Jie Zhou, Jiwen Lu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this paper, we propose a training-free framework for vision-and-language navigation (VLN). Existing zero-shot VLN methods are mainly designed for discrete environments or involve unsupervised training in continuous simulator environments, which makes it challenging to generalize and deploy them in real-world scenarios. To achieve a training-free framework in continuous environments, our framework formulates navigation guidance as graph constraint optimization by decomposing instructions into explicit spatial constraints. The constraint-driven paradigm decodes spatial semantics through constraint solving, enabling zero-shot adaptation to unseen environments. Specifically, we construct a spatial constraint library covering all types of spatial relationship mentioned in VLN instructions. The human instruction is decomposed into a directed acyclic graph, with waypoint nodes, object nodes and edges, which are used as queries to retrieve the library to build the graph constraints. The graph constraint optimization is solved by the constraint solver to determine the positions of waypoints, obtaining the robot's navigation path and final goal. To handle cases of no solution or multiple solutions, we construct a navigation tree and the backtracking mechanism. Extensive experiments on standard benchmarks demonstrate significant improvements in success rate and navigation efficiency compared to state-of-the-art zero-shot VLN methods. We further conduct real-world experiments to show that our framework can effectively generalize to new environments and instruction sets, paving the way for a more robust and autonomous navigation framework.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.10454) | **Categories:** cs.RO, cs.CV

---
