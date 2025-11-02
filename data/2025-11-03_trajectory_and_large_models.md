# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-03

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (9)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (4)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization](https://arxiv.org/abs/2510.26023)
*Zhipeng Bao, Qianwen Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26023) | **Categories:** cs.AI, cs.RO

---

### [2] [Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles](https://arxiv.org/abs/2510.26242)
*Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26242) | **Categories:** cs.AI

---

### [3] [Estimating cognitive biases with attention-aware inverse planning](https://arxiv.org/abs/2510.25951)
*Sounak Banerjee, Daphne Cornelisse, Deepak Gopinath, Emily Sumner, Jonathan DeCastro, Guy Rosman, Eugene Vinitsky, Mark K. Ho*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: People's goal-directed behaviors are influenced by their cognitive biases, and autonomous systems that interact with people should be aware of this. For example, people's attention to objects in their environment will be biased in a way that systematically affects how they perform everyday tasks such as driving to work. Here, building on recent work in computational cognitive science, we formally articulate the attention-aware inverse planning problem, in which the goal is to estimate a person's attentional biases from their actions. We demonstrate how attention-aware inverse planning systematically differs from standard inverse reinforcement learning and how cognitive biases can be inferred from behavior. Finally, we present an approach to attention-aware inverse planning that combines deep reinforcement learning with computational cognitive modeling. We use this approach to infer the attentional strategies of RL agents in real-life driving scenarios selected from the Waymo Open Dataset, demonstrating the scalability of estimating cognitive biases with attention-aware inverse planning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.25951) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios](https://arxiv.org/abs/2510.26125)
*Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Kate Tolstaya, Sarah Tang, Brandyn White, Ben Sapp, Mingxing Tan, Jyh-Jing Hwang, Drago Anguelov*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-based end-to-end (E2E) driving has garnered significant interest in the research community due to its scalability and synergy with multimodal large language models (MLLMs). However, current E2E driving benchmarks primarily feature nominal scenarios, failing to adequately test the true potential of these systems. Furthermore, existing open-loop evaluation metrics often fall short in capturing the multi-modal nature of driving or effectively evaluating performance in long-tail scenarios. To address these gaps, we introduce the Waymo Open Dataset for End-to-End Driving (WOD-E2E). WOD-E2E contains 4,021 driving segments (approximately 12 hours), specifically curated for challenging long-tail scenarios that that are rare in daily life with an occurring frequency of less than 0.03%. Concretely, each segment in WOD-E2E includes the high-level routing information, ego states, and 360-degree camera views from 8 surrounding cameras. To evaluate the E2E driving performance on these long-tail situations, we propose a novel open-loop evaluation metric: Rater Feedback Score (RFS). Unlike conventional metrics that measure the distance between predicted way points and the logs, RFS measures how closely the predicted trajectory matches rater-annotated trajectory preference labels. We have released rater preference labels for all WOD-E2E validation set segments, while the held out test set labels have been used for the 2025 WOD-E2E Challenge. Through our work, we aim to foster state of the art research into generalizable, robust, and safe end-to-end autonomous driving agents capable of handling complex real-world situations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26125) | **Categories:** cs.CV, cs.AI

---

### [2] [MoTDiff: High-resolution Motion Trajectory estimation from a single blurred image using Diffusion models](https://arxiv.org/abs/2510.26173)
*Wontae Choi, Jaelin Lee, Hyung Sup Yun, Byeungwoo Jeon, Il Yong Chun*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate estimation of motion information is crucial in diverse computational imaging and computer vision applications. Researchers have investigated various methods to extract motion information from a single blurred image, including blur kernels and optical flow. However, existing motion representations are often of low quality, i.e., coarse-grained and inaccurate. In this paper, we propose the first high-resolution (HR) Motion Trajectory estimation framework using Diffusion models (MoTDiff). Different from existing motion representations, we aim to estimate an HR motion trajectory with high-quality from a single motion-blurred image. The proposed MoTDiff consists of two key components: 1) a new conditional diffusion framework that uses multi-scale feature maps extracted from a single blurred image as a condition, and 2) a new training method that can promote precise identification of a fine-grained motion trajectory, consistent estimation of overall shape and position of a motion path, and pixel connectivity along a motion trajectory. Our experiments demonstrate that the proposed MoTDiff can outperform state-of-the-art methods in both blind image deblurring and coded exposure photography applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26173) | **Categories:** cs.CV

---

### [3] [Beyond Imitation: Constraint-Aware Trajectory Generation with Flow Matching For End-to-End Autonomous Driving](https://arxiv.org/abs/2510.26292)
*Lin Liu, Guanyi Yu, Ziying Song, Junqiao Li, Caiyan Jia, Feiyang Jia, Peiliang Wu, Yandan Luo*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planning is a critical component of end-to-end autonomous driving. However, prevailing imitation learning methods often suffer from mode collapse, failing to produce diverse trajectory hypotheses. Meanwhile, existing generative approaches struggle to incorporate crucial safety and physical constraints directly into the generative process, necessitating an additional optimization stage to refine their outputs. To address these limitations, we propose CATG, a novel planning framework that leverages Constrained Flow Matching. Concretely, CATG explicitly models the flow matching process, which inherently mitigates mode collapse and allows for flexible guidance from various conditioning signals. Our primary contribution is the novel imposition of explicit constraints directly within the flow matching process, ensuring that the generated trajectories adhere to vital safety and kinematic rules. Secondly, CATG parameterizes driving aggressiveness as a control signal during generation, enabling precise manipulation of trajectory style. Notably, on the NavSim v2 challenge, CATG achieved 2nd place with an EPDMS score of 51.31 and was honored with the Innovation Award.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26292) | **Categories:** cs.CV

---

### [4] [Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](https://arxiv.org/abs/2510.26027)
*Ali Rasekh, Erfan Bagheri Soula, Omid Daliran, Simon Gottschalk, Mohsen Fayyaz*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite significant advances in Multimodal Large Language Models (MLLMs), understanding complex temporal dynamics in videos remains a major challenge. Our experiments show that current Video Large Language Model (Video-LLM) architectures have critical limitations in temporal understanding, struggling with tasks that require detailed comprehension of action sequences and temporal progression. In this work, we propose a Video-LLM architecture that introduces stacked temporal attention modules directly within the vision encoder. This design incorporates a temporal attention in vision encoder, enabling the model to better capture the progression of actions and the relationships between frames before passing visual tokens to the LLM. Our results show that this approach significantly improves temporal reasoning and outperforms existing models in video question answering tasks, specifically in action recognition. We improve on benchmarks including VITATECS, MVBench, and Video-MME by up to +5.5%. By enhancing the vision encoder with temporal structure, we address a critical gap in video understanding for Video-LLMs. Project page and code are available at: https://alirasekh.github.io/STAVEQ2/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26027) | **Categories:** cs.CV

---

### [5] [On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations](https://arxiv.org/abs/2510.00037)
*Jianing Guo, Zhenhong Wu, Chang Tu, Yiyao Ma, Xiangqi Kong, Zhiqian Liu, Jiaming Ji, Shuning Zhang, Yuanpei Chen, Kai Chen, Qi Dou, Yaodong Yang, Xianglong Liu, Huijie Zhao, Weifeng Lv, Simin Li*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In Vision-Language-Action (VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness with a diffusion-based action head. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust VLAs, and a 10.4% gain under mixed perturbations. Our RobustVLA is particularly effective on real-world FR5 robot with limited demonstrations, showing absolute gains by 65.6% under perturbations of four modalities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00037) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [6] [PointSt3R: Point Tracking through 3D Grounded Correspondence](https://arxiv.org/abs/2510.26443)
*Rhodri Guerrier, Adam W. Harley, Dima Damen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in foundational 3D reconstruction models, such as DUSt3R and MASt3R, have shown great potential in 2D and 3D correspondence in static scenes. In this paper, we propose to adapt them for the task of point tracking through 3D grounded correspondence. We first demonstrate that these models are competitive point trackers when focusing on static points, present in current point tracking benchmarks ($+33.5\%$ on EgoPoints vs. CoTracker2). We propose to combine the reconstruction loss with training for dynamic correspondence along with a visibility head, and fine-tuning MASt3R for point tracking using a relatively small amount of synthetic data. Importantly, we only train and evaluate on pairs of frames where one contains the query point, effectively removing any temporal context. Using a mix of dynamic and static point correspondences, we achieve competitive or superior point tracking results on four datasets (e.g. competitive on TAP-Vid-DAVIS 73.8 $\delta_{avg}$ / 85.8\% occlusion acc. for PointSt3R compared to 75.7 / 88.3\% for CoTracker2; and significantly outperform CoTracker3 on EgoPoints 61.3 vs 54.2 and RGB-S 87.0 vs 82.8). We also present results on 3D point tracking along with several ablations on training datasets and percentage of dynamic correspondences.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26443) | **Categories:** cs.CV

---

### [7] [Dynamic Context-Aware Scene Reasoning Using Vision-Language Alignment in Zero-Shot Real-World Scenarios](https://arxiv.org/abs/2510.26580)
*Manjunath Prasad Holenarasipura Rajiv, B. M. Vidyavathi*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In real-world environments, AI systems often face unfamiliar scenarios without labeled data, creating a major challenge for conventional scene understanding models. The inability to generalize across unseen contexts limits the deployment of vision-based applications in dynamic, unstructured settings. This work introduces a Dynamic Context-Aware Scene Reasoning framework that leverages Vision-Language Alignment to address zero-shot real-world scenarios. The goal is to enable intelligent systems to infer and adapt to new environments without prior task-specific training. The proposed approach integrates pre-trained vision transformers and large language models to align visual semantics with natural language descriptions, enhancing contextual comprehension. A dynamic reasoning module refines predictions by combining global scene cues and object-level interactions guided by linguistic priors. Extensive experiments on zero-shot benchmarks such as COCO, Visual Genome, and Open Images demonstrate up to 18% improvement in scene understanding accuracy over baseline models in complex and unseen environments. Results also show robust performance in ambiguous or cluttered scenes due to the synergistic fusion of vision and language. This framework offers a scalable and interpretable approach for context-aware reasoning, advancing zero-shot generalization in dynamic real-world settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26580) | **Categories:** cs.CV

---

### [8] [Emu3.5: Native Multimodal Models are World Learners](https://arxiv.org/abs/2510.26583)
*Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, Xinlong Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce Emu3.5, a large-scale multimodal world model that natively predicts the next state across vision and language. Emu3.5 is pre-trained end-to-end with a unified next-token prediction objective on a corpus of vision-language interleaved data containing over 10 trillion tokens, primarily derived from sequential frames and transcripts of internet videos. The model naturally accepts interleaved vision-language inputs and generates interleaved vision-language outputs. Emu3.5 is further post-trained with large-scale reinforcement learning to enhance multimodal reasoning and generation. To improve inference efficiency, we propose Discrete Diffusion Adaptation (DiDA), which converts token-by-token decoding into bidirectional parallel prediction, accelerating per-image inference by about 20x without sacrificing performance. Emu3.5 exhibits strong native multimodal capabilities, including long-horizon vision-language generation, any-to-image (X2I) generation, and complex text-rich image generation. It also exhibits generalizable world-modeling abilities, enabling spatiotemporally consistent world exploration and open-world embodied manipulation across diverse scenarios and tasks. For comparison, Emu3.5 achieves performance comparable to Gemini 2.5 Flash Image (Nano Banana) on image generation and editing tasks and demonstrates superior results on a suite of interleaved generation tasks. We open-source Emu3.5 at https://github.com/baaivision/Emu3.5 to support community research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26583) | **Categories:** cs.CV

---

### [9] [All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles](https://arxiv.org/abs/2510.26641)
*Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Hazim Alzorgan, Ahmad Sarlak, Mahlagha Fazeli, Abolfazl Razi*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26641) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Network-Constrained Policy Optimization for Adaptive Multi-agent Vehicle Routing](https://arxiv.org/abs/2510.26089)
*Fazel Arasteh, Arian Haghparast, Manos Papagelis*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traffic congestion in urban road networks leads to longer trip times and higher emissions, especially during peak periods. While the Shortest Path First (SPF) algorithm is optimal for a single vehicle in a static network, it performs poorly in dynamic, multi-vehicle settings, often worsening congestion by routing all vehicles along identical paths. We address dynamic vehicle routing through a multi-agent reinforcement learning (MARL) framework for coordinated, network-aware fleet navigation. We first propose Adaptive Navigation (AN), a decentralized MARL model where each intersection agent provides routing guidance based on (i) local traffic and (ii) neighborhood state modeled using Graph Attention Networks (GAT). To improve scalability in large networks, we further propose Hierarchical Hub-based Adaptive Navigation (HHAN), an extension of AN that assigns agents only to key intersections (hubs). Vehicles are routed hub-to-hub under agent control, while SPF handles micro-routing within each hub region. For hub coordination, HHAN adopts centralized training with decentralized execution (CTDE) under the Attentive Q-Mixing (A-QMIX) framework, which aggregates asynchronous vehicle decisions via attention. Hub agents use flow-aware state features that combine local congestion and predictive dynamics for proactive routing. Experiments on synthetic grids and real urban maps (Toronto, Manhattan) show that AN reduces average travel time versus SPF and learning baselines, maintaining 100% routing success. HHAN scales to networks with hundreds of intersections, achieving up to 15.9% improvement under heavy traffic. These findings highlight the potential of network-constrained MARL for scalable, coordinated, and congestion-aware routing in intelligent transportation systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26089) | **Categories:** cs.LG, cs.AI, cs.MA

---

### [2] [CorVS: Person Identification via Video Trajectory-Sensor Correspondence in a Real-World Warehouse](https://arxiv.org/abs/2510.26369)
*Kazuma Kano, Yuki Mori, Shin Katayama, Kenta Urano, Takuro Yonezawa, Nobuo Kawaguchi*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Worker location data is key to higher productivity in industrial sites. Cameras are a promising tool for localization in logistics warehouses since they also offer valuable environmental contexts such as package status. However, identifying individuals with only visual data is often impractical. Accordingly, several prior studies identified people in videos by comparing their trajectories and wearable sensor measurements. While this approach has advantages such as independence from appearance, the existing methods may break down under real-world conditions. To overcome this challenge, we propose CorVS, a novel data-driven person identification method based on correspondence between visual tracking trajectories and sensor measurements. Firstly, our deep learning model predicts correspondence probabilities and reliabilities for every pair of a trajectory and sensor measurements. Secondly, our algorithm matches the trajectories and sensor measurements over time using the predicted probabilities and reliabilities. We developed a dataset with actual warehouse operations and demonstrated the method's effectiveness for real-world applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26369) | **Categories:** cs.LG, cs.CV, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [DARTS: A Drone-Based AI-Powered Real-Time Traffic Incident Detection System](https://arxiv.org/abs/2510.26004)
*Bai Li, Achilleas Kourtellis, Rong Cao, Joseph Post, Brian Porter, Yu Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Rapid and reliable incident detection is critical for reducing crash-related fatalities, injuries, and congestion. However, conventional methods, such as closed-circuit television, dashcam footage, and sensor-based detection, separate detection from verification, suffer from limited flexibility, and require dense infrastructure or high penetration rates, restricting adaptability and scalability to shifting incident hotspots. To overcome these challenges, we developed DARTS, a drone-based, AI-powered real-time traffic incident detection system. DARTS integrates drones' high mobility and aerial perspective for adaptive surveillance, thermal imaging for better low-visibility performance and privacy protection, and a lightweight deep learning framework for real-time vehicle trajectory extraction and incident detection. The system achieved 99% detection accuracy on a self-collected dataset and supports simultaneous online visual verification, severity assessment, and incident-induced congestion propagation monitoring via a web-based interface. In a field test on Interstate 75 in Florida, DARTS detected and verified a rear-end collision 12 minutes earlier than the local transportation management center and monitored incident-induced congestion propagation, suggesting potential to support faster emergency response and enable proactive traffic control to reduce congestion and secondary crash risk. Crucially, DARTS's flexible deployment architecture reduces dependence on frequent physical patrols, indicating potential scalability and cost-effectiveness for use in remote areas and resource-constrained settings. This study presents a promising step toward a more flexible and integrated real-time traffic incident detection system, with significant implications for the operational efficiency and responsiveness of modern transportation management.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26004) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [2] [Accelerating Real-World Overtaking in F1TENTH Racing Employing Reinforcement Learning Methods](https://arxiv.org/abs/2510.26040)
*Emily Steiner, Daniel van der Spuy, Futian Zhou, Afereti Pama, Minas Liarokapis, Henry Williams*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While autonomous racing performance in Time-Trial scenarios has seen significant progress and development, autonomous wheel-to-wheel racing and overtaking are still severely limited. These limitations are particularly apparent in real-life driving scenarios where state-of-the-art algorithms struggle to safely or reliably complete overtaking manoeuvres. This is important, as reliable navigation around other vehicles is vital for safe autonomous wheel-to-wheel racing. The F1Tenth Competition provides a useful opportunity for developing wheel-to-wheel racing algorithms on a standardised physical platform. The competition format makes it possible to evaluate overtaking and wheel-to-wheel racing algorithms against the state-of-the-art. This research presents a novel racing and overtaking agent capable of learning to reliably navigate a track and overtake opponents in both simulation and reality. The agent was deployed on an F1Tenth vehicle and competed against opponents running varying competitive algorithms in the real world. The results demonstrate that the agent's training against opponents enables deliberate overtaking behaviours with an overtaking rate of 87% compared 56% for an agent trained just to race.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26040) | **Categories:** cs.RO, cs.LG

---

### [3] [Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling](https://arxiv.org/abs/2510.26139)
*Minseo Kwon, Young J. Kim*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Task and Motion Planning (TAMP) integrates high-level task planning with low-level motion feasibility, but existing methods are costly in long-horizon problems due to excessive motion sampling. While LLMs provide commonsense priors, they lack 3D spatial reasoning and cannot ensure geometric or dynamic feasibility. We propose a kinodynamic TAMP framework based on a hybrid state tree that uniformly represents symbolic and numeric states during planning, enabling task and motion decisions to be jointly decided. Kinodynamic constraints embedded in the TAMP problem are verified by an off-the-shelf motion planner and physics simulator, and a VLM guides exploring a TAMP solution and backtracks the search based on visual rendering of the states. Experiments on the simulated domains and in the real world show 32.14% - 1166.67% increased average success rates compared to traditional and LLM-based TAMP planners and reduced planning time on complex problems, with ablations further highlighting the benefits of VLM guidance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26139) | **Categories:** cs.RO

---

### [4] [Adaptive Trajectory Refinement for Optimization-based Local Planning in Narrow Passages](https://arxiv.org/abs/2510.26142)
*Hahjin Lee, Young J. Kim*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory planning for mobile robots in cluttered environments remains a major challenge due to narrow passages, where conventional methods often fail or generate suboptimal paths. To address this issue, we propose the adaptive trajectory refinement algorithm, which consists of two main stages. First, to ensure safety at the path-segment level, a segment-wise conservative collision test is applied, where risk-prone trajectory path segments are recursively subdivided until collision risks are eliminated. Second, to guarantee pose-level safety, pose correction based on penetration direction and line search is applied, ensuring that each pose in the trajectory is collision-free and maximally clear from obstacles. Simulation results demonstrate that the proposed method achieves up to 1.69x higher success rates and up to 3.79x faster planning times than state-of-the-art approaches. Furthermore, real-world experiments confirm that the robot can safely pass through narrow passages while maintaining rapid planning performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26142) | **Categories:** cs.RO

---


## eess.SY [eess.SY]
### [1] [Efficient Collision-Avoidance Constraints for Ellipsoidal Obstacles in Optimal Control: Application to Path-Following MPC and UAVs](https://arxiv.org/abs/2510.26531)
*David Leprich, Mario Rosenfelder, Markus Herrmann-Wicklmayr, Kathrin Flaßkamp, Peter Eberhard, Henrik Ebel*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This article proposes a modular optimal control framework for local three-dimensional ellipsoidal obstacle avoidance, exemplarily applied to model predictive path-following control. Static as well as moving obstacles are considered. Central to the approach is a computationally efficient and continuously differentiable condition for detecting collisions with ellipsoidal obstacles. A novel two-stage optimization approach mitigates numerical issues arising from the structure of the resulting optimal control problem. The effectiveness of the approach is demonstrated through simulations and real-world experiments with the Crazyflie quadrotor. This represents the first hardware demonstration of an MPC controller of this kind for UAVs in a three-dimensional task.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26531) | **Categories:** eess.SY, cs.RO, cs.SY, 93-XX

---
