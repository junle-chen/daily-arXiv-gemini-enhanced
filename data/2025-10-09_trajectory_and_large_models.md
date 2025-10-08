# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-09

## 目录

- [人工智能 (Artificial Intelligence) (6)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (3)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Joint Communication Scheduling and Velocity Control for Multi-UAV-Assisted Post-Disaster Monitoring: An Attention-Based In-Context Learning Approach](https://arxiv.org/abs/2510.05698)
*Yousef Emami, Seyedsina Nabavirazavi, Jingjing Zheng, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently, Unmanned Aerial Vehicles (UAVs) are increasingly being investigated to collect sensory data in post-disaster monitoring scenarios, such as tsunamis, where early actions are critical to limit coastal damage. A major challenge is to design the data collection schedules and flight velocities, as unfavorable schedules and velocities can lead to transmission errors and buffer overflows of the ground sensors, ultimately resulting in significant packet loss. Meanwhile, online Deep Reinforcement Learning (DRL) solutions have a complex training process and a mismatch between simulation and reality that does not meet the urgent requirements of tsunami monitoring. Recent advances in Large Language Models (LLMs) offer a compelling alternative. With their strong reasoning and generalization capabilities, LLMs can adapt to new tasks through In-Context Learning (ICL), which enables task adaptation through natural language prompts and example-based guidance without retraining. However, LLM models have input data limitations and thus require customized approaches. In this paper, a joint optimization of data collection schedules and velocities control for multiple UAVs is proposed to minimize data loss. The battery level of the ground sensors, the length of the queues, and the channel conditions, as well as the trajectories of the UAVs, are taken into account. Attention-Based In-Context Learning for Velocity Control and Data Collection Schedule (AIC-VDS) is proposed as an alternative to DRL in emergencies. The simulation results show that the proposed AIC-VDS outperforms both the Deep-Q-Network (DQN) and maximum channel gain baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05698) | **Categories:** cs.AI, 53-01, C.2

---

### [2] [Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents](https://arxiv.org/abs/2510.06078)
*Tao Zhe, Rui Liu, Fateme Memar, Xiao Luo, Wei Fan, Xinyue Ye, Zhongren Peng, Dongjie Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06078) | **Categories:** cs.AI

---

### [3] [AInstein: Assessing the Feasibility of AI-Generated Approaches to Research Problems](https://arxiv.org/abs/2510.05432)
*Shambhavi Mishra, Gaurav Sahu, Marco Pedersoli, Laurent Charlin, Jose Dolz, Christopher Pal*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet it remains unclear whether such success reflects genuine reasoning or sophisticated recall. We introduce AInstein, a framework for testing whether LLMs can generate valid solutions to AI research problems using only their pretrained parametric knowledge -- without domain-specific fine-tuning, retrieval augmentation, or other external aids. Our approach extracts distilled problem statements from high-quality ICLR 2025 submissions, then tasks specialized solver agents with proposing and refining technical solutions through iterative critique loops, mimicking the cycles of proposal, review, and revision central to scientific inquiry. We evaluate AInstein on 1,214 ICLR papers stratified by acceptance tier (Oral, Spotlight, Poster), using an LLM-as-a-judge paradigm guided by a structured rubric, complemented by targeted manual checks. Performance is assessed with three metrics: Success Rate (does the solution address the problem?), Rediscovery (does it align with human-proposed methods?), and Novelty (does it yield valid, original approaches?). Our results reveal that while LLMs can rediscover feasible solutions and occasionally propose creative alternatives, their problem-solving ability remains fragile and highly sensitive to framing. These findings provide the first large-scale evidence on the extent to which LLMs can act as autonomous scientific problem-solvers, highlighting both their latent potential and their current limitations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05432) | **Categories:** cs.AI

---

### [4] [From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Future Research Directions](https://arxiv.org/abs/2510.05596)
*Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Dusit Niyato, Geng Sun, Xianbin Wang, Shiwen Mao, Abbas Jamalipour*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Self-evolving agentic artificial intelligence (AI) offers a new paradigm for future wireless systems by enabling autonomous agents to continually adapt and improve without human intervention. Unlike static AI models, self-evolving agents embed an autonomous evolution cycle that updates models, tools, and workflows in response to environmental dynamics. This paper presents a comprehensive overview of self-evolving agentic AI, highlighting its layered architecture, life cycle, and key techniques, including tool intelligence, workflow optimization, self-reflection, and evolutionary learning. We further propose a multi-agent cooperative self-evolving agentic AI framework, where multiple large language models (LLMs) are assigned role-specialized prompts under the coordination of a supervisor agent. Through structured dialogue, iterative feedback, and systematic validation, the system autonomously executes the entire life cycle without human intervention. A case study on antenna evolution in low-altitude wireless networks (LAWNs) demonstrates how the framework autonomously upgrades fixed antenna optimization into movable antenna optimization. Experimental results show that the proposed self-evolving agentic AI autonomously improves beam gain and restores degraded performance by up to 52.02%, consistently surpassing the fixed baseline with little to no human intervention and validating its adaptability and robustness for next-generation wireless intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05596) | **Categories:** cs.AI

---

### [5] [D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](https://arxiv.org/abs/2510.05684)
*Suwhan Choi, Jaeyoon Jung, Haebin Seong, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu, Yunsung Lee*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at https://worv-ai.github.io/d2e/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05684) | **Categories:** cs.AI, cs.CV, cs.RO

---

### [6] [The Safety Challenge of World Models for Embodied AI Agents: A Review](https://arxiv.org/abs/2510.05865)
*Lorenzo Baraldi, Zifan Zeng, Chongzhe Zhang, Aradhana Nayak, Hongbo Zhu, Feng Liu, Qunli Zhang, Peng Wang, Shiming Liu, Zheng Hu, Angelo Cangelosi, Lorenzo Baraldi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05865) | **Categories:** cs.AI, cs.CV, cs.RO

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [See the past: Time-Reversed Scene Reconstruction from Thermal Traces Using Visual Language Models](https://arxiv.org/abs/2510.05408)
*Kebin Contreras, Luis Toscano-Palomino, Mauro Dalla Mura, Jorge Bacca*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recovering the past from present observations is an intriguing challenge with potential applications in forensics and scene analysis. Thermal imaging, operating in the infrared range, provides access to otherwise invisible information. Since humans are typically warmer (37 C -98.6 F) than their surroundings, interactions such as sitting, touching, or leaning leave residual heat traces. These fading imprints serve as passive temporal codes, allowing for the inference of recent events that exceed the capabilities of RGB cameras. This work proposes a time-reversed reconstruction framework that uses paired RGB and thermal images to recover scene states from a few seconds earlier. The proposed approach couples Visual-Language Models (VLMs) with a constrained diffusion process, where one VLM generates scene descriptions and another guides image reconstruction, ensuring semantic and structural consistency. The method is evaluated in three controlled scenarios, demonstrating the feasibility of reconstructing plausible past frames up to 120 seconds earlier, providing a first step toward time-reversed imaging from thermal traces.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05408) | **Categories:** cs.CV, cs.AI

---

### [2] [Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics](https://arxiv.org/abs/2510.05558)
*Christopher Hoang, Mengye Ren*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Object recognition and motion understanding are key components of perception that complement each other. While self-supervised learning methods have shown promise in their ability to learn from unlabeled data, they have primarily focused on obtaining rich representations for either recognition or motion rather than both in tandem. On the other hand, latent dynamics modeling has been used in decision making to learn latent representations of observations and their transformations over time for control and planning tasks. In this work, we present Midway Network, a new self-supervised learning architecture that is the first to learn strong visual representations for both object recognition and motion understanding solely from natural videos, by extending latent dynamics modeling to this domain. Midway Network leverages a midway top-down path to infer motion latents between video frames, as well as a dense forward prediction objective and hierarchical structure to tackle the complex, multi-object scenes of natural videos. We demonstrate that after pretraining on two large-scale natural video datasets, Midway Network achieves strong performance on both semantic segmentation and optical flow tasks relative to prior self-supervised learning methods. We also show that Midway Network's learned dynamics can capture high-level correspondence via a novel analysis method based on forward feature perturbation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05558) | **Categories:** cs.CV, cs.LG

---

### [3] [Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](https://arxiv.org/abs/2510.05836)
*Ruyang Liu, Shangkun Sun, Haoran Tang, Ge Li, Wei Gao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Long-form video understanding has always been a challenging problem due to the significant redundancy in both temporal and spatial contents. This challenge is further exacerbated by the limited context length of Multimodal Large Language Models (MLLMs). To address this issue, many previous works have attempted to extract key video information, where the "key" is typically semantic-aware and heavily dependent on the CLIP model as prior. In this paper, we propose Flow4Agent, a novel framework that pioneeringly incorporates motion priors from optical flow to facilitate LLM-based long video understanding. Flow4Agent mitigates the redundancy in long videos at both temporal and spatial levels through two core modules: Temporal Granularity Optimization (TGO) adaptively refines framelevel hierarchies, which first leverages coarse flow priors to group similar visual contents and then applies semantic priors to filter out highly irrelevant scene information. Motion Token Pruning (MTP) further refines the intra-frame visual representations, pruning high-redundancy video tokens using fine-grained optical flow information. Extensive experiments demonstrate that our Flow4Agent outperforms existing methods across a wide range of video MLLM benchmarks, especially for hour-level video understanding tasks, achieving 64.7% on Video-MME, 71.4% on MLVU and 60.4% on LongVideoBench.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05836) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [NorMuon: Making Muon more efficient and scalable](https://arxiv.org/abs/2510.05491)
*Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The choice of optimizer significantly impacts the training efficiency and computational costs of large language models (LLMs). Recently, the Muon optimizer has demonstrated promising results by orthogonalizing parameter updates, improving optimization geometry through better conditioning. Despite Muon's emergence as a candidate successor to Adam, the potential for jointly leveraging their strengths has not been systematically explored. In this work, we bridge this gap by proposing NorMuon (Neuron-wise Normalized Muon), an optimizer that synergistically combines orthogonalization with neuron-level adaptive learning rates. Our analysis reveals that while Muon effectively reduces condition numbers, the resulting updates exhibit highly non-uniform neuron norms, causing certain neurons to dominate the optimization process. NorMuon addresses this imbalance by maintaining second-order momentum statistics for each neuron and applying row-wise normalization after orthogonalization, ensuring balanced parameter utilization while preserving Muon's conditioning benefits. To enable practical deployment at scale, we develop an efficient distributed implementation under the FSDP2 framework that strategically distributes orthogonalization computations across devices. Experiments across multiple model scales demonstrate that NorMuon consistently outperforms both Adam and Muon, achieving 21.74% better training efficiency than Adam and 11.31% improvement over Muon on 1.1 B pretraining setting, while maintaining a comparable memory footprint to Muon. Our findings suggest that orthogonalization and adaptive learning rates are complementary rather than competing approaches, opening new avenues for optimizer design in large-scale deep learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05491) | **Categories:** cs.LG, cs.CL

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Active Semantic Perception](https://arxiv.org/abs/2510.05430)
*Huayi Tang, Pratik Chaudhari*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We develop an approach for active semantic perception which refers to using the semantics of the scene for tasks such as exploration. We build a compact, hierarchical multi-layer scene graph that can represent large, complex indoor environments at various levels of abstraction, e.g., nodes corresponding to rooms, objects, walls, windows etc. as well as fine-grained details of their geometry. We develop a procedure based on large language models (LLMs) to sample plausible scene graphs of unobserved regions that are consistent with partial observations of the scene. These samples are used to compute an information gain of a potential waypoint for sophisticated spatial reasoning, e.g., the two doors in the living room can lead to either a kitchen or a bedroom. We evaluate this approach in complex, realistic 3D indoor environments in simulation. We show using qualitative and quantitative experiments that our approach can pin down the semantics of the environment quicker and more accurately than baseline approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05430) | **Categories:** cs.RO

---

### [2] [Precise and Efficient Collision Prediction under Uncertainty in Autonomous Driving](https://arxiv.org/abs/2510.05729)
*Marc Kaufeld, Johannes Betz*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This research introduces two efficient methods to estimate the collision risk of planned trajectories in autonomous driving under uncertain driving conditions. Deterministic collision checks of planned trajectories are often inaccurate or overly conservative, as noisy perception, localization errors, and uncertain predictions of other traffic participants introduce significant uncertainty into the planning process. This paper presents two semi-analytic methods to compute the collision probability of planned trajectories with arbitrary convex obstacles. The first approach evaluates the probability of spatial overlap between an autonomous vehicle and surrounding obstacles, while the second estimates the collision probability based on stochastic boundary crossings. Both formulations incorporate full state uncertainties, including position, orientation, and velocity, and achieve high accuracy at computational costs suitable for real-time planning. Simulation studies verify that the proposed methods closely match Monte Carlo results while providing significant runtime advantages, enabling their use in risk-aware trajectory planning. The collision estimation methods are available as open-source software: https://github.com/TUM-AVS/Collision-Probability-Estimation

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.05729) | **Categories:** cs.RO

---

### [3] [EmbodiedCoder: Parameterized Embodied Mobile Manipulation via Modern Coding Model](https://arxiv.org/abs/2510.06207)
*Zefu Lin, Rongxu Cui, Chen Hanning, Xiangyu Wang, Junjia Xu, Xiaojuan Jin, Chen Wenbo, Hui Zhou, Lue Fan, Wenling Li, Zhaoxiang Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in control robot methods, from end-to-end vision-language-action frameworks to modular systems with predefined primitives, have advanced robots' ability to follow natural language instructions. Nonetheless, many approaches still struggle to scale to diverse environments, as they often rely on large annotated datasets and offer limited interpretability.In this work, we introduce EmbodiedCoder, a training-free framework for open-world mobile robot manipulation that leverages coding models to directly generate executable robot trajectories. By grounding high-level instructions in code, EmbodiedCoder enables flexible object geometry parameterization and manipulation trajectory synthesis without additional data collection or fine-tuning.This coding-based paradigm provides a transparent and generalizable way to connect perception with manipulation. Experiments on real mobile robots show that EmbodiedCoder achieves robust performance across diverse long-term tasks and generalizes effectively to novel objects and environments.Our results demonstrate an interpretable approach for bridging high-level reasoning and low-level control, moving beyond fixed primitives toward versatile robot intelligence. See the project page at: https://anonymous.4open.science/w/Embodied-Coder/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.06207) | **Categories:** cs.RO

---
