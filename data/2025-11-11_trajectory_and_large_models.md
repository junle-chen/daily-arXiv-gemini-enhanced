# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-11

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (2)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Real-Time Reasoning Agents in Evolving Environments](https://arxiv.org/abs/2511.04898)
*Yule Wen, Yixin Ye, Yanzhe Zhang, Diyi Yang, Hao Zhu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Agents in the real world must make not only logical but also timely judgments. This requires continuous awareness of the dynamic environment: hazards emerge, opportunities arise, and other agents act, while the agent's reasoning is still unfolding. Despite advances in language model reasoning, existing approaches fail to account for this dynamic nature. We introduce real-time reasoning as a new problem formulation for agents in evolving environments and build Real-Time Reasoning Gym to demonstrate it. We study two paradigms for deploying language models in agents: (1) reactive agents, which employ language models with bounded reasoning computation for rapid responses, and (2) planning agents, which allow extended reasoning computation for complex problems. Our experiments show that even state-of-the-art models struggle with making logical and timely judgments in either paradigm. To address this limitation, we propose AgileThinker, which simultaneously engages both reasoning paradigms. AgileThinker consistently outperforms agents engaging only one reasoning paradigm as the task difficulty and time pressure rise, effectively balancing reasoning depth and response latency. Our work establishes real-time reasoning as a critical testbed for developing practical agents and provides a foundation for research in temporally constrained AI systems, highlighting a path toward real-time capable agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.04898) | **Categories:** cs.AI

---

### [2] [Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance](https://arxiv.org/abs/2511.05311)
*Valeriu Dimidov, Faisal Hawlader, Sasan Jafarnejad, Raphaël Frank*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05311) | **Categories:** cs.AI, cs.LG, cs.RO, cs.SE

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Dense Motion Captioning](https://arxiv.org/abs/2511.05369)
*Shiyao Xu, Benedetta Liberatori, Gül Varol, Paolo Rota*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in 3D human motion and language integration have primarily focused on text-to-motion generation, leaving the task of motion understanding relatively unexplored. We introduce Dense Motion Captioning, a novel task that aims to temporally localize and caption actions within 3D human motion sequences. Current datasets fall short in providing detailed temporal annotations and predominantly consist of short sequences featuring few actions. To overcome these limitations, we present the Complex Motion Dataset (CompMo), the first large-scale dataset featuring richly annotated, complex motion sequences with precise temporal boundaries. Built through a carefully designed data generation pipeline, CompMo includes 60,000 motion sequences, each composed of multiple actions ranging from at least two to ten, accurately annotated with their temporal extents. We further present DEMO, a model that integrates a large language model with a simple motion adapter, trained to generate dense, temporally grounded captions. Our experiments show that DEMO substantially outperforms existing methods on CompMo as well as on adapted benchmarks, establishing a robust baseline for future research in 3D motion understanding and captioning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05369) | **Categories:** cs.CV, I.2.10; I.4.8; I.5.4

---

### [2] [Pressure2Motion: Hierarchical Motion Synthesis from Ground Pressure with Text Guidance](https://arxiv.org/abs/2511.05038)
*Zhengxuan Li, Qinhui Yang, Yiyu Zhuang, Chuan Guo, Xinxin Zuo, Xiaoxiao Long, Yao Yao, Xun Cao, Qiu Shen, Hao Zhu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present Pressure2Motion, a novel motion capture algorithm that synthesizes human motion from a ground pressure sequence and text prompt. It eliminates the need for specialized lighting setups, cameras, or wearable devices, making it suitable for privacy-preserving, low-light, and low-cost motion capture scenarios. Such a task is severely ill-posed due to the indeterminate nature of the pressure signals to full-body motion. To address this issue, we introduce Pressure2Motion, a generative model that leverages pressure features as input and utilizes a text prompt as a high-level guiding constraint. Specifically, our model utilizes a dual-level feature extractor that accurately interprets pressure data, followed by a hierarchical diffusion model that discerns broad-scale movement trajectories and subtle posture adjustments. Both the physical cues gained from the pressure sequence and the semantic guidance derived from descriptive texts are leveraged to guide the motion generation with precision. To the best of our knowledge, Pressure2Motion is a pioneering work in leveraging both pressure data and linguistic priors for motion generation, and the established MPL benchmark is the first benchmark for this task. Experiments show our method generates high-fidelity, physically plausible motions, establishing a new state-of-the-art for this task. The codes and benchmarks will be publicly released upon publication.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05038) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [QuAnTS: Question Answering on Time Series](https://arxiv.org/abs/2511.05124)
*Felix Divo, Maurice Kraus, Anh Q. Nguyen, Hao Xue, Imran Razzak, Flora D. Salim, Kristian Kersting, Devendra Singh Dhami*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Text offers intuitive access to information. This can, in particular, complement the density of numerical time series, thereby allowing improved interactions with time series models to enhance accessibility and decision-making. While the creation of question-answering datasets and models has recently seen remarkable growth, most research focuses on question answering (QA) on vision and text, with time series receiving minute attention. To bridge this gap, we propose a challenging novel time series QA (TSQA) dataset, QuAnTS, for Question Answering on Time Series data. Specifically, we pose a wide variety of questions and answers about human motion in the form of tracked skeleton trajectories. We verify that the large-scale QuAnTS dataset is well-formed and comprehensive through extensive experiments. Thoroughly evaluating existing and newly proposed baselines then lays the groundwork for a deeper exploration of TSQA using QuAnTS. Additionally, we provide human performances as a key reference for gauging the practical usability of such models. We hope to encourage future research on interacting with time series models through text, enabling better decision-making and more transparent systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05124) | **Categories:** cs.LG, I.2.6; I.2.7

---

### [2] [SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning](https://arxiv.org/abs/2511.05355)
*Tzu-Yuan Huang, Armin Lederer, Dai-Jie Wu, Xiaobing Dai, Sihua Zhang, Stefan Sosnowski, Shao-Hua Sun, Sandra Hirche*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Flow matching (FM) has shown promising results in data-driven planning. However, it inherently lacks formal guarantees for ensuring state and action constraints, whose satisfaction is a fundamental and crucial requirement for the safety and admissibility of planned trajectories on various systems. Moreover, existing FM planners do not ensure the dynamical consistency, which potentially renders trajectories inexecutable. We address these shortcomings by proposing SAD-Flower, a novel framework for generating Safe, Admissible, and Dynamically consistent trajectories. Our approach relies on an augmentation of the flow with a virtual control input. Thereby, principled guidance can be derived using techniques from nonlinear control theory, providing formal guarantees for state constraints, action constraints, and dynamic consistency. Crucially, SAD-Flower operates without retraining, enabling test-time satisfaction of unseen constraints. Through extensive experiments across several tasks, we demonstrate that SAD-Flower outperforms various generative-model-based baselines in ensuring constraint satisfaction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05355) | **Categories:** cs.LG, cs.RO, cs.SY, eess.SY

---


## 机器人学 (Robotics) [cs.RO]
### [1] [ReGen: Generative Robot Simulation via Inverse Design](https://arxiv.org/abs/2511.04769)
*Phat Nguyen, Tsun-Hsuan Wang, Zhang-Wei Hong, Erfan Aasi, Andrew Silva, Guy Rosman, Sertac Karaman, Daniela Rus*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Simulation plays a key role in scaling robot learning and validating policies, but constructing simulations remains a labor-intensive process. This paper introduces ReGen, a generative simulation framework that automates simulation design via inverse design. Given a robot's behavior -- such as a motion trajectory or an objective function -- and its textual description, ReGen infers plausible scenarios and environments that could have caused the behavior. ReGen leverages large language models to synthesize scenarios by expanding a directed graph that encodes cause-and-effect relationships, relevant entities, and their properties. This structured graph is then translated into a symbolic program, which configures and executes a robot simulation environment. Our framework supports (i) augmenting simulations based on ego-agent behaviors, (ii) controllable, counterfactual scenario generation, (iii) reasoning about agent cognition and mental states, and (iv) reasoning with distinct sensing modalities, such as braking due to faulty GPS signals. We demonstrate ReGen in autonomous driving and robot manipulation tasks, generating more diverse, complex simulated environments compared to existing simulations with high success rates, and enabling controllable generation for corner cases. This approach enhances the validation of robot policies and supports data or simulation augmentation, advancing scalable robot learning for improved generalization and robustness. We provide code and example videos at: https://regen-sim.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.04769) | **Categories:** cs.RO

---

### [2] [Conformalized Non-uniform Sampling Strategies for Accelerated Sampling-based Motion Planning](https://arxiv.org/abs/2511.04835)
*Shubham Natraj, Bruno Sinopoli, Yiannis Kantaros*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Sampling-based motion planners (SBMPs) are widely used to compute dynamically feasible robot paths. However, their reliance on uniform sampling often leads to poor efficiency and slow planning in complex environments. We introduce a novel non-uniform sampling strategy that integrates into existing SBMPs by biasing sampling toward `certified' regions. These regions are constructed by (i) generating an initial, possibly infeasible, path using any heuristic path predictor (e.g., A* or vision-language models) and (ii) applying conformal prediction to quantify the predictor's uncertainty. This process yields prediction sets around the initial-guess path that are guaranteed, with user-specified probability, to contain the optimal solution. To our knowledge, this is the first non-uniform sampling approach for SBMPs that provides such probabilistically correct guarantees on the sampling regions. Extensive evaluations demonstrate that our method consistently finds feasible paths faster and generalizes better to unseen environments than existing baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.04835) | **Categories:** cs.RO

---

### [3] [iFlyBot-VLM Technical Report](https://arxiv.org/abs/2511.04976)
*Xin Nie, Zhiyuan Cheng, Yuan Zhang, Chao Ji, Jiajia Wu, Yuhan Zhang, Jia Pan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.04976) | **Categories:** cs.RO

---

### [4] [Follow-Me in Micro-Mobility with End-to-End Imitation Learning](https://arxiv.org/abs/2511.05158)
*Sahar Salimpour, Iacopo Catalano, Tomi Westerlund, Mohsen Falahi, Jorge Peña Queralta*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous micro-mobility platforms face challenges from the perspective of the typical deployment environment: large indoor spaces or urban areas that are potentially crowded and highly dynamic. While social navigation algorithms have progressed significantly, optimizing user comfort and overall user experience over other typical metrics in robotics (e.g., time or distance traveled) is understudied. Specifically, these metrics are critical in commercial applications. In this paper, we show how imitation learning delivers smoother and overall better controllers, versus previously used manually-tuned controllers. We demonstrate how DAAV's autonomous wheelchair achieves state-of-the-art comfort in follow-me mode, in which it follows a human operator assisting persons with reduced mobility (PRM). This paper analyzes different neural network architectures for end-to-end control and demonstrates their usability in real-world production-level deployments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05158) | **Categories:** cs.RO, cs.LG

---

### [5] [Let Me Show You: Learning by Retrieving from Egocentric Video for Robotic Manipulation](https://arxiv.org/abs/2511.05199)
*Yichen Zhu, Feifei Feng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robots operating in complex and uncertain environments face considerable challenges. Advanced robotic systems often rely on extensive datasets to learn manipulation tasks. In contrast, when humans are faced with unfamiliar tasks, such as assembling a chair, a common approach is to learn by watching video demonstrations. In this paper, we propose a novel method for learning robot policies by Retrieving-from-Video (RfV), using analogies from human demonstrations to address manipulation tasks. Our system constructs a video bank comprising recordings of humans performing diverse daily tasks. To enrich the knowledge from these videos, we extract mid-level information, such as object affordance masks and hand motion trajectories, which serve as additional inputs to enhance the robot model's learning and generalization capabilities. We further feature a dual-component system: a video retriever that taps into an external video bank to fetch task-relevant video based on task specification, and a policy generator that integrates this retrieved knowledge into the learning cycle. This approach enables robots to craft adaptive responses to various scenarios and generalize to tasks beyond those in the training data. Through rigorous testing in multiple simulated and real-world settings, our system demonstrates a marked improvement in performance over conventional robotic systems, showcasing a significant breakthrough in the field of robotics.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.05199) | **Categories:** cs.RO

---
