# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-04

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (4)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Realistic pedestrian-driver interaction modelling using multi-agent RL with human perceptual-motor constraints](https://arxiv.org/abs/2510.27383)
*Yueyang Wang, Mehmet Dogar, Gustav Markkula*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modelling pedestrian-driver interactions is critical for understanding human road user behaviour and developing safe autonomous vehicle systems. Existing approaches often rely on rule-based logic, game-theoretic models, or 'black-box' machine learning methods. However, these models typically lack flexibility or overlook the underlying mechanisms, such as sensory and motor constraints, which shape how pedestrians and drivers perceive and act in interactive scenarios. In this study, we propose a multi-agent reinforcement learning (RL) framework that integrates both visual and motor constraints of pedestrian and driver agents. Using a real-world dataset from an unsignalised pedestrian crossing, we evaluate four model variants, one without constraints, two with either motor or visual constraints, and one with both, across behavioural metrics of interaction realism. Results show that the combined model with both visual and motor constraints performs best. Motor constraints lead to smoother movements that resemble human speed adjustments during crossing interactions. The addition of visual constraints introduces perceptual uncertainty and field-of-view limitations, leading the agents to exhibit more cautious and variable behaviour, such as less abrupt deceleration. In this data-limited setting, our model outperforms a supervised behavioural cloning model, demonstrating that our approach can be effective without large training datasets. Finally, our framework accounts for individual differences by modelling parameters controlling the human constraints as population-level distributions, a perspective that has not been explored in previous work on pedestrian-vehicle interaction modelling. Overall, our work demonstrates that multi-agent RL with human constraints is a promising modelling approach for simulating realistic road user interactions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27383) | **Categories:** cs.AI

---

### [2] [GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](https://arxiv.org/abs/2510.27210)
*Tao Liu, Chongyu Wang, Rongjie Li, Yingchen Yu, Xuming He, Bai Song*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While Multimodal Large Language Models (MLLMs) have advanced GUI navigation agents, current approaches face limitations in cross-domain generalization and effective history utilization. We present a reasoning-enhanced framework that systematically integrates structured reasoning, action prediction, and history summarization. The structured reasoning component generates coherent Chain-of-Thought analyses combining progress estimation and decision reasoning, which inform both immediate action predictions and compact history summaries for future steps. Based on this framework, we train a GUI agent, \textbf{GUI-Rise}, through supervised fine-tuning on pseudo-labeled trajectories and reinforcement learning with Group Relative Policy Optimization (GRPO). This framework employs specialized rewards, including a history-aware objective, directly linking summary quality to subsequent action performance. Comprehensive evaluations on standard benchmarks demonstrate state-of-the-art results under identical training data conditions, with particularly strong performance in out-of-domain scenarios. These findings validate our framework's ability to maintain robust reasoning and generalization across diverse GUI navigation tasks. Code is available at https://leon022.github.io/GUI-Rise.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27210) | **Categories:** cs.AI, cs.CV

---

### [3] [Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning](https://arxiv.org/abs/2510.27623)
*Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27623) | **Categories:** cs.AI, cs.CL, cs.CV

---

### [4] [Interaction as Intelligence Part II: Asynchronous Human-Agent Rollout for Long-Horizon Task Training](https://arxiv.org/abs/2510.27630)
*Dayuan Fu, Yunze Wu, Xiaojie Cai, Lyumanshan Ye, Shijie Xia, Zhen Huang, Weiye Si, Tianze Xu, Jie Sun, Keyu Li, Mohan Jiang, Junfei Wang, Qishuo Hua, Pengrui Lu, Yang Xiao, Pengfei Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Model (LLM) agents have recently shown strong potential in domains such as automated coding, deep research, and graphical user interface manipulation. However, training them to succeed on long-horizon, domain-specialized tasks remains challenging. Current methods primarily fall into two categories. The first relies on dense human annotations through behavior cloning, which is prohibitively expensive for long-horizon tasks that can take days or months. The second depends on outcome-driven sampling, which often collapses due to the rarity of valid positive trajectories on domain-specialized tasks. We introduce Apollo, a sampling framework that integrates asynchronous human guidance with action-level data filtering. Instead of requiring annotators to shadow every step, Apollo allows them to intervene only when the agent drifts from a promising trajectory, by providing prior knowledge, strategic advice, etc. This lightweight design makes it possible to sustain interactions for over 30 hours and produces valuable trajectories at a lower cost. Apollo then applies supervision control to filter out sub-optimal actions and prevent error propagation. Together, these components enable reliable and effective data collection in long-horizon environments. To demonstrate the effectiveness of Apollo, we evaluate it using InnovatorBench. Our experiments show that when applied to train the GLM-4.5 model on InnovatorBench, Apollo achieves more than a 50% improvement over the untrained baseline and a 28% improvement over a variant trained without human interaction. These results highlight the critical role of human-in-the-loop sampling and the robustness of Apollo's design in handling long-horizon, domain-specialized tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27630) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Enhancing Spatio-Temporal Zero-shot Action Recognition with Language-driven Description Attributes](https://arxiv.org/abs/2510.27255)
*Yehna Kim andYoung-Eun Kim, Seong-Whan Lee*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language Models (VLMs) have demonstrated impressive capabilities in zero-shot action recognition by learning to associate video embeddings with class embeddings. However, a significant challenge arises when relying solely on action classes to provide semantic context, particularly due to the presence of multi-semantic words, which can introduce ambiguity in understanding the intended concepts of actions. To address this issue, we propose an innovative approach that harnesses web-crawled descriptions, leveraging a large-language model to extract relevant keywords. This method reduces the need for human annotators and eliminates the laborious manual process of attribute data creation. Additionally, we introduce a spatio-temporal interaction module designed to focus on objects and action units, facilitating alignment between description attributes and video content. In our zero-shot experiments, our model achieves impressive results, attaining accuracies of 81.0%, 53.1%, and 68.9% on UCF-101, HMDB-51, and Kinetics-600, respectively, underscoring the model's adaptability and effectiveness across various downstream tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27255) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Mind the Gaps: Auditing and Reducing Group Inequity in Large-Scale Mobility Prediction](https://arxiv.org/abs/2510.26940)
*Ashwin Kumar, Hanyu Zhang, David A. Schweidel, William Yeoh*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Next location prediction underpins a growing number of mobility, retail, and public-health applications, yet its societal impacts remain largely unexplored. In this paper, we audit state-of-the-art mobility prediction models trained on a large-scale dataset, highlighting hidden disparities based on user demographics. Drawing from aggregate census data, we compute the difference in predictive performance on racial and ethnic user groups and show a systematic disparity resulting from the underlying dataset, resulting in large differences in accuracy based on location and user groups. To address this, we propose Fairness-Guided Incremental Sampling (FGIS), a group-aware sampling strategy designed for incremental data collection settings. Because individual-level demographic labels are unavailable, we introduce Size-Aware K-Means (SAKM), a clustering method that partitions users in latent mobility space while enforcing census-derived group proportions. This yields proxy racial labels for the four largest groups in the state: Asian, Black, Hispanic, and White. Built on these labels, our sampling algorithm prioritizes users based on expected performance gains and current group representation. This method incrementally constructs training datasets that reduce demographic performance gaps while preserving overall accuracy. Our method reduces total disparity between groups by up to 40\% with minimal accuracy trade-offs, as evaluated on a state-of-art MetaPath2Vec model and a transformer-encoder model. Improvements are most significant in early sampling stages, highlighting the potential for fairness-aware strategies to deliver meaningful gains even in low-resource settings. Our findings expose structural inequities in mobility prediction pipelines and demonstrate how lightweight, data-centric interventions can improve fairness with little added complexity, especially for low-data applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26940) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [NaviTrace: Evaluating Embodied Navigation of Vision-Language Models](https://arxiv.org/abs/2510.26909)
*Tim Windecker, Manthan Patel, Moritz Reuss, Richard Schwarzkopf, Cesar Cadena, Rudolf Lioutikov, Marco Hutter, Jonas Frey*

Main category: cs.RO

TL;DR: NaviTrace是一个视觉问答基准，用于评估视觉语言模型在机器人导航中的能力，通过语义感知的轨迹评分来揭示模型与人类表现的差距。


<details>
  <summary>Details</summary>
Motivation: 现有机器人导航模型评估方法成本高、仿真过于简单且基准有限，阻碍了通用机器人的发展。

Method: 提出了NaviTrace基准，包含1000个场景和3000多条专家轨迹，模型接收指令和机器人类型，输出2D导航轨迹，并使用语义感知的轨迹评分进行评估。

Result: 对八个先进的视觉语言模型进行了评估，揭示了模型在空间定位和目标定位方面与人类表现存在差距。

Conclusion: NaviTrace建立了一个可扩展和可复现的机器人导航基准，为评估和改进视觉语言模型在实际机器人应用中的导航能力提供了平台。

Abstract: 视觉语言模型在各种任务和场景中表现出前所未有的性能和泛化能力。将这些基础模型集成到机器人导航系统中，为构建通用机器人开辟了道路。然而，评估这些模型的导航能力仍然受到代价高昂的现实世界试验、过于简化的模拟和有限的基准的限制。我们引入了NaviTrace，这是一个高质量的视觉问答基准，其中模型接收指令和机器人类型（人类、腿式机器人、轮式机器人、自行车），并且必须在图像空间中输出2D导航轨迹。在1000个场景和3000多个专家轨迹中，我们使用新引入的语义感知轨迹评分系统地评估了八个最先进的视觉语言模型。该指标结合了动态时间规整距离、目标端点误差和源自每像素语义的机器人条件惩罚，并且与人类偏好相关。我们的评估揭示了由于较差的空间定位和目标定位而导致与人类表现的持续差距。NaviTrace为现实世界机器人导航建立了一个可扩展且可重现的基准。基准和排行榜可在https://leggedrobotics.github.io/navitrace_webpage/找到。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26909) | **Categories:** cs.RO

---

### [2] [Heterogeneous Robot Collaboration in Unstructured Environments with Grounded Generative Intelligence](https://arxiv.org/abs/2510.26915)
*Zachary Ravichandran, Fernando Cladera, Ankit Prabhu, Jason Hughes, Varun Murali, Camillo Taylor, George J. Pappas, Vijay Kumar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Heterogeneous robot teams operating in realistic settings often must accomplish complex missions requiring collaboration and adaptation to information acquired online. Because robot teams frequently operate in unstructured environments -- uncertain, open-world settings without prior maps -- subtasks must be grounded in robot capabilities and the physical world. While heterogeneous teams have typically been designed for fixed specifications, generative intelligence opens the possibility of teams that can accomplish a wide range of missions described in natural language. However, current large language model (LLM)-enabled teaming methods typically assume well-structured and known environments, limiting deployment in unstructured environments. We present SPINE-HT, a framework that addresses these limitations by grounding the reasoning abilities of LLMs in the context of a heterogeneous robot team through a three-stage process. Given language specifications describing mission goals and team capabilities, an LLM generates grounded subtasks which are validated for feasibility. Subtasks are then assigned to robots based on capabilities such as traversability or perception and refined given feedback collected during online operation. In simulation experiments with closed-loop perception and control, our framework achieves nearly twice the success rate compared to prior LLM-enabled heterogeneous teaming approaches. In real-world experiments with a Clearpath Jackal, a Clearpath Husky, a Boston Dynamics Spot, and a high-altitude UAV, our method achieves an 87\% success rate in missions requiring reasoning about robot capabilities and refining subtasks with online feedback. More information is provided at https://zacravichandran.github.io/SPINE-HT.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.26915) | **Categories:** cs.RO, cs.AI

---

### [3] [Modified-Emergency Index (MEI): A Criticality Metric for Autonomous Driving in Lateral Conflict](https://arxiv.org/abs/2510.27333)
*Hao Cheng, Yanbo Jiang, Qingyuan Shi, Qingwen Meng, Keyu Chen, Wenhao Yu, Jianqiang Wang, Sifa Zheng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective, reliable, and efficient evaluation of autonomous driving safety is essential to demonstrate its trustworthiness. Criticality metrics provide an objective means of assessing safety. However, as existing metrics primarily target longitudinal conflicts, accurately quantifying the risks of lateral conflicts - prevalent in urban settings - remains challenging. This paper proposes the Modified-Emergency Index (MEI), a metric designed to quantify evasive effort in lateral conflicts. Compared to the original Emergency Index (EI), MEI refines the estimation of the time available for evasive maneuvers, enabling more precise risk quantification. We validate MEI on a public lateral conflict dataset based on Argoverse-2, from which we extract over 1,500 high-quality AV conflict cases, including more than 500 critical events. MEI is then compared with the well-established ACT and the widely used PET metrics. Results show that MEI consistently outperforms them in accurately quantifying criticality and capturing risk evolution. Overall, these findings highlight MEI as a promising metric for evaluating urban conflicts and enhancing the safety assessment framework for autonomous driving. The open-source implementation is available at https://github.com/AutoChengh/MEI.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27333) | **Categories:** cs.RO

---

### [4] [Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs](https://arxiv.org/abs/2510.27558)
*Sushil Samuel Dinesh, Shinkyu Park*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a framework that leverages pre-trained foundation models for robotic manipulation without domain-specific training. The framework integrates off-the-shelf models, combining multimodal perception from foundation models with a general-purpose reasoning model capable of robust task sequencing. Scene graphs, dynamically maintained within the framework, provide spatial awareness and enable consistent reasoning about the environment. The framework is evaluated through a series of tabletop robotic manipulation experiments, and the results highlight its potential for building robotic manipulation systems directly on top of off-the-shelf foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.27558) | **Categories:** cs.RO, cs.AI, cs.LG

---
