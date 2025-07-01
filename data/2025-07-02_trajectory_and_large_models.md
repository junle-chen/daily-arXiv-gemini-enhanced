# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-02

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (4)](#cs-lg)
- [机器人学 (Robotics) (13)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [GATSim: Urban Mobility Simulation with Generative Agents](https://arxiv.org/abs/2506.23306)
*Qi Liu, Can Li, Wanjing Ma*

Main category: cs.AI

TL;DR: GATSim 提出了一种新的框架，利用生成式智能体进行城市交通模拟，能够产生可信的出行行为。


<details>
  <summary>Details</summary>
Motivation: 传统的基于智能体的城市交通模拟依赖于刚性的、基于规则的系统，无法捕捉人类出行决策的复杂性、适应性和行为多样性。

Method: GATSim 框架，结合城市交通基础模型与智能体认知系统和交通仿真环境。

Result: 实验表明，在出行场景中，生成式智能体的性能与人类标注者相比具有竞争力，并且能够自然地产生宏观的交通演变模式。

Conclusion: 生成式智能体能够产生可信的出行行为，并在出行场景中与人类标注者表现出竞争性，同时自然地产生宏观交通演化模式。

Abstract: 传统的基于智能体的城市交通模拟依赖于刚性的、基于规则的系统，无法捕捉人类出行决策的复杂性、适应性和行为多样性。大型语言模型和人工智能智能体技术的最新进展为创建具有推理能力、持久记忆和自适应学习机制的智能体提供了机会。我们提出了 GATSim（生成式智能体交通模拟），这是一个新颖的框架，它利用这些进展来创建具有丰富行为特征的生成式智能体，用于城市交通模拟。与传统方法不同，GATSim 智能体具有不同的社会经济属性、个人生活方式和不断变化的偏好，这些属性通过心理学上的记忆系统、工具使用能力和终身学习机制来影响他们的出行决策。本研究的主要贡献包括：（1）一个综合的架构，将城市交通基础模型与智能体认知系统和交通仿真环境相结合，（2）一个功能齐全的原型实现，以及（3）系统的验证，证明生成式智能体能够产生可信的出行行为。通过设计的反思过程，本研究中的生成式智能体可以将特定的出行体验转化为广义的见解，从而实现现实的行为随时间推移的适应，并具有专门的活动计划机制和针对城市交通环境量身定制的实时反应行为。实验表明，生成式智能体在出行场景中的表现与人类标注者相比具有竞争力，同时自然地产生宏观的交通演变模式。原型系统的代码已在 https://github.com/qiliuchn/gatsim 上共享。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23306) | **Categories:** cs.AI

---

### [2] [Advancing Learnable Multi-Agent Pathfinding Solvers with Active Fine-Tuning](https://arxiv.org/abs/2506.23793)
*Anton Andreychuk, Konstantin Yakovlev, Aleksandr Panov, Alexey Skrynnik*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-agent pathfinding (MAPF) is a common abstraction of multi-robot trajectory planning problems, where multiple homogeneous robots simultaneously move in the shared environment. While solving MAPF optimally has been proven to be NP-hard, scalable, and efficient, solvers are vital for real-world applications like logistics, search-and-rescue, etc. To this end, decentralized suboptimal MAPF solvers that leverage machine learning have come on stage. Building on the success of the recently introduced MAPF-GPT, a pure imitation learning solver, we introduce MAPF-GPT-DDG. This novel approach effectively fine-tunes the pre-trained MAPF model using centralized expert data. Leveraging a novel delta-data generation mechanism, MAPF-GPT-DDG accelerates training while significantly improving performance at test time. Our experiments demonstrate that MAPF-GPT-DDG surpasses all existing learning-based MAPF solvers, including the original MAPF-GPT, regarding solution quality across many testing scenarios. Remarkably, it can work with MAPF instances involving up to 1 million agents in a single environment, setting a new milestone for scalability in MAPF domains.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23793) | **Categories:** cs.AI, cs.LG, cs.MA

---

### [3] [Bootstrapping Human-Like Planning via LLMs](https://arxiv.org/abs/2506.22604)
*David Porfirio, Vincent Hsiao, Morgan Fine-Morris, Leslie Smith, Laura M. Hiatt*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robot end users increasingly require accessible means of specifying tasks for robots to perform. Two common end-user programming paradigms include drag-and-drop interfaces and natural language programming. Although natural language interfaces harness an intuitive form of human communication, drag-and-drop interfaces enable users to meticulously and precisely dictate the key actions of the robot's task. In this paper, we investigate the degree to which both approaches can be combined. Specifically, we construct a large language model (LLM)-based pipeline that accepts natural language as input and produces human-like action sequences as output, specified at a level of granularity that a human would produce. We then compare these generated action sequences to another dataset of hand-specified action sequences. Although our results reveal that larger models tend to outperform smaller ones in the production of human-like action sequences, smaller models nonetheless achieve satisfactory performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22604) | **Categories:** cs.AI, cs.HC, cs.RO

---

### [4] [MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning](https://arxiv.org/abs/2506.22992)
*Yulun Jiang, Yekun Chai, Maria Brbić, Michael Moor*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The ability to process information from multiple modalities and to reason through it step-by-step remains a critical challenge in advancing artificial intelligence. However, existing reasoning benchmarks focus on text-only reasoning, or employ multimodal questions that can be answered by directly retrieving information from a non-text modality. Thus, complex reasoning remains poorly understood in multimodal domains. Here, we present MARBLE, a challenging multimodal reasoning benchmark that is designed to scrutinize multimodal language models (MLLMs) in their ability to carefully reason step-by-step through complex multimodal problems and environments. MARBLE is composed of two highly challenging tasks, M-Portal and M-Cube, that require the crafting and understanding of multistep plans under spatial, visual, and physical constraints. We find that current MLLMs perform poorly on MARBLE -- all the 12 advanced models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Only in simplified subtasks some models outperform the random baseline, indicating that complex reasoning is still a challenge for existing MLLMs. Moreover, we show that perception remains a bottleneck, where MLLMs occasionally fail to extract information from the visual inputs. By shedding a light on the limitations of MLLMs, we hope that MARBLE will spur the development of the next generation of models with the ability to reason and plan across many, multimodal reasoning steps.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22992) | **Categories:** cs.AI, cs.CL, cs.CV

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset](https://arxiv.org/abs/2506.22554)
*Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero, Morteza Behrooz, Julia Buffalini, Fabio Maria Carlucci, Joy Chen, Junming Chen, Zhang Chen, Shiyang Cheng, Praveen Chowdary, Joe Chuang, Antony D'Avirro, Jon Daly, Ning Dong, Mark Duppenthaler, Cynthia Gao, Jeff Girard, Martin Gleize, Sahir Gomez, Hongyu Gong, Srivathsan Govindarajan, Brandon Han, Sen He, Denise Hernandez, Yordan Hristov, Rongjie Huang, Hirofumi Inaguma, Somya Jain, Raj Janardhan, Qingyao Jia, Christopher Klaiber, Dejan Kovachev, Moneish Kumar, Hang Li, Yilei Li, Pavel Litvin, Wei Liu, Guangyao Ma, Jing Ma, Martin Ma, Xutai Ma, Lucas Mantovani, Sagar Miglani, Sreyas Mohan, Louis-Philippe Morency, Evonne Ng, Kam-Woh Ng, Tu Anh Nguyen, Amia Oberai, Benjamin Peloquin, Juan Pino, Jovan Popovic, Omid Poursaeed, Fabian Prada, Alice Rakotoarison, Alexander Richard, Christophe Ropers, Safiyyah Saleem, Vasu Sharma, Alex Shcherbyna, Jia Shen, Jie Shen, Anastasis Stathopoulos, Anna Sun, Paden Tomasello, Tuan Tran, Arina Turkatenko, Bo Wan, Chao Wang, Jeff Wang, Mary Williamson, Carleigh Wood, Tao Xiang, Yilin Yang, Julien Yao, Chen Zhang, Jiemin Zhang, Xinyue Zhang, Jason Zheng, Pavlo Zhyzheria, Jan Zikes, Michael Zollhoefer*

Main category: cs.CV

TL;DR: 本文提出了一个大规模的面对面互动数据集，并开发了一套模型来生成与人类语音对齐的二元运动手势和面部表情，从而推进更直观的人机交互。


<details>
  <summary>Details</summary>
Motivation: Developing socially intelligent AI technologies requires models that can comprehend and generate dyadic behavioral dynamics.

Method: The authors develop a suite of models that utilize the Seamless Interaction Dataset to generate dyadic motion gestures and facial expressions aligned with human speech. These models can take as input both the speech and visual behavior of their interlocutors.

Result: The paper introduces the Seamless Interaction Dataset, a large-scale collection of over 4,000 hours of face-to-face interaction footage. The developed models generate dyadic motion gestures and facial expressions aligned with human speech.

Conclusion: The paper demonstrates the potential for more intuitive and responsive human-AI interactions through the development of dyadic motion models.

Abstract: 人类交流涉及语言和非语言信号的复杂互动，这对于传达意义和实现人际目标至关重要。为了开发具有社交智能的AI技术，至关重要的是开发能够理解和生成二元行为动态的模型。为此，我们推出了Seamless Interaction Dataset，这是一个大规模的数据集，包含来自4000多名参与者在各种情境下的4000多个小时的面对面互动视频。该数据集能够开发理解二元具身动态的AI技术，从而在虚拟代理、远程呈现体验和多模态内容分析工具中取得突破。我们还开发了一套模型，利用该数据集生成与人类语音对齐的二元运动手势和面部表情。这些模型可以将对话者的语音和视觉行为作为输入。我们展示了一个带有来自LLM模型的语音的变体，以及与2D和3D渲染方法的集成，使我们更接近交互式虚拟代理。此外，我们还描述了我们运动模型的可控变体，这些变体可以适应情绪反应和表达水平，以及生成更多语义相关的手势。最后，我们讨论了评估这些二元运动模型质量的方法，这些方法展示了更直观和响应式人机交互的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22554) | **Categories:** cs.CV, cs.AI

---

### [2] [RoboPearls: Editable Video Simulation for Robot Manipulation](https://arxiv.org/abs/2506.22756)
*Tao Tang, Likui Zhang, Youpeng Wen, Kaidong Zhang, Jia-Wang Bian, xia zhou, Tianyi Yan, Kun Zhan, Peng Jia, Hefeng Wu, Liang Lin, Xiaodan Liang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The development of generalist robot manipulation policies has seen significant progress, driven by large-scale demonstration data across diverse environments. However, the high cost and inefficiency of collecting real-world demonstrations hinder the scalability of data acquisition. While existing simulation platforms enable controlled environments for robotic learning, the challenge of bridging the sim-to-real gap remains. To address these challenges, we propose RoboPearls, an editable video simulation framework for robotic manipulation. Built on 3D Gaussian Splatting (3DGS), RoboPearls enables the construction of photo-realistic, view-consistent simulations from demonstration videos, and supports a wide range of simulation operators, including various object manipulations, powered by advanced modules like Incremental Semantic Distillation (ISD) and 3D regularized NNFM Loss (3D-NNFM). Moreover, by incorporating large language models (LLMs), RoboPearls automates the simulation production process in a user-friendly manner through flexible command interpretation and execution. Furthermore, RoboPearls employs a vision-language model (VLM) to analyze robotic learning issues to close the simulation loop for performance enhancement. To demonstrate the effectiveness of RoboPearls, we conduct extensive experiments on multiple datasets and scenes, including RLBench, COLOSSEUM, Ego4D, Open X-Embodiment, and a real-world robot, which demonstrate our satisfactory simulation performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22756) | **Categories:** cs.CV, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](https://arxiv.org/abs/2506.22716)
*Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian Del Carmen Hipolito Garcia, Menglin Xia, Laks V. S. Lakshmanan, Qingyun Wu, Victor Rühle*

Main category: cs.LG

TL;DR: 提出了一种名为BEST-Route的新型路由框架，该框架通过选择模型和采样响应的数量来降低LLM的部署成本，实验表明该方法在性能影响很小的情况下显著降低了成本。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型（LLM）是强大的工具，但通常大规模部署成本高昂。LLM查询路由通过将查询动态分配给不同成本和质量的模型来缓解这种情况，以获得所需的权衡。

Method: 提出了一种新的路由框架BEST-Route，该框架根据查询难度和质量阈值选择模型以及从中采样的响应数量。

Result: 实验表明，该方法在性能下降小于1%的情况下，可以将成本降低高达60%。

Conclusion: 该方法在实际数据集上进行了实验，结果表明，在性能下降小于1%的情况下，该方法可以将成本降低高达60%。

Abstract: 大型语言模型（LLM）是强大的工具，但通常大规模部署成本高昂。LLM查询路由通过将查询动态分配给不同成本和质量的模型来缓解这种情况，以获得所需的权衡。先前的查询路由方法仅从所选模型生成一个响应，而来自小型（廉价）模型的单个响应通常不足以胜过来自大型（昂贵）模型的响应，因此它们最终过度使用大型模型，错失了潜在的成本节约。然而，众所周知，对于小型模型，生成多个响应并选择最佳响应可以提高质量，同时保持比单个大型模型响应更便宜。我们利用这一想法提出了BEST-Route，这是一个新的路由框架，它根据查询难度和质量阈值选择模型以及从中采样的响应数量。在实际数据集上的实验表明，我们的方法可以将成本降低高达60%，而性能下降不到1%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22716) | **Categories:** cs.LG, cs.AI, cs.CL, cs.DB

---

### [2] [Latent Factorization of Tensors with Threshold Distance Weighted Loss for Traffic Data Estimation](https://arxiv.org/abs/2506.22441)
*Lei Yang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Intelligent transportation systems (ITS) rely heavily on complete and high-quality spatiotemporal traffic data to achieve optimal performance. Nevertheless, in real-word traffic data collection processes, issues such as communication failures and sensor malfunctions often lead to incomplete or corrupted datasets, thereby posing significant challenges to the advancement of ITS. Among various methods for imputing missing spatiotemporal traffic data, the latent factorization of tensors (LFT) model has emerged as a widely adopted and effective solution. However, conventional LFT models typically employ the standard L2-norm in their learning objective, which makes them vulnerable to the influence of outliers. To overcome this limitation, this paper proposes a threshold distance weighted (TDW) loss-incorporated Latent Factorization of Tensors (TDWLFT) model. The proposed loss function effectively reduces the model's sensitivity to outliers by assigning differentiated weights to individual samples. Extensive experiments conducted on two traffic speed datasets sourced from diverse urban environments confirm that the proposed TDWLFT model consistently outperforms state-of-the-art approaches in terms of both in both prediction accuracy and computational efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22441) | **Categories:** cs.LG, cs.AI

---

### [3] [Towards Time Series Generation Conditioned on Unstructured Natural Language](https://arxiv.org/abs/2506.22927)
*Jaeyun Woo, Jiseok Lee, Brian Kenji Iwana*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generative Artificial Intelligence (AI) has rapidly become a powerful tool, capable of generating various types of data, such as images and text. However, despite the significant advancement of generative AI, time series generative AI remains underdeveloped, even though the application of time series is essential in finance, climate, and numerous fields. In this research, we propose a novel method of generating time series conditioned on unstructured natural language descriptions. We use a diffusion model combined with a language model to generate time series from the text. Through the proposed method, we demonstrate that time series generation based on natural language is possible. The proposed method can provide various applications such as custom forecasting, time series manipulation, data augmentation, and transfer learning. Furthermore, we construct and propose a new public dataset for time series generation, consisting of 63,010 time series-description pairs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22927) | **Categories:** cs.LG

---

### [4] [Cybersecurity-Focused Anomaly Detection in Connected Autonomous Vehicles Using Machine Learning](https://arxiv.org/abs/2506.22984)
*Prathyush Kumar Reddy Lebaku, Lu Gao, Yunpeng Zhang, Zhixia Li, Yongxin Liu, Tanvir Arafin*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Anomaly detection in connected autonomous vehicles (CAVs) is crucial for maintaining safe and reliable transportation networks, as CAVs can be susceptible to sensor malfunctions, cyber-attacks, and unexpected environmental disruptions. This study explores an anomaly detection approach by simulating vehicle behavior, generating a dataset that represents typical and atypical vehicular interactions. The dataset includes time-series data of position, speed, and acceleration for multiple connected autonomous vehicles. We utilized machine learning models to effectively identify abnormal driving patterns. First, we applied a stacked Long Short-Term Memory (LSTM) model to capture temporal dependencies and sequence-based anomalies. The stacked LSTM model processed the sequential data to learn standard driving behaviors. Additionally, we deployed a Random Forest model to support anomaly detection by offering ensemble-based predictions, which enhanced model interpretability and performance. The Random Forest model achieved an R2 of 0.9830, MAE of 5.746, and a 95th percentile anomaly threshold of 14.18, while the stacked LSTM model attained an R2 of 0.9998, MAE of 82.425, and a 95th percentile anomaly threshold of 265.63. These results demonstrate the models' effectiveness in accurately predicting vehicle trajectories and detecting anomalies in autonomous driving scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22984) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [InfGen: Scenario Generation as Next Token Group Prediction](https://arxiv.org/abs/2506.23316)
*Zhenghao Peng, Yuxin Liu, Bolei Zhou*

Main category: cs.RO

TL;DR: InfGen是一个基于Transformer的交通仿真框架，可以生成逼真、多样且适应性强的交通场景。


<details>
  <summary>Details</summary>
Motivation: 现有的数据驱动仿真方法依赖于静态初始化或日志回放数据，限制了它们模拟具有不断变化的智能体群体的动态、长时程场景的能力。

Method: InfGen采用自回归方式输出智能体的状态和轨迹，将整个场景表示为令牌序列，并使用Transformer模型来模拟交通。

Result: InfGen能够生成逼真、多样且适应性强的交通行为。

Conclusion: InfGen生成的场景能够提高强化学习策略的鲁棒性和泛化能力，验证了其作为自动驾驶高保真仿真环境的有效性。

Abstract: 逼真且交互式的交通仿真对于训练和评估自动驾驶系统至关重要。然而，大多数现有的数据驱动仿真方法依赖于静态初始化或日志回放数据，限制了它们模拟具有不断变化的智能体群体的动态、长时程场景的能力。我们提出了InfGen，一个以自回归方式输出智能体状态和轨迹的场景生成框架。InfGen将整个场景表示为令牌序列，包括交通灯信号、智能体状态和运动向量，并使用Transformer模型来模拟交通随时间的变化。这种设计使InfGen能够不断地将新的智能体插入到交通中，支持无限的场景生成。实验表明，InfGen能够生成逼真、多样且适应性强的交通行为。此外，在InfGen生成的场景中训练的强化学习策略实现了卓越的鲁棒性和泛化能力，验证了其作为自动驾驶高保真仿真环境的效用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23316) | **Categories:** cs.RO, cs.CV

---

### [2] [Mode Collapse Happens: Evaluating Critical Interactions in Joint Trajectory Prediction Models](https://arxiv.org/abs/2506.23164)
*Maarten Hugenholtz, Anna Meszaros, Jens Kober, Zlatan Ajanovic*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous Vehicle decisions rely on multimodal prediction models that account for multiple route options and the inherent uncertainty in human behavior. However, models can suffer from mode collapse, where only the most likely mode is predicted, posing significant safety risks. While existing methods employ various strategies to generate diverse predictions, they often overlook the diversity in interaction modes among agents. Additionally, traditional metrics for evaluating prediction models are dataset-dependent and do not evaluate inter-agent interactions quantitatively. To our knowledge, none of the existing metrics explicitly evaluates mode collapse. In this paper, we propose a novel evaluation framework that assesses mode collapse in joint trajectory predictions, focusing on safety-critical interactions. We introduce metrics for mode collapse, mode correctness, and coverage, emphasizing the sequential dimension of predictions. By testing four multi-agent trajectory prediction models, we demonstrate that mode collapse indeed happens. When looking at the sequential dimension, although prediction accuracy improves closer to interaction events, there are still cases where the models are unable to predict the correct interaction mode, even just before the interaction mode becomes inevitable. We hope that our framework can help researchers gain new insights and advance the development of more consistent and accurate prediction models, thus enhancing the safety of autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23164) | **Categories:** cs.RO, cs.AI

---

### [3] [Predictive Risk Analysis and Safe Trajectory Planning for Intelligent and Connected Vehicles](https://arxiv.org/abs/2506.23999)
*Zeyu Han, Mengchi Cai, Chaoyi Chen, Qingwen Meng, Guangwei Wang, Ying Liu, Qing Xu, Jianqiang Wang, Keqiang Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The safe trajectory planning of intelligent and connected vehicles is a key component in autonomous driving technology. Modeling the environment risk information by field is a promising and effective approach for safe trajectory planning. However, existing risk assessment theories only analyze the risk by current information, ignoring future prediction. This paper proposes a predictive risk analysis and safe trajectory planning framework for intelligent and connected vehicles. This framework first predicts future trajectories of objects by a local risk-aware algorithm, following with a spatiotemporal-discretised predictive risk analysis using the prediction results. Then the safe trajectory is generated based on the predictive risk analysis. Finally, simulation and vehicle experiments confirm the efficacy and real-time practicability of our approach.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23999) | **Categories:** cs.RO

---

### [4] [DriveBLIP2: Attention-Guided Explanation Generation for Complex Driving Scenarios](https://arxiv.org/abs/2506.22494)
*Shihong Ling, Yue Wan, Xiaowei Jia, Na Du*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces a new framework, DriveBLIP2, built upon the BLIP2-OPT architecture, to generate accurate and contextually relevant explanations for emerging driving scenarios. While existing vision-language models perform well in general tasks, they encounter difficulties in understanding complex, multi-object environments, particularly in real-time applications such as autonomous driving, where the rapid identification of key objects is crucial. To address this limitation, an Attention Map Generator is proposed to highlight significant objects relevant to driving decisions within critical video frames. By directing the model's focus to these key regions, the generated attention map helps produce clear and relevant explanations, enabling drivers to better understand the vehicle's decision-making process in critical situations. Evaluations on the DRAMA dataset reveal significant improvements in explanation quality, as indicated by higher BLEU, ROUGE, CIDEr, and SPICE scores compared to baseline models. These findings underscore the potential of targeted attention mechanisms in vision-language models for enhancing explainability in real-time autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22494) | **Categories:** cs.RO, cs.CV, cs.LG

---

### [5] [SPI-BoTER: Error Compensation for Industrial Robots via Sparse Attention Masking and Hybrid Loss with Spatial-Physical Information](https://arxiv.org/abs/2506.22788)
*Xuao Hou, Yongquan Jia, Shijin Zhang, Yuqiang Wu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The widespread application of industrial robots in fields such as cutting and welding has imposed increasingly stringent requirements on the trajectory accuracy of end-effectors. However, current error compensation methods face several critical challenges, including overly simplified mechanism modeling, a lack of physical consistency in data-driven approaches, and substantial data requirements. These issues make it difficult to achieve both high accuracy and strong generalization simultaneously. To address these challenges, this paper proposes a Spatial-Physical Informed Attention Residual Network (SPI-BoTER). This method integrates the kinematic equations of the robotic manipulator with a Transformer architecture enhanced by sparse self-attention masks. A parameter-adaptive hybrid loss function incorporating spatial and physical information is employed to iteratively optimize the network during training, enabling high-precision error compensation under small-sample conditions. Additionally, inverse joint angle compensation is performed using a gradient descent-based optimization method. Experimental results on a small-sample dataset from a UR5 robotic arm (724 samples, with a train:test:validation split of 8:1:1) demonstrate the superior performance of the proposed method. It achieves a 3D absolute positioning error of 0.2515 mm with a standard deviation of 0.15 mm, representing a 35.16\% reduction in error compared to conventional deep neural network (DNN) methods. Furthermore, the inverse angle compensation algorithm converges to an accuracy of 0.01 mm within an average of 147 iterations. This study presents a solution that combines physical interpretability with data adaptability for high-precision control of industrial robots, offering promising potential for the reliable execution of precision tasks in intelligent manufacturing.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22788) | **Categories:** cs.RO

---

### [6] [Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation](https://arxiv.org/abs/2506.22827)
*André Schakkal, Ben Zandonati, Zhutian Yang, Navid Azizan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Enabling humanoid robots to reliably execute complex multi-step manipulation tasks is crucial for their effective deployment in industrial and household environments. This paper presents a hierarchical planning and control framework designed to achieve reliable multi-step humanoid manipulation. The proposed system comprises three layers: (1) a low-level RL-based controller responsible for tracking whole-body motion targets; (2) a mid-level set of skill policies trained via imitation learning that produce motion targets for different steps of a task; and (3) a high-level vision-language planning module that determines which skills should be executed and also monitors their completion in real-time using pretrained vision-language models (VLMs). Experimental validation is performed on a Unitree G1 humanoid robot executing a non-prehensile pick-and-place task. Over 40 real-world trials, the hierarchical system achieved a 72.5% success rate in completing the full manipulation sequence. These experiments confirm the feasibility of the proposed hierarchical system, highlighting the benefits of VLM-based skill planning and monitoring for multi-step manipulation scenarios. See https://vlp-humanoid.github.io/ for video demonstrations of the policy rollout.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22827) | **Categories:** cs.RO

---

### [7] [Safe Reinforcement Learning with a Predictive Safety Filter for Motion Planning and Control: A Drifting Vehicle Example](https://arxiv.org/abs/2506.22894)
*Bei Zhou, Baha Zarrouki, Mattia Piccinini, Cheng Hu, Lei Xie, Johannes Betz*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous drifting is a complex and crucial maneuver for safety-critical scenarios like slippery roads and emergency collision avoidance, requiring precise motion planning and control. Traditional motion planning methods often struggle with the high instability and unpredictability of drifting, particularly when operating at high speeds. Recent learning-based approaches have attempted to tackle this issue but often rely on expert knowledge or have limited exploration capabilities. Additionally, they do not effectively address safety concerns during learning and deployment. To overcome these limitations, we propose a novel Safe Reinforcement Learning (RL)-based motion planner for autonomous drifting. Our approach integrates an RL agent with model-based drift dynamics to determine desired drift motion states, while incorporating a Predictive Safety Filter (PSF) that adjusts the agent's actions online to prevent unsafe states. This ensures safe and efficient learning, and stable drift operation. We validate the effectiveness of our method through simulations on a Matlab-Carsim platform, demonstrating significant improvements in drift performance, reduced tracking errors, and computational efficiency compared to traditional methods. This strategy promises to extend the capabilities of autonomous vehicles in safety-critical maneuvers.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.22894) | **Categories:** cs.RO

---

### [8] [ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation](https://arxiv.org/abs/2506.23126)
*Suning Huang, Qianzhong Chen, Xiaohan Zhang, Jiankai Sun, Mac Schwager*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: 3D world models (i.e., learning-based 3D dynamics models) offer a promising approach to generalizable robotic manipulation by capturing the underlying physics of environment evolution conditioned on robot actions. However, existing 3D world models are primarily limited to single-material dynamics using a particle-based Graph Neural Network model, and often require time-consuming 3D scene reconstruction to obtain 3D particle tracks for training. In this work, we present ParticleFormer, a Transformer-based point cloud world model trained with a hybrid point cloud reconstruction loss, supervising both global and local dynamics features in multi-material, multi-object robot interactions. ParticleFormer captures fine-grained multi-object interactions between rigid, deformable, and flexible materials, trained directly from real-world robot perception data without an elaborate scene reconstruction. We demonstrate the model's effectiveness both in 3D scene forecasting tasks, and in downstream manipulation tasks using a Model Predictive Control (MPC) policy. In addition, we extend existing dynamics learning benchmarks to include diverse multi-material, multi-object interaction scenarios. We validate our method on six simulation and three real-world experiments, where it consistently outperforms leading baselines by achieving superior dynamics prediction accuracy and less rollout error in downstream visuomotor tasks. Experimental videos are available at https://particleformer.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23126) | **Categories:** cs.RO

---

### [9] [Flatness-based Finite-Horizon Multi-UAV Formation Trajectory Planning and Directionally Aware Collision Avoidance Tracking](https://arxiv.org/abs/2506.23129)
*Hossein B. Jond, Logan Beaver, Martin Jiroušek, Naiemeh Ahmadlou, Veli Bakırcıoğlu, Martin Saska*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Collision-free optimal formation control of unmanned aerial vehicle (UAV) teams is challenging. The state-of-the-art optimal control approaches often rely on numerical methods sensitive to initial guesses. This paper presents an innovative collision-free finite-time formation control scheme for multiple UAVs leveraging the differential flatness of the UAV dynamics, eliminating the need for numerical methods. We formulate a finite-time optimal control problem to plan a formation trajectory for feasible initial states. This formation trajectory planning optimal control problem involves a collective performance index to meet the formation requirements of achieving relative positions and velocity consensus. It is solved by applying Pontryagin's principle. Subsequently, a collision-constrained regulating problem is addressed to ensure collision-free tracking of the planned formation trajectory. The tracking problem incorporates a directionally aware collision avoidance strategy that prioritizes avoiding UAVs in the forward path and relative approach. It assigns lower priority to those on the sides with an oblique relative approach and disregards UAVs behind and not in the relative approach. The simulation results for a four-UAV team (re)formation problem confirm the efficacy of the proposed control scheme.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23129) | **Categories:** cs.RO

---

### [10] [Risk-Based Filtering of Valuable Driving Situations in the Waymo Open Motion Dataset](https://arxiv.org/abs/2506.23433)
*Tim Puphal, Vipul Ramtekkar, Kenji Nishimiya*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Improving automated vehicle software requires driving data rich in valuable road user interactions. In this paper, we propose a risk-based filtering approach that helps identify such valuable driving situations from large datasets. Specifically, we use a probabilistic risk model to detect high-risk situations. Our method stands out by considering a) first-order situations (where one vehicle directly influences another and induces risk) and b) second-order situations (where influence propagates through an intermediary vehicle). In experiments, we show that our approach effectively selects valuable driving situations in the Waymo Open Motion Dataset. Compared to the two baseline interaction metrics of Kalman difficulty and Tracks-To-Predict (TTP), our filtering approach identifies complex and complementary situations, enriching the quality in automated vehicle testing. The risk data is made open-source: https://github.com/HRI-EU/RiskBasedFiltering.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23433) | **Categories:** cs.RO

---

### [11] [Online Human Action Detection during Escorting](https://arxiv.org/abs/2506.23573)
*Siddhartha Mondal, Avik Mitra, Chayan Sarkar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The deployment of robot assistants in large indoor spaces has seen significant growth, with escorting tasks becoming a key application. However, most current escorting robots primarily rely on navigation-focused strategies, assuming that the person being escorted will follow without issue. In crowded environments, this assumption often falls short, as individuals may struggle to keep pace, become obstructed, get distracted, or need to stop unexpectedly. As a result, conventional robotic systems are often unable to provide effective escorting services due to their limited understanding of human movement dynamics. To address these challenges, an effective escorting robot must continuously detect and interpret human actions during the escorting process and adjust its movement accordingly. However, there is currently no existing dataset designed specifically for human action detection in the context of escorting. Given that escorting often occurs in crowded environments, where other individuals may enter the robot's camera view, the robot also needs to identify the specific human it is escorting (the subject) before predicting their actions. Since no existing model performs both person re-identification and action prediction in real-time, we propose a novel neural network architecture that can accomplish both tasks. This enables the robot to adjust its speed dynamically based on the escortee's movements and seamlessly resume escorting after any disruption. In comparative evaluations against strong baselines, our system demonstrates superior efficiency and effectiveness, showcasing its potential to significantly improve robotic escorting services in complex, real-world scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23573) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [12] [Validation of AI-Based 3D Human Pose Estimation in a Cyber-Physical Environment](https://arxiv.org/abs/2506.23739)
*Lisa Marie Otto, Michael Kaiser, Daniel Seebacher, Steffen Müller*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Ensuring safe and realistic interactions between automated driving systems and vulnerable road users (VRUs) in urban environments requires advanced testing methodologies. This paper presents a test environment that combines a Vehiclein-the-Loop (ViL) test bench with a motion laboratory, demonstrating the feasibility of cyber-physical (CP) testing of vehicle-pedestrian and vehicle-cyclist interactions. Building upon previous work focused on pedestrian localization, we further validate a human pose estimation (HPE) approach through a comparative analysis of real-world (RW) and virtual representations of VRUs. The study examines the perception of full-body motion using a commercial monocular camera-based 3Dskeletal detection AI. The virtual scene is generated in Unreal Engine 5, where VRUs are animated in real time and projected onto a screen to stimulate the camera. The proposed stimulation technique ensures the correct perspective, enabling realistic vehicle perception. To assess the accuracy and consistency of HPE across RW and CP domains, we analyze the reliability of detections as well as variations in movement trajectories and joint estimation stability. The validation includes dynamic test scenarios where human avatars, both walking and cycling, are monitored under controlled conditions. Our results show a strong alignment in HPE between RW and CP test conditions for stable motion patterns, while notable inaccuracies persist under dynamic movements and occlusions, particularly for complex cyclist postures. These findings contribute to refining CP testing approaches for evaluating next-generation AI-based vehicle perception and to enhancing interaction models of automated vehicles and VRUs in CP environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23739) | **Categories:** cs.RO, cs.CE, cs.HC

---

### [13] [Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving](https://arxiv.org/abs/2506.23771)
*Guizhe Jin, Zhuoren Li, Bo Leng, Ran Yu, Lu Xiong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.23771) | **Categories:** cs.RO, cs.AI

---
