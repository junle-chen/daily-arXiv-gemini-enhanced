# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-30

## 目录

- [人工智能 (Artificial Intelligence) (8)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [cs.MA (1)](#cs-ma)
- [机器人学 (Robotics) (6)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Decentralized Multi-Agent Goal Assignment for Path Planning using Large Language Models](https://arxiv.org/abs/2510.23824)
*Murad Ismayilov, Edwin Meriaux, Shuo Wen, Gregory Dudek*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Coordinating multiple autonomous agents in shared environments under decentralized conditions is a long-standing challenge in robotics and artificial intelligence. This work addresses the problem of decentralized goal assignment for multi-agent path planning, where agents independently generate ranked preferences over goals based on structured representations of the environment, including grid visualizations and scenario data. After this reasoning phase, agents exchange their goal rankings, and assignments are determined by a fixed, deterministic conflict-resolution rule (e.g., agent index ordering), without negotiation or iterative coordination. We systematically compare greedy heuristics, optimal assignment, and large language model (LLM)-based agents in fully observable grid-world settings. Our results show that LLM-based agents, when provided with well-designed prompts and relevant quantitative information, can achieve near-optimal makespans and consistently outperform traditional heuristics. These findings underscore the potential of language models for decentralized goal assignment in multi-agent path planning and highlight the importance of information structure in such systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23824) | **Categories:** cs.AI

---

### [2] [UniPlanner: A Unified Motion Planning Framework for Autonomous Vehicle Decision-Making Systems via Multi-Dataset Integration](https://arxiv.org/abs/2510.24166)
*Xin Yang, Yuhang Zhang, Wei Li, Xin Lin, Wenbin Zou, Chen Xu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Motion planning is a critical component of autonomous vehicle decision-making systems, directly determining trajectory safety and driving efficiency. While deep learning approaches have advanced planning capabilities, existing methods remain confined to single-dataset training, limiting their robustness in planning.   Through systematic analysis, we discover that vehicular trajectory distributions and history-future correlations demonstrate remarkable consistency across different datasets. Based on these findings, we propose UniPlanner, the first planning framework designed for multi-dataset integration in autonomous vehicle decision-making. UniPlanner achieves unified cross-dataset learning through three synergistic innovations.   First, the History-Future Trajectory Dictionary Network (HFTDN) aggregates history-future trajectory pairs from multiple datasets, using historical trajectory similarity to retrieve relevant futures and generate cross-dataset planning guidance.   Second, the Gradient-Free Trajectory Mapper (GFTM) learns robust history-future correlations from multiple datasets, transforming historical trajectories into universal planning priors. Its gradient-free design ensures the introduction of valuable priors while preventing shortcut learning, making the planning knowledge safely transferable. Third, the Sparse-to-Dense (S2D) paradigm implements adaptive dropout to selectively suppress planning priors during training for robust learning, while enabling full prior utilization during inference to maximize planning performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24166) | **Categories:** cs.AI

---

### [3] [Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents](https://arxiv.org/abs/2510.23691)
*Zihao Wang, Xujing Li, Yining Ye, Junjie Fang, Haoming Wang, Longxiang Liu, Shihao Liang, Junting Lu, Zhiyong Wu, Jiazhan Feng, Wanjun Zhong, Zili Li, Yu Wang, Yu Miao, Bo Zhou, Yuanfan Li, Hao Wang, Zhongkai Zhao, Faming Wu, Zhengxuan Jiang, Weihao Tan, Heyuan Yao, Shi Yan, Xiangyang Li, Yitao Liang, Yujia Qin, Guang Shi*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present Game-TARS, a generalist game agent trained with a unified, scalable action space anchored to human-aligned native keyboard-mouse inputs. Unlike API- or GUI-based approaches, this paradigm enables large-scale continual pre-training across heterogeneous domains, including OS, web, and simulation games. Game-TARS is pre-trained on over 500B tokens with diverse trajectories and multimodal data. Key techniques include a decaying continual loss to reduce causal confusion and an efficient Sparse-Thinking strategy that balances reasoning depth and inference cost. Experiments show that Game-TARS achieves about 2 times the success rate over the previous sota model on open-world Minecraft tasks, is close to the generality of fresh humans in unseen web 3d games, and outperforms GPT-5, Gemini-2.5-Pro, and Claude-4-Sonnet in FPS benchmarks. Scaling results on training-time and test-time confirm that the unified action space sustains improvements when scaled to cross-game and multimodal data. Our results demonstrate that simple, scalable action representations combined with large-scale pre-training provide a promising path toward generalist agents with broad computer-use abilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23691) | **Categories:** cs.AI

---

### [4] [Hybrid Modeling, Sim-to-Real Reinforcement Learning, and Large Language Model Driven Control for Digital Twins](https://arxiv.org/abs/2510.23882)
*Adil Rasheed, Oscar Ravik, Omer San*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This work investigates the use of digital twins for dynamical system modeling and control, integrating physics-based, data-driven, and hybrid approaches with both traditional and AI-driven controllers. Using a miniature greenhouse as a test platform, four predictive models Linear, Physics-Based Modeling (PBM), Long Short Term Memory (LSTM), and Hybrid Analysis and Modeling (HAM) are developed and compared under interpolation and extrapolation scenarios. Three control strategies Model Predictive Control (MPC), Reinforcement Learning (RL), and Large Language Model (LLM) based control are also implemented to assess trade-offs in precision, adaptability, and implementation effort. Results show that in modeling HAM provides the most balanced performance across accuracy, generalization, and computational efficiency, while LSTM achieves high precision at greater resource cost. Among controllers, MPC delivers robust and predictable performance, RL demonstrates strong adaptability, and LLM-based controllers offer flexible human-AI interaction when coupled with predictive tools.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23882) | **Categories:** cs.AI

---

### [5] [Learning Individual Movement Shifts After Urban Disruptions with Social Infrastructure Reliance](https://arxiv.org/abs/2510.23989)
*Shangde Gao, Zelin Xu, Zhe Jiang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Shifts in individual movement patterns following disruptive events can reveal changing demands for community resources. However, predicting such shifts before disruptive events remains challenging for several reasons. First, measures are lacking for individuals' heterogeneous social infrastructure resilience (SIR), which directly influences their movement patterns, and commonly used features are often limited or unavailable at scale, e.g., sociodemographic characteristics. Second, the complex interactions between individual movement patterns and spatial contexts have not been sufficiently captured. Third, individual-level movement may be spatially sparse and not well-suited to traditional decision-making methods for movement predictions. This study incorporates individuals' SIR into a conditioned deep learning model to capture the complex relationships between individual movement patterns and local spatial context using large-scale, sparse individual-level data. Our experiments demonstrate that incorporating individuals' SIR and spatial context can enhance the model's ability to predict post-event individual movement patterns. The conditioned model can capture the divergent shifts in movement patterns among individuals who exhibit similar pre-event patterns but differ in SIR.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23989) | **Categories:** cs.AI

---

### [6] [Modeling Electric Vehicle Car-Following Behavior: Classical vs Machine Learning Approach](https://arxiv.org/abs/2510.24085)
*Md. Shihab Uddin, Md Nazmus Shakib, Rahul Bhadani*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The increasing adoption of electric vehicles (EVs) necessitates an understanding of their driving behavior to enhance traffic safety and develop smart driving systems. This study compares classical and machine learning models for EV car following behavior. Classical models include the Intelligent Driver Model (IDM), Optimum Velocity Model (OVM), Optimal Velocity Relative Velocity (OVRV), and a simplified CACC model, while the machine learning approach employs a Random Forest Regressor. Using a real world dataset of an EV following an internal combustion engine (ICE) vehicle under varied driving conditions, we calibrated classical model parameters by minimizing the RMSE between predictions and real data. The Random Forest model predicts acceleration using spacing, speed, and gap type as inputs. Results demonstrate the Random Forest's superior accuracy, achieving RMSEs of 0.0046 (medium gap), 0.0016 (long gap), and 0.0025 (extra long gap). Among physics based models, CACC performed best, with an RMSE of 2.67 for long gaps. These findings highlight the machine learning model's performance across all scenarios. Such models are valuable for simulating EV behavior and analyzing mixed autonomy traffic dynamics in EV integrated environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24085) | **Categories:** cs.AI, cs.LG

---

### [7] [BLM$_1$: A Boundless Large Model for Cross-Space, Cross-Task, and Cross-Embodiment Learning](https://arxiv.org/abs/2510.24161)
*Wentao Tan, Bowen Wang, Heng Zhi, Chenyu Liu, Zhe Li, Jian Liu, Zengrong Lin, Yukun Dai, Yipeng Chen, Wenjie Yang, Enci Xie, Hao Xue, Baixu Ji, Chen Xu, Zhibin Wang, Tianshi Wang, Lei Zhu, Heng Tao Shen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal large language models (MLLMs) have advanced vision-language reasoning and are increasingly deployed in embodied agents. However, significant limitations remain: MLLMs generalize poorly across digital-physical spaces and embodiments; vision-language-action models (VLAs) produce low-level actions yet lack robust high-level embodied reasoning; and most embodied large language models (ELLMs) are constrained to digital-space with poor generalization to the physical world. Thus, unified models that operate seamlessly across digital and physical spaces while generalizing across embodiments and tasks remain absent. We introduce the \textbf{Boundless Large Model (BLM$_1$)}, a multimodal spatial foundation model that preserves instruction following and reasoning, incorporates embodied knowledge, and supports robust cross-embodiment control. BLM$_1$ integrates three key capabilities -- \textit{cross-space transfer, cross-task learning, and cross-embodiment generalization} -- via a two-stage training paradigm. Stage I injects embodied knowledge into the MLLM through curated digital corpora while maintaining language competence. Stage II trains a policy module through an intent-bridging interface that extracts high-level semantics from the MLLM to guide control, without fine-tuning the MLLM backbone. This process is supported by a self-collected cross-embodiment demonstration suite spanning four robot embodiments and six progressively challenging tasks. Evaluations across digital and physical benchmarks show that a single BLM$_1$ instance outperforms four model families -- MLLMs, ELLMs, VLAs, and GMLMs -- achieving $\sim\!\textbf{6%}$ gains in digital tasks and $\sim\!\textbf{3%}$ in physical tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24161) | **Categories:** cs.AI, cs.MM, cs.RO

---

### [8] [APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training](https://arxiv.org/abs/2510.24397)
*Jiarui Qin, Yunjia Xi, Junjie Huang, Renting Rui, Di Yin, Weiwen Liu, Yong Yu, Weinan Zhang, Xing Sun*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the rapid development of LLM-based agents, there is a growing trend to incorporate agent-specific data into the pre-training stage of LLMs, aiming to better align LLMs with real-world autonomous task execution. However, current pre-training benchmarks primarily focus on isolated and static skills, e.g., common knowledge or mathematical/code reasoning, and fail to reflect model's agentic capabilities. On the other hand, agent benchmarks are typically designed for post-trained models, requiring multi-turn task execution abilities that base models struggle to support. Thus, there is a compelling need for a benchmark that can evaluate agentic potentials during pre-training and guide the model training more effectively. To address this gap, we propose APTBench, a framework that converts real-world agent tasks and successful trajectories into multiple-choice or text completion questions tailored for base models. It focuses on core agentic abilities, e.g., planning and action, and covers key agent scenarios, software engineering and deep research. Compared to existing general-purpose benchmarks, APTBench offers a more predictive signal of a model's downstream performance as an agent, while remaining significantly more lightweight and cost-effective than full-scale, end-to-end agent evaluations after post-training.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24397) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Enhancing Vision-Language Models for Autonomous Driving through Task-Specific Prompting and Spatial Reasoning](https://arxiv.org/abs/2510.24152)
*Aodi Wu, Xubo Luo*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This technical report presents our solution for the RoboSense Challenge at IROS 2025, which evaluates Vision-Language Models (VLMs) on autonomous driving scene understanding across perception, prediction, planning, and corruption detection tasks. We propose a systematic framework built on four core components. First, a Mixture-of-Prompts router classifies questions and dispatches them to task-specific expert prompts, eliminating interference across diverse question types. Second, task-specific prompts embed explicit coordinate systems, spatial reasoning rules, role-playing, Chain-of-Thought/Tree-of-Thought reasoning, and few-shot examples tailored to each task. Third, a visual assembly module composes multi-view images with object crops, magenta markers, and adaptive historical frames based on question requirements. Fourth, we configure model inference parameters (temperature, top-p, message roles) per task to optimize output quality. Implemented on Qwen2.5-VL-72B, our approach achieves 70.87% average accuracy on Phase-1 (clean data) and 72.85% on Phase-2 (corrupted data), demonstrating that structured prompting and spatial grounding substantially enhance VLM performance on safety-critical autonomous driving tasks. Code and prompt are available at https://github.com/wuaodi/UCAS-CSU-phase2.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24152) | **Categories:** cs.CV, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation](https://arxiv.org/abs/2510.23636)
*Thaweerath Phisannupawong, Joshua Julian Damanik, Han-Lim Choi*

Main category: cs.LG

TL;DR: 本文提出了一种轻量级的基于大型语言模型的多模态航班延误预测方法，该方法结合了轨迹表示和文本航空信息，实现了亚分钟级的预测误差。


<details>
  <summary>Details</summary>
Motivation: 航班延误预测已成为空中交通管理的关键，因为延误突出了影响整体网络性能的低效率。

Method: 该方法从空中交通管制员的角度出发，监测飞机进入终端区域后的延误情况，通过将轨迹数据转换为语言模态来捕获空域条件，整合了轨迹表示和文本航空信息，包括航班信息、天气报告和机场公告。

Result: 实验结果表明，该模型通过有效利用与延误来源相关的上下文信息，始终实现亚分钟级的预测误差。

Conclusion: 该框架表明，语言理解与轨迹信息的跨模态适应相结合，可以提高延误预测的准确性。此外，该方法还显示出在实际操作中的实用性和可扩展性，支持接收到新的操作信息时进行实时更新，从而改进预测。

Abstract: 航班延误预测已成为空中交通管理中的一个重点，因为延误突显了影响整体网络性能的低效率。本文提出了一种轻量级的基于大型语言模型的多模态航班延误预测方法，该方法从空中交通管制员的角度出发，监测飞机进入终端区域后的延误情况。该方法通过将轨迹数据转换为语言模态来捕获空域条件，整合了轨迹表示和文本航空信息，包括航班信息、天气报告和机场公告。实验结果表明，该模型通过有效利用与延误来源相关的上下文信息，始终实现亚分钟级的预测误差。该框架表明，语言理解与轨迹信息的跨模态适应相结合，可以提高延误预测的准确性。此外，该方法还显示出在实际操作中的实用性和可扩展性，支持接收到新的操作信息时进行实时更新，从而改进预测。

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23636) | **Categories:** cs.LG, cs.AI, cs.CL

---

### [2] [NUM2EVENT: Interpretable Event Reasoning from Numerical time-series](https://arxiv.org/abs/2510.23630)
*Ninghui Feng, Yiyan Qi*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) have recently demonstrated impressive multimodal reasoning capabilities, yet their understanding of purely numerical time-series signals remains limited. Existing approaches mainly focus on forecasting or trend description, without uncovering the latent events that drive numerical changes or explaining the reasoning process behind them. In this work, we introduce the task of number-to-event reasoning and decoding, which aims to infer interpretable structured events from numerical inputs, even when current text is unavailable. To address the data scarcity and semantic alignment challenges, we propose a reasoning-aware framework that integrates an agent-guided event extractor (AGE), a marked multivariate Hawkes-based synthetic generator (EveDTS), and a two-stage fine-tuning pipeline combining a time-series encoder with a structured decoder. Our model explicitly reasons over numerical changes, generates intermediate explanations, and outputs structured event hypotheses. Experiments on multi-domain datasets show that our method substantially outperforms strong LLM baselines in event-level precision and recall. These results suggest a new direction for bridging quantitative reasoning and semantic understanding, enabling LLMs to explain and predict events directly from numerical dynamics.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23630) | **Categories:** cs.LG, cs.AI, cs.CL

---

### [3] [LLMComp: A Language Modeling Paradigm for Error-Bounded Scientific Data Compression](https://arxiv.org/abs/2510.23632)
*Guozhong Li, Muhannad Alhumaidi, Spiros Skiadopoulos, Panos Kalnis*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid growth of high-resolution scientific simulations and observation systems is generating massive spatiotemporal datasets, making efficient, error-bounded compression increasingly important. Meanwhile, decoder-only large language models (LLMs) have demonstrated remarkable capabilities in modeling complex sequential data. In this paper, we propose LLMCOMP, a novel lossy compression paradigm that leverages decoder-only large LLMs to model scientific data. LLMCOMP first quantizes 3D fields into discrete tokens, arranges them via Z-order curves to preserve locality, and applies coverage-guided sampling to enhance training efficiency. An autoregressive transformer is then trained with spatial-temporal embeddings to model token transitions. During compression, the model performs top-k prediction, storing only rank indices and fallback corrections to ensure strict error bounds. Experiments on multiple reanalysis datasets show that LLMCOMP consistently outperforms state-of-the-art compressors, achieving up to 30% higher compression ratios under strict error bounds. These results highlight the potential of LLMs as general-purpose compressors for high-fidelity scientific data.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23632) | **Categories:** cs.LG, cs.AI

---

### [4] [Error Adjustment Based on Spatiotemporal Correlation Fusion for Traffic Forecasting](https://arxiv.org/abs/2510.23656)
*Fuqiang Liu, Weiping Ding, Luis Miranda-Moreno, Lijun Sun*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Deep neural networks (DNNs) play a significant role in an increasing body of research on traffic forecasting due to their effectively capturing spatiotemporal patterns embedded in traffic data. A general assumption of training the said forecasting models via mean squared error estimation is that the errors across time steps and spatial positions are uncorrelated. However, this assumption does not really hold because of the autocorrelation caused by both the temporality and spatiality of traffic data. This gap limits the performance of DNN-based forecasting models and is overlooked by current studies. To fill up this gap, this paper proposes Spatiotemporally Autocorrelated Error Adjustment (SAEA), a novel and general framework designed to systematically adjust autocorrelated prediction errors in traffic forecasting. Unlike existing approaches that assume prediction errors follow a random Gaussian noise distribution, SAEA models these errors as a spatiotemporal vector autoregressive (VAR) process to capture their intrinsic dependencies. First, it explicitly captures both spatial and temporal error correlations by a coefficient matrix, which is then embedded into a newly formulated cost function. Second, a structurally sparse regularization is introduced to incorporate prior spatial information, ensuring that the learned coefficient matrix aligns with the inherent road network structure. Finally, an inference process with test-time error adjustment is designed to dynamically refine predictions, mitigating the impact of autocorrelated errors in real-time forecasting. The effectiveness of the proposed approach is verified on different traffic datasets. Results across a wide range of traffic forecasting models show that our method enhances performance in almost all cases.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23656) | **Categories:** cs.LG, cs.AI

---

### [5] [Parallel BiLSTM-Transformer networks for forecasting chaotic dynamics](https://arxiv.org/abs/2510.23685)
*Junwen Ma, Mingyu Ge, Yisen Wang, Yong Zhang, Weicheng Fu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The nonlinear nature of chaotic systems results in extreme sensitivity to initial conditions and highly intricate dynamical behaviors, posing fundamental challenges for accurately predicting their evolution. To overcome the limitation that conventional approaches fail to capture both local features and global dependencies in chaotic time series simultaneously, this study proposes a parallel predictive framework integrating Transformer and Bidirectional Long Short-Term Memory (BiLSTM) networks. The hybrid model employs a dual-branch architecture, where the Transformer branch mainly captures long-range dependencies while the BiLSTM branch focuses on extracting local temporal features. The complementary representations from the two branches are fused in a dedicated feature-fusion layer to enhance predictive accuracy. As illustrating examples, the model's performance is systematically evaluated on two representative tasks in the Lorenz system. The first is autonomous evolution prediction, in which the model recursively extrapolates system trajectories from the time-delay embeddings of the state vector to evaluate long-term tracking accuracy and stability. The second is inference of unmeasured variable, where the model reconstructs the unobserved states from the time-delay embeddings of partial observations to assess its state-completion capability. The results consistently indicate that the proposed hybrid framework outperforms both single-branch architectures across tasks, demonstrating its robustness and effectiveness in chaotic system prediction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23685) | **Categories:** cs.LG, cs.AI

---


## cs.MA [cs.MA]
### [1] [Coordinated Autonomous Drones for Human-Centered Fire Evacuation in Partially Observable Urban Environments](https://arxiv.org/abs/2510.23899)
*Maria G. Mendoza, Addison Kalanther, Daniel Bostwick, Emma Stephan, Chinmay Maheshwari, Shankar Sastry*

Main category: cs.MA

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous drone technology holds significant promise for enhancing search and rescue operations during evacuations by guiding humans toward safety and supporting broader emergency response efforts. However, their application in dynamic, real-time evacuation support remains limited. Existing models often overlook the psychological and emotional complexity of human behavior under extreme stress. In real-world fire scenarios, evacuees frequently deviate from designated safe routes due to panic and uncertainty. To address these challenges, this paper presents a multi-agent coordination framework in which autonomous Unmanned Aerial Vehicles (UAVs) assist human evacuees in real-time by locating, intercepting, and guiding them to safety under uncertain conditions. We model the problem as a Partially Observable Markov Decision Process (POMDP), where two heterogeneous UAV agents, a high-level rescuer (HLR) and a low-level rescuer (LLR), coordinate through shared observations and complementary capabilities. Human behavior is captured using an agent-based model grounded in empirical psychology, where panic dynamically affects decision-making and movement in response to environmental stimuli. The environment features stochastic fire spread, unknown evacuee locations, and limited visibility, requiring UAVs to plan over long horizons to search for humans and adapt in real-time. Our framework employs the Proximal Policy Optimization (PPO) algorithm with recurrent policies to enable robust decision-making in partially observable settings. Simulation results demonstrate that the UAV team can rapidly locate and intercept evacuees, significantly reducing the time required for them to reach safety compared to scenarios without UAV assistance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23899) | **Categories:** cs.MA, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [ZTRS: Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring](https://arxiv.org/abs/2510.24108)
*Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Jingde Chen, Nadine Chang, Maying Shen, Jingyu Song, Zuxuan Wu, Shiyi Lan, Jose M. Alvarez*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end autonomous driving maps raw sensor inputs directly into ego-vehicle trajectories to avoid cascading errors from perception modules and to leverage rich semantic cues. Existing frameworks largely rely on Imitation Learning (IL), which can be limited by sub-optimal expert demonstrations and covariate shift during deployment. On the other hand, Reinforcement Learning (RL) has recently shown potential in scaling up with simulations, but is typically confined to low-dimensional symbolic inputs (e.g. 3D objects and maps), falling short of full end-to-end learning from raw sensor data. We introduce ZTRS (Zero-Imitation End-to-End Autonomous Driving with Trajectory Scoring), a framework that combines the strengths of both worlds: sensor inputs without losing information and RL training for robust planning. To the best of our knowledge, ZTRS is the first framework that eliminates IL entirely by only learning from rewards while operating directly on high-dimensional sensor data. ZTRS utilizes offline reinforcement learning with our proposed Exhaustive Policy Optimization (EPO), a variant of policy gradient tailored for enumerable actions and rewards. ZTRS demonstrates strong performance across three benchmarks: Navtest (generic real-world open-loop planning), Navhard (open-loop planning in challenging real-world and synthetic scenarios), and HUGSIM (simulated closed-loop driving). Specifically, ZTRS achieves the state-of-the-art result on Navhard and outperforms IL-based baselines on HUGSIM. Code will be available at https://github.com/woxihuanjiangguo/ZTRS.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24108) | **Categories:** cs.RO, cs.CV

---

### [2] [RoboOmni: Proactive Robot Manipulation in Omni-modal Context](https://arxiv.org/abs/2510.23763)
*Siyin Wang, Jinlan Fu, Feihong Liu, Xinzhe He, Huangxuan Wu, Junhao Shi, Kexin Huang, Zhaoye Fei, Jingjing Gong, Zuxuan Wu, Yugang Jiang, See-Kiong Ng, Tat-Seng Chua, Xipeng Qiu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.23763) | **Categories:** cs.RO, cs.CL, cs.CV

---

### [3] [SynAD: Enhancing Real-World End-to-End Autonomous Driving Models through Synthetic Data Integration](https://arxiv.org/abs/2510.24052)
*Jongsuk Kim, Jaeyoung Lee, Gyojin Han, Dongjae Lee, Minki Jeong, Junmo Kim*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in deep learning and the availability of high-quality real-world driving datasets have propelled end-to-end autonomous driving. Despite this progress, relying solely on real-world data limits the variety of driving scenarios for training. Synthetic scenario generation has emerged as a promising solution to enrich the diversity of training data; however, its application within E2E AD models remains largely unexplored. This is primarily due to the absence of a designated ego vehicle and the associated sensor inputs, such as camera or LiDAR, typically provided in real-world scenarios. To address this gap, we introduce SynAD, the first framework designed to enhance real-world E2E AD models using synthetic data. Our method designates the agent with the most comprehensive driving information as the ego vehicle in a multi-agent synthetic scenario. We further project path-level scenarios onto maps and employ a newly developed Map-to-BEV Network to derive bird's-eye-view features without relying on sensor inputs. Finally, we devise a training strategy that effectively integrates these map-based synthetic data with real driving data. Experimental results demonstrate that SynAD effectively integrates all components and notably enhances safety performance. By bridging synthetic scenario generation and E2E AD, SynAD paves the way for more comprehensive and robust autonomous driving models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24052) | **Categories:** cs.RO, cs.AI

---

### [4] [Dynamically-Consistent Trajectory Optimization for Legged Robots via Contact Point Decomposition](https://arxiv.org/abs/2510.24069)
*Sangmin Kim, Hajun Kim, Gijeong Kim, Min-Gyu Kim, Hae-Won Park*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: To generate reliable motion for legged robots through trajectory optimization, it is crucial to simultaneously compute the robot's path and contact sequence, as well as accurately consider the dynamics in the problem formulation. In this paper, we present a phase-based trajectory optimization that ensures the feasibility of translational dynamics and friction cone constraints throughout the entire trajectory. Specifically, our approach leverages the superposition properties of linear differential equations to decouple the translational dynamics for each contact point, which operates under different phase sequences. Furthermore, we utilize the differentiation matrix of B{\'e}zier polynomials to derive an analytical relationship between the robot's position and force, thereby ensuring the consistent satisfaction of translational dynamics. Additionally, by exploiting the convex closure property of B{\'e}zier polynomials, our method ensures compliance with friction cone constraints. Using the aforementioned approach, the proposed trajectory optimization framework can generate dynamically reliable motions with various gait sequences for legged robots. We validate our framework using a quadruped robot model, focusing on the feasibility of dynamics and motion generation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24069) | **Categories:** cs.RO

---

### [5] [LagMemo: Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation](https://arxiv.org/abs/2510.24118)
*Haotian Zhou, Xiaole Wang, He Li, Fusheng Sun, Shengyu Guo, Guolei Qi, Jianghuan Xu, Huijing Zhao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Navigating to a designated goal using visual information is a fundamental capability for intelligent robots. Most classical visual navigation methods are restricted to single-goal, single-modality, and closed set goal settings. To address the practical demands of multi-modal, open-vocabulary goal queries and multi-goal visual navigation, we propose LagMemo, a navigation system that leverages a language 3D Gaussian Splatting memory. During exploration, LagMemo constructs a unified 3D language memory. With incoming task goals, the system queries the memory, predicts candidate goal locations, and integrates a local perception-based verification mechanism to dynamically match and validate goals during navigation. For fair and rigorous evaluation, we curate GOAT-Core, a high-quality core split distilled from GOAT-Bench tailored to multi-modal open-vocabulary multi-goal visual navigation. Experimental results show that LagMemo's memory module enables effective multi-modal open-vocabulary goal localization, and that LagMemo outperforms state-of-the-art methods in multi-goal visual navigation. Project page: https://weekgoodday.github.io/lagmemo

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24118) | **Categories:** cs.RO, cs.AI

---

### [6] [Multi-Agent Scenario Generation in Roundabouts with a Transformer-enhanced Conditional Variational Autoencoder](https://arxiv.org/abs/2510.24671)
*Li Li, Tobias Brinkmann, Till Temmen, Markus Eisenbarth, Jakob Andert*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the increasing integration of intelligent driving functions into serial-produced vehicles, ensuring their functionality and robustness poses greater challenges. Compared to traditional road testing, scenario-based virtual testing offers significant advantages in terms of time and cost efficiency, reproducibility, and exploration of edge cases. We propose a Transformer-enhanced Conditional Variational Autoencoder (CVAE-T) model for generating multi-agent traffic scenarios in roundabouts, which are characterized by high vehicle dynamics and complex layouts, yet remain relatively underexplored in current research. The results show that the proposed model can accurately reconstruct original scenarios and generate realistic, diverse synthetic scenarios. Besides, two Key-Performance-Indicators (KPIs) are employed to evaluate the interactive behavior in the generated scenarios. Analysis of the latent space reveals partial disentanglement, with several latent dimensions exhibiting distinct and interpretable effects on scenario attributes such as vehicle entry timing, exit timing, and velocity profiles. The results demonstrate the model's capability to generate scenarios for the validation of intelligent driving functions involving multi-agent interactions, as well as to augment data for their development and iterative improvement.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.24671) | **Categories:** cs.RO, cs.AI

---
