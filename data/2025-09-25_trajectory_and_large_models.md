# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-25

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Synthesizing Attitudes, Predicting Actions (SAPA): Behavioral Theory-Guided LLMs for Ridesourcing Mode Choice Modeling](https://arxiv.org/abs/2509.18181)
*Mustafa Sameen, Xiaojian Zhang, Xilei Zhao*

Main category: cs.AI

TL;DR: 本文提出了一种名为SAPA的框架，该框架利用大型语言模型（LLM）合成潜在态度，以更准确地预测网约车模式选择。


<details>
  <summary>Details</summary>
Motivation: 现有网约车模式选择模型预测精度有限，无法捕捉关键心理因素，且面临严重的类别不平衡问题。

Method: 该论文提出SAPA框架，首先使用LLM从原始出行调查数据中生成定性出行者角色，然后训练倾向得分模型，并使用LLM为理论驱动的潜在变量分配定量分数，最后通过分类器整合倾向得分、潜在变量分数和可观察的出行属性来预测网约车模式选择。

Result: 在大型多年出行调查上的实验表明，SAPA显著优于现有基线模型，在PR-AUC指标上，网约车选择预测提高了高达75.9%。

Conclusion: 该研究提供了一种强大的工具，可以准确预测网约车模式选择，并提供了一种易于转移到各种应用的方法。

Abstract: 为了设计和实施有效的交通管理政策，减少拥堵，改善交通，更有效地分配资源，准确地模拟网约车模式选择至关重要。现有的网约车模式选择预测模型通常预测精度有限，因为它们无法捕捉关键的心理因素，并且由于网约车出行仅占个人日常出行的很小一部分，因此面临着严重的类别不平衡的挑战。为了解决这些局限性，本文介绍了一种综合态度、预测行为（SAPA）框架，这是一种分层方法，它使用大型语言模型（LLM）来综合理论驱动的潜在态度，以预测网约车选择。SAPA首先使用LLM从原始出行调查数据中生成定性出行者角色，然后训练一个基于人口统计和行为特征（由这些角色丰富）的倾向得分模型，以生成个人层面的得分。接下来，LLM为理论驱动的潜在变量（例如，时间和成本敏感性）分配定量分数，最终的分类器集成了倾向得分、潜在变量得分（及其交互项）和可观察的出行属性，以预测网约车模式选择。在大型多年出行调查上的实验表明，SAPA显著优于最先进的基线模型，在PR-AUC方面，在保留的测试集上，网约车选择预测提高了高达75.9%。这项研究提供了一种强大的工具，可以准确预测网约车模式选择，并提供了一种易于转移到各种应用的方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18181) | **Categories:** cs.AI, cs.LG

---

### [2] [MMCD: Multi-Modal Collaborative Decision-Making for Connected Autonomy with Knowledge Distillation](https://arxiv.org/abs/2509.18198)
*Rui Liu, Zikang Wang, Peng Gao, Yu Shen, Pratap Tokekar, Ming Lin*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous systems have advanced significantly, but challenges persist in accident-prone environments where robust decision-making is crucial. A single vehicle's limited sensor range and obstructed views increase the likelihood of accidents. Multi-vehicle connected systems and multi-modal approaches, leveraging RGB images and LiDAR point clouds, have emerged as promising solutions. However, existing methods often assume the availability of all data modalities and connected vehicles during both training and testing, which is impractical due to potential sensor failures or missing connected vehicles. To address these challenges, we introduce a novel framework MMCD (Multi-Modal Collaborative Decision-making) for connected autonomy. Our framework fuses multi-modal observations from ego and collaborative vehicles to enhance decision-making under challenging conditions. To ensure robust performance when certain data modalities are unavailable during testing, we propose an approach based on cross-modal knowledge distillation with a teacher-student model structure. The teacher model is trained with multiple data modalities, while the student model is designed to operate effectively with reduced modalities. In experiments on $\textit{connected autonomous driving with ground vehicles}$ and $\textit{aerial-ground vehicles collaboration}$, our method improves driving safety by up to ${\it 20.7}\%$, surpassing the best-existing baseline in detecting potential accidents and making safe driving decisions. More information can be found on our website https://ruiiu.github.io/mmcd.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18198) | **Categories:** cs.AI, cs.MA, cs.RO

---

### [3] [FERA: Foil Fencing Referee Assistant Using Pose-Based Multi-Label Move Recognition and Rule Reasoning](https://arxiv.org/abs/2509.18527)
*Ziwen Chen, Zhong Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The sport of fencing, like many other sports, faces challenges in refereeing: subjective calls, human errors, bias, and limited availability in practice environments. We present FERA (Fencing Referee Assistant), a prototype AI referee for foil fencing which integrates pose-based multi-label action recognition and rule-based reasoning. FERA extracts 2D joint positions from video, normalizes them, computes a 101-dimensional kinematic feature set, and applies a Transformer for multi-label move and blade classification. To determine priority and scoring, FERA applies a distilled language model with encoded right-of-way rules, producing both a decision and an explanation for each exchange. With limited hand-labeled data, a 5-fold cross-validation achieves an average macro-F1 score of 0.549, outperforming multiple baselines, including a Temporal Convolutional Network (TCN), BiLSTM, and a vanilla Transformer. While not ready for deployment, these results demonstrate a promising path towards automated referee assistance in foil fencing and new opportunities for AI applications, such as coaching in the field of fencing.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18527) | **Categories:** cs.AI

---

### [4] [Code Driven Planning with Domain-Adaptive Critic](https://arxiv.org/abs/2509.19077)
*Zikang Tian, Shaohui Peng, Du Huang, Jiaming Guo, Ruizhi Chen, Rui Zhang, Xishan Zhang, Yuxuan Guo, Zidong Du, Qi Guo, Ling Li, Yewen Pu, Xing Hu, Yunji Chen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) have been widely adopted as task planners for AI agents in sequential decision-making problems, leveraging their extensive world knowledge. However, the gap between their general knowledge and environment-specific requirements often leads to inaccurate plans. To address this, existing approaches rely on frequent LLM queries to iteratively refine plans based on immediate environmental feedback, which incurs substantial query costs. However, this refinement is typically guided by short-term environmental feedback, limiting LLMs from developing plans aligned with long-term rewards. We propose Code Driven Planning with Domain-Adaptive Critic (CoPiC). Instead of relying on frequent queries, CoPiC employs LLMs to generate a diverse set of high-level planning programs, which iteratively produce and refine candidate plans. A trained domain-adaptive critic then evaluates these candidates and selects the one most aligned with long-term rewards for execution. Using high-level planning programs as planner and domain-adaptive critic as estimator, CoPiC improves planning while significantly reducing query costs. Results in ALFWorld, NetHack, and StarCraft II Unit Building show that CoPiC outperforms advanced LLM-based baselines, AdaPlanner and Reflexion, achieving an average (1) 23.33% improvement in success rate and (2) 91.27% reduction in query costs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19077) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [TinyBEV: Cross Modal Knowledge Distillation for Efficient Multi Task Bird's Eye View Perception and Planning](https://arxiv.org/abs/2509.18372)
*Reeshad Khan, John Gauch*

Main category: cs.CV

TL;DR: TinyBEV提出了一种轻量级的、仅使用摄像头的BEV框架，通过知识蒸馏将大型规划模型的能力转移到小型实时模型中。


<details>
  <summary>Details</summary>
Motivation: 现有技术无法在资源受限的环境中实现全栈自动驾驶功能。

Method: 提出了一种模型无关的多阶段蒸馏策略，结合特征级、输出级和自适应区域感知监督。

Result: TinyBEV在nuScenes数据集上实现了具有竞争力的检测、运动预测和碰撞率结果，同时运行速度提高了5倍（11 FPS），并且仅需要摄像头输入。

Conclusion: TinyBEV证明了全栈驾驶智能可以在资源受限的环境中保留，缩小了大型多模态感知规划模型与可部署的实时自动驾驶之间的差距。

Abstract: 我们提出了TinyBEV，一个统一的、仅使用摄像头的鸟瞰图（BEV）框架，它将大型面向规划的教师模型（UniAD [19]）的全栈能力提炼成一个紧凑的、实时的学生模型。与之前高效的仅使用摄像头的基线（如VAD[23]和VADv2[7]）不同，TinyBEV支持完整的自主堆栈：3D检测、HD地图分割、运动预测、 occupancy预测和目标导向的规划，所有这些都在一个简化的28M参数骨干网络中实现，与UniAD [19] 相比，参数减少了78%。我们的模型无关、多阶段蒸馏策略结合了特征级、输出级和自适应区域感知监督，以有效地将高容量多模态知识转移到轻量级BEV表示中。在nuScenes[4]上，Tiny-BEV在检测方面实现了39.0 mAP，在运动预测方面实现了1.08 minADE，碰撞率为0.32，同时运行速度提高了5倍（11 FPS），并且只需要摄像头输入。这些结果表明，全栈驾驶智能可以在资源受限的环境中保留，从而弥合了大规模多模态感知规划模型与可部署的实时自主系统之间的差距。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18372) | **Categories:** cs.CV

---

### [2] [BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking](https://arxiv.org/abs/2509.18387)
*Thomas Gossard, Filip Radovic, Andreas Ziegler, Andrea Zell*

Main category: cs.CV

TL;DR: 本文提出了一种新的运动模糊标注策略，并将模糊属性加入标注信息中，从而提高了球类检测的准确性和轨迹预测的可靠性。


<details>
  <summary>Details</summary>
Motivation: 现有的标注方法将运动模糊球体标注在模糊边缘，忽略了运动线索，导致检测系统性能下降，尤其是在球拍运动中。

Method: 本文提出将球体标注在模糊条纹的中心，并明确标注模糊属性。此外，还提出了BlurBall模型，该模型通过结合注意力机制，可以联合估计球的位置和运动模糊属性。

Result: 实验结果表明，本文提出的标注方法可以提高各种模型的检测性能。BlurBall模型在球体检测方面取得了最先进的结果，并且能够实现更可靠的轨迹预测。

Conclusion: 利用模糊信息不仅可以提高检测精度，还可以实现更可靠的轨迹预测，从而有益于实时体育分析。

Abstract: 运动模糊降低了快速移动物体的清晰度，给检测系统带来了挑战，尤其是在球拍运动中，球通常表现为条纹而不是清晰的点。现有的标注惯例将球标记在模糊的前缘，引入了不对称性，忽略了与速度相关的有价值的运动线索。本文介绍了一种新的标注策略，该策略将球放置在模糊条纹的中心，并明确地注释模糊属性。使用这种惯例，我们发布了一个新的乒乓球检测数据集。我们证明了这种标注方法始终如一地提高了各种模型的检测性能。此外，我们还介绍了BlurBall，该模型可以联合估计球的位置和运动模糊属性。通过在多帧输入上结合诸如Squeeze-and-Excitation之类的注意力机制，我们在球检测中获得了最先进的结果。利用模糊不仅可以提高检测精度，还可以实现更可靠的轨迹预测，从而有益于实时体育分析。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18387) | **Categories:** cs.CV

---

### [3] [Live-E2T: Real-time Threat Monitoring in Video via Deduplicated Event Reasoning and Chain-of-Thought](https://arxiv.org/abs/2509.18571)
*Yuhan Wang, Cheng Liu, Zihan Zhao, Weichao Wu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Real-time threat monitoring identifies threatening behaviors in video streams and provides reasoning and assessment of threat events through explanatory text. However, prevailing methodologies, whether based on supervised learning or generative models, struggle to concurrently satisfy the demanding requirements of real-time performance and decision explainability. To bridge this gap, we introduce Live-E2T, a novel framework that unifies these two objectives through three synergistic mechanisms. First, we deconstruct video frames into structured Human-Object-Interaction-Place semantic tuples. This approach creates a compact, semantically focused representation, circumventing the information degradation common in conventional feature compression. Second, an efficient online event deduplication and updating mechanism is proposed to filter spatio-temporal redundancies, ensuring the system's real time responsiveness. Finally, we fine-tune a Large Language Model using a Chain-of-Thought strategy, endow it with the capability for transparent and logical reasoning over event sequences to produce coherent threat assessment reports. Extensive experiments on benchmark datasets, including XD-Violence and UCF-Crime, demonstrate that Live-E2T significantly outperforms state-of-the-art methods in terms of threat detection accuracy, real-time efficiency, and the crucial dimension of explainability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18571) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [From Parameters to Performance: A Data-Driven Study on LLM Structure and Development](https://arxiv.org/abs/2509.18136)
*Suqing Wang, Zuchao Li, Luohe Shi, Bo Du, Hai Zhao, Yun Li, Qianren Wang*

Main category: cs.LG

TL;DR: 该论文构建了一个大规模数据集，用于分析大型语言模型的结构配置与其性能之间的关系。


<details>
  <summary>Details</summary>
Motivation: 目前缺乏关于结构配置如何影响大型语言模型性能的系统性、数据驱动的研究。

Method: 通过构建包含各种开源LLM结构及其在多个基准测试中性能的大规模数据集，进行系统的数据挖掘驱动分析。

Result: 研究验证并量化了结构配置与性能之间的关系，并使用机制可解释性技术进一步证实了研究结果。

Conclusion: 该研究旨在通过提供数据驱动的LLM优化见解，指导未来模型的有针对性开发和应用。

Abstract: 大型语言模型（LLMs）在各个领域取得了显著的成功，推动了重大的技术进步和创新。尽管模型规模和能力迅速增长，但关于结构配置如何影响性能的系统性、数据驱动的研究仍然稀缺。为了解决这一差距，我们提出了一个大规模数据集，其中包含各种开源LLM结构及其在多个基准测试中的性能。利用该数据集，我们进行了系统的、数据挖掘驱动的分析，以验证和量化结构配置与性能之间的关系。我们的研究首先回顾了LLM的历史发展，并探讨了潜在的未来趋势。然后，我们分析了各种结构选择如何影响跨基准测试的性能，并使用机制可解释性技术进一步证实了我们的发现。通过提供数据驱动的LLM优化见解，我们的工作旨在指导未来模型的有针对性开发和应用。我们将在https://huggingface.co/datasets/DX0369/LLM-Structure-Performance-Dataset上发布我们的数据集

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18136) | **Categories:** cs.LG, cs.AI

---

### [2] [MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](https://arxiv.org/abs/2509.18119)
*Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, Yuxiao Dong*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Building general-purpose graphical user interface (GUI) agents has become increasingly promising with the progress in vision language models. However, developing effective mobile GUI agents with reinforcement learning (RL) remains challenging due to the heavy-tailed distribution of task difficulty and the inefficiency of large-scale environment sampling. We present an online agentic reinforcement learning framework MOBILERL to enhance GUI agents in mobile environments. Its core component is the Difficulty-Adaptive GRPO (ADAGRPO) algorithm. In ADAGRPO, we design difficulty-adaptive positive replay and failure curriculum filtering to adapt the model to different task difficulties. We introduce the shortest path reward adjustment strategy to reshape rewards concerning the task length in multi-turn agentic tasks. Those strategies jointly stabilize RL training, improve sample efficiency, and generate strong performance across diverse mobile apps and tasks. We apply MOBILERL to two open models (Qwen2.5-VL-7B-Instruct and GLM-4.1V-9B-Base). The resultant MOBILERL-9B model achieves state-of-the-art results in terms of success rates on both AndroidWorld (75.8%) and AndroidLab (46.8%). The MOBILERL framework is adopted in the AutoGLM products, and also open-sourced at https://github.com/THUDM/MobileRL.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18119) | **Categories:** cs.LG, cs.AI

---

### [3] [MobiGPT: A Foundation Model for Mobile Wireless Networks](https://arxiv.org/abs/2509.18166)
*Xiaoqian Qi, Haoye Chai, Yong Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the rapid development of mobile communication technologies, future mobile networks will offer vast services and resources for commuting, production, daily life, and entertainment. Accurate and efficient forecasting of mobile data (e.g., cell traffic, user behavior, channel quality) helps operators monitor network state changes, orchestrate wireless resources, and schedule infrastructure and users, thereby improving supply efficiency and service quality. However, current forecasting paradigms rely on customized designs with tailored models for exclusive data types. Such approaches increase complexity and deployment costs under large-scale, heterogeneous networks involving base stations, users, and channels. In this paper, we design a foundation model for mobile data forecasting, MobiGPT, with a unified structure capable of forecasting three data types: base station traffic, user app usage, and channel quality. We propose a soft-prompt learning method to help the model understand features of different data types, and introduce a temporal masking mechanism to guide the model through three forecasting tasks: short-term prediction, long-term prediction, and distribution generation, supporting diverse optimization scenarios. Evaluations on real-world datasets with over 100,000 samples show that MobiGPT achieves accurate multi-type forecasting. Compared to existing models, it improves forecasting accuracy by 27.37%, 20.08%, and 7.27%, reflecting strong generalization. Moreover, MobiGPT exhibits superior zero/few-shot performance in unseen scenarios, with over 21.51% improvement, validating its strong transferability as a foundation model.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18166) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [AD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback](https://arxiv.org/abs/2509.18384)
*Yunhao Yang, Junyuan Hong, Gabriel Jacob Perin, Zhiwen Fan, Li Yin, Zhangyang Wang, Ufuk Topcu*

Main category: cs.RO

TL;DR: LAD-VF 提出了一种无需微调的框架，利用形式验证反馈进行自动提示工程，从而提高 LLM 在机器人任务中的规范依从性。


<details>
  <summary>Details</summary>
Motivation: 当前的大语言模型在物理世界中进行规划时，由于幻觉或对齐不足，常常违反安全和法规约束。传统的数据驱动对齐方法需要昂贵的人工标注，而最近的形式反馈方法仍然依赖于资源密集型的微调。

Method: LAD-VF 引入了一种基于形式验证信息的文本损失，并将其与 LLM-AutoDiff 集成，从而迭代地优化提示，而不是模型参数。

Result: 在机器人导航和操作任务中的实验表明，LAD-VF 显著提高了规范依从性，将成功率从 60% 提高到 90% 以上。

Conclusion: LAD-VF 提供了一种可扩展且可解释的途径，可以构建可信赖的、经过形式验证的 LLM 驱动的控制系统。

Abstract: 大型语言模型（LLM）可以将自然语言指令翻译成机器人、自动驾驶和其他领域的可执行行动计划。然而，在物理世界中部署 LLM 驱动的规划需要严格遵守安全和法规约束，但当前的模型由于幻觉或弱对齐而经常违反这些约束。传统的数据驱动对齐方法（如直接偏好优化 DPO）需要昂贵的人工标注，而最近的形式反馈方法仍然依赖于资源密集型的微调。在本文中，我们提出了一种无需微调的框架 LAD-VF，该框架利用形式验证反馈进行自动提示工程。通过引入与 LLM-AutoDiff 集成的形式验证信息文本损失，LAD-VF 迭代地优化提示，而不是模型参数。这产生了三个关键好处：（i）无需微调即可实现可扩展的适应；（ii）与模块化 LLM 架构兼容；（iii）通过可审计的提示实现可解释的改进。在机器人导航和操作任务中的实验表明，LAD-VF 显著提高了规范依从性，将成功率从 60% 提高到 90% 以上。因此，我们的方法提供了一种可扩展且可解释的途径，可以构建可信赖的、经过形式验证的 LLM 驱动的控制系统。

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18384) | **Categories:** cs.RO, cs.FL

---

### [2] [PEEK: Guiding and Minimal Image Representations for Zero-Shot Generalization of Robot Manipulation Policies](https://arxiv.org/abs/2509.18282)
*Jesse Zhang, Marius Memmel, Kevin Kim, Dieter Fox, Jesse Thomason, Fabio Ramos, Erdem Bıyık, Abhishek Gupta, Anqi Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robotic manipulation policies often fail to generalize because they must simultaneously learn where to attend, what actions to take, and how to execute them. We argue that high-level reasoning about where and what can be offloaded to vision-language models (VLMs), leaving policies to specialize in how to act. We present PEEK (Policy-agnostic Extraction of Essential Keypoints), which fine-tunes VLMs to predict a unified point-based intermediate representation: 1. end-effector paths specifying what actions to take, and 2. task-relevant masks indicating where to focus. These annotations are directly overlaid onto robot observations, making the representation policy-agnostic and transferable across architectures. To enable scalable training, we introduce an automatic annotation pipeline, generating labeled data across 20+ robot datasets spanning 9 embodiments. In real-world evaluations, PEEK consistently boosts zero-shot generalization, including a 41.4x real-world improvement for a 3D policy trained only in simulation, and 2-3.5x gains for both large VLAs and small manipulation policies. By letting VLMs absorb semantic and visual complexity, PEEK equips manipulation policies with the minimal cues they need--where, what, and how. Website at https://peek-robot.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18282) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [3] [Spatial Envelope MPC: High Performance Driving without a Reference](https://arxiv.org/abs/2509.18506)
*Siyuan Yu, Congkai Shen, Yufei Xi, James Dallas, Michael Thompson, John Subosits, Hiroshi Yasuda, Tulga Ersal*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a novel envelope based model predictive control (MPC) framework designed to enable autonomous vehicles to handle high performance driving across a wide range of scenarios without a predefined reference. In high performance autonomous driving, safe operation at the vehicle's dynamic limits requires a real time planning and control framework capable of accounting for key vehicle dynamics and environmental constraints when following a predefined reference trajectory is suboptimal or even infeasible. State of the art planning and control frameworks, however, are predominantly reference based, which limits their performance in such situations. To address this gap, this work first introduces a computationally efficient vehicle dynamics model tailored for optimization based control and a continuously differentiable mathematical formulation that accurately captures the entire drivable envelope. This novel model and formulation allow for the direct integration of dynamic feasibility and safety constraints into a unified planning and control framework, thereby removing the necessity for predefined references. The challenge of envelope planning, which refers to maximally approximating the safe drivable area, is tackled by combining reinforcement learning with optimization techniques. The framework is validated through both simulations and real world experiments, demonstrating its high performance across a variety of tasks, including racing, emergency collision avoidance and off road navigation. These results highlight the framework's scalability and broad applicability across a diverse set of scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18506) | **Categories:** cs.RO

---

### [4] [VLN-Zero: Rapid Exploration and Cache-Enabled Neurosymbolic Vision-Language Planning for Zero-Shot Transfer in Robot Navigation](https://arxiv.org/abs/2509.18592)
*Neel P. Bhatt, Yunhao Yang, Rohan Siva, Pranay Samineni, Daniel Milan, Zhangyang Wang, Ufuk Topcu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Rapid adaptation in unseen environments is essential for scalable real-world autonomy, yet existing approaches rely on exhaustive exploration or rigid navigation policies that fail to generalize. We present VLN-Zero, a two-phase vision-language navigation framework that leverages vision-language models to efficiently construct symbolic scene graphs and enable zero-shot neurosymbolic navigation. In the exploration phase, structured prompts guide VLM-based search toward informative and diverse trajectories, yielding compact scene graph representations. In the deployment phase, a neurosymbolic planner reasons over the scene graph and environmental observations to generate executable plans, while a cache-enabled execution module accelerates adaptation by reusing previously computed task-location trajectories. By combining rapid exploration, symbolic reasoning, and cache-enabled execution, the proposed framework overcomes the computational inefficiency and poor generalization of prior vision-language navigation methods, enabling robust and scalable decision-making in unseen environments. VLN-Zero achieves 2x higher success rate compared to state-of-the-art zero-shot models, outperforms most fine-tuned baselines, and reaches goal locations in half the time with 55% fewer VLM calls on average compared to state-of-the-art models across diverse environments. Codebase, datasets, and videos for VLN-Zero are available at: https://vln-zero.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18592) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG, cs.SY, eess.SY

---

### [5] [PIE: Perception and Interaction Enhanced End-to-End Motion Planning for Autonomous Driving](https://arxiv.org/abs/2509.18609)
*Chengran Yuan, Zijian Lu, Zhanqi Zhang, Yimin Zhao, Zefan Huang, Shuo Sun, Jiawei Sun, Jiahui Li, Christina Dao Wen Lee, Dongen Li, Marcelo H. Ang Jr*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end motion planning is promising for simplifying complex autonomous driving pipelines. However, challenges such as scene understanding and effective prediction for decision-making continue to present substantial obstacles to its large-scale deployment. In this paper, we present PIE, a pioneering framework that integrates advanced perception, reasoning, and intention modeling to dynamically capture interactions between the ego vehicle and surrounding agents. It incorporates a bidirectional Mamba fusion that addresses data compression losses in multimodal fusion of camera and LiDAR inputs, alongside a novel reasoning-enhanced decoder integrating Mamba and Mixture-of-Experts to facilitate scene-compliant anchor selection and optimize adaptive trajectory inference. PIE adopts an action-motion interaction module to effectively utilize state predictions of surrounding agents to refine ego planning. The proposed framework is thoroughly validated on the NAVSIM benchmark. PIE, without using any ensemble and data augmentation techniques, achieves an 88.9 PDM score and 85.6 EPDM score, surpassing the performance of prior state-of-the-art methods. Comprehensive quantitative and qualitative analyses demonstrate that PIE is capable of reliably generating feasible and high-quality ego trajectories.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18609) | **Categories:** cs.RO

---

### [6] [SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones](https://arxiv.org/abs/2509.18610)
*Maximilian Adang, JunEn Low, Ola Shorinwa, Mac Schwager*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large vision-language models have driven remarkable progress in open-vocabulary robot policies, e.g., generalist robot manipulation policies, that enable robots to complete complex tasks specified in natural language. Despite these successes, open-vocabulary autonomous drone navigation remains an unsolved challenge due to the scarcity of large-scale demonstrations, real-time control demands of drones for stabilization, and lack of reliable external pose estimation modules. In this work, we present SINGER for language-guided autonomous drone navigation in the open world using only onboard sensing and compute. To train robust, open-vocabulary navigation policies, SINGER leverages three central components: (i) a photorealistic language-embedded flight simulator with minimal sim-to-real gap using Gaussian Splatting for efficient data generation, (ii) an RRT-inspired multi-trajectory generation expert for collision-free navigation demonstrations, and these are used to train (iii) a lightweight end-to-end visuomotor policy for real-time closed-loop control. Through extensive hardware flight experiments, we demonstrate superior zero-shot sim-to-real transfer of our policy to unseen environments and unseen language-conditioned goal objects. When trained on ~700k-1M observation action pairs of language conditioned visuomotor data and deployed on hardware, SINGER outperforms a velocity-controlled semantic guidance baseline by reaching the query 23.33% more on average, and maintains the query in the field of view 16.67% more on average, with 10% fewer collisions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18610) | **Categories:** cs.RO

---

### [7] [The Case for Negative Data: From Crash Reports to Counterfactuals for Reasonable Driving](https://arxiv.org/abs/2509.18626)
*Jay Patrikar, Apoorva Sharma, Sushant Veer, Boyi Li, Sebastian Scherer, Marco Pavone*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning-based autonomous driving systems are trained mostly on incident-free data, offering little guidance near safety-performance boundaries. Real crash reports contain precisely the contrastive evidence needed, but they are hard to use: narratives are unstructured, third-person, and poorly grounded to sensor views. We address these challenges by normalizing crash narratives to ego-centric language and converting both logs and crashes into a unified scene-action representation suitable for retrieval. At decision time, our system adjudicates proposed actions by retrieving relevant precedents from this unified index; an agentic counterfactual extension proposes plausible alternatives, retrieves for each, and reasons across outcomes before deciding. On a nuScenes benchmark, precedent retrieval substantially improves calibration, with recall on contextually preferred actions rising from 24% to 53%. The counterfactual variant preserves these gains while sharpening decisions near risk.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18626) | **Categories:** cs.RO, cs.AI

---

### [8] [Distributionally Robust Safe Motion Planning with Contextual Information](https://arxiv.org/abs/2509.18666)
*Kaizer Rahaman, Simran Kumari, Ashish R. Hota*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present a distributionally robust approach for collision avoidance by incorporating contextual information. Specifically, we embed the conditional distribution of future trajectory of the obstacle conditioned on the motion of the ego agent in a reproducing kernel Hilbert space (RKHS) via the conditional kernel mean embedding operator. Then, we define an ambiguity set containing all distributions whose embedding in the RKHS is within a certain distance from the empirical estimate of conditional mean embedding learnt from past data. Consequently, a distributionally robust collision avoidance constraint is formulated, and included in the receding horizon based motion planning formulation of the ego agent. Simulation results show that the proposed approach is more successful in avoiding collision compared to approaches that do not include contextual information and/or distributional robustness in their formulation in several challenging scenarios.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18666) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [9] [3D Flow Diffusion Policy: Visuomotor Policy Learning via Generating Flow in 3D Space](https://arxiv.org/abs/2509.18676)
*Sangjun Noh, Dongwoo Nam, Kangmin Kim, Geonhyup Lee, Yeonguk Yu, Raeyoung Kang, Kyoobin Lee*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning robust visuomotor policies that generalize across diverse objects and interaction dynamics remains a central challenge in robotic manipulation. Most existing approaches rely on direct observation-to-action mappings or compress perceptual inputs into global or object-centric features, which often overlook localized motion cues critical for precise and contact-rich manipulation. We present 3D Flow Diffusion Policy (3D FDP), a novel framework that leverages scene-level 3D flow as a structured intermediate representation to capture fine-grained local motion cues. Our approach predicts the temporal trajectories of sampled query points and conditions action generation on these interaction-aware flows, implemented jointly within a unified diffusion architecture. This design grounds manipulation in localized dynamics while enabling the policy to reason about broader scene-level consequences of actions. Extensive experiments on the MetaWorld benchmark show that 3D FDP achieves state-of-the-art performance across 50 tasks, particularly excelling on medium and hard settings. Beyond simulation, we validate our method on eight real-robot tasks, where it consistently outperforms prior baselines in contact-rich and non-prehensile scenarios. These results highlight 3D flow as a powerful structural prior for learning generalizable visuomotor policies, supporting the development of more robust and versatile robotic manipulation. Robot demonstrations, additional results, and code can be found at https://sites.google.com/view/3dfdp/home.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18676) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [10] [Lang2Morph: Language-Driven Morphological Design of Robotic Hands](https://arxiv.org/abs/2509.18937)
*Yanyuan Qiao, Kieran Gilday, Yutong Xie, Josie Hughes*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Designing robotic hand morphologies for diverse manipulation tasks requires balancing dexterity, manufacturability, and task-specific functionality. While open-source frameworks and parametric tools support reproducible design, they still rely on expert heuristics and manual tuning. Automated methods using optimization are often compute-intensive, simulation-dependent, and rarely target dexterous hands. Large language models (LLMs), with their broad knowledge of human-object interactions and strong generative capabilities, offer a promising alternative for zero-shot design reasoning. In this paper, we present Lang2Morph, a language-driven pipeline for robotic hand design. It uses LLMs to translate natural-language task descriptions into symbolic structures and OPH-compatible parameters, enabling 3D-printable task-specific morphologies. The pipeline consists of: (i) Morphology Design, which maps tasks into semantic tags, structural grammars, and OPH-compatible parameters; and (ii) Selection and Refinement, which evaluates design candidates based on semantic alignment and size compatibility, and optionally applies LLM-guided refinement when needed. We evaluate Lang2Morph across varied tasks, and results show that our approach can generate diverse, task-relevant morphologies. To our knowledge, this is the first attempt to develop an LLM-based framework for task-conditioned robotic hand design.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.18937) | **Categories:** cs.RO

---
