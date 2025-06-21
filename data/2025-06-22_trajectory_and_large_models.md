# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-22

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [cs.CR (1)](#cs-cr)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [cs.IT (1)](#cs-it)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Truncated Proximal Policy Optimization](https://arxiv.org/abs/2506.15050)
*Tiantian Fan, Lingjun Liu, Yu Yue, Jiaze Chen, Chengyi Wang, Qiying Yu, Chi Zhang, Zhiqi Lin, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Bole Ma, Mofan Zhang, Gaohong Liu, Ru Zhang, Haotian Zhou, Cong Xie, Ruidong Zhu, Zhi Zhang, Xin Liu, Mingxuan Wang, Lin Yan, Yonghui Wu*

Main category: cs.AI

TL;DR: T-PPO通过截断优化和扩展优势估计，显著提高了大型语言模型推理训练的效率。


<details>
  <summary>Details</summary>
Motivation: 近端策略优化（PPO）由于其固有的on-policy性质，可能非常耗时，而响应长度的增加进一步加剧了这种情况。完全同步的长生成过程存在硬件利用率低的问题，在等待完整rollout期间，资源经常处于闲置状态。

Method: 提出了截断近端策略优化（T-PPO），通过简化策略更新和长度限制的响应生成来提高训练效率。此外，还提出了扩展广义优势估计（EGAE），用于从不完整的响应中进行优势估计，并设计了一种计算优化的机制，允许独立优化策略和价值模型。

Result: 在AIME 2024上，T-PPO将推理LLM的训练效率提高了2.5倍，并且优于其现有的竞争对手。

Conclusion: T-PPO将推理LLM的训练效率提高了2.5倍，并优于现有的竞争对手。

Abstract: 最近，测试时扩展的大型语言模型（LLM）通过生成长的思维链（CoT）展示了卓越的科学和专业任务推理能力。作为开发这些推理模型的关键组成部分，强化学习（RL），以近端策略优化（PPO）及其变体为例，允许模型通过试错来学习。然而，PPO由于其固有的on-policy性质，可能非常耗时，而响应长度的增加进一步加剧了这种情况。在这项工作中，我们提出了截断近端策略优化（T-PPO），它是PPO的一个新扩展，通过简化策略更新和长度限制的响应生成来提高训练效率。T-PPO缓解了硬件利用率低的问题，这是完全同步的长生成程序的固有缺陷，在等待完整rollout期间，资源经常处于闲置状态。我们的贡献有两方面。首先，我们提出了扩展广义优势估计（EGAE），用于从不完整的响应中进行优势估计，同时保持策略学习的完整性。其次，我们设计了一种计算优化的机制，允许独立优化策略和价值模型。通过选择性地过滤prompt和截断的tokens，这种机制减少了冗余计算，并在不牺牲收敛性能的情况下加速了训练过程。我们通过在AIME 2024上使用32B的基础模型证明了T-PPO的有效性和效率。实验结果表明，T-PPO将推理LLM的训练效率提高了2.5倍，并且优于其现有的竞争对手。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15050) | **Categories:** cs.AI

---


## cs.CR [cs.CR]
### [1] [Advanced Prediction of Hypersonic Missile Trajectories with CNN-LSTM-GRU Architectures](https://arxiv.org/abs/2506.15043)
*Amir Hossein Baradaran*

Main category: cs.CR

TL;DR: 本文提出了一种混合深度学习方法，用于高精度预测高超音速导弹的复杂轨迹，从而显著提升防御能力。


<details>
  <summary>Details</summary>
Motivation: 高超音速导弹以其极高的速度和机动性构成了严峻的挑战，因此准确的轨迹预测对于有效的对抗措施至关重要。

Method: 提出了一种新颖的混合深度学习方法，集成了卷积神经网络（CNN）、长短期记忆（LSTM）网络和门控循环单元（GRU）。

Result: 该方法成功地高精度预测了高超音速导弹的复杂轨迹。

Conclusion: 先进的机器学习技术可以增强防御系统的预测能力。

Abstract: 国防工业的进步对于确保国家安全至关重要，它能够提供强大的保护以应对新兴威胁。其中，高超音速导弹由于其极高的速度和机动性构成了严峻的挑战，因此准确的轨迹预测对于有效的对抗措施至关重要。本文通过采用一种新颖的混合深度学习方法来应对这一挑战，该方法集成了卷积神经网络（CNN）、长短期记忆（LSTM）网络和门控循环单元（GRU）。通过利用这些架构的优势，所提出的方法成功地高精度预测了高超音速导弹的复杂轨迹，为防御策略和导弹拦截技术做出了重大贡献。这项研究证明了先进的机器学习技术在增强防御系统预测能力方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15043) | **Categories:** cs.CR, cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review](https://arxiv.org/abs/2506.14831)
*Céline Finet, Stephane Da Silva Martins, Jean-Bernard Hayet, Ioannis Karamouzas, Javad Amirian, Sylvie Le Hégarat-Mascle, Julien Pettré, Emanuel Aldea*

Main category: cs.CV

TL;DR: 本文综述了 2020-2024 年基于深度学习的多智能体轨迹预测研究，并突出了该领域的挑战和未来方向。


<details>
  <summary>Details</summary>
Motivation: 随着以人为本的轨迹预测 (HTP) 中强大的数据驱动方法的出现，更深入地了解多智能体交互触手可及，这对自动导航和人群建模等领域具有重要意义。

Method: 根据现有方法的架构设计、输入表示和整体预测策略对其进行分类，特别关注使用 ETH/UCY 基准评估的模型。

Result: 回顾了基于深度学习的多智能体轨迹预测的最新进展，重点介绍了 2020 年至 2024 年间发表的研究。

Conclusion: 强调了多智能体 HTP 领域的关键挑战和未来研究方向。

Abstract: 随着以人为本的轨迹预测 (HTP) 中强大的数据驱动方法的出现，更深入地了解多智能体交互触手可及，这对自动导航和人群建模等领域具有重要意义。本文回顾了基于深度学习的多智能体轨迹预测的最新进展，重点介绍了 2020 年至 2024 年间发表的研究。我们根据现有方法的架构设计、输入表示和整体预测策略对其进行分类，特别关注使用 ETH/UCY 基准评估的模型。此外，我们还强调了多智能体 HTP 领域的关键挑战和未来研究方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14831) | **Categories:** cs.CV, cs.LG, cs.RO

---

### [2] [MapFM: Foundation Model-Driven HD Mapping with Multi-Task Contextual Learning](https://arxiv.org/abs/2506.15313)
*Leonid Ivanov, Vasily Yuryev, Dmitry Yudin*

Main category: cs.CV

TL;DR: MapFM 通过结合基础模型和多任务学习，提高了在线矢量化高清地图生成的准确性和质量。


<details>
  <summary>Details</summary>
Motivation: 高清地图和鸟瞰图中的语义地图对于自动驾驶中的精确定位、规划和决策至关重要。

Method: 提出了一种名为 MapFM 的增强型端到端模型，用于在线矢量化高清地图生成。

Result: 通过结合强大的基础模型进行相机图像编码，显著提高了特征表示质量。

Conclusion: 通过引入多任务学习和上下文监督，MapFM 提高了预测矢量化高清地图的准确性和质量。

Abstract: 在自动驾驶中，高清（HD）地图和鸟瞰（BEV）视图中的语义地图对于精确定位、规划和决策至关重要。本文介绍了一种名为 MapFM 的增强型端到端模型，用于在线矢量化高清地图生成。我们通过结合强大的基础模型进行相机图像编码，显著提高了特征表示质量。为了进一步丰富模型对环境的理解并提高预测质量，我们集成了辅助预测头，用于在 BEV 表示中进行语义分割。这种多任务学习方法提供了更丰富的上下文监督，从而实现了更全面的场景表示，并最终提高了预测矢量化高清地图的准确性和质量。源代码可在 https://github.com/LIvanoff/MapFM 获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15313) | **Categories:** cs.CV, cs.AI

---


## cs.IT [cs.IT]
### [1] [LLM Agent for Hyper-Parameter Optimization](https://arxiv.org/abs/2506.15167)
*Wanzhe Wang, Jianqiu Peng, Menghao Hu, Weihuang Zhong, Tong Zhang, Shuai Wang, Yixin Zhang, Mingjie Shao, Wanli Ni*

Main category: cs.IT

TL;DR: 提出了一种基于LLM的智能体，用于自动调整WS-PSO-CM算法的超参数，以提高无线电地图无人机通信性能。


<details>
  <summary>Details</summary>
Motivation: 现有的基于启发式的WS-PSO-CM算法超参数调整方法自动化程度低，性能不佳。

Method: 设计了一个基于大型语言模型（LLM）的智能体，用于自动超参数调整，采用了迭代框架和模型上下文协议（MCP）。

Result: 实验结果表明，通过LLM智能体生成的超参数实现的最小和速率显著高于人工启发式和随机生成方法。

Conclusion: LLM智能体能够有效找到高性能超参数，显著优于人工和随机方法。

Abstract: 超参数对于通信算法的性能至关重要。然而，目前针对无线电地图无人机（UAV）轨迹和通信的带有交叉和变异的warm-start粒子群优化（WS-PSO-CM）算法的超参数调整方法主要基于启发式，自动化程度低，性能不尽如人意。在本文中，我们设计了一个基于大型语言模型（LLM）的智能体，用于自动超参数调整，其中应用了迭代框架和模型上下文协议（MCP）。特别地，首先通过配置文件设置LLM智能体，该文件指定任务、背景和输出格式。然后，LLM智能体由提示需求驱动，并迭代地调用WS-PSO-CM算法进行探索。最后，LLM智能体自主终止循环并返回一组超参数。我们的实验结果表明，通过LLM智能体生成的超参数实现的最小和速率显著高于人工启发式和随机生成方法。这表明具有PSO知识和WS-PSO-CM算法背景的LLM智能体在寻找高性能超参数方面非常有用。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15167) | **Categories:** cs.IT, cs.AI, math.IT

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models](https://arxiv.org/abs/2506.15065)
*Trishna Chakraborty, Udita Ghosh, Xiaopan Zhang, Fahim Faisal Niloy, Yue Dong, Jiachen Li, Amit K. Roy-Chowdhury, Chengyu Song*

Main category: cs.LG

TL;DR: 该论文首次系统研究了基于 LLM 的具身代理在场景任务不一致情况下执行长时程任务时的幻觉问题。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型越来越多地被用作具身代理的认知核心，但由于未能将用户指令与观察到的物理环境联系起来而导致的幻觉，可能导致导航错误。

Method: 构建了一个幻觉探测集，能够诱导比基本提示高出 40 倍的幻觉率。

Result: 在两个模拟环境中评估了 12 个模型，发现模型表现出推理能力，但未能解决场景任务不一致问题。

Conclusion: 模型在处理场景任务不一致时存在根本性局限，无法解决这些不一致问题。

Abstract: 大型语言模型（LLM）越来越多地被用作具身代理的认知核心。然而，由于未能将用户指令与观察到的物理环境联系起来而导致的幻觉，可能导致导航错误，例如搜索不存在的冰箱。在本文中，我们首次系统地研究了基于 LLM 的具身代理在场景任务不一致情况下执行长时程任务时的幻觉问题。我们的目标是了解幻觉发生的程度，哪些类型的不一致会触发幻觉，以及当前模型如何响应。为了实现这些目标，我们构建了一个幻觉探测集，该探测集建立在现有基准之上，能够诱导比基本提示高出 40 倍的幻觉率。通过在两个模拟环境中评估 12 个模型，我们发现模型表现出推理能力，但未能解决场景任务不一致问题——突显了处理不可行任务的根本局限性。我们还为每种场景提供了理想模型行为的可行见解，为开发更强大、更可靠的规划策略提供了指导。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15065) | **Categories:** cs.LG, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Robust Instant Policy: Leveraging Student's t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation](https://arxiv.org/abs/2506.15157)
*Hanbit Oh, Andrea M. Salcedo-Vázquez, Ixchel G. Ramirez-Alpizar, Yukiyasu Domae*

Main category: cs.RO

TL;DR: 提出了一种鲁棒瞬时策略（RIP），通过利用Student的t回归模型抵抗大型语言模型（LLM）的幻觉，从而提高上下文模仿学习的可靠性。


<details>
  <summary>Details</summary>
Motivation: 基于LLM的瞬时策略存在幻觉问题，导致生成的轨迹偏离给定的演示，从而降低了其在机器人领域的可靠性。

Method: 提出了一种新的鲁棒上下文模仿学习算法，称为鲁棒瞬时策略（RIP），它利用Student的t回归模型来抵抗瞬时策略的幻觉轨迹，从而实现可靠的轨迹生成。

Result: RIP在任务成功率方面显著优于最先进的IL方法，至少提高了26%，特别是在日常任务的低数据场景中。

Conclusion: RIP在模拟和真实环境中的实验表明，其性能显著优于最先进的IL方法，特别是在日常任务的低数据场景中，任务成功率至少提高了26%。

Abstract: 模仿学习（IL）旨在通过观察少量人类演示，使机器人能够自主执行任务。最近，一种IL的变体，称为上下文IL，利用现成的（off-the-shelf）大型语言模型（LLM）作为瞬时策略，从一些给定的演示中理解上下文以执行新任务，而不是用大规模演示显式更新网络模型。然而，由于基于LLM的瞬时策略等幻觉问题，其在机器人领域的可靠性受到损害，这些问题偶尔会产生偏离给定演示的不良轨迹。为了缓解这个问题，我们提出了一种新的鲁棒上下文模仿学习算法，称为鲁棒瞬时策略（RIP），它利用Student的t回归模型来抵抗瞬时策略的幻觉轨迹，从而实现可靠的轨迹生成。具体来说，RIP生成多个候选机器人轨迹，以从LLM完成给定的任务，并使用Student的t分布聚合它们，这有利于忽略异常值（即幻觉）；因此，生成了针对幻觉的鲁棒轨迹。我们的实验在模拟和真实环境进行，结果表明，RIP显著优于最先进的IL方法，特别是在日常任务的低数据场景中，任务成功率至少提高了26%。视频结果可在https://sites.google.com/view/robustinstantpolicy查看。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15157) | **Categories:** cs.RO, cs.CV

---

### [2] [Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers](https://arxiv.org/abs/2506.14855)
*Tommaso Belvedere, Michael Ziegltrum, Giulio Turrisi, Valerio Modugno*

Main category: cs.RO

TL;DR: F-MPPI通过增加局部线性反馈增益来增强标准MPPI，从而实现鲁棒、高频的机器人控制。


<details>
  <summary>Details</summary>
Motivation: Model Predictive Path Integral control is a powerful sampling-based approach suitable for complex robotic tasks. However, its applicability in real-time, highfrequency robotic control scenarios is limited by computational demands.

Method: This paper introduces Feedback-MPPI (F-MPPI), a novel framework that augments standard MPPI by computing local linear feedback gains derived from sensitivity analysis inspired by Riccati-based feedback used in gradient-based MPC.

Result: Results illustrate that incorporating local feedback significantly improves control performance and stability.

Conclusion: Incorporating local feedback significantly improves control performance and stability, enabling robust, high-frequency operation suitable for complex robotic systems.

Abstract: 模型预测路径积分控制是一种强大的基于采样的方法，适用于复杂的机器人任务，因为它在处理非线性动力学和非凸成本方面具有灵活性。然而，由于计算需求，它在实时、高频机器人控制场景中的适用性受到限制。本文介绍了一种新的框架Feedback-MPPI (F-MPPI)，该框架通过计算局部线性反馈增益来增强标准MPPI，这些增益来自灵敏度分析，灵感来自基于梯度的MPC中使用的基于Riccati的反馈。这些增益允许在当前状态下进行快速闭环校正，而无需在每个时间步进行完全重新优化。我们通过在两个机器人平台上的模拟和真实实验证明了F-MPPI的有效性：一个四足机器人在不平坦地形上执行动态运动，一个四旋翼飞行器在机载计算的情况下执行激进的机动。结果表明，结合局部反馈可以显著提高控制性能和稳定性，从而实现适用于复杂机器人系统的鲁棒、高频操作。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14855) | **Categories:** cs.RO, cs.AI

---

### [3] [Towards Perception-based Collision Avoidance for UAVs when Guiding the Visually Impaired](https://arxiv.org/abs/2506.14857)
*Suman Raj, Swapnil Padhi, Ruchi Bhoot, Prince Modi, Yogesh Simmhan*

Main category: cs.RO

TL;DR: 本文提出了一种基于多 DNN 框架的无人机辅助视障人士在城市环境中导航的路径规划系统。


<details>
  <summary>Details</summary>
Motivation: 本文研究了使用无人机辅助视障人士（VIP）在户外城市环境中导航。

Method: 我们使用几何公式表示该问题，并提出了一个基于多 DNN 的框架，用于无人机和 VIP 的避障。

Result: 在大学校园环境中的无人机-人系统评估验证了我们的算法在三种场景中的可行性；当 VIP 在人行道上行走、靠近停放的车辆以及在拥挤的街道上。

Conclusion: 在大学校园环境中的无人机-人系统评估验证了我们的算法在三种场景中的可行性：当 VIP 在人行道上行走、靠近停放的车辆以及在拥挤的街道上。

Abstract: 无人机结合机载传感器、机器学习和计算机视觉算法的自主导航正在影响农业、物流和灾害管理等多个领域。在本文中，我们研究了使用无人机辅助视障人士（VIP）在户外城市环境中导航。具体来说，我们提出了一个基于感知的路径规划系统，用于在 VIP 附近进行局部规划，并结合基于 GPS 和地图的全局规划器进行粗略规划。我们使用几何公式表示该问题，并提出了一个基于多 DNN 的框架，用于无人机和 VIP 的避障。我们在大学校园环境中的无人机-人系统上进行的评估验证了我们的算法在三种场景中的可行性：当 VIP 在人行道上行走、靠近停放的车辆以及在拥挤的街道上。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14857) | **Categories:** cs.RO, cs.CV

---

### [4] [FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization](https://arxiv.org/abs/2506.14968)
*Rajat Kumar Jenamani, Tom Silver, Ben Dodson, Shiqin Tong, Anthony Song, Yuting Yang, Ziang Liu, Benjamin Howe, Aimee Whitneck, Tapomayukh Bhattacharjee*

Main category: cs.RO

TL;DR: FEAST是一个灵活的膳食辅助系统，可以通过模块化硬件、多样化的交互方式和参数化的行为树进行个性化定制，以满足个体护理对象的独特需求。


<details>
  <summary>Details</summary>
Motivation: 由于活动、环境、食物种类和用户偏好的多样性，居家膳食辅助仍然具有挑战性。

Method: FEAST是一种灵活的膳食辅助系统，可以通过模块化硬件、多样化的交互方式和参数化的行为树进行个性化定制。

Result: FEAST提供了广泛的透明和安全的调整，并且优于仅限于固定定制的现有技术。

Conclusion: 用户可以成功地个性化FEAST以满足他们的个人需求和偏好。

Abstract: 物理护理机器人有望改善全球数百万需要喂食帮助的人的生活质量。然而，由于部署期间出现的各种活动（例如，吃饭、喝水、擦嘴）、环境（例如，社交、看电视）、食物种类和用户偏好，居家膳食辅助仍然具有挑战性。在这项工作中，我们提出了FEAST，这是一种灵活的膳食辅助系统，可以在实际应用中进行个性化定制，以满足个体护理对象的独特需求。我们的系统与两位社区研究员合作开发，并以对不同护理对象群体的前期研究为基础，以适应性、透明性和安全性这三个在实际应用中进行个性化定制的关键原则为指导。FEAST通过以下方式体现了这些原则：（i）模块化硬件，可以在辅助喂食、饮水和擦嘴之间切换，（ii）多样化的交互方式，包括Web界面、头部手势和物理按钮，以适应不同的功能能力和偏好，以及（iii）可以使用大型语言模型安全透明地调整的参数化行为树。我们根据前期研究中确定的个性化需求评估了我们的系统，表明FEAST提供了广泛的透明和安全的调整，并且优于仅限于固定定制的现有技术。为了证明其实际应用性，我们与两位护理对象（他们是社区研究员）进行了一项家庭用户研究，在三个不同的场景中，每人喂食三餐。我们还通过与一位以前不熟悉该系统的职业治疗师进行评估，来评估FEAST的生态有效性。在所有情况下，用户都可以成功地个性化FEAST以满足他们的个人需求和偏好。网站：https://emprise.cs.cornell.edu/feast

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.14968) | **Categories:** cs.RO, cs.AI

---

### [5] [DyNaVLM: Zero-Shot Vision-Language Navigation System with Dynamic Viewpoints and Self-Refining Graph Memory](https://arxiv.org/abs/2506.15096)
*Zihe Ji, Huangxuan Lin, Yue Gao*

Main category: cs.RO

TL;DR: DyNaVLM 是一种新的视觉语言导航框架，它使用动态动作空间和协作图记忆来实现高效的真实世界机器人导航，无需任何训练。


<details>
  <summary>Details</summary>
Motivation: 先前的视觉语言导航方法受限于固定的角度或距离间隔。

Method: DyNaVLM 是一种端到端视觉语言导航框架，它使用视觉语言模型 (VLM) 并允许智能体通过视觉语言推理自由选择导航目标。该框架的核心是一个自完善的图记忆，它可以将对象位置存储为可执行的拓扑关系，通过分布式图更新实现跨机器人内存共享，并通过检索增强来增强 VLM 的决策能力。

Result: DyNaVLM 在 GOAT 和 ObjectNav 基准测试中表现出高性能。真实世界的测试进一步验证了其鲁棒性和泛化能力。

Conclusion: 该系统通过动态动作空间、协作图记忆和无训练部署，为可扩展的具身机器人建立了一个新范例，弥合了离散 VLN 任务和连续真实世界导航之间的差距。

Abstract: 我们提出了 DyNaVLM，这是一个使用视觉语言模型 (VLM) 的端到端视觉语言导航框架。与先前受固定角度或距离间隔约束的方法不同，我们的系统使智能体能够通过视觉语言推理自由选择导航目标。它的核心在于一个自完善的图记忆，该记忆 1) 将对象位置存储为可执行的拓扑关系，2) 通过分布式图更新实现跨机器人内存共享，以及 3) 通过检索增强来增强 VLM 的决策能力。DyNaVLM 无需特定于任务的训练或微调即可运行，在 GOAT 和 ObjectNav 基准测试中表现出高性能。真实世界的测试进一步验证了其鲁棒性和泛化能力。该系统的三项创新：动态动作空间公式、协作图记忆和无训练部署，为可扩展的具身机器人建立了一个新范例，弥合了离散 VLN 任务和连续真实世界导航之间的差距。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15096) | **Categories:** cs.RO

---


## eess.SY [eess.SY]
### [1] [Model Predictive Path-Following Control for a Quadrotor](https://arxiv.org/abs/2506.15447)
*David Leprich, Mario Rosenfelder, Mario Hermle, Jingshan Chen, Peter Eberhard*

Main category: eess.SY

TL;DR: 本文提出了一种基于模型预测控制的路径跟踪方法，并成功应用于Crazyflie四旋翼飞行器，通过实验验证了其有效性，并扩展到走廊路径跟踪。


<details>
  <summary>Details</summary>
Motivation: 现有的路径跟踪解决方案缺乏显式处理状态和输入约束的能力，通常采用保守的两阶段方法，或者仅适用于线性系统。为了解决这些问题，本文旨在提出一种能够处理约束并在四旋翼飞行器上实现实时路径跟踪的方案。

Method: 提出了一种基于模型预测控制（MPC）的路径跟踪框架，并将其应用于Crazyflie四旋翼飞行器。该框架包含一个底层的姿态控制器，以满足四旋翼控制的实时性需求。

Result: 通过实际实验证明了所提出的基于MPC的路径跟踪方法在四旋翼飞行器上的有效性。此外，作为原始方法的扩展，提出了一种走廊路径跟踪方法，允许在路径精确跟踪可能过于严格的情况下偏离路径。

Conclusion: 通过实际实验验证了该方法在四旋翼飞行器上的有效性，并提出了一种走廊路径跟踪方法，允许在路径跟踪过于严格的情况下偏离路径。

Abstract: 无人机辅助流程的自动化是一个复杂的任务。许多解决方案依赖于轨迹生成和跟踪，相比之下，路径跟踪控制是一种特别有前途的方法，它为无人机和其他车辆的自动化任务提供了一种直观和自然的方法。虽然已经提出了不同的路径跟踪问题解决方案，但它们中的大多数缺乏显式处理状态和输入约束的能力，以保守的两阶段方法制定，或者仅适用于线性系统。为了应对这些挑战，本文建立在基于模型预测控制的路径跟踪框架之上，并将其应用扩展到Crazyflie四旋翼飞行器，并在硬件实验中进行了研究。模型预测路径跟踪控制公式中包含一个包含底层姿态控制器的级联控制结构，以满足四旋翼控制的具有挑战性的实时需求。通过实际实验证明了该方法的有效性，据作者所知，这代表了这种基于MPC的路径跟踪方法在四旋翼飞行器上的新颖应用。此外，作为对原始方法的扩展，为了允许在路径精确跟踪可能过于严格的情况下偏离路径，提出了一种走廊路径跟踪方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.15447) | **Categories:** eess.SY, cs.RO, cs.SY, 93-XX

---
