# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-13

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (1)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (9)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [AIA Forecaster: Technical Report](https://arxiv.org/abs/2511.07678)
*Rohan Alur, Bradly C. Stadie, Daniel Kang, Ryan Chen, Matt McManus, Michael Rickert, Tyler Lee, Michael Federici, Richard Zhu, Dennis Fogerty, Hayley Williamson, Nina Lozinski, Aaron Linsky, Jasjeet S. Sekhon*

Main category: cs.AI

TL;DR: AIA Forecaster是一个基于大型语言模型（LLM）的预测系统，它结合了代理搜索、预测协调和统计校准技术，在预测性能上达到了人类超预测者的水平。


<details>
  <summary>Details</summary>
Motivation: 本文旨在解决如何利用非结构化数据，通过大型语言模型实现专家级别的判断预测。

Method: 该方法结合了三个核心要素：高质量新闻来源上的代理搜索、协调相同事件不同预测的监督代理，以及用于对抗大型语言模型中行为偏差的统计校准技术。

Result: 在ForecastBench基准测试中，AIA Forecaster的性能与人类超预测者相当，超过了之前的LLM基线。此外，在更具挑战性的预测市场基准测试中，AIA Forecaster虽然不如市场共识，但与市场共识的集成优于单独的市场共识。

Conclusion: 该研究建立了一种新的AI预测技术，达到了专家水平，并为未来的研究提供了可转移的实践建议。

Abstract: 本技术报告介绍了AIA Forecaster，这是一个基于大型语言模型（LLM）的系统，它使用非结构化数据进行判断预测。AIA Forecaster方法结合了三个核心要素：在高质量新闻来源上进行代理搜索，一个协调相同事件不同预测的监督代理，以及一套用于对抗大型语言模型中行为偏差的统计校准技术。在ForecastBench基准测试（Karger et al., 2024）中，AIA Forecaster的性能与人类超预测者相当，超过了之前的LLM基线。除了报告ForecastBench的结果外，我们还介绍了一个来自流动预测市场的、更具挑战性的预测基准。虽然AIA Forecaster在这个基准上表现不如市场共识，但AIA Forecaster与市场共识的集成优于单独的市场共识，表明我们的预测器提供了额外的有用信息。我们的工作建立了一种新的AI预测技术，并为未来的研究提供了可转移的实践建议。据我们所知，这是第一个经过验证的、大规模实现专家级预测的工作。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07678) | **Categories:** cs.AI

---

### [2] [Towards AI-Assisted Generation of Military Training Scenarios](https://arxiv.org/abs/2511.07690)
*Soham Hans, Volkan Ustun, Benjamin Nye, James Sterrett, Matthew Green*

Main category: cs.AI

TL;DR: 本文提出了一种基于大型语言模型（LLM）的多智能体、多模态推理框架，用于生成复杂且可适应的模拟训练场景，从而显著提升军事训练的自动化水平。


<details>
  <summary>Details</summary>
Motivation: 传统上，基于仿真的训练依赖于创建复杂且适应性强的场景，但这是一个费力且资源密集的过程。以往的研究探索了军事训练的场景生成，但 pre-LLM AI 工具难以生成足够复杂或适应性强的场景。

Method: 该框架将场景生成分解为一系列子问题，并为每个子问题定义了 AI 工具的角色：(1) 生成选项供人工作者选择，(2) 生成候选产品供人工批准或修改，或 (3) 完全自动生成文本制品。该框架采用基于 LLM 的专用智能体来解决不同的子问题。每个智能体接收来自先前子问题智能体的输入，整合基于文本的场景细节和视觉信息（例如，地图特征、单位位置），并应用专门的推理来生成适当的输出。后续智能体按顺序处理这些输出，保持逻辑一致性并确保准确的文档生成。

Result: 通过生成作战命令（OPORD）的机动方案和行动部分，并估计地图位置和行动作为先决条件，验证了该框架的可行性和准确性。结果表明，基于 LLM 的多智能体系统有潜力生成连贯、细致的文档，并动态适应变化的环境，从而推进军事训练场景生成的自动化。

Conclusion: 该研究表明，基于 LLM 的多智能体系统能够有效生成复杂且可适应的模拟训练场景，显著提升军事训练的自动化水平，具有巨大的应用潜力。

Abstract: 为了在基于仿真的训练中实现专家级的表现，需要创建复杂且适应性强的场景，但这通常是一个费力且资源密集的过程。虽然之前的研究探索了军事训练的场景生成，但 pre-LLM AI 工具难以生成足够复杂或适应性强的场景。本文介绍了一种多智能体、多模态推理框架，该框架利用大型语言模型 (LLM) 来生成关键的训练工件，例如作战命令 (OPORD)。我们通过将场景生成分解为子问题的层次结构来构建我们的框架，并为每个子问题定义 AI 工具的角色：(1) 生成选项供人工作者选择，(2) 生成候选产品供人工批准或修改，或 (3) 完全自动生成文本制品。我们的框架采用基于 LLM 的专用智能体来解决不同的子问题。每个智能体接收来自先前子问题智能体的输入，整合基于文本的场景细节和视觉信息（例如，地图特征、单位位置），并应用专门的推理来生成适当的输出。后续智能体按顺序处理这些输出，保持逻辑一致性并确保准确的文档生成。这种多智能体策略克服了在处理此类高度复杂的任务时基本提示或单智能体方法的局限性。我们通过一个概念验证来验证我们的框架，该验证生成 OPORD 的机动方案和行动部分，同时估计地图位置和行动作为先决条件，证明了其可行性和准确性。我们的结果表明，LLM 驱动的多智能体系统有潜力生成连贯、细致的文档，并动态适应变化的环境，从而推进军事训练场景生成的自动化。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07690) | **Categories:** cs.AI

---

### [3] [SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder](https://arxiv.org/abs/2511.07896)
*Dengcan Liu, Jiahao Li, Zheren Fu, Yi Tu, Jiajun Li, Zhendong Mao, Yongdong Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reward models (RMs) are a core component in the post-training of large language models (LLMs), serving as proxies for human preference evaluation and guiding model alignment. However, training reliable RMs under limited resources remains challenging due to the reliance on large-scale preference annotations and the high cost of fine-tuning LLMs. To address this, we propose SparseRM, which leverages Sparse Autoencoder (SAE) to extract preference-relevant information encoded in model representations, enabling the construction of a lightweight and interpretable reward model. SparseRM first employs SAE to decompose LLM representations into interpretable directions that capture preference-relevant features. The representations are then projected onto these directions to compute alignment scores, which quantify the strength of each preference feature in the representations. A simple reward head aggregates these scores to predict preference scores. Experiments on three preference modeling tasks show that SparseRM achieves superior performance over most mainstream RMs while using less than 1% of trainable parameters. Moreover, it integrates seamlessly into downstream alignment pipelines, highlighting its potential for efficient alignment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07896) | **Categories:** cs.AI, cs.CL

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Sparse3DPR: Training-Free 3D Hierarchical Scene Parsing and Task-Adaptive Subgraph Reasoning from Sparse RGB Views](https://arxiv.org/abs/2511.07813)
*Haida Feng, Hao Wei, Zewen Xu, Haolin Wang, Chade Li, Yihong Wu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently, large language models (LLMs) have been explored widely for 3D scene understanding. Among them, training-free approaches are gaining attention for their flexibility and generalization over training-based methods. However, they typically struggle with accuracy and efficiency in practical deployment. To address the problems, we propose Sparse3DPR, a novel training-free framework for open-ended scene understanding, which leverages the reasoning capabilities of pre-trained LLMs and requires only sparse-view RGB inputs. Specifically, we introduce a hierarchical plane-enhanced scene graph that supports open vocabulary and adopts dominant planar structures as spatial anchors, which enables clearer reasoning chains and more reliable high-level inferences. Furthermore, we design a task-adaptive subgraph extraction method to filter query-irrelevant information dynamically, reducing contextual noise and improving 3D scene reasoning efficiency and accuracy. Experimental results demonstrate the superiority of Sparse3DPR, which achieves a 28.7% EM@1 improvement and a 78.2% speedup compared with ConceptGraphs on the Space3D-Bench. Moreover, Sparse3DPR obtains comparable performance to training-based methods on ScanQA, with additional real-world experiments confirming its robustness and generalization capability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07813) | **Categories:** cs.CV, cs.AI

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Counterfactual Forecasting of Human Behavior using Generative AI and Causal Graphs](https://arxiv.org/abs/2511.07484)
*Dharmateja Priyadarshi Uddandarao, Ravi Kiran Vadlamani*

Main category: cs.LG

TL;DR: 该论文提出了一个新颖的反事实用户行为预测框架，它结合了结构因果模型与基于Transformer的生成式人工智能。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在解决用户行为预测问题，并量化不同干预措施对用户行为的影响。

Method: 该方法构建因果图来模拟用户互动、采用指标和产品特征之间的关系，并利用生成模型在反事实条件下生成逼真的行为轨迹。

Result: 在网络互动、移动应用和电子商务数据集上的实验表明，该方法优于传统的预测和提升建模技术。

Conclusion: 该框架通过因果路径可视化提高了可解释性，使产品团队能够在部署前有效地模拟和评估可能的干预措施。

Abstract: 本研究提出了一个新颖的反事实用户行为预测框架，该框架结合了结构因果模型与基于Transformer的生成式人工智能。为了模拟虚构情境，该方法创建了因果图，这些因果图描绘了用户互动、采用指标和产品特征之间的联系。该框架通过使用以因果变量为条件的生成模型，生成反事实条件下的真实行为轨迹。通过在来自网络互动、移动应用和电子商务的数据集上进行测试，该方法优于传统的预测和提升建模技术。由于该框架通过因果路径可视化提高了可解释性，因此产品团队可以在部署前有效地模拟和评估可能的干预措施。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07484) | **Categories:** cs.LG, cs.CE, stat.ME

---


## 机器人学 (Robotics) [cs.RO]
### [1] [ViPRA: Video Prediction for Robot Actions](https://arxiv.org/abs/2511.07732)
*Sandeep Routray, Hengkai Pan, Unnat Jain, Shikhar Bahl, Deepak Pathak*

Main category: cs.RO

TL;DR: ViPRA提出了一种预训练-微调框架，通过预测未来视觉观测和运动中心潜在动作，从无动作视频中学习连续机器人控制。


<details>
  <summary>Details</summary>
Motivation: 现有视频缺乏标注动作，限制了其在机器人学习中的应用。

Method: ViPRA训练一个视频-语言模型来预测未来视觉观测和运动中心潜在动作，并使用感知损失和光流一致性来训练这些潜在动作。

Result: ViPRA在SIMPLER基准测试上获得了16%的提升，在真实世界操作任务中获得了13%的提升。

Conclusion: ViPRA避免了昂贵的动作标注，支持跨embodiment泛化，并通过分块动作解码实现了高达22 Hz的平滑、高频连续控制。

Abstract: ViPRA提出了一种简单的预训练-微调框架，该框架可以从这些无动作视频中学习连续机器人控制。该框架训练一个视频-语言模型来预测未来的视觉观测和以运动为中心的潜在动作，这些潜在动作作为场景动态的中间表示。我们使用感知损失和光流一致性来训练这些潜在动作，以确保它们反映物理基础的行为。对于下游控制，我们引入了一个分块流动匹配解码器，该解码器使用仅100到200个远程操作演示将潜在动作映射到特定于机器人的连续动作序列。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07732) | **Categories:** cs.RO, cs.AI, cs.CL, cs.CV, cs.LG

---

### [2] [LLM-GROP: Visually Grounded Robot Task and Motion Planning with Large Language Models](https://arxiv.org/abs/2511.07727)
*Xiaohan Zhang, Yan Ding, Yohei Hayamizu, Zainab Altaweel, Yifeng Zhu, Yuke Zhu, Peter Stone, Chris Paxton, Shiqi Zhang*

Main category: cs.RO

TL;DR: 本文提出了一种结合常识知识和计算机视觉的 TAMP 框架，用于解决多物体移动操作任务。


<details>
  <summary>Details</summary>
Motivation: 解决移动操作中，机器人如何在不明确目标下，根据常识知识放置物体的问题。

Method: 利用大型语言模型（LLM）的常识知识辅助任务和运动规划，并使用计算机视觉方法学习选择基座位置的策略。

Result: 在真实和模拟环境中进行了定量实验，机器人完成了 84.4% 的真实物体重排试验，但性能仍低于有经验的服务员。

Conclusion: 本文提出的 TAMP 框架能够处理多物体移动操作任务，并适应新情况，但性能仍有提升空间。

Abstract: 任务规划和运动规划是机器人学中两个最重要的问题。任务规划方法帮助机器人实现高层目标，而运动规划方法保证底层可行性。任务和运动规划 (TAMP) 方法交错进行任务规划和运动规划这两个过程，以确保目标达成和运动可行性。在 TAMP 的背景下，我们关注的是多物体的移动操作 (MoMa)，其中有必要交错导航和操作的动作。特别地，我们的目标是计算每个物体应该放置在哪里以及如何放置，给定不明确的目标，例如“用叉子、刀子和盘子摆放餐桌”。我们利用来自大型语言模型 (LLM) 的丰富的常识知识，例如关于餐具如何组织的知识，以促进任务级别和运动级别的规划。此外，我们使用计算机视觉方法来学习选择基座位置的策略，以方便 MoMa 行为，其中基座位置对应于机器人在其操作空间中的“足迹”和方向。总而言之，本文为 MoMa 任务提供了一个有原则的 TAMP 框架，该框架考虑了关于物体重新排列的常识，并且能够适应包括许多需要移动的物体的新情况。我们在真实环境和模拟环境中进行了定量实验。我们评估了完成长时程物体重排任务的成功率和效率。虽然机器人完成了 84.4% 的真实物体重排试验，但主观的人工评估表明，机器人的性能仍然低于经验丰富的服务员。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07727) | **Categories:** cs.RO

---

### [3] [Virtual Traffic Lights for Multi-Robot Navigation: Decentralized Planning with Centralized Conflict Resolution](https://arxiv.org/abs/2511.07811)
*Sagar Gupta, Thanh Vinh Nguyen, Thieu Long Phan, Vidul Attri, Archit Gupta, Niroshinie Fernando, Kevin Lee, Seng W. Loke, Ronny Kutadinata, Benjamin Champion, Akansel Cosgun*

Main category: cs.RO

TL;DR: 该论文提出了一种混合多机器人协调框架，结合分散路径规划和集中式冲突解决，提高了机器人到达目标的成功率并减少了死锁。


<details>
  <summary>Details</summary>
Motivation: 解决多机器人协调中的冲突和死锁问题，提高机器人到达目标的成功率。

Method: 提出一种混合多机器人协调框架，机器人自主规划路径并与中央节点共享，中央系统检测潜在冲突，并指示冲突机器人停止以避免死锁。

Result: 在多机器人仿真实验中，该方法提高了机器人到达目标的成功率，减少了死锁。在两足机器人和轮式Duckiebots的真实实验中验证了系统的有效性。

Conclusion: 该混合多机器人协调框架能够有效解决冲突和死锁问题，提高多机器人系统的效率和可靠性。

Abstract: 我们提出了一种混合多机器人协调框架，该框架结合了分散式路径规划和集中式冲突解决。在我们的方法中，每个机器人自主地规划其路径，并将此信息与中央节点共享。中央系统检测潜在的冲突，并且一次只允许一个冲突机器人继续前进，指示其他机器人停止在冲突区域之外以避免死锁。与传统的集中式规划方法不同，我们的系统不指示机器人路径，而是提供停止命令，其功能类似于虚拟交通灯。在多个机器人的仿真实验中，我们的方法提高了机器人到达目标的成功率，同时减少了死锁。此外，我们还在两个四足机器人以及轮式Duckiebots的真实实验中成功验证了该系统。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07811) | **Categories:** cs.RO

---

### [4] [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://arxiv.org/abs/2511.07820)
*Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando Castañeda, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Zi Wang, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi "Jim" Fan, Yuke Zhu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited behavior set, and are trained on a handful of GPUs over several days. We show that scaling up model capacity, data, and compute yields a generalist humanoid controller capable of creating natural and robust whole-body movements. Specifically, we posit motion tracking as a natural and scalable task for humanoid control, leverageing dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (from 1.2M to 42M parameters), dataset volume (over 100M frames, 700 hours of high-quality motion data), and compute (9k GPU hours). Beyond demonstrating the benefits of scale, we show the practical utility of our model through two mechanisms: (1) a real-time universal kinematic planner that bridges motion tracking to downstream task execution, enabling natural and interactive control, and (2) a unified token space that supports various motion input interfaces, such as VR teleoperation devices, human videos, and vision-language-action (VLA) models, all using the same policy. Scaling motion tracking exhibits favorable properties: performance improves steadily with increased compute and data diversity, and learned representations generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07820) | **Categories:** cs.RO, cs.AI, cs.CV, cs.GR, eess.SY

---

### [5] [Local Path Planning with Dynamic Obstacle Avoidance in Unstructured Environments](https://arxiv.org/abs/2511.07927)
*Okan Arif Guvenkaya, Selim Ahmet Iz, Mustafa Unel*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Obstacle avoidance and path planning are essential for guiding unmanned ground vehicles (UGVs) through environments that are densely populated with dynamic obstacles. This paper develops a novel approach that combines tangentbased path planning and extrapolation methods to create a new decision-making algorithm for local path planning. In the assumed scenario, a UGV has a prior knowledge of its initial and target points within the dynamic environment. A global path has already been computed, and the robot is provided with waypoints along this path. As the UGV travels between these waypoints, the algorithm aims to avoid collisions with dynamic obstacles. These obstacles follow polynomial trajectories, with their initial positions randomized in the local map and velocities randomized between O and the allowable physical velocity limit of the robot, along with some random accelerations. The developed algorithm is tested in several scenarios where many dynamic obstacles move randomly in the environment. Simulation results show the effectiveness of the proposed local path planning strategy by gradually generating a collision free path which allows the robot to navigate safely between initial and the target locations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.07927) | **Categories:** cs.RO, math.DS, math.OC

---

### [6] [Effective Game-Theoretic Motion Planning via Nested Search](https://arxiv.org/abs/2511.08001)
*Avishav Engle, Andrey Zhitnikov, Oren Salzman, Omer Ben-Porat, Kiril Solovey*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: To facilitate effective, safe deployment in the real world, individual robots must reason about interactions with other agents, which often occur without explicit communication. Recent work has identified game theory, particularly the concept of Nash Equilibrium (NE), as a key enabler for behavior-aware decision-making. Yet, existing work falls short of fully unleashing the power of game-theoretic reasoning. Specifically, popular optimization-based methods require simplified robot dynamics and tend to get trapped in local minima due to convexification. Other works that rely on payoff matrices suffer from poor scalability due to the explicit enumeration of all possible trajectories. To bridge this gap, we introduce Game-Theoretic Nested Search (GTNS), a novel, scalable, and provably correct approach for computing NEs in general dynamical systems. GTNS efficiently searches the action space of all agents involved, while discarding trajectories that violate the NE constraint (no unilateral deviation) through an inner search over a lower-dimensional space. Our algorithm enables explicit selection among equilibria by utilizing a user-specified global objective, thereby capturing a rich set of realistic interactions. We demonstrate the approach on a variety of autonomous driving and racing scenarios where we achieve solutions in mere seconds on commodity hardware.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.08001) | **Categories:** cs.RO, cs.MA

---

### [7] [PerspAct: Enhancing LLM Situated Collaboration Skills through Perspective Taking and Active Vision](https://arxiv.org/abs/2511.08098)
*Sabrina Patania, Luca Annese, Anita Pellegrini, Silvia Serino, Anna Lambiase, Luca Pallonetto, Silvia Rossi, Simone Colombani, Tom Foulsham, Azzurra Ruggeri, Dimitri Ognibene*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in Large Language Models (LLMs) and multimodal foundation models have significantly broadened their application in robotics and collaborative systems. However, effective multi-agent interaction necessitates robust perspective-taking capabilities, enabling models to interpret both physical and epistemic viewpoints. Current training paradigms often neglect these interactive contexts, resulting in challenges when models must reason about the subjectivity of individual perspectives or navigate environments with multiple observers. This study evaluates whether explicitly incorporating diverse points of view using the ReAct framework, an approach that integrates reasoning and acting, can enhance an LLM's ability to understand and ground the demands of other agents. We extend the classic Director task by introducing active visual exploration across a suite of seven scenarios of increasing perspective-taking complexity. These scenarios are designed to challenge the agent's capacity to resolve referential ambiguity based on visual access and interaction, under varying state representations and prompting strategies, including ReAct-style reasoning. Our results demonstrate that explicit perspective cues, combined with active exploration strategies, significantly improve the model's interpretative accuracy and collaborative effectiveness. These findings highlight the potential of integrating active perception with perspective-taking mechanisms in advancing LLMs' application in robotics and multi-agent systems, setting a foundation for future research into adaptive and context-aware AI systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.08098) | **Categories:** cs.RO, cs.AI, cs.CL, cs.HC

---

### [8] [Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](https://arxiv.org/abs/2511.08214)
*Yi Huang, Zhan Qu, Lihui Jiang, Bingbing Liu, Hongbo Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end autonomous driving systems, predominantly trained through imitation learning, have demonstrated considerable effectiveness in leveraging large-scale expert driving data. Despite their success in open-loop evaluations, these systems often exhibit significant performance degradation in closed-loop scenarios due to causal confusion. This confusion is fundamentally exacerbated by the overreliance of the imitation learning paradigm on expert trajectories, which often contain unattributable noise and interfere with the modeling of causal relationships between environmental contexts and appropriate driving actions.   To address this fundamental limitation, we propose Perception-Guided Self-Supervision (PGS) - a simple yet effective training paradigm that leverages perception outputs as the primary supervisory signals, explicitly modeling causal relationships in decision-making. The proposed framework aligns both the inputs and outputs of the decision-making module with perception results, such as lane centerlines and the predicted motions of surrounding agents, by introducing positive and negative self-supervision for the ego trajectory. This alignment is specifically designed to mitigate causal confusion arising from the inherent noise in expert trajectories.   Equipped with perception-driven supervision, our method, built on a standard end-to-end architecture, achieves a Driving Score of 78.08 and a mean success rate of 48.64% on the challenging closed-loop Bench2Drive benchmark, significantly outperforming existing state-of-the-art methods, including those employing more complex network architectures and inference pipelines. These results underscore the effectiveness and robustness of the proposed PGS framework and point to a promising direction for addressing causal confusion and enhancing real-world generalization in autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.08214) | **Categories:** cs.RO

---

### [9] [X-IONet: Cross-Platform Inertial Odometry Network with Dual-Stage Attention](https://arxiv.org/abs/2511.08277)
*Dehan Shen, Changhao Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning-based inertial odometry has achieved remarkable progress in pedestrian navigation. However, extending these methods to quadruped robots remains challenging due to their distinct and highly dynamic motion patterns. Models that perform well on pedestrian data often experience severe degradation when deployed on legged platforms. To tackle this challenge, we introduce X-IONet, a cross-platform inertial odometry framework that operates solely using a single Inertial Measurement Unit (IMU). X-IONet incorporates a rule-based expert selection module to classify motion platforms and route IMU sequences to platform-specific expert networks. The displacement prediction network features a dual-stage attention architecture that jointly models long-range temporal dependencies and inter-axis correlations, enabling accurate motion representation. It outputs both displacement and associated uncertainty, which are further fused through an Extended Kalman Filter (EKF) for robust state estimation. Extensive experiments on public pedestrian datasets and a self-collected quadruped robot dataset demonstrate that X-IONet achieves state-of-the-art performance, reducing Absolute Trajectory Error (ATE) by 14.3% and Relative Trajectory Error (RTE) by 11.4% on pedestrian data, and by 52.8% and 41.3% on quadruped robot data. These results highlight the effectiveness of X-IONet in advancing accurate and robust inertial navigation across both human and legged robot platforms.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.08277) | **Categories:** cs.RO, cs.LG

---
