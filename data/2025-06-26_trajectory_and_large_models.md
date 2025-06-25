# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-26

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (5)](#cs-cv)
- [机器学习 (Machine Learning) (1)](#cs-lg)
- [机器人学 (Robotics) (2)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Temporal-IRL: Modeling Port Congestion and Berth Scheduling with Inverse Reinforcement Learning](https://arxiv.org/abs/2506.19843)
*Guo Li, Zixiang Xu, Wei Zhang, Yikuan Hu, Xinyu Yang, Nikolay Aristov, Mingjie Tang, Elenna R Dugundji*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Predicting port congestion is crucial for maintaining reliable global supply chains. Accurate forecasts enableimprovedshipment planning, reducedelaysand costs, and optimizeinventoryanddistributionstrategies, thereby ensuring timely deliveries and enhancing supply chain resilience. To achieve accurate predictions, analyzing vessel behavior and their stay times at specific port terminals is essential, focusing particularly on berth scheduling under various conditions. Crucially, the model must capture and learn the underlying priorities and patterns of berth scheduling. Berth scheduling and planning are influenced by a range of factors, including incoming vessel size, waiting times, and the status of vessels within the port terminal. By observing historical Automatic Identification System (AIS) positions of vessels, we reconstruct berth schedules, which are subsequently utilized to determine the reward function via Inverse Reinforcement Learning (IRL). For this purpose, we modeled a specific terminal at the Port of New York/New Jersey and developed Temporal-IRL. This Temporal-IRL model learns berth scheduling to predict vessel sequencing at the terminal and estimate vessel port stay, encompassing both waiting and berthing times, to forecast port congestion. Utilizing data from Maher Terminal spanning January 2015 to September 2023, we trained and tested the model, achieving demonstrably excellent results.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19843) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Damba-ST: Domain-Adaptive Mamba for Efficient Urban Spatio-Temporal Prediction](https://arxiv.org/abs/2506.18939)
*Rui An, Yifeng Zhang, Ziran Liang, Wenqi Fan, Yuxuan Liang, Xuequn Shang, Qing Li*

Main category: cs.CV

TL;DR: Damba-ST是一种领域自适应的Mamba模型，用于高效城市时空预测，具有强大的泛化能力和效率。


<details>
  <summary>Details</summary>
Motivation: 现有的基于Transformer的模型面临二次计算复杂度和高内存开销的问题，限制了它们的可扩展性和实际部署。直接应用Mamba作为时空骨干会导致负迁移和严重的性能下降，这主要是由于时空异质性和Mamba隐藏状态更新的递归机制限制了跨域泛化。

Method: 提出了一种新颖的基于Mamba的领域自适应模型Damba-ST，用于高效的城市时空预测。它包含两个核心创新：(1) 一种领域自适应状态空间模型，将潜在表示空间划分为一个共享子空间（用于学习跨域共性）和独立的领域特定子空间（用于捕获域内判别特征）；(2) 三个不同的领域适配器，它们充当领域感知代理，以桥接不同的领域分布并促进跨领域共性的对齐。

Result: Damba-ST在预测任务上实现了最先进的性能，并展示了强大的零样本泛化能力。

Conclusion: Damba-ST在预测任务上实现了最先进的性能，并展示了强大的零样本泛化能力，无需大量重新训练或微调即可在新的城市环境中无缝部署。

Abstract: 训练能够很好地泛化到不同区域和城市的城市时空基础模型，对于在未见过的或数据稀缺的区域部署城市服务至关重要。最近的研究通常侧重于融合跨领域时空数据来训练统一的基于Transformer的模型。然而，这些模型面临二次计算复杂度和高内存开销的问题，限制了它们的可扩展性和实际部署。受到具有线性时间复杂度的状态空间模型Mamba的效率的启发，我们探索了其在高效城市时空预测中的潜力。然而，直接应用Mamba作为时空骨干会导致负迁移和严重的性能下降。这主要是由于时空异质性和Mamba隐藏状态更新的递归机制限制了跨域泛化。为了克服这些挑战，我们提出了一种新颖的基于Mamba的领域自适应模型Damba-ST，用于高效的城市时空预测。Damba-ST保留了Mamba的线性复杂度优势，同时显着增强了其对异构领域的适应性。具体来说，我们引入了两个核心创新：(1) 一种领域自适应状态空间模型，它将潜在表示空间划分为一个共享子空间（用于学习跨域共性）和独立的领域特定子空间（用于捕获域内判别特征）；(2) 三个不同的领域适配器，它们充当领域感知代理，以桥接不同的领域分布并促进跨领域共性的对齐。 广泛的实验证明了Damba-ST的泛化性和效率。它在预测任务上实现了最先进的性能，并展示了强大的零样本泛化能力，无需大量重新训练或微调即可在新的城市环境中无缝部署。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.18939) | **Categories:** cs.CV, cs.AI

---

### [2] [Trajectory Prediction in Dynamic Object Tracking: A Critical Study](https://arxiv.org/abs/2506.19341)
*Zhongping Dong, Liming Chen, Mohand Tahar Kechadi*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This study provides a detailed analysis of current advancements in dynamic object tracking (DOT) and trajectory prediction (TP) methodologies, including their applications and challenges. It covers various approaches, such as feature-based, segmentation-based, estimation-based, and learning-based methods, evaluating their effectiveness, deployment, and limitations in real-world scenarios. The study highlights the significant impact of these technologies in automotive and autonomous vehicles, surveillance and security, healthcare, and industrial automation, contributing to safety and efficiency. Despite the progress, challenges such as improved generalization, computational efficiency, reduced data dependency, and ethical considerations still exist. The study suggests future research directions to address these challenges, emphasizing the importance of multimodal data integration, semantic information fusion, and developing context-aware systems, along with ethical and privacy-preserving frameworks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19341) | **Categories:** cs.CV

---

### [3] [Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding](https://arxiv.org/abs/2506.19288)
*Runwei Guan, Ningwei Ouyang, Tianhao Xu, Shaofeng Liang, Wei Dai, Yafeng Sun, Shang Gao, Songning Lai, Shanliang Yao, Xuming Hu, Ryan Wen Liu, Yutao Yue, Hui Xiong*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Automated waterway environment perception is crucial for enabling unmanned surface vessels (USVs) to understand their surroundings and make informed decisions. Most existing waterway perception models primarily focus on instance-level object perception paradigms (e.g., detection, segmentation). However, due to the complexity of waterway environments, current perception datasets and models fail to achieve global semantic understanding of waterways, limiting large-scale monitoring and structured log generation. With the advancement of vision-language models (VLMs), we leverage image captioning to introduce WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size. Additionally, we propose Da Yu, an edge-deployable multi-modal large language model for USVs, where we propose a novel vision-to-language projector called Nano Transformer Adaptor (NTA). NTA effectively balances computational efficiency with the capacity for both global and fine-grained local modeling of visual features, thereby significantly enhancing the model's ability to generate long-form textual outputs. Da Yu achieves an optimal balance between performance and efficiency, surpassing state-of-the-art models on WaterCaption and several other captioning benchmarks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19288) | **Categories:** cs.CV, cs.RO

---

### [4] [Unified Vision-Language-Action Model](https://arxiv.org/abs/2506.19850)
*Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, Zhaoxiang Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning--especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing pi0-FAST's 85.5%. We further demonstrate its broad applicability on real-world ALOHA manipulation and autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19850) | **Categories:** cs.CV, cs.RO

---

### [5] [Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System](https://arxiv.org/abs/2506.19433)
*Lixuan He, Haoyu Dong, Zhenxing Chen, Yangcheng Yu, Jie Feng, Yong Li*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via https://github.com/tsinghua-fib-lab/Mem4Nav.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19433) | **Categories:** cs.CV, cs.AI, cs.CL

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Path Learning with Trajectory Advantage Regression](https://arxiv.org/abs/2506.19375)
*Kohei Miyaguchi*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this paper, we propose trajectory advantage regression, a method of offline path learning and path attribution based on reinforcement learning. The proposed method can be used to solve path optimization problems while algorithmically only solving a regression problem.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19375) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Scaffolding Dexterous Manipulation with Vision-Language Models](https://arxiv.org/abs/2506.19212)
*Vincent de Bakker, Joey Hejna, Tyler Ga Wei Lum, Onur Celik, Aleksandar Taranovic, Denis Blessing, Gerhard Neumann, Jeannette Bohg, Dorsa Sadigh*

Main category: cs.RO

TL;DR: 该论文提出了一种利用视觉语言模型生成轨迹，并用强化学习训练机器人灵巧操作的方法，无需人工演示或手动设计的奖励函数。


<details>
  <summary>Details</summary>
Motivation: 灵巧的机器人手对于执行复杂的操纵任务至关重要，但由于演示收集和高维控制的挑战，训练仍然很困难。强化学习虽然可以通过在模拟中生成经验来缓解数据瓶颈，但通常依赖于精心设计的、特定于任务的奖励函数，这阻碍了可扩展性和泛化。

Method: 该方法利用视觉语言模型（VLM）识别任务相关的关键点，并合成手部运动和物体运动的3D轨迹，然后在模拟环境中训练一个低级别的残差强化学习策略来高精度地跟踪这些粗略的轨迹。

Result: 在多个涉及铰接物体和语义理解的模拟任务中，该方法能够学习到鲁棒的灵巧操作策略。此外，该方法可以转移到真实世界的机器人手上，而无需任何人工演示或手动设计的奖励。

Conclusion: 该方法能够在模拟环境中学习到鲁棒的灵巧操作策略，并且无需人工演示或手动设计的奖励函数即可转移到真实世界的机器人手上。

Abstract: 灵巧的机器人手对于执行复杂的操纵任务至关重要，但训练它们仍然很困难。强化学习可以通过在模拟中生成经验来缓解数据瓶颈，但通常依赖于精心设计的、特定于任务的奖励函数，这阻碍了可扩展性和泛化。因此，目前灵巧操作的工作通常从参考轨迹引导。这些轨迹指定了目标手部姿势，以指导强化学习策略的探索，以及使密集、任务无关的奖励成为可能的物体姿势。然而，寻找合适的轨迹仍然是一个重大挑战，特别是对于灵巧的手。然而，显式参考轨迹中的精确细节通常是不必要的，因为强化学习最终会改进运动。我们的关键见解是，现代视觉语言模型（VLM）已经编码了指定任务和有效指导探索所需的常识空间和语义知识。给定一个任务描述（例如，“打开柜子”）和一个视觉场景，我们的方法使用现成的VLM首先识别任务相关的关键点（例如，把手、按钮），然后合成手部运动和物体运动的3D轨迹。随后，我们在模拟中训练一个低级别的残差强化学习策略，以高精度地跟踪这些粗略的轨迹或“支架”。在多个涉及铰接物体和语义理解的模拟任务中，我们证明了我们的方法能够学习到鲁棒的灵巧操作策略。此外，我们展示了我们的方法可以转移到真实世界的机器人手上，而无需任何人工演示或手动设计的奖励。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19212) | **Categories:** cs.RO

---

### [2] [CronusVLA: Transferring Latent Motion Across Time for Multi-Frame Prediction in Manipulation](https://arxiv.org/abs/2506.19816)
*Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, Jiangmiao Pang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent vision-language-action (VLA) models built on pretrained vision-language models (VLMs) have demonstrated strong generalization across manipulation tasks. However, they remain constrained by a single-frame observation paradigm and cannot fully benefit from the motion information offered by aggregated multi-frame historical observations, as the large vision-language backbone introduces substantial computational cost and inference latency. We propose CronusVLA, a unified framework that extends single-frame VLA models to the multi-frame paradigm through an efficient post-training stage. CronusVLA comprises three key components: (1) single-frame pretraining on large-scale embodied datasets with autoregressive action tokens prediction, which establishes an embodied vision-language foundation; (2) multi-frame encoding, adapting the prediction of vision-language backbones from discrete action tokens to motion features during post-training, and aggregating motion features from historical frames into a feature chunking; (3) cross-frame decoding, which maps the feature chunking to accurate actions via a shared decoder with cross-attention. By reducing redundant token computation and caching past motion features, CronusVLA achieves efficient inference. As an application of motion features, we further propose an action adaptation mechanism based on feature-action retrieval to improve model performance during finetuning. CronusVLA achieves state-of-the-art performance on SimplerEnv with 70.9% success rate, and 12.7% improvement over OpenVLA on LIBERO. Real-world Franka experiments also show the strong performance and robustness.

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.19816) | **Categories:** cs.RO, cs.CV

---
