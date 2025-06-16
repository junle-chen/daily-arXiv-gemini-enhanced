# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-17

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [cs.AR (1)](#cs-ar)
- [计算语言学 (Computation and Language) (2)](#cs-cl)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [cs.MA (1)](#cs-ma)
- [机器人学 (Robotics) (4)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [FocalAD: Local Motion Planning for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.11419)
*Bin Sun, Boao Zhang, Jiayi Lu, Xinjie Feng, Jiachen Shang, Rui Cao, Mengchao Zheng, Chuanye Wang, Shichun Yang, Yaoguang Cao, Ziying Song*

Main category: cs.AI

TL;DR: FocalAD通过关注局部关键邻居并优化局部运动表征，提升了端到端自动驾驶的性能，尤其在鲁棒性方面有显著提升。


<details>
  <summary>Details</summary>
Motivation: 现有方法依赖全局聚合运动特征，忽略了规划决策主要受少量局部交互代理的影响。

Method: 提出Ego-Local-Agents Interactor (ELAI)和Focal-Local-Agents Loss (FLA Loss)。ELAI进行基于图的自我中心交互表示，FLA Loss增加决策关键邻居代理的权重。

Result: 在Adv-nuScenes数据集上，平均碰撞率比DiffusionDrive降低41.9%，比SparseDrive降低15.6%。

Conclusion: FocalAD在nuScenes和Bench2Drive数据集上优于现有技术，尤其在Adv-nuScenes数据集上，碰撞率显著降低。

Abstract: 在端到端自动驾驶中，运动预测在自我车辆规划中起着关键作用。然而，现有方法通常依赖于全局聚合的运动特征，忽略了规划决策主要受到少量局部交互代理的影响。未能关注这些关键的局部交互可能会掩盖潜在风险并削弱规划的可靠性。在这项工作中，我们提出了FocalAD，这是一种新颖的端到端自动驾驶框架，它专注于关键的本地邻居，并通过增强本地运动表示来改进规划。具体来说，FocalAD包括两个核心模块：Ego-Local-Agents Interactor (ELAI)和Focal-Local-Agents Loss (FLA Loss)。ELAI进行基于图的自我中心交互表示，该表示捕获与本地邻居的运动动态，以增强自我规划和代理运动查询。FLA Loss增加了决策关键邻近代理的权重，引导模型优先考虑那些与规划更相关的代理。大量实验表明，FocalAD在开放循环nuScenes数据集和闭环Bench2Drive基准测试中优于现有的最先进方法。值得注意的是，在以鲁棒性为重点的Adv-nuScenes数据集上，FocalAD取得了更大的改进，与DiffusionDrive相比，平均碰撞率降低了41.9%，与SparseDrive相比降低了15.6%。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11419) | **Categories:** cs.AI, cs.RO

---


## cs.AR [cs.AR]
### [1] [Real-World Deployment of a Lane Change Prediction Architecture Based on Knowledge Graph Embeddings and Bayesian Inference](https://arxiv.org/abs/2506.11925)
*M. Manzour, Catherine M. Elias, Omar M. Shehata, R. Izquierdo, M. A. Sotelo*

Main category: cs.AR

TL;DR: 本文提出了一种基于知识图嵌入和贝叶斯推理的车道变换预测系统，并在真实硬件上进行了验证。


<details>
  <summary>Details</summary>
Motivation: 弥合算法进步与实际道路部署之间的差距。

Method: 该方法基于知识图嵌入（KGEs）和贝叶斯推断。

Result: 实 world 硬件实验验证表明，该预测系统能够提前三到四秒预测目标车辆的变道行为。

Conclusion: 该预测系统能够提前三到四秒预测目标车辆的变道行为，为自我车辆提供足够的反应时间，并允许目标车辆安全地进行变道。

Abstract: 近年来，车道变换预测的研究获得了很大的发展。然而，大多数研究仅限于仿真或从数据集中获得的结果，这使得算法进步与实际道路部署之间存在差距。这项工作通过在真实硬件上展示一个基于知识图嵌入（KGEs）和贝叶斯推理的车道变换预测系统来弥合这一差距。此外，自我车辆采用纵向制动动作，以确保自身和周围车辆的安全。我们的架构包括两个模块：（i）感知模块，用于感知环境，导出输入数值特征，并将其转换为语言类别；并将其传递给预测模块；（ii）预训练的预测模块，该模块执行 KGE 和贝叶斯推理模型以预测目标车辆的动作，并将预测转换为纵向制动动作。真实世界的硬件实验验证表明，我们的预测系统能够提前三到四秒预测目标车辆的变道行为，为自我车辆提供足够的反应时间，并允许目标车辆安全地进行变道。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11925) | **Categories:** cs.AR, cs.AI, cs.CV, cs.LG

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization](https://arxiv.org/abs/2506.11109)
*Yile Chen, Yicheng Tao, Yue Jiang, Shuai Liu, Han Yu, Gao Cong*

Main category: cs.CL

TL;DR: QT-Mob通过学习语义丰富的位置tokens和优化微调目标，显著提升了LLM在移动分析任务中的性能。


<details>
  <summary>Details</summary>
Motivation: 现有方法面临两个主要限制：位置的语义表示不足（即离散ID）以及LLM内移动信号的建模不足（即单个模板指令微调）。

Method: 提出了一种新的框架QT-Mob，通过位置标记化模块学习紧凑且语义丰富的tokens来表示位置，并结合了一系列互补的微调目标，以对齐学习到的tokens与LLM中的内部表示。

Result: 在三个真实世界数据集上的实验表明，在下一个位置预测和移动恢复任务中，QT-Mob 均优于现有的深度学习和基于LLM的方法。

Conclusion: 提出的QT-Mob框架提升了LLM解释移动数据的能力，并为各种移动分析任务提供了一种更通用的方法。

Abstract: 位置服务的广泛采用产生了大量的移动数据，为在城市环境中建模用户移动动态提供了重要的机会。最近的进展集中于调整大型语言模型（LLM）用于移动分析。然而，现有方法面临两个主要限制：位置的语义表示不足（即离散ID）以及LLM内移动信号的建模不足（即单个模板指令微调）。为了解决这些问题，我们提出了一种新的框架QT-Mob，该框架显著增强了LLM用于移动分析的能力。QT-Mob 引入了一个位置标记化模块，该模块学习紧凑、语义丰富的tokens来表示位置，在保留上下文信息的同时确保与 LLM 的兼容性。此外，QT-Mob 结合了一系列互补的微调目标，这些目标将学习到的tokens与 LLM 中的内部表示对齐，从而提高了模型对顺序移动模式和位置语义的理解。所提出的 QT-Mob 框架不仅增强了 LLM 解释移动数据的能力，而且为各种移动分析任务提供了一种更通用的方法。在三个真实世界数据集上的实验表明，在下一个位置预测和移动恢复任务中，QT-Mob 均优于现有的深度学习和基于LLM的方法。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11109) | **Categories:** cs.CL, cs.AI

---

### [2] [Don't Pay Attention](https://arxiv.org/abs/2506.11305)
*Mohammad Hammoud, Devang Acharya*

Main category: cs.CL

TL;DR: Avey是一种新的神经架构，它打破了注意力和递归，能够有效处理任意长度的序列，并在长程依赖捕获方面表现出色。


<details>
  <summary>Details</summary>
Motivation: Transformer在处理超出固定上下文窗口的序列时面临挑战，并且其注意力机制具有二次复杂度。这些挑战重新激发了人们对RNN类架构的兴趣，尽管RNN类架构由于其固有的递归性质而具有有限的并行性。

Method: Avey包含一个排序器和一个自回归神经处理器，它们协同识别和上下文化任何给定token的最相关token，而不管它们在序列中的位置。

Result: 实验结果表明，Avey在各种标准短程NLP基准测试中与Transformer相比具有优势，同时在捕获长程依赖方面表现出色。

Conclusion: Avey在处理长程依赖方面表现出色，并在各种标准短程NLP基准测试中与Transformer相比具有优势。

Abstract: Transformer已成为大型语言模型的实际标准，并在各个领域的各种下游任务中得到广泛应用。尽管Transformer具有许多优点，如固有的训练并行性，但由于其无法有效处理超出固定上下文窗口的序列以及注意力机制的二次复杂度，Transformer仍然面临着关键挑战。这些挑战重新激发了人们对RNN类架构的兴趣，RNN类架构提供随序列长度的线性缩放和改进的长程依赖处理，但由于其固有的递归性质，并行性有限。在本文中，我们提出了一种新的神经基础架构Avey，它打破了注意力和递归。Avey包含一个排序器和一个自回归神经处理器，它们协同识别和上下文化任何给定token的最相关token，而不管它们在序列中的位置。具体来说，Avey将序列长度与上下文宽度分离，从而能够有效处理任意长度的序列。实验结果表明，Avey在各种标准短程NLP基准测试中与Transformer相比具有优势，同时在捕获长程依赖方面表现出色。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11305) | **Categories:** cs.CL, cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Technical Report for Argoverse2 Scenario Mining Challenges on Iterative Error Correction and Spatially-Aware Prompting](https://arxiv.org/abs/2506.11124)
*Yifei Chen, Ross Greer*

Main category: cs.CV

TL;DR: 该论文提出了一种改进的场景挖掘框架，通过容错迭代代码生成和专门的提示工程，提高了大型语言模型在自动驾驶数据集上挖掘场景的可靠性和精度。


<details>
  <summary>Details</summary>
Motivation: 从Argoverse 2等广泛的自动驾驶数据集中挖掘场景，对于开发和验证自动驾驶系统至关重要。RefAV框架通过使用大型语言模型（LLM）将自然语言查询转换为可执行代码来识别相关场景，代表了一种有前景的方法。然而，这种方法面临着挑战，包括LLM生成的代码产生的运行时错误，以及在解释描述复杂多对象空间关系的函数的参数时的不准确性。

Method: 提出了一种容错的迭代代码生成机制，通过用错误反馈重新提示LLM来改进代码；以及专门的提示工程，以提高LLM对空间关系函数的理解和正确应用。

Result: 在Argoverse 2验证集上，使用Qwen2.5-VL-7B、Gemini 2.5 Flash和Gemini 2.5 Pro等多种LLM进行的实验表明，该方法在多个指标上均有持续提升；最值得注意的是，所提出的系统在使用Gemini 2.5 Pro的官方测试集上，HOTA-Temporal得分达到了52.37。

Conclusion: 在Argoverse 2验证集上的实验表明，该方法在多个指标上均有提升；在使用Gemini 2.5 Pro的官方测试集上，HOTA-Temporal得分达到了52.37。结果表明，该技术对于可靠、高精度的场景挖掘是有效的。

Abstract: 从Argoverse 2等大规模自动驾驶数据集中挖掘场景对于自动驾驶系统的开发和验证至关重要。RefAV框架利用大型语言模型（LLM）将自然语言查询转化为可执行代码，从而识别相关场景，但面临LLM生成代码的运行时错误以及复杂空间关系函数参数解释不准确等问题。本报告提出了两项关键改进：一是容错迭代代码生成机制，通过错误反馈重新提示LLM来优化代码；二是专门设计的提示工程，提高LLM对空间关系函数的理解和应用。在Argoverse 2验证集上，使用Qwen2.5-VL-7B、Gemini 2.5 Flash和Gemini 2.5 Pro等多种LLM进行的实验表明，该方法在多个指标上均有提升，尤其是在使用Gemini 2.5 Pro的官方测试集上，HOTA-Temporal得分达到了52.37。这些结果表明，该技术对于可靠、高精度的场景挖掘是有效的。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11124) | **Categories:** cs.CV, cs.SE

---

### [2] [TARDIS STRIDE: A Spatio-Temporal Road Image Dataset for Exploration and Autonomy](https://arxiv.org/abs/2506.11302)
*Héctor Carrión, Yutong Bai, Víctor A. Hernández Castro, Kishan Panaganti, Ayush Zenith, Matthew Trang, Tony Zhang, Pietro Perona, Jitendra Malik*

Main category: cs.CV

TL;DR: 该论文提出了一个时空道路图像数据集（STRIDE）和一个基于Transformer的世界模型（TARDIS），用于模拟现实世界环境中的空间和时间动态。


<details>
  <summary>Details</summary>
Motivation: Modeling real-world environments presents unique challenges as they dynamically change across both space and time.

Method: The authors benchmark the STRIDE dataset via TARDIS, a transformer-based generative world model that integrates spatial and temporal dynamics through a unified autoregressive framework trained on STRIDE.

Result: The TARDIS model demonstrates robust performance across a range of agentic tasks such as controllable photorealistic image synthesis, instruction following, autonomous self-control, and state-of-the-art georeferencing.

Conclusion: The paper demonstrates robust performance across a range of agentic tasks such as controllable photorealistic image synthesis, instruction following, autonomous self-control, and state-of-the-art georeferencing, suggesting a promising direction towards sophisticated generalist agents capable of understanding and manipulating the spatial and temporal aspects of their material environments with enhanced embodied reasoning capabilities.

Abstract: 世界模型旨在模拟环境并实现有效的智能体行为。然而，对现实世界环境进行建模提出了独特的挑战，因为它们在空间和时间上都会动态变化。为了捕捉这些组合的动态，我们引入了一个时空道路图像数据集（STRIDE），将 360 度全景图像排列成丰富的互连观察、状态和动作节点。利用这种结构，我们可以同时模拟以自我为中心的视图、位置坐标和跨空间和时间的移动命令之间的关系。我们通过 TARDIS 对该数据集进行基准测试，TARDIS 是一种基于 Transformer 的生成世界模型，它通过在 STRIDE 上训练的统一自回归框架集成空间和时间动态。我们展示了在各种智能体任务中的强大性能，例如可控的逼真图像合成、指令跟随、自主自控和最先进的地理参考。这些结果表明，朝着复杂的通用智能体发展前景广阔——能够理解和操纵其物质环境的空间和时间方面——并具有增强的具身推理能力。训练代码、数据集和模型检查点可在 https://huggingface.co/datasets/Tera-AI/STRIDE 获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11302) | **Categories:** cs.CV, cs.AI

---

### [3] [DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs](https://arxiv.org/abs/2506.11558)
*Bo-Cheng Chiu, Jen-Jee Chen, Yu-Chee Tseng, Feng-Chi Chen*

Main category: cs.CV

TL;DR: DaMO: 一种数据高效的视频语言模型，通过时间感知的Fuseformer和渐进式训练，在时间推理和多模态理解方面表现出色。


<details>
  <summary>Details</summary>
Motivation: 现有的视频大型语言模型在细粒度的时间推理方面存在局限性，限制了它们将响应精确地归因于特定视频时刻的能力，尤其是在有限的监督下。因此，本研究旨在设计一种更精确的时间推理和多模态理解的视频LLM。

Method: 该论文提出了一种数据高效的视频语言模型DaMO，它采用了一种时间感知的Fuseformer，该Fuseformer采用分层双流架构，逐步捕获每个模态中的时间动态，并有效地融合互补的视觉和音频信息。为了进一步提高计算效率，DaMO集成了一个全局残差，减少了空间冗余，同时保留了必要的语义细节。

Result: 在时间定位和视频问答基准测试中的综合实验表明，DaMO始终优于现有方法，尤其是在需要精确的时间对齐和推理的任务中。

Conclusion: 该研究表明，DaMO在时间和视频问答基准测试中始终优于现有方法，尤其是在需要精确的时间对齐和推理的任务中，为数据高效的视频语言建模提供了一个有希望的方向。

Abstract: 大型语言模型（LLM）最近已扩展到视频领域，从而实现了复杂的视频语言理解。然而，现有的视频LLM通常在细粒度的时间推理方面表现出局限性，限制了它们将响应精确地归因于特定视频时刻的能力，尤其是在有限的监督下。我们介绍DaMO，这是一种数据高效的视频LLM，专门为准确的时间推理和多模态理解而设计。所提出的时间感知Fuseformer的核心采用分层双流架构，该架构逐步捕获每个模态中的时间动态，并有效地融合互补的视觉和音频信息。为了进一步提高计算效率，DaMO集成了一个全局残差，减少了空间冗余，同时保留了必要的语义细节。我们通过结构化的四阶段渐进式训练范例来训练DaMO，逐步使模型具备多模态对齐、语义基础和时间推理能力。这项工作还贡献了多个数据集，这些数据集从现有的数据集中扩充而来，其中包含GPT生成的、时间上接地的QA对，用于需要时间监督的任务。在时间定位和视频问答基准测试中的综合实验表明，DaMO始终优于现有方法，尤其是在需要精确的时间对齐和推理的任务中。我们的工作为数据高效的视频语言建模建立了一个有希望的方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11558) | **Categories:** cs.CV, cs.AI, cs.CL

---

### [4] [AgentSense: Virtual Sensor Data Generation Using LLM Agent in Simulated Home Environments](https://arxiv.org/abs/2506.11773)
*Zikang Leng, Megha Thukral, Yaqi Liu, Hrudhai Rajasekhar, Shruthi K. Hiremath, Thomas Plötz*

Main category: cs.CV

TL;DR: AgentSense通过生成虚拟传感器数据，有效提升了人体活动识别系统的性能，尤其是在真实数据匮乏时。


<details>
  <summary>Details</summary>
Motivation: 缺乏大规模、多样化的标注数据集是开发稳健且泛化的智能家居人体活动识别（HAR）系统的主要障碍，家庭布局、传感器配置和用户行为的差异增加了复杂性。

Method: 提出AgentSense虚拟数据生成流程，利用大型语言模型生成多样化的人物角色，创建日常行为，分解为低级动作序列，并在扩展的VirtualHome模拟家庭环境中执行，生成虚拟传感器数据。

Result: 在五个基准HAR数据集上，利用虚拟传感器数据显著提高了性能，尤其是在真实数据有限的情况下。

Conclusion: 虚拟数据与少量真实数据结合，可使模型性能与完全使用真实数据训练的模型相媲美，证明了虚拟数据在解决环境感知中大规模标注数据集缺失问题上的潜力。

Abstract: 开发稳健且泛化的智能家居人体活动识别（HAR）系统的一个主要障碍是缺乏大规模、多样化的标注数据集。家庭布局、传感器配置和用户行为的差异增加了复杂性，因为个体遵循不同的日常习惯并以不同的方式执行活动。构建能够很好地泛化的HAR系统需要捕获用户和环境多样性的训练数据。为了应对这些挑战，我们介绍了AgentSense，这是一个虚拟数据生成流程，通过利用大型语言模型生成多样化的人物角色。这些人物角色用于创建日常行为，然后分解为低级动作序列。随后，这些动作在模拟的家庭环境中执行，我们扩展了VirtualHome，使其配备了虚拟环境传感器，能够记录智能体展开活动时的状态。总的来说，AgentSense能够生成丰富的虚拟传感器数据集，代表了广泛的用户和家庭环境。在五个基准HAR数据集上，我们表明，利用我们的虚拟传感器数据可以显著提高性能，尤其是在真实数据有限的情况下。值得注意的是，在虚拟数据和仅仅几天的真实数据的组合上训练的模型，其性能可与在整个真实数据集上训练的模型相媲美。这些结果证明并证明了虚拟数据在解决环境感知中最紧迫的挑战之一方面的潜力，即在不需要任何手动数据收集工作的情况下，明显缺乏大规模的带注释数据集。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11773) | **Categories:** cs.CV, cs.HC

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613)
*Edward Turner, Anna Soligo, Mia Taylor, Senthooran Rajamanoharan, Neel Nanda*

Main category: cs.LG

TL;DR: 该研究通过提炼模型，分离出对齐损害的最小变化，为未来理解和减轻大型语言模型中的对齐风险奠定基础。


<details>
  <summary>Details</summary>
Motivation: 最近的工作发现了涌现性未对齐（EM）：在狭义有害数据集上对大型语言模型进行微调可能导致它们变得广泛未对齐。发布前对专家的调查显示，这是非常出乎意料的，表明我们对模型对齐的理解存在严重差距。

Method: 使用新的狭义未对齐数据集，创建了一组改进的模型，实现了99%的连贯性（之前为67%），使用较小的0.5B参数模型（之前为32B），并使用单个rank-1 LoRA适配器诱导未对齐。

Result: 证明了EM在不同的模型大小、三个模型系列和包括完全监督微调在内的众多训练协议中稳健地发生。利用这些更清晰的模型，分离出一个机械相变，并证明它对应于所有研究生物中的稳健行为相变。

Conclusion: 通过提炼干净的模型，分离出最小的对齐损害变化，并了解它是如何学习的，从而为未来研究理解和减轻LLM中的对齐风险奠定了基础。

Abstract: 最近的研究发现了涌现性未对齐（EM）：在狭义有害的数据集上对大型语言模型进行微调可能导致它们变得广泛未对齐。发布前对专家的调查显示，这是非常出乎意料的，表明我们对模型对齐的理解存在严重差距。在这项工作中，我们既提高了理解，又为未来的研究提供了工具。使用新的狭义未对齐数据集，我们创建了一组改进的模型，实现了99%的连贯性（之前为67%），使用较小的0.5B参数模型（之前为32B），并使用单个rank-1 LoRA适配器诱导未对齐。我们证明了EM在不同的模型大小、三个模型系列和包括完全监督微调在内的众多训练协议中稳健地发生。利用这些更清晰的模型，我们分离出一个机械相变，并证明它对应于所有研究生物中的稳健行为相变。对齐大型语言模型对于前沿人工智能安全至关重要，但EM揭示了我们距离稳健地实现这一目标还有多远。通过提炼干净的模型，分离出最小的对齐损害变化，并了解它是如何学习的，我们为未来研究理解和减轻LLM中的对齐风险奠定了基础。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11613) | **Categories:** cs.LG, cs.AI

---

### [2] [Large Language models for Time Series Analysis: Techniques, Applications, and Challenges](https://arxiv.org/abs/2506.11040)
*Feifei Shi, Xueyan Yin, Kang Wang, Wanyu Tu, Qifu Sun, Huansheng Ning*

Main category: cs.LG

TL;DR: 该综述概述了预训练LLM在时间序列分析中的应用，并探讨了未来的发展方向。


<details>
  <summary>Details</summary>
Motivation: 传统时间序列分析方法在非线性特征表示和长期依赖捕获方面存在局限性；通用时间序列LLM的开发受到数据多样性、注释稀缺性和计算需求的阻碍。

Method: 该综述从工作流程的角度组织和系统化了LLM驱动的时间序列分析技术，涵盖了LLM的输入、优化和轻量级阶段。

Result: 该综述建立了一个AI驱动的时间序列分析的演进路线图，并 критически 考察了新的实际应用，强调了可以指导未来研究和创新的关键开放挑战。

Conclusion: 该综述总结了当前基于预训练LLM的时间序列分析的进展，并为未来发展方向提供了指导。

Abstract: 时间序列分析在金融预测和生物医学监控等领域至关重要，但传统方法受到有限的非线性特征表示和长期依赖捕获的限制。大型语言模型（LLM）的出现通过利用其跨模态知识整合和固有的注意力机制进行时间序列分析，提供了变革的潜力。然而，从头开始开发用于时间序列的通用LLM仍然受到数据多样性、注释稀缺性和计算需求的阻碍。本文对预训练的LLM驱动的时间序列分析进行了系统综述，重点关注使能技术、潜在应用和开放挑战。首先，它建立了一个AI驱动的时间序列分析的演进路线图，从早期的机器学习时代，到新兴的LLM驱动的范式，再到原生时间基础模型的开发。其次，它从工作流程的角度组织和系统化了LLM驱动的时间序列分析的技术格局，涵盖了LLM的输入、优化和轻量级阶段。最后，它 критически 考察了新的实际应用，并强调了可以指导未来研究和创新的关键开放挑战。这项工作不仅为当前的进展提供了有价值的见解，而且概述了未来发展的有希望的方向。它为学术和工业研究人员提供了一个基础参考，为开发更高效、更通用和可解释的LLM驱动的时间序列分析系统铺平了道路。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11040) | **Categories:** cs.LG, cs.CL, cs.ET

---

### [3] [Prioritizing Alignment Paradigms over Task-Specific Model Customization in Time-Series LLMs](https://arxiv.org/abs/2506.11512)
*Wei Li, Yunyao Cheng, Xinli Hao, Chaohong Ma, Yuxuan Liang, Bin Yang, Christian S. Jensen, Xiaofeng Meng*

Main category: cs.LG

TL;DR: 本文提出了一种基于时间序列数据内在原语对齐范式的时间序列推理新方法，以提高经济性、灵活性和效率。


<details>
  <summary>Details</summary>
Motivation: 现有方法通常侧重于特定任务的模型定制，而忽略了时间序列数据本身，这对于深入推理至关重要，导致成本高昂、不灵活和效率低下。

Method: 该文提出了三种对齐范式：内射对齐、桥接对齐和内部对齐，分别侧重于时间序列原语的不同方面：领域、特征和表示。

Result: 该文通过优先考虑时间序列数据的内在原语，解决了当前时间序列推理方法的局限性。

Conclusion: 通过对时间序列数据内在结构的系统性考虑，该文提出了一种更经济、灵活和高效的时间序列推理方法。

Abstract: 大型语言模型（LLM）的最新进展使得在医疗、金融和时空等各种实际应用中进行时间序列推理成为可能。然而，现有方法通常侧重于特定任务的模型定制，例如预测和异常检测，而忽略了数据本身，即时间序列原语，这对于深入推理至关重要。本文提倡一种根本性的转变，即在使用LLM进行时间序列推理时，优先考虑基于时间序列数据内在原语的对齐范式，而不是特定任务的模型定制。这种调整通过在任务工程之前系统地考虑数据的内在结构，解决了当前时间序列推理方法的局限性，这些方法通常成本高昂、不灵活和效率低下。为此，我们提出了三种对齐范式：内射对齐、桥接对齐和内部对齐，分别侧重于时间序列原语的不同方面：领域、特征和表示，以激活LLM的时间序列推理能力，从而实现经济、灵活和高效的推理。我们进一步建议从业者采用面向对齐的方法来利用该指令选择适当的对齐范式。此外，我们将相关文献分为这些对齐范式，并概述了有希望的研究方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11512) | **Categories:** cs.LG, cs.AI

---

### [4] [Delayformer: spatiotemporal transformation for predicting high-dimensional dynamics](https://arxiv.org/abs/2506.11528)
*Zijian Wang, Peng Tao, Luonan Chen*

Main category: cs.LG

TL;DR: Delayformer通过新颖的多元时空信息转换和Transformer架构，实现了优于现有方法的时序预测性能。


<details>
  <summary>Details</summary>
Motivation: 在有限和噪声数据的情况下，由于高维系统中变量的非线性和复杂交互，准确预测所有变量的动态是一个具有挑战性的任务。当前包括深度学习方法在内的方法在这种情况下通常表现不佳。

Method: 该研究提出了一种新颖的多元时空信息（mvSTI）转换，将每个观测变量转换为延迟嵌入状态（向量），并进一步交叉学习来自不同变量的这些状态。然后，它利用一个共享的视觉Transformer（ViT）编码器以延迟嵌入的形式交叉表示来自观测变量的动态状态，并采用不同的线性解码器来预测下一个状态，即并行预测所有原始变量。

Result: Delayformer在合成和真实世界数据集上的预测任务中优于当前最先进的方法，并通过跨领域预测任务展示了其作为基础时间序列模型的潜力。

Conclusion: Delayformer在合成和真实世界数据集上的预测任务中优于当前最先进的方法，并通过跨领域预测任务展示了其作为基础时间序列模型的潜力。

Abstract: 在各种科学和工程领域中，预测时间序列非常重要。然而，在有限和噪声数据的情况下，由于高维系统中变量的非线性和复杂交互，准确预测所有变量的动态是一项具有挑战性的任务。当前的方法，包括深度学习方法，在这种情况下通常表现不佳。本研究介绍了一种Delayformer框架，用于同时预测所有变量的动态，通过开发一种新颖的多元时空信息（mvSTI）转换，该转换将每个观测变量转换为延迟嵌入状态（向量），并进一步交叉学习来自不同变量的这些状态。从动力系统的角度来看，Delayformer预测系统状态而不是单个变量，因此在理论上和计算上克服了这种非线性和交叉交互问题。具体来说，它首先利用一个共享的视觉Transformer（ViT）编码器以延迟嵌入的形式交叉表示来自观测变量的动态状态，然后采用不同的线性解码器来预测下一个状态，即并行预测所有原始变量。通过利用延迟嵌入理论的理论基础和Transformer的表示能力，Delayformer在合成和真实世界数据集上的预测任务中优于当前最先进的方法。此外，通过跨领域预测任务展示了Delayformer作为基础时间序列模型的潜力，突出了其在各种场景中的广泛适用性。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11528) | **Categories:** cs.LG

---

### [5] [Self-Regulating Cars: Automating Traffic Control in Free Flow Road Networks](https://arxiv.org/abs/2506.11973)
*Ankit Bhardwaj, Rohail Asim, Sachin Chauhan, Yasir Zaki, Lakshminarayanan Subramanian*

Main category: cs.LG

TL;DR: 提出了一种基于强化学习的自调节车辆速度控制方法，用于优化自由流道路网络的交通吞吐量并减少拥堵。


<details>
  <summary>Details</summary>
Motivation: 由于通勤者涌入的增加和基础设施的限制，郊区高速公路等自由流道路网络正日益面临交通拥堵。

Method: 提出了一种基于强化学习的交通控制协议，该协议动态调节车辆速度以优化吞吐量并防止拥堵，而无需新的物理基础设施。

Result: 在高保真 PTV Vissim 仿真器中，该方法在实际高速公路网络中进行了评估，结果表明，与无控制设置相比，总吞吐量提高了 5%，平均延误减少了 13%，总停车次数减少了 3%。

Conclusion: 该方法在实际高速公路网络中进行了评估，结果表明，与无控制设置相比，总吞吐量提高了5%，平均延误减少了13%，总停车次数减少了3%。

Abstract: 由于通勤流量的增长和基础设施的限制，郊区高速公路等自由流道路网络正面临日益严重的交通拥堵。传统的控制机制，如交通信号灯或局部启发式算法，在这些高速、无信号灯环境中无效或不可行。我们引入了自调节汽车，这是一种基于强化学习的交通控制协议，可以动态调节车辆速度，以优化吞吐量并防止拥堵，而无需新的物理基础设施。我们的方法将经典交通流理论、间隙接受模型和微观仿真集成到一个物理信息强化学习框架中。通过将道路抽象成超段，智能体可以从瞬时交通观测中捕捉到涌现的流量动态，并学习稳健的速度调节策略。在高保真 PTV Vissim 仿真器中，在一个真实的高速公路网络上进行了评估，结果表明，与无控制设置相比，我们的方法将总吞吐量提高了 5%，平均延误减少了 13%，总停车次数减少了 3%。它还实现了更平稳、抗拥堵的交通流，同时推广到不同的交通模式，证明了其在可扩展的、机器学习驱动的交通管理方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11973) | **Categories:** cs.LG

---


## cs.MA [cs.MA]
### [1] [AutoGen Driven Multi Agent Framework for Iterative Crime Data Analysis and Prediction](https://arxiv.org/abs/2506.11475)
*Syeda Kisaa Fatima, Tehreem Zubair, Noman Ahmed, Asifullah Khan*

Main category: cs.MA

TL;DR: LUCID-MA 使用多智能体对话离线分析犯罪数据，实现自主学习和预测。


<details>
  <summary>Details</summary>
Motivation: 通过多智能体的对话协作分析和理解犯罪数据。

Method: LUCID-MA 框架，由分析助理、反馈组件和预测组件组成。使用 LLaMA-2-13B-Chat-GPTQ 模型，通过精心设计的提示，使智能体能够进行 100 轮的通信，并在较少的人工干预下进行自我改进。还结合了一个评分函数来评估智能体的性能，并提供可视化图表来跟踪学习进度。

Result: 该系统能够突出时空犯罪模式，审查和完善分析结果，并预测未来犯罪趋势。

Conclusion: AutoGen 风格的智能体在社会科学领域实现了自主、可扩展和迭代的分析，并通过离线执行保持数据隐私。

Abstract: 本文介绍了一种创新的人工智能驱动框架 LUCID-MA（通过多智能体对话学习和理解犯罪）。我们的系统由三个核心组件组成：一个突出时空犯罪模式的分析助理，一个审查和改进分析结果的反馈组件，以及一个预测未来犯罪趋势的预测组件。借助精心设计的提示和 LLaMA-2-13B-Chat-GPTQ 模型，该系统完全离线运行，并允许智能体在较少的人工交互下，通过 100 轮的通信进行自我改进。该系统还结合了一个评分函数来评估智能体的性能，并提供可视化图表来跟踪学习进度。这项工作展示了 AutoGen 风格的智能体在社会科学领域进行自主、可扩展和迭代分析的潜力，并通过离线执行来维护数据隐私。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11475) | **Categories:** cs.MA, cs.CL, cs.CV

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving](https://arxiv.org/abs/2506.11234)
*Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull*

Main category: cs.RO

TL;DR: Poutine是一个为长尾驾驶场景设计的视觉-语言模型，通过VLT预训练和RL微调在Waymo自动驾驶挑战赛中获得第一名。


<details>
  <summary>Details</summary>
Motivation: 解决长尾驾驶场景下的端到端自动驾驶问题。

Method: 提出了一种名为Poutine的视觉-语言模型，该模型通过两阶段训练：首先在大量正常驾驶数据和少量长尾驾驶数据上进行自监督VLT预训练，然后使用群体相对策略优化（GRPO）进行微调。

Result: Poutine-Base在验证集上达到了8.12的评分，接近Waymo的专家真实评分。最终的Poutine模型在Waymo测试集上达到了7.99的评分，在2025 Waymo Vision-Based End-to-End Driving Challenge中名列第一。

Conclusion: 可扩展的VLT预训练和轻量级的RL微调能够实现强大且可泛化的自动驾驶。

Abstract: 我们提出了Poutine，一个30亿参数的视觉-语言模型（VLM），专为长尾驾驶场景下的端到端自动驾驶而设计。Poutine的训练分为两个阶段。为了获得强大的基础驾驶能力，我们以自监督的视觉-语言-轨迹（VLT）下一token预测方式，在83小时的CoVLA正常驾驶数据和11小时的Waymo长尾驾驶数据上训练Poutine-Base。伴随的语言标注由一个720亿参数的VLM自动生成。Poutine通过使用来自Waymo验证集的少于500个偏好标记帧，利用群体相对策略优化（GRPO）对Poutine-Base进行微调而获得。我们表明，VLT预训练和RL微调对于在长尾中获得强大的驾驶性能至关重要。Poutine-Base在验证集上获得了8.12的评分（RFS），几乎与Waymo的专家真实RFS相匹配。最终的Poutine模型在官方Waymo测试集上获得了7.99的RFS，在2025 Waymo Vision-Based End-to-End Driving Challenge中以显著优势名列第一。这些结果突出了可扩展的VLT预训练和轻量级RL微调在实现鲁棒和可泛化的自动驾驶方面的潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11234) | **Categories:** cs.RO, cs.CV

---

### [2] [Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis](https://arxiv.org/abs/2506.11526)
*Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, Steven Peters, Andrea Stocco, Bassam Alrifaee, Marco Pavone, Johannes Betz*

Main category: cs.RO

TL;DR: 本文综述了基础模型在自动驾驶场景生成与分析中的应用，为该领域的研究者提供了一个全面的参考。


<details>
  <summary>Details</summary>
Motivation: 在复杂的环境中，自动驾驶汽车的安全导航取决于处理各种不同的、罕见的驾驶场景。传统的场景生成方法通常产生有限的多样性和不真实的、对安全至关重要的案例。而基础模型的出现，为合成和解释复杂的驾驶场景提供了新的可能性。

Method: 本文对使用各种基础模型进行自动驾驶场景生成和分析的方法进行了综述，整理了一个统一的分类标准，并回顾了相关的方法论、开源数据集、仿真平台和基准挑战，以及专门为场景生成和分析定制的评估指标。

Result: 本文调研了截至2025年5月，基础模型在自动驾驶场景生成和分析中的应用，并建立了一个持续维护的存储库，其中包含补充材料。

Conclusion: 本文总结了截至2025年5月，在自动驾驶领域中，使用各种基础模型（包括大型语言模型、视觉语言模型、多模态大型语言模型、扩散模型和世界模型）进行场景生成和分析的应用，并突出了开放性挑战和未来的研究方向。

Abstract: 本文对自动驾驶领域中，基础模型在场景生成和场景分析中的应用进行了综述。自动驾驶汽车在复杂环境中安全导航，依赖于处理各种罕见驾驶场景。传统的场景生成方法多样性有限，且难以生成真实的安全性关键案例。而新兴的基础模型能够处理异构输入，从而合成和解释复杂驾驶场景。本文调研了截至2025年5月，大型语言模型、视觉语言模型、多模态大型语言模型、扩散模型和世界模型等基础模型在自动驾驶场景生成和分析中的应用，提出了统一的分类标准，并回顾了方法论、开源数据集、仿真平台、基准挑战以及评估指标。最后，本文总结了开放性挑战和未来的研究方向。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11526) | **Categories:** cs.RO, cs.AI

---

### [3] [Your Ride, Your Rules: Psychology and Cognition Enabled Automated Driving Systems](https://arxiv.org/abs/2506.11842)
*Zhipeng Bao, Qianwen Li*

Main category: cs.RO

TL;DR: PACE-ADS 框架通过感知和响应乘员状态，提升自动驾驶汽车的个性化和舒适性。


<details>
  <summary>Details</summary>
Motivation: 目前的自动驾驶汽车缺乏与乘员有效的双向沟通，限制了个性化和从固定状态的恢复，这降低了舒适性和信任度，可能会减缓自动驾驶汽车的更广泛采用。

Method: PACE-ADS包含三个基于基础模型的代理：驾驶员代理、心理学家代理和协调员代理。驾驶员代理分析驾驶环境，心理学家代理解释乘员的心理信号和认知命令，协调员代理整合这些输入以产生高级行为决策和操作参数。

Result: 在涉及交通信号灯、行人、工作区域和车辆跟随的各种场景中，PACE-ADS 调整驾驶风格以适应乘员状态，提高乘坐舒适性，并通过自主推理或人工指导实现从固定状态的安全恢复。

Conclusion: 基于LLM的框架有希望弥合机器自主和以人为中心的驾驶之间的差距。

Abstract: 尽管自动驾驶技术迅速发展，但目前的自动驾驶汽车（AV）缺乏与乘员有效的双向沟通，这限制了个性化和从固定状态的恢复。这降低了舒适性和信任度，可能会减缓自动驾驶汽车的更广泛采用。我们提出了 PACE-ADS（心理学和认知赋能的自动驾驶系统），这是一个以人为中心的自主框架，使自动驾驶汽车能够感知、解释和响应外部交通和内部乘员状态。PACE-ADS 包含三个基于基础模型的代理：驾驶员代理分析驾驶环境，心理学家代理解释乘员的心理信号（例如，脑电图、心率、面部表情）和认知命令（例如，语音），协调员代理整合这些输入以产生高级行为决策和操作参数。PACE-ADS 不是取代现有的自动驾驶模块，而是通过在行为层面运行来补充它们，并将低级控制委托给本地自动驾驶系统。这种分离实现了闭环适应，并支持跨不同平台的集成。我们在涉及交通信号灯、行人、工作区域和车辆跟随的各种场景中对 PACE-ADS 进行了仿真评估。结果表明，PACE-ADS 调整驾驶风格以适应乘员状态，提高乘坐舒适性，并通过自主推理或人工指导实现从固定状态的安全恢复。我们的研究结果强调了基于 LLM 的框架在弥合机器自主和以人为中心的驾驶之间的差距方面的希望。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11842) | **Categories:** cs.RO

---

### [4] [Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation](https://arxiv.org/abs/2506.11261)
*Shizhe Chen, Ricardo Garcia, Paul Pacaud, Cordelia Schmid*

Main category: cs.RO

TL;DR: Gondola是一种基于LLM的接地视觉-语言规划模型，用于可泛化的机器人操作，它使用多视角图像和历史计划来生成行动计划，并在GemBench上表现出色。


<details>
  <summary>Details</summary>
Motivation: 机器人操作在跨未见物体、环境和由不同语言指令指定的任务中泛化面临重大挑战。现有方法通常在视觉环境中生成基于视觉的计划方面存在不足，并且通常受到单视图图像输入和精确物体接地的限制。

Method: Gondola采用多视角图像和历史计划，以生成包含目标物体和位置的交错文本和分割掩码的下一个行动计划。

Result: Gondola在GemBench数据集的四个泛化层面上均优于最先进的基于LLM的方法。

Conclusion: Gondola在GemBench数据集的四个泛化层面上均优于最先进的基于LLM的方法，包括新的放置、刚性物体、铰接物体和长程任务。

Abstract: 机器人操作在跨未见物体、环境和由不同语言指令指定的任务中泛化面临重大挑战。为了提高泛化能力，最近的研究已经将大型语言模型（LLM）纳入规划和行动执行中。虽然这些方法很有希望，但它们通常在视觉环境中生成基于视觉的计划方面存在不足。尽管已经努力对LLM进行视觉指导调整以进行机器人操作，但现有方法通常受到单视图图像输入的限制，并且难以进行精确的物体定位。在这项工作中，我们介绍Gondola，一种基于LLM的新型基于接地的视觉-语言规划模型，用于可泛化的机器人操作。Gondola采用多视角图像和历史计划，以生成包含目标物体和位置的交错文本和分割掩码的下一个行动计划。为了支持Gondola的训练，我们使用RLBench模拟器构建了三种类型的数据集，即机器人接地规划、多视角指代表达式和伪长程任务数据集。Gondola在GemBench数据集的四个泛化层面上均优于最先进的基于LLM的方法，包括新的放置、刚性物体、铰接物体和长程任务。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.11261) | **Categories:** cs.RO, cs.AI, cs.CV

---
