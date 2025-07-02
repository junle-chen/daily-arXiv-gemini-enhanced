# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-03

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (6)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [VoyagerVision: Investigating the Role of Multi-modal Information for Open-ended Learning Systems](https://arxiv.org/abs/2507.00079)
*Ethan Smyth, Alessandro Suglia*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Open-endedness is an active field of research in the pursuit of capable Artificial General Intelligence (AGI), allowing models to pursue tasks of their own choosing. Simultaneously, recent advancements in Large Language Models (LLMs) such as GPT-4o [9] have allowed such models to be capable of interpreting image inputs. Implementations such as OMNI-EPIC [4] have made use of such features, providing an LLM with pixel data of an agent's POV to parse the environment and allow it to solve tasks. This paper proposes that providing these visual inputs to a model gives it greater ability to interpret spatial environments, and as such, can increase the number of tasks it can successfully perform, extending its open-ended potential. To this aim, this paper proposes VoyagerVision -- a multi-modal model capable of creating structures within Minecraft using screenshots as a form of visual feedback, building on the foundation of Voyager. VoyagerVision was capable of creating an average of 2.75 unique structures within fifty iterations of the system, as Voyager was incapable of this, it is an extension in an entirely new direction. Additionally, in a set of building unit tests VoyagerVision was successful in half of all attempts in flat worlds, with most failures arising in more complex structures. Project website is available at https://esmyth-dev.github.io/VoyagerVision.github.io/

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00079) | **Categories:** cs.AI, cs.LG

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](https://arxiv.org/abs/2507.00603)
*Yupeng Zheng, Pengxuan Yang, Zebin Xing, Qichao Zhang, Yuhang Zheng, Yinfeng Gao, Pengfei Li, Teng Zhang, Zhongpu Xia, Peng Jia, Dongbin Zhao*

Main category: cs.CV

TL;DR: World4Drive利用视觉基础模型构建潜在世界模型，实现了无需感知标注的端到端自动驾驶，并在多个基准测试中取得了显著的性能提升。


<details>
  <summary>Details</summary>
Motivation: 端到端自动驾驶通常依赖于昂贵的感知监督来提取场景信息。因此，构建一个信息丰富的驾驶世界模型，以实现无需感知标注的端到端规划，是一个重要的研究挑战。

Method: World4Drive采用视觉基础模型构建潜在世界模型，用于生成和评估多模态规划轨迹。它首先提取场景特征，然后基于这些特征和驾驶意图生成轨迹，并在潜在空间中预测未来的状态。最后，引入世界模型选择器来评估和选择最佳轨迹。

Result: World4Drive在nuScenes和NavSim基准测试中实现了最先进的性能，L2误差相对降低了18.1%，碰撞率降低了46.7%，训练收敛速度提高了3.75倍。

Conclusion: World4Drive在无需人工标注的情况下，在nuScenes和NavSim基准测试中实现了最先进的性能，L2误差相对降低了18.1%，碰撞率降低了46.7%，训练收敛速度提高了3.75倍。

Abstract: 端到端自动驾驶直接从原始传感器数据生成规划轨迹，但通常依赖于昂贵的感知监督来提取场景信息。由此产生了一个关键的研究挑战：构建一个信息丰富的驾驶世界模型，以通过自监督学习实现无需感知标注的端到端规划。在本文中，我们提出了World4Drive，一个端到端自动驾驶框架，它采用视觉基础模型来构建潜在世界模型，用于生成和评估多模态规划轨迹。具体来说，World4Drive首先提取场景特征，包括驾驶意图和世界潜在表示，这些表示通过视觉基础模型提供的空间语义先验来丰富。然后，它基于当前的场景特征和驾驶意图生成多模态规划轨迹，并在潜在空间中预测多个意图驱动的未来状态。最后，它引入了一个世界模型选择器模块来评估和选择最佳轨迹。我们通过实际的未来观察和从潜在空间重建的预测观察之间的自监督对齐，实现了无需感知标注的端到端规划。World4Drive在开放循环的nuScenes和闭环的NavSim基准测试中，无需手动感知标注即可实现最先进的性能，L2误差相对降低了18.1%，碰撞率降低了46.7%，训练收敛速度提高了3.75倍。代码可在https://github.com/ucaszyp/World4Drive上访问。

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00603) | **Categories:** cs.CV

---

### [2] [Box-QAymo: Box-Referring VQA Dataset for Autonomous Driving](https://arxiv.org/abs/2507.00525)
*Djamahl Etchegaray, Yuxia Fu, Zi Huang, Yadan Luo*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Interpretable communication is essential for safe and trustworthy autonomous driving, yet current vision-language models (VLMs) often operate under idealized assumptions and struggle to capture user intent in real-world scenarios. Existing driving-oriented VQA datasets are limited to full-scene descriptions or waypoint prediction, preventing the assessment of whether VLMs can respond to localized user-driven queries. We introduce Box-QAymo, a box-referring dataset and benchmark designed to both evaluate and finetune VLMs on spatial and temporal reasoning over user-specified objects. Users express intent by drawing bounding boxes, offering a fast and intuitive interface for focused queries in complex scenes. Specifically, we propose a hierarchical evaluation protocol that begins with binary sanity-check questions to assess basic model capacities, and progresses to (1) attribute prediction for box-referred objects, (2) motion understanding of target instances, and (3) spatiotemporal motion reasoning over inter-object dynamics across frames. To support this, we crowd-sourced fine-grained object classes and visual attributes that reflect the complexity drivers encounter, and extract object trajectories to construct temporally grounded QA pairs. Rigorous quality control through negative sampling, temporal consistency checks, and difficulty-aware balancing guarantee dataset robustness and diversity. Our comprehensive evaluation reveals significant limitations in current VLMs when queried about perception questions, highlighting the gap in achieving real-world performance. This work provides a foundation for developing more robust and interpretable autonomous driving systems that can communicate effectively with users under real-world conditions. Project page and dataset are available at https://djamahl99.github.io/qaymo-pages/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00525) | **Categories:** cs.CV, cs.AI

---

### [3] [Overtake Detection in Trucks Using CAN Bus Signals: A Comparative Study of Machine Learning Methods](https://arxiv.org/abs/2507.00593)
*Fernando Alonso-Fernandez, Talha Hanif Butt, Prayag Tiwari*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe overtaking manoeuvres in trucks are vital for preventing accidents and ensuring efficient traffic flow. Accurate prediction of such manoeuvres is essential for Advanced Driver Assistance Systems (ADAS) to make timely and informed decisions. In this study, we focus on overtake detection using Controller Area Network (CAN) bus data collected from five in-service trucks provided by the Volvo Group. We evaluate three common classifiers for vehicle manoeuvre detection, Artificial Neural Networks (ANN), Random Forest (RF), and Support Vector Machines (SVM), and analyse how different preprocessing configurations affect performance. We find that variability in traffic conditions strongly influences the signal patterns, particularly in the no-overtake class, affecting classification performance if training data lacks adequate diversity. Since the data were collected under unconstrained, real-world conditions, class diversity cannot be guaranteed a priori. However, training with data from multiple vehicles improves generalisation and reduces condition-specific bias. Our pertruck analysis also reveals that classification accuracy, especially for overtakes, depends on the amount of training data per vehicle. To address this, we apply a score-level fusion strategy, which yields the best per-truck performance across most cases. Overall, we achieve an accuracy via fusion of TNR=93% (True Negative Rate) and TPR=86.5% (True Positive Rate). This research has been part of the BIG FUN project, which explores how Artificial Intelligence can be applied to logged vehicle data to understand and predict driver behaviour, particularly in relation to Camera Monitor Systems (CMS), being introduced as digital replacements for traditional exterior mirrors.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00593) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [IDRIFTNET: Physics-Driven Spatiotemporal Deep Learning for Iceberg Drift Forecasting](https://arxiv.org/abs/2507.00036)
*Rohan Putatunda, Sanjay Purushotham, Ratnaksha Lele, Vandana P. Janeja*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Drifting icebergs in the polar oceans play a key role in the Earth's climate system, impacting freshwater fluxes into the ocean and regional ecosystems while also posing a challenge to polar navigation. However, accurately forecasting iceberg trajectories remains a formidable challenge, primarily due to the scarcity of spatiotemporal data and the complex, nonlinear nature of iceberg motion, which is also impacted by environmental variables. The iceberg motion is influenced by multiple dynamic environmental factors, creating a highly variable system that makes trajectory identification complex. These limitations hinder the ability of deep learning models to effectively capture the underlying dynamics and provide reliable predictive outcomes. To address these challenges, we propose a hybrid IDRIFTNET model, a physics-driven deep learning model that combines an analytical formulation of iceberg drift physics, with an augmented residual learning model. The model learns the pattern of mismatch between the analytical solution and ground-truth observations, which is combined with a rotate-augmented spectral neural network that captures both global and local patterns from the data to forecast future iceberg drift positions. We compare IDRIFTNET model performance with state-of-the-art models on two Antarctic icebergs: A23A and B22A. Our findings demonstrate that IDRIFTNET outperforms other models by achieving a lower Final Displacement Error (FDE) and Average Displacement Error (ADE) across a variety of time points. These results highlight IDRIFTNET's effectiveness in capturing the complex, nonlinear drift of icebergs for forecasting iceberg trajectories under limited data and dynamic environmental conditions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00036) | **Categories:** cs.LG, physics.ao-ph

---

### [2] [HiT-JEPA: A Hierarchical Self-supervised Trajectory Embedding Framework for Similarity Computation](https://arxiv.org/abs/2507.00028)
*Lihuan Li, Hao Xue, Shuang Ao, Yang Song, Flora Salim*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The representation of urban trajectory data plays a critical role in effectively analyzing spatial movement patterns. Despite considerable progress, the challenge of designing trajectory representations that can capture diverse and complementary information remains an open research problem. Existing methods struggle in incorporating trajectory fine-grained details and high-level summary in a single model, limiting their ability to attend to both long-term dependencies while preserving local nuances. To address this, we propose HiT-JEPA (Hierarchical Interactions of Trajectory Semantics via a Joint Embedding Predictive Architecture), a unified framework for learning multi-scale urban trajectory representations across semantic abstraction levels. HiT-JEPA adopts a three-layer hierarchy that progressively captures point-level fine-grained details, intermediate patterns, and high-level trajectory abstractions, enabling the model to integrate both local dynamics and global semantics in one coherent structure. Extensive experiments on multiple real-world datasets for trajectory similarity computation show that HiT-JEPA's hierarchical design yields richer, multi-scale representations. Code is available at: https://anonymous.4open.science/r/HiT-JEPA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00028) | **Categories:** cs.LG, cs.AI, cs.CV

---

### [3] [LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing](https://arxiv.org/abs/2507.00029)
*Wenbing Li, Zikai Song, Hang Zhou, Yunyao Zhang, Junqing Yu, Wei Yang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent efforts to combine low-rank adaptation (LoRA) with mixture-of-experts (MoE) for adapting large language models (LLMs) to multiple tasks still exhibit prevailing limitations: they either swap entire attention/feed-forward layers for switch experts or bolt on parallel expert branches, diluting parameter efficiency and task fidelity. We propose the LoRA-Mixer, a modular and lightweight MoE framework that integrates LoRA experts. Our core innovation lies in replacing the projection matrices of the attention module's input/output linear layers with dynamically routed, task-specific LoRA experts. This design ensures seamless compatibility with diverse foundation models, including transformers and state space models (SSMs), by leveraging their inherent linear projection structures. The framework supports two operational paradigms: (1) joint optimization of LoRA experts and routing mechanisms via a novel hard-soft routing strategy, or (2) direct deployment of pre-trained, frozen LoRA modules sourced from external repositories. To enable robust router training with limited data while ensuring stable routing decisions and maximizing expert reuse, we introduce an adaptive Specialization Balance Loss (SBL) that jointly optimizes expert balance and task-specific alignment. Extensive experiments on seven benchmark datasets, including MedQA, CoLA, SST-2, GSM8K, ARC-E, ARC-C, and HumanEval, demonstrate the effectiveness of LoRA-Mixer. On datasets such as GSM8K, HumanEval, and MedQA, LoRA-Mixer achieves significant improvements of 7.61%, 4.88%, and 3.08% over the base models, respectively. Compared with state-of-the-art methods, LoRA-Mixer achieves additional improvements of 1.09%, 1.45%, and 1.68%, respectively, using only 48% of the parameters, demonstrating its efficiency and strong performance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00029) | **Categories:** cs.LG, cs.AI

---

### [4] [Enhancing Spatio-Temporal Forecasting with Spatial Neighbourhood Fusion:A Case Study on COVID-19 Mobility in Peru](https://arxiv.org/abs/2507.00031)
*Chuan Li, Jiang You, Hassine Moungla, Vincent Gauthier, Miguel Nunez-del-Prado, Hugo Alatrista-Salas*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate modeling of human mobility is critical for understanding epidemic spread and deploying timely interventions. In this work, we leverage a large-scale spatio-temporal dataset collected from Peru's national Digital Contact Tracing (DCT) application during the COVID-19 pandemic to forecast mobility flows across urban regions. A key challenge lies in the spatial sparsity of hourly mobility counts across hexagonal grid cells, which limits the predictive power of conventional time series models. To address this, we propose a lightweight and model-agnostic Spatial Neighbourhood Fusion (SPN) technique that augments each cell's features with aggregated signals from its immediate H3 neighbors. We evaluate this strategy on three forecasting backbones: NLinear, PatchTST, and K-U-Net, under various historical input lengths. Experimental results show that SPN consistently improves forecasting performance, achieving up to 9.85 percent reduction in test MSE. Our findings demonstrate that spatial smoothing of sparse mobility signals provides a simple yet effective path toward robust spatio-temporal forecasting during public health crises.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00031) | **Categories:** cs.LG

---

### [5] [The language of time: a language model perspective on time-series foundation models](https://arxiv.org/abs/2507.00078)
*Yi Xie, Yun Xiong, Zejian Shi, Hao Niu, Zhengfu Liu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the rise of large language models, the paradigm of training foundation models with massive parameter counts on vast datasets has been adopted in multiple domains to achieve remarkable success. Time series foundation models represent a significant extension of this paradigm, demonstrating exceptional expressive power, generalization, and cross-domain transferability. However, this gives rise to a fundamental paradox: time series data reflect distinct dynamical systems, making cross-domain transfer intuitively implausible, yet this is contradicted by the models' empirical success. To resolve this paradox, this paper investigates, from both theoretical and experimental perspectives, the representation learning mechanisms and generalization capabilities of patch-based time series foundation models. We argue that such models are not merely applying a new architecture but are fundamentally generalizing the representation paradigm of language models by extending deterministic vector-based representations to latent probabilistic distributional forms. Our theoretical analysis supports this framework by demonstrating that continuous time-series patches can be faithfully quantized into a discrete vocabulary whose key statistical properties are highly consistent with those of natural language. This generalization allows time series models to inherit the robust representation and transfer abilities of large language models, thereby explaining their superior performance in temporal tasks. Ultimately, our work provides a rigorous theoretical cornerstone for understanding, evaluating, and improving the safety and reliability of large-scale time series foundation models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00078) | **Categories:** cs.LG, cs.AI, cs.CL

---

### [6] [A Joint Topology-Data Fusion Graph Network for Robust Traffic Speed Prediction with Data Anomalism](https://arxiv.org/abs/2507.00085)
*Ruiyuan Jiang, Dongyao Jia, Eng Gee Lim, Pengfei Fan, Yuli Zhang, Shangbo Wang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate traffic prediction is essential for Intelligent Transportation Systems (ITS), yet current methods struggle with the inherent complexity and non-linearity of traffic dynamics, making it difficult to integrate spatial and temporal characteristics. Furthermore, existing approaches use static techniques to address non-stationary and anomalous historical data, which limits adaptability and undermines data smoothing. To overcome these challenges, we propose the Graph Fusion Enhanced Network (GFEN), an innovative framework for network-level traffic speed prediction. GFEN introduces a novel topological spatiotemporal graph fusion technique that meticulously extracts and merges spatial and temporal correlations from both data distribution and network topology using trainable methods, enabling the modeling of multi-scale spatiotemporal features. Additionally, GFEN employs a hybrid methodology combining a k-th order difference-based mathematical framework with an attention-based deep learning structure to adaptively smooth historical observations and dynamically mitigate data anomalies and non-stationarity. Extensive experiments demonstrate that GFEN surpasses state-of-the-art methods by approximately 6.3% in prediction accuracy and exhibits convergence rates nearly twice as fast as recent hybrid models, confirming its superior performance and potential to significantly enhance traffic prediction system efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00085) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [When Digital Twins Meet Large Language Models: Realistic, Interactive, and Editable Simulation for Autonomous Driving](https://arxiv.org/abs/2507.00319)
*Tanmay Vilas Samak, Chinmay Vilas Samak, Bing Li, Venkat Krovi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Simulation frameworks have been key enablers for the development and validation of autonomous driving systems. However, existing methods struggle to comprehensively address the autonomy-oriented requirements of balancing: (i) dynamical fidelity, (ii) photorealistic rendering, (iii) context-relevant scenario orchestration, and (iv) real-time performance. To address these limitations, we present a unified framework for creating and curating high-fidelity digital twins to accelerate advancements in autonomous driving research. Our framework leverages a mix of physics-based and data-driven techniques for developing and simulating digital twins of autonomous vehicles and their operating environments. It is capable of reconstructing real-world scenes and assets (real2sim) with geometric and photorealistic accuracy and infusing them with various physical properties to enable real-time dynamical simulation of the ensuing driving scenarios. Additionally, it also incorporates a large language model (LLM) interface to flexibly edit the driving scenarios online via natural language prompts. We analyze the presented framework in terms of its fidelity, performance, and serviceability. Results indicate that our framework can reconstruct 3D scenes and assets with up to 97% structural similarity, while maintaining frame rates above 60 Hz. We also demonstrate that it can handle natural language prompts to generate diverse driving scenarios with up to 95% repeatability and 85% generalizability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00319) | **Categories:** cs.RO

---

### [2] [VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers](https://arxiv.org/abs/2507.01016)
*Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, Tong He*

Main category: cs.RO

TL;DR: 本文提出了一种基于大规模合成数据训练的向量量化动作标记器，显著提升了机器人控制的性能和效率。


<details>
  <summary>Details</summary>
Motivation: 旨在解决现有方法在捕捉丰富的时空动态方面的不足，并加速推理过程，生成更流畅和连贯的动作输出。

Method: 提出了一种基于向量量化的动作标记器，该标记器建立在目前最大规模的动作轨迹数据集上。

Result: 实验结果表明，随着合成轨迹数据量的增加，标记器在下游任务中的性能显著提高，在长时程场景中的两个真实世界任务上的成功率提高了30%。

Conclusion: 该研究表明，利用大规模合成数据训练的基于向量量化的动作标记器，能够有效提升机器人控制在各种应用领域中的效率和可靠性。

Abstract: 本文介绍了一种创新的基于向量量化的动作标记器，该标记器建立在迄今为止最大规模的动作轨迹数据集上，利用的数据比以往方法多100倍以上。这种广泛的数据集使我们的标记器能够捕获丰富的时空动态，从而产生一个不仅加速推理，而且生成更流畅和连贯的动作输出的模型。一旦经过训练，该标记器可以零样本方式无缝地适应各种下游任务，从短时程反应行为到长时程规划。我们工作的一个关键发现是，合成和真实动作轨迹之间的领域差距很小，这使我们能够有效地利用大量合成数据进行训练，而不会影响真实世界的性能。为了验证我们的方法，我们进行了广泛的模拟环境和真实机器人平台实验。结果表明，随着合成轨迹数据量的增加，我们的标记器在下游任务中的性能显着提高——最值得注意的是，在长时程场景中的两个真实世界任务上的成功率提高了30%。这些发现突出了我们的动作标记器作为实时具身智能系统的稳健且可扩展的解决方案的潜力，为在各种应用领域中实现更高效和可靠的机器人控制铺平了道路。

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.01016) | **Categories:** cs.RO, cs.CV

---

### [3] [Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations](https://arxiv.org/abs/2507.00990)
*Shivansh Patel, Shraddhaa Mohan, Hanlin Mai, Unnat Jain, Svetlana Lazebnik, Yunzhu Li*

Main category: cs.RO

TL;DR: RIGVid系统使机器人能够通过模仿AI生成的视频来执行复杂的操作任务，无需物理演示或机器人特定训练。


<details>
  <summary>Details</summary>
Motivation: 该论文旨在解决机器人执行复杂操作任务时，需要大量物理演示或机器人特定训练的问题。

Method: 提出了一种名为RIGVid的系统，该系统通过模仿AI生成的视频，使机器人能够执行复杂的操作任务。该系统包括：视频扩散模型生成潜在的演示视频，视觉-语言模型（VLM）自动过滤不符合指令的结果，6D姿态跟踪器从视频中提取对象轨迹，并将轨迹以与机器人无关的方式重新定位到机器人。

Result: 实验结果表明，经过滤的生成视频与真实演示一样有效，并且性能随着生成质量的提高而提高。此外，依赖生成的视频优于使用VLM的关键点预测等更紧凑的替代方案，并且强大的6D姿态跟踪优于其他提取轨迹的方法，例如密集特征点跟踪。

Conclusion: 利用AI生成的视频可以有效地指导机器人完成复杂的操作任务，且效果与真实演示相当。

Abstract: 本文介绍了一种名为Robots Imitating Generated Videos (RIGVid) 的系统，该系统使机器人能够仅通过模仿AI生成的视频来执行复杂的操纵任务（例如倾倒、擦拭和混合），而无需任何物理演示或机器人特定的训练。给定语言命令和初始场景图像，视频扩散模型会生成潜在的演示视频，并且视觉语言模型 (VLM) 会自动过滤掉不符合命令的结果。然后，6D姿态跟踪器从视频中提取对象轨迹，并且这些轨迹以与具体实现无关的方式重新定位到机器人。通过广泛的真实世界评估，我们表明，经过滤的生成视频与真实演示一样有效，并且性能随着生成质量的提高而提高。我们还表明，依赖生成的视频优于使用VLM的关键点预测等更紧凑的替代方案，并且强大的6D姿态跟踪优于其他提取轨迹的方法，例如密集特征点跟踪。这些发现表明，最先进的现成模型生成的视频可以为机器人操作提供有效的监督来源。

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00990) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [4] [Sim2Real Diffusion: Learning Cross-Domain Adaptive Representations for Transferable Autonomous Driving](https://arxiv.org/abs/2507.00236)
*Chinmay Vilas Samak, Tanmay Vilas Samak, Bing Li, Venkat Krovi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Simulation-based design, optimization, and validation of autonomous driving algorithms have proven to be crucial for their iterative improvement over the years. Nevertheless, the ultimate measure of effectiveness is their successful transition from simulation to reality (sim2real). However, existing sim2real transfer methods struggle to comprehensively address the autonomy-oriented requirements of balancing: (i) conditioned domain adaptation, (ii) robust performance with limited examples, (iii) modularity in handling multiple domain representations, and (iv) real-time performance. To alleviate these pain points, we present a unified framework for learning cross-domain adaptive representations for sim2real transferable autonomous driving algorithms using conditional latent diffusion models. Our framework offers options to leverage: (i) alternate foundation models, (ii) a few-shot fine-tuning pipeline, and (iii) textual as well as image prompts for mapping across given source and target domains. It is also capable of generating diverse high-quality samples when diffusing across parameter spaces such as times of day, weather conditions, seasons, and operational design domains. We systematically analyze the presented framework and report our findings in the form of critical quantitative metrics and ablation studies, as well as insightful qualitative examples and remarks. Additionally, we demonstrate the serviceability of the proposed approach in bridging the sim2real gap for end-to-end autonomous driving using a behavioral cloning case study. Our experiments indicate that the proposed framework is capable of bridging the perceptual sim2real gap by over 40%. We hope that our approach underscores the potential of generative diffusion models in sim2real transfer, offering a pathway toward more robust and adaptive autonomous driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00236) | **Categories:** cs.RO

---

### [5] [PI-WAN: A Physics-Informed Wind-Adaptive Network for Quadrotor Dynamics Prediction in Unknown Environments](https://arxiv.org/abs/2507.00816)
*Mengyun Wang, Bo Wang, Yifeng Niu, Chang Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate dynamics modeling is essential for quadrotors to achieve precise trajectory tracking in various applications. Traditional physical knowledge-driven modeling methods face substantial limitations in unknown environments characterized by variable payloads, wind disturbances, and external perturbations. On the other hand, data-driven modeling methods suffer from poor generalization when handling out-of-distribution (OoD) data, restricting their effectiveness in unknown scenarios. To address these challenges, we introduce the Physics-Informed Wind-Adaptive Network (PI-WAN), which combines knowledge-driven and data-driven modeling methods by embedding physical constraints directly into the training process for robust quadrotor dynamics learning. Specifically, PI-WAN employs a Temporal Convolutional Network (TCN) architecture that efficiently captures temporal dependencies from historical flight data, while a physics-informed loss function applies physical principles to improve model generalization and robustness across previously unseen conditions. By incorporating real-time prediction results into a model predictive control (MPC) framework, we achieve improvements in closed-loop tracking performance. Comprehensive simulations and real-world flight experiments demonstrate that our approach outperforms baseline methods in terms of prediction accuracy, tracking precision, and robustness to unknown environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.00816) | **Categories:** cs.RO, cs.AI

---
