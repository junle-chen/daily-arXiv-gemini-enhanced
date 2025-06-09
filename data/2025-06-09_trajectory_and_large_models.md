# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-06-08

## 目录

- [Computer Vision (11)](#cs-cv)

## Computer Vision [cs.CV]
### [1] [Dynamic Epsilon Scheduling: A Multi-Factor Adaptive Perturbation Budget for Adversarial Training](https://arxiv.org/abs/2506.04263)
*Alan Mitkiy, James Smith, Hana Satou, Hiroshi Tanaka, Emily Johnson, F Monkey*

Main category: cs.CV

TL;DR: 本文提出了一种动态调整对抗扰动预算的对抗训练框架，提高了模型的鲁棒性和泛化能力。


<details>
  <summary>Details</summary>
Motivation: 现有的对抗训练方法依赖于固定的扰动预算，无法考虑实例特定的鲁棒性特征。

Method: 本文提出了一种名为动态Epsilon调度（DES）的框架，该框架通过结合梯度代理、预测置信度和模型不确定性等因素，自适应地调整每个实例和每个训练迭代的对抗扰动预算。

Result: 在CIFAR-10和CIFAR-100上的实验结果表明，与固定epsilon基线和先前的自适应方法相比，该方法能够持续提高对抗鲁棒性和标准准确率。

Conclusion: 本文提出了一种新的对抗训练方法，通过动态调整扰动预算，提高了模型的鲁棒性和泛化能力。

Abstract: 对抗训练是防御深度神经网络免受对抗样本攻击的最有效策略之一。现有的对抗训练方法的一个关键限制在于它们依赖于固定的扰动预算，这无法解释特定实例的鲁棒性特征。虽然诸如IAAT和MMA之类的工作引入了实例级别的自适应，但它们通常依赖于数据鲁棒性的启发式或静态近似。在本文中，我们提出了一种动态Epsilon调度（DES）框架，该框架可以自适应地调整每个实例和每个训练迭代的对抗扰动预算。DES集成了三个关键因素：（1）通过基于梯度的代理近似的到决策边界的距离，（2）从softmax熵得出的预测置信度，以及（3）通过蒙特卡洛dropout估计的模型不确定性。通过将这些线索整合到统一的调度策略中，DES可以动态地调整扰动预算，以指导更有效的对抗学习。在CIFAR-10和CIFAR-100上的实验结果表明，与固定epsilon基线和先前的自适应方法相比，我们的方法能够持续提高对抗鲁棒性和标准准确率。此外，我们还提供了有关调度策略的稳定性和收敛性的理论见解。这项工作为实例感知、数据驱动的对抗训练方法开辟了一条新途径。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04263) | **Categories:** cs.CV, cs.LG

---

### [2] [RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought](https://arxiv.org/abs/2506.04277)
*Yi Lu, Jiawang Cao, Yongliang Wu, Bozheng Li, Licheng Tang, Yangguang Ji, Chong Wu, Jay Wu, Wenbo Zhu*

Main category: cs.CV

TL;DR: RSVP：一个用于统一多模态推理与分割的新框架，通过视觉提示实现可解释的推理分割。


<details>
  <summary>Details</summary>
Motivation: 多模态大型语言模型(MLLM)在推理能力方面表现出了显著的能力，但缺乏显式的视觉基础和分割机制，这在认知推理和视觉感知之间造成了差距。

Method: RSVP是一个两阶段的结构化框架，它集成了推理驱动的定位和分割细化。

Result: RSVP在ReasonSeg上超过了最先进的方法高达+6.5 gIoU和+9.2 cIoU，并在zero-shot设置下在SegInW上实现了49.7 mAP。

Conclusion: RSVP通过显式地建模多模态推理和分割之间的交互，为可解释的推理分割引入了一种新的范例，并实现了最先进的性能。

Abstract: 多模态大型语言模型(MLLM)在推理能力方面表现出了显著的能力，但缺乏显式的视觉基础和分割机制，这在认知推理和视觉感知之间造成了差距。为了弥合这一差距，我们引入了通过视觉提示进行推理分割(RSVP)，这是一个新的框架，它统一了多步骤多模态推理与有基础的视觉理解。RSVP是一个两阶段的结构化框架，它集成了推理驱动的定位和分割细化。在推理阶段，RSVP采用多模态的思维链视觉提示，以帮助MLLM理解查询和推断目标，生成可解释的区域建议，从而增强视觉基础。在分割阶段，RSVP使用视觉-语言分割模块(VLSM)细化这些建议，无缝地集成文本和视觉线索，以生成精确的分割掩码。通过显式地建模多模态推理和分割之间的交互，RSVP为可解释的推理分割引入了一种新的范例。它利用了MLLM固有的定位能力，使模型不仅能够推理对象，还能够生成结构化的视觉表示。我们广泛的实验表明，RSVP实现了最先进的性能，在ReasonSeg上超过了最先进的方法高达+6.5 gIoU和+9.2 cIoU，并在zero-shot设置下在SegInW上实现了49.7 mAP。这些结果验证了RSVP作为一个有效的和可扩展的框架，用于整合认知推理与结构化的视觉理解。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04277) | **Categories:** cs.CV, cs.AI

---

### [3] [Evaluating MLLMs with Multimodal Multi-image Reasoning Benchmark](https://arxiv.org/abs/2506.04280)
*Ziming Cheng, Binrui Xu, Lisheng Gong, Zuhe Song, Tianshuo Zhou, Shiqi Zhong, Siyu Ren, Mingxiang Chen, Xiangchao Meng, Yuxin Zhang, Yanlin Li, Lei Ren, Wei Chen, Zhiyuan Huang, Mingjie Zhan, Xiaojie Wang, Fangxiang Feng*

Main category: cs.CV

TL;DR: MMRB基准测试表明，开源多模态大语言模型在多图推理能力上显著落后于商业模型，且现有奖励模型难以处理多图奖励排序。


<details>
  <summary>Details</summary>
Motivation: 现有的多模态大语言模型基准主要集中于单图视觉推理或仅有最终答案评估的多图理解任务，对多图输入的多模态大语言模型的推理能力探索不足。

Method: 提出了多模态多图推理基准（MMRB），包含92个子任务，涵盖空间、时间和语义推理，并使用GPT-4o生成并由人工专家改进的多解、CoT风格注释。此外，还提出了一个句子级匹配框架，使用开源LLM来支持快速和可扩展的评估。

Result: 对40个多模态大语言模型（包括9个推理专用模型和8个奖励模型）进行了广泛的基线实验，结果表明开源多模态大语言模型在多图推理任务中仍然显著落后于商业模型。此外，当前的多模态奖励模型几乎无法处理多图奖励排序任务。

Conclusion: 开源多模态大语言模型在多图推理任务中显著落后于商业模型，且当前多模态奖励模型几乎无法处理多图奖励排序任务。

Abstract: 随着能力的增强和应用的普及，多模态大型语言模型（MLLM）越来越需要在多个图像上进行处理和推理。然而，现有的MLLM基准主要集中于单图像视觉推理或仅有最终答案评估的多图像理解任务，这使得MLLM在多图像输入上的推理能力在很大程度上未被探索。为了解决这个问题，我们推出了多模态多图像推理基准（MMRB），这是第一个旨在评估跨多个图像的结构化视觉推理的基准。MMRB包含92个子任务，涵盖空间、时间和语义推理，具有由GPT-4o生成并由人工专家改进的多解、CoT风格的注释。一个衍生的子集旨在评估多图像场景中的多模态奖励模型。为了支持快速和可扩展的评估，我们提出了一个使用开源LLM的句子级匹配框架。对40个MLLM（包括9个推理专用模型和8个奖励模型）的广泛基线实验表明，在多图像推理任务中，开源MLLM仍然显著落后于商业MLLM。此外，当前的多模态奖励模型几乎无法处理多图像奖励排序任务。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04280) | **Categories:** cs.CV, cs.AI, 68T50, I.2.7

---

### [4] [HuGeDiff: 3D Human Generation via Diffusion with Gaussian Splatting](https://arxiv.org/abs/2506.04351)
*Maksym Ivashechkin, Oscar Mendez, Richard Bowden*

Main category: cs.CV

TL;DR: 该论文提出了一种弱监督流水线，用于快速生成高质量、可控的3D人体模型。


<details>
  <summary>Details</summary>
Motivation: 当前方法在生成精确的3D人体方面存在挑战，包括细节不足、手和面部渲染不准确、人体逼真度不足以及外观控制性差。此外，缺乏多样性、逼真度和标注的人体图像数据也阻碍了3D人体模型的发展。

Method: 该方法包括三个步骤：1) 使用图像扩散模型生成逼真且可控的人体图像数据集；2) 提出一种高效的图像特征到3D点云的映射方法，该方法基于Transformer架构；3) 训练一个以文本提示为条件的点云扩散模型。

Result: 实验结果表明，与现有方法相比，该方法在3D人体生成速度上提高了几个数量级，并且在文本提示对齐、逼真度和渲染质量方面都有显著提高。

Conclusion: 该论文提出了一种弱监督流水线，通过图像扩散模型生成逼真的人体图像数据集，并使用基于Transformer的架构将图像特征映射到3D点云，最后训练一个以文本提示为条件的点云扩散模型，从而实现了更快速、更高质量的3D人体生成。

Abstract: 三维人体生成是计算机视觉和图形学中一个重要的问题，有着广泛的应用。尽管生成式人工智能（如扩散模型）或神经辐射场或高斯溅射等渲染方法取得了最新进展，但从文本提示控制精确的三维人体的生成仍然是一个开放的挑战。目前的方法在精细细节、手和面部的精确渲染、人体逼真度以及外观的可控性方面存在困难。缺乏多样性、逼真度和标注的人体图像数据仍然是一个挑战，阻碍了基础三维人体模型的发展。我们提出了一种弱监督流水线，试图解决这些挑战。第一步，我们使用最先进的图像扩散模型生成具有可控属性（如外观、种族、性别等）的逼真人体图像数据集。接下来，我们提出了一种有效的从图像特征到三维点云的映射方法，使用基于Transformer的架构。最后，我们通过训练一个以用于生成原始样本的相同文本提示为条件的点云扩散模型来闭环。我们证明，与最先进的方法相比，三维人体生成的速度提高了几个数量级，并且文本提示对齐、逼真度和渲染质量也得到了显著提高。我们将提供代码和数据集。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04351) | **Categories:** cs.CV

---

### [5] [ReXVQA: A Large-scale Visual Question Answering Benchmark for Generalist Chest X-ray Understanding](https://arxiv.org/abs/2506.04353)
*Ankit Pal, Jung-Oh Lee, Xiaoman Zhang, Malaikannan Sankarasubbu, Seunghyeon Roh, Won Jung Kim, Meesun Lee, Pranav Rajpurkar*

Main category: cs.CV

TL;DR: ReXVQA是一个新的胸部X光片视觉问答基准，AI模型在该基准上的表现超越了人类专家。


<details>
  <summary>Details</summary>
Motivation: 现有的VQA方法过度依赖模板查询，缺乏多样性和临床真实性。

Method: 提出了ReXVQA，一个包含约696,000个问题和160,000张胸部X光片研究的视觉问答基准。

Result: MedGemma模型取得了83.24%的总体准确率，超过了放射科住院医师的77.27%。

Conclusion: AI模型在胸部X光片判读方面超越了人类专家，但AI模型和人类专家之间存在不同的表现模式。

Abstract: 我们提出了ReXVQA，这是胸部放射学中最大、最全面的视觉问答（VQA）基准，包含约696,000个问题，并配有训练、验证和测试集中160,000个胸部X光片研究。与先前严重依赖基于模板的查询的工作不同，ReXVQA引入了一个多样化且临床真实的Task套件，反映了五个核心放射学推理技能：存在评估、位置分析、否定检测、鉴别诊断和几何推理。我们评估了八个最先进的多模态大型语言模型，包括MedGemma-4B-it、Qwen2.5-VL、Janus-Pro-7B和Eagle2-9B。性能最佳的模型（MedGemma）实现了83.24%的总体准确率。为了弥合AI性能和临床专业知识之间的差距，我们对200个随机抽样的病例进行了全面的人类读者研究，涉及3名放射科住院医师。我们的评估表明，与人类读者（最佳放射科住院医师：77.27%）相比，MedGemma取得了卓越的性能（83.84%的准确率），这代表着一个重要的里程碑，即AI性能超过了胸部X光片解读方面的专家人类评估。读者研究揭示了AI模型和人类专家之间不同的性能模式，放射科医生之间具有很强的读者间一致性，同时显示了人类读者和AI模型之间更可变的一致性模式。ReXVQA为评估通用放射AI系统建立了一个新标准，提供公共排行榜、细粒度评估拆分、结构化解释和类别级别细分。该基准为下一代AI系统奠定了基础，这些系统能够模仿超出狭窄病理分类的专家级临床推理。我们的数据集将在https://huggingface.co/datasets/rajpurkarlab/ReXVQA上开源。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04353) | **Categories:** cs.CV, cs.AI, cs.CE, cs.CL, cs.LG

---

### [6] [WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning](https://arxiv.org/abs/2506.04363)
*Delong Chen, Willy Chung, Yejin Bang, Ziwei Ji, Pascale Fung*

Main category: cs.CV

TL;DR: WorldPrediction是一个新的视频基准，用于评估AI模型的世界建模和程序规划能力，结果表明当前模型与人类水平差距很大。


<details>
  <summary>Details</summary>
Motivation: 当前AI模型，特别是生成模型，如何学习世界模型并在不同的环境中进行程序规划尚不清楚。

Method: 引入WorldPrediction，这是一个基于视频的基准，用于评估不同AI模型的世界建模和程序规划能力。

Result: 当前前沿模型在WorldPrediction-WM上仅达到57%的准确率，在WorldPrediction-PP上仅达到38%，而人类可以完美解决这两项任务。

Conclusion: 当前前沿模型在WorldPrediction-WM上仅达到57%的准确率，在WorldPrediction-PP上仅达到38%，而人类可以完美解决这两项任务。

Abstract: 人类拥有一个内在的“世界模型”，使我们能够根据世界状态进行行动规划。人工智能体也需要拥有这样的世界模型来进行行动规划。目前的人工智能模型，特别是生成模型，如何学习这种世界模型并在不同的环境中进行程序规划尚不清楚。我们引入了WorldPrediction，这是一个基于视频的基准，用于评估不同AI模型的世界建模和程序规划能力。与之前主要关注低级世界建模和机器人运动规划的基准相比，WorldPrediction是第一个强调具有时间和语义抽象的动作的基准。给定初始和最终的世界状态，任务是从一组反事实的干扰项中区分出正确的动作（WorldPrediction-WM）或正确排序的动作序列（WorldPrediction-PP）。这种判别性任务设置使我们能够评估不同类型的世界模型和规划器，并实现对不同假设的全面比较。该基准使用视觉观察来表示状态和动作。为了防止模型利用背景场景中的低级连续性线索，我们提供了“动作等价物”——在不同上下文中观察到的相同动作——作为选择的候选对象。该基准基于部分可观察半MDP的形式框架，确保了评估的更好可靠性和鲁棒性。我们对基准进行了广泛的人工过滤和验证，结果表明，当前的前沿模型在WorldPrediction-WM上的准确率仅为57%，在WorldPrediction-PP上的准确率仅为38%，而人类能够完美地解决这两项任务。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04363) | **Categories:** cs.CV

---

### [7] [Ice Hockey Puck Localization Using Contextual Cues](https://arxiv.org/abs/2506.04365)
*Liam Salass, Jerrin Bright, Amir Nazemi, Yuhao Chen, John Zelek, David Clausi*

Main category: cs.CV

TL;DR: 本文提出了一种利用球员行为上下文线索进行冰球检测的新方法，并在PuckDataset数据集上取得了最先进的性能。


<details>
  <summary>Details</summary>
Motivation: 由于冰球体积小、频繁遮挡、运动模糊、广播伪影以及因相机变焦和广播相机视角变化而导致的不一致性，冰球广播视频中的冰球检测提出了重大挑战。以往的研究侧重于基于外观或基于运动的冰球线索，而没有明确地对来自球员行为的线索进行建模。球员们会持续地转动身体，并将视线 направ к шайбе.

Method: 提出了一种名为PLUCC的新方法，该方法利用上下文线索进行尺度感知的单帧冰球检测。PLUCC包括三个组成部分：(a) 上下文编码器，它利用球员的朝向和位置作为有用的先验知识；(b) 特征金字塔编码器，它从双编码器中提取多尺度特征；(c) 门控解码器，它结合了潜在特征和通道门控机制。

Result: 在PuckDataset数据集上进行的PLUCC实验结果表明，检测性能达到了最先进的水平，平均精度提高了12.2%，RSLE平均精度提高了25%，超过了之前的基线方法。

Conclusion: 这项研究表明，上下文理解在提高冰球检测性能方面起着关键作用，对自动化体育分析具有广泛的影响。

Abstract: 冰球广播视频中的冰球检测由于冰球体积小、频繁遮挡、运动模糊、广播伪影以及因相机变焦和广播相机视角变化而导致的不一致性而面临重大挑战。以往的研究侧重于基于外观或基于运动的冰球线索，而没有明确地对来自球员行为的线索进行建模。受这种强烈的上下文线索的推动，我们提出了一种名为Puck Localization Using Contextual Cues (PLUCC) 的新方法，该方法用于进行尺度感知和上下文驱动的单帧冰球检测。PLUCC包括三个组成部分：(a) 上下文编码器，它利用球员的朝向和位置作为有用的先验知识；(b) 特征金字塔编码器，它从双编码器中提取多尺度特征；(c) 门控解码器，它结合了潜在特征和通道门控机制。为了进行评估，除了标准的平均精度外，我们还提出了冰场空间定位误差 (RSLE)，这是一种尺度不变的基于单应性的度量，用于消除冰场空间评估中的透视偏差。PLUCC在PuckDataset数据集上的实验结果表明，检测性能达到了最先进的水平，平均精度提高了12.2%，RSLE平均精度提高了25%，超过了之前的基线方法。我们的研究表明，上下文理解在提高冰球检测性能方面起着关键作用，对自动化体育分析具有广泛的影响。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04365) | **Categories:** cs.CV, cs.AI

---

### [8] [Fine-Tuning Video Transformers for Word-Level Bangla Sign Language: A Comparative Analysis for Classification Tasks](https://arxiv.org/abs/2506.04367)
*Jubayer Ahmed Bhuiyan Shawon, Hasan Mahmud, Kamrul Hasan*

Main category: cs.CV

TL;DR: 该研究通过微调视频Transformer模型，显著提高了孟加拉语手语识别的准确性，特别是在VideoMAE模型上取得了优异成果。


<details>
  <summary>Details</summary>
Motivation: 孟加拉语手语（BdSL）是孟加拉国听障人士的主要交流方式。本研究旨在提高BdSL识别的准确性和可扩展性。

Method: 研究人员在BdSLW60和BdSLW401数据集上微调了VideoMAE、ViViT和TimeSformer等视频Transformer架构，并采用了数据增强和分层交叉验证等技术。

Result: VideoMAE模型在修正帧率的BdSLW60数据集上达到了95.5%的准确率，在BdSLW401数据集的前向手势上达到了81.04%的准确率。

Conclusion: 视频Transformer模型在孟加拉语手语识别中显著优于传统方法，VideoMAE在BdSLW60和BdSLW401数据集上取得了最高的准确率。

Abstract: 本研究旨在通过自动识别和分类图像或视频中的手势，并将其转换为文本或语音，从而提高听障人士的可访问性。在孟加拉国，孟加拉语手语（BdSL）是许多听障人士的主要交流方式。本研究在BdSLW60（一个包含60个常用手势的小规模BdSL数据集）上微调了最先进的视频Transformer架构——VideoMAE、ViViT和TimeSformer。我们将视频标准化为30 FPS，生成了9307个用户试验片段。为了评估可扩展性和鲁棒性，还在包含401个手势类别的大规模数据集BdSLW401上对模型进行了微调。此外，我们还针对包括LSA64和WLASL在内的公共数据集进行了基准测试。应用了随机裁剪、水平翻转和短边缩放等数据增强技术，以提高模型的鲁棒性。为了确保模型选择过程中各折叠之间的平衡评估，我们在训练集上采用了10折分层交叉验证，同时使用来自未见过用户U4和U8的保留测试数据进行与说话人无关的评估。结果表明，视频Transformer模型明显优于传统的机器学习和深度学习方法。性能受数据集大小、视频质量、帧分布、帧率和模型架构等因素的影响。在这些模型中，VideoMAE变体在帧率校正的BdSLW60数据集上获得了95.5%的最高准确率，在BdSLW401的前向手势上获得了81.04%的最高准确率——证明了可扩展且准确的BdSL识别的强大潜力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04367) | **Categories:** cs.CV

---

### [9] [Normalize Filters! Classical Wisdom for Deep Vision](https://arxiv.org/abs/2506.04401)
*Gustavo Perez, Stella X. Yu*

Main category: cs.CV

TL;DR: 本文提出了一种滤波器归一化方法，通过将经典滤波原则整合到深度学习中，提高了模型在图像强度变化下的鲁棒性和泛化能力。


<details>
  <summary>Details</summary>
Motivation: 当图像经过大气传输时，深度网络中端到端学习的卷积滤波器的响应会失真，导致不正确的结果。

Method: 我们提出滤波器归一化，然后进行可学习的缩放和移位，类似于批量归一化。

Result: 通过将经典滤波原则整合到深度学习中，我们的方法在人工和自然强度变化基准上取得了显着改进。我们的 ResNet34 甚至可以大大优于 CLIP。

Conclusion: 未归一化的滤波器会降低性能，而滤波器归一化可以规范学习，促进多样性，并提高鲁棒性和泛化能力。

Abstract: 经典的图像滤波器，例如用于平均或差分的滤波器，都经过仔细的归一化，以确保一致性、可解释性，并避免诸如强度偏移、光晕或振铃之类的伪影。相比之下，在深度网络中端到端学习的卷积滤波器缺乏这种约束。尽管它们可能类似于小波和斑点/边缘检测器，但它们没有以相同或任何方式进行归一化。因此，当图像经过大气传输时，它们的响应会失真，导致不正确的结果。我们通过提出滤波器归一化来解决此限制，然后进行可学习的缩放和移位，类似于批量归一化。这种简单而有效的修改确保了滤波器是大气等变的，从而实现了共域对称性。通过将经典滤波原则整合到深度学习中（适用于卷积神经网络和卷积相关的视觉 Transformer），我们的方法在人工和自然强度变化基准上取得了显着改进。我们的 ResNet34 甚至可以大大优于 CLIP。我们的分析表明，未归一化的滤波器会降低性能，而滤波器归一化可以规范学习，促进多样性，并提高鲁棒性和泛化能力。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04401) | **Categories:** cs.CV

---

### [10] [HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](https://arxiv.org/abs/2506.04421)
*Hermann Kumbong, Xian Liu, Tsung-Yi Lin, Ming-Yu Liu, Xihui Liu, Ziwei Liu, Daniel Y. Fu, Christopher Ré, David W. Romero*

Main category: cs.CV

TL;DR: HMAR通过分层掩码自回归建模，实现了高质量、快速采样的图像生成，并在速度、内存占用和灵活性方面优于现有方法。


<details>
  <summary>Details</summary>
Motivation: 视觉自回归建模(VAR)在弥合自回归图像模型和扩散模型之间的速度和质量差距方面显示出了希望。然而，由于分辨率尺度中所有token的并行生成，这种公式导致图像质量降低；序列长度以超线性方式缩放图像分辨率；并且需要重新训练才能改变采样计划。

Method: HMAR将下一尺度预测重新定义为马尔可夫过程，其中每个分辨率尺度的预测仅以其直接前置分辨率中的token为条件，而不是以所有前置分辨率中的token为条件。在预测分辨率尺度时，HMAR使用可控的多步掩码生成程序，以在每个步骤中生成token的子集。

Result: 在ImageNet 256x256和512x512基准测试中，HMAR模型与参数匹配的VAR、扩散模型和自回归模型相匹配或优于它们。我们开发了高效的IO感知块稀疏注意内核，使HMAR的训练和推理时间比VAR分别快2.5倍和1.75倍以上，并且推理内存占用降低了3倍以上。最后，HMAR比VAR更灵活；它的采样计划可以在没有进一步训练的情况下改变，并且它可以以零样本的方式应用于图像编辑任务。

Conclusion: HMAR模型在ImageNet 256x256和512x512基准测试中，性能与参数匹配的VAR、扩散模型和自回归模型相当或更优，训练和推理速度分别比VAR快2.5倍和1.75倍以上，并且推理内存占用降低了3倍以上。此外，HMAR还比VAR更灵活，无需进一步训练即可更改其采样计划，并且可以以零样本方式应用于图像编辑任务。

Abstract: 视觉自回归建模（VAR）在弥合自回归图像模型和扩散模型之间的速度和质量差距方面显示出前景。VAR通过将图像分解为连续的分辨率尺度来重新制定自回归建模。在推理过程中，通过预测下一个（更高分辨率）尺度中的所有token来生成图像，该预测以所有先前（较低分辨率）尺度中的所有token为条件。然而，由于分辨率尺度中所有token的并行生成，这种公式导致图像质量降低；序列长度以超线性方式缩放图像分辨率；并且需要重新训练才能改变采样计划。 我们引入了分层掩码自回归建模（HMAR），这是一种新的图像生成算法，它使用下一尺度预测和掩码预测来缓解这些问题，从而以快速采样生成高质量图像。HMAR将下一尺度预测重新定义为马尔可夫过程，其中每个分辨率尺度的预测仅以其直接前置分辨率中的token为条件，而不是以所有前置分辨率中的token为条件。在预测分辨率尺度时，HMAR使用可控的多步掩码生成程序，以在每个步骤中生成token的子集。在ImageNet 256x256和512x512基准测试中，HMAR模型与参数匹配的VAR、扩散和自回归基线模型相匹配或优于它们。我们开发了高效的IO感知块稀疏注意内核，使HMAR的训练和推理时间比VAR分别快2.5倍和1.75倍以上，并且推理内存占用降低了3倍以上。最后，HMAR比VAR更灵活；它的采样计划可以在没有进一步训练的情况下改变，并且它可以以零样本的方式应用于图像编辑任务。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04421) | **Categories:** cs.CV, cs.AI, cs.LG

---

### [11] [Photoreal Scene Reconstruction from an Egocentric Device](https://arxiv.org/abs/2506.04444)
*Zhaoyang Lv, Maurizio Monge, Ka Chen, Yufeng Zhu, Michael Goesele, Jakob Engel, Zhao Dong, Richard Newcombe*

Main category: cs.CV

TL;DR: 本文提出了一种结合视觉惯性束调整和物理图像形成模型的高斯溅射方法，用于提高第一人称视角设备高动态范围场景三维重建的质量。


<details>
  <summary>Details</summary>
Motivation: 现有的使用第一人称视角设备进行高动态范围场景三维重建的方法忽略了像素级精确重建所需的关键细节，并且通常假设RGB相机为全局快门相机，忽略了滚动快门效应。

Method: 提出了一种新的基于高斯溅射的三维重建方法，该方法结合了视觉惯性束调整(VIBA)和基于物理的图像形成模型，以解决滚动快门相机的时间戳和运动校准问题，并处理高动态范围。

Result: 在各种室内和室外光照条件下，使用Project Aria设备和Meta Quest3设备进行的大量实验表明，结合VIBA可以稳定地提高1 dB的PSNR，而使用提出的图像形成模型可以额外提高1 dB。

Conclusion: 通过结合视觉惯性束调整(VIBA)和基于物理的图像形成模型，该方法能够显著提高高动态范围场景下使用第一人称视角设备进行三维重建的质量。

Abstract: 本文研究了使用第一人称视角设备进行高动态范围场景照片级真实感重建所面临的挑战。现有方法通常假设使用设备视觉惯性里程计系统估计的帧率6DoF位姿，这可能会忽略像素级精确重建所需的关键细节。本研究提出了两个重要的发现。首先，与将RGB相机视为全局快门相机的传统方法不同，我们强调了使用视觉惯性束调整(VIBA)来校准滚动快门RGB传感相机在高速率轨迹格式下的精确时间戳和运动的重要性，这确保了滚动快门相机的物理特性的精确校准。其次，我们将基于物理的图像形成模型融入到高斯溅射中，有效地解决了传感器特性，包括RGB相机的滚动快门效应和传感器测量的动态范围。我们提出的公式适用于广泛使用的高斯溅射表示的变体。我们使用开源的Project Aria设备在各种室内和室外照明条件下对我们的流程进行了全面的评估，并在Meta Quest3设备上进一步验证了它。在所有实验中，我们观察到，通过结合VIBA，视觉效果稳定地提高了+1 dB PSNR，而通过我们提出的图像形成模型，视觉效果又提高了+1 dB。我们的完整实现、评估数据集和记录配置文件可在http://www.projectaria.com/photoreal-reconstruction/获得。

</details>

[**[PDF]**](https://arxiv.org/pdf/2506.04444) | **Categories:** cs.CV, cs.AI, cs.GR, cs.HC, cs.MM

---
