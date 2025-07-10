# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-07-11

## 目录

- [人工智能 (Artificial Intelligence) (1)](#cs-ai)
- [计算机视觉 (Computer Vision) (6)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (5)](#cs-ro)
- [eess.SY (1)](#eess-sy)
- [q-bio.QM (1)](#q-bio-qm)
- [stat.AP (1)](#stat-ap)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation](https://arxiv.org/abs/2507.06993)
*Jieren Deng, Aleksandar Cvetkovic, Pak Kiu Chung, Dragomir Yankov, Chiqun Zhang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traditional travel-planning systems are often static and fragmented, leaving them ill-equipped to handle real-world complexities such as evolving environmental conditions and unexpected itinerary disruptions. In this paper, we identify three gaps between existing service providers causing frustrating user experience: intelligent trip planning, precision "last-100-meter" navigation, and dynamic itinerary adaptation. We propose three cooperative agents: a Travel Planning Agent that employs grid-based spatial grounding and map analysis to help resolve complex multi-modal user queries; a Destination Assistant Agent that provides fine-grained guidance for the final navigation leg of each journey; and a Local Discovery Agent that leverages image embeddings and Retrieval-Augmented Generation (RAG) to detect and respond to trip plan disruptions. With evaluations and experiments, our system demonstrates substantial improvements in query interpretation, navigation accuracy, and disruption resilience, underscoring its promise for applications from urban exploration to emergency response.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06993) | **Categories:** cs.AI, cs.CV

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting](https://arxiv.org/abs/2507.05116)
*Juyi Lin, Amir Taherin, Arash Akbari, Arman Akbari, Lei Lu, Guangyu Chen, Taskin Padir, Xiaomeng Yang, Weiwei Chen, Yiqian Li, Xue Lin, David Kaeli, Pu Zhao, Yanzhi Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent large-scale Vision Language Action (VLA) models have shown superior performance in robotic manipulation tasks guided by natural language. However, their generalization remains limited when applied to novel objects or unfamiliar environments that lie outside the training distribution. To address this, many existing approaches integrate additional components such as depth estimation, segmentation, or even diffusion to improve generalization, at the cost of adding significant computation overhead, resulting in low efficiency. This motivates the exploration of efficient action prediction methods, which are independent of additional high-level visual representations or diffusion techniques. In this work, we propose VOTE, an efficient and general framework for the optimization and acceleration of VLA models. In details, we propose a novel tokenizer-free fine-tuning approach for parallel accurate action prediction, which reduces computational overhead and accelerates inference speed. Additionally, we adopt an ensemble voting strategy for the action sampling, which significantly improves model performance and enhances generalization. Experimental results show that our method achieves state-of-the-art performance with 35$\times$ faster inference and 145 Hz throughput. All the details and codes will be open-sourced.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.05116) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [ILNet: Trajectory Prediction with Inverse Learning Attention for Enhancing Intention Capture](https://arxiv.org/abs/2507.06531)
*Mingjin Zeng, Nan Ouyang, Wenkang Wan, Lei Ao, Qing Cai, Kai Sheng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction for multi-agent interaction scenarios is a crucial challenge. Most advanced methods model agent interactions by efficiently factorized attention based on the temporal and agent axes. However, this static and foward modeling lacks explicit interactive spatio-temporal coordination, capturing only obvious and immediate behavioral intentions. Alternatively, the modern trajectory prediction framework refines the successive predictions by a fixed-anchor selection strategy, which is difficult to adapt in different future environments. It is acknowledged that human drivers dynamically adjust initial driving decisions based on further assumptions about the intentions of surrounding vehicles. Motivated by human driving behaviors, this paper proposes ILNet, a multi-agent trajectory prediction method with Inverse Learning (IL) attention and Dynamic Anchor Selection (DAS) module. IL Attention employs an inverse learning paradigm to model interactions at neighboring moments, introducing proposed intentions to dynamically encode the spatio-temporal coordination of interactions, thereby enhancing the model's ability to capture complex interaction patterns. Then, the learnable DAS module is proposed to extract multiple trajectory change keypoints as anchors in parallel with almost no increase in parameters. Experimental results show that the ILNet achieves state-of-the-art performance on the INTERACTION and Argoverse motion forecasting datasets. Particularly, in challenged interaction scenarios, ILNet achieves higher accuracy and more multimodal distributions of trajectories over fewer parameters. Our codes are available at https://github.com/mjZeng11/ILNet.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06531) | **Categories:** cs.CV

---

### [3] [Physics-Grounded Motion Forecasting via Equation Discovery for Trajectory-Guided Image-to-Video Generation](https://arxiv.org/abs/2507.06830)
*Tao Feng, Xianbing Zhao, Zhenhua Chen, Tien Tsin Wong, Hamid Rezatofighi, Gholamreza Haffari, Lizhen Qu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in diffusion-based and autoregressive video generation models have achieved remarkable visual realism. However, these models typically lack accurate physical alignment, failing to replicate real-world dynamics in object motion. This limitation arises primarily from their reliance on learned statistical correlations rather than capturing mechanisms adhering to physical laws. To address this issue, we introduce a novel framework that integrates symbolic regression (SR) and trajectory-guided image-to-video (I2V) models for physics-grounded video forecasting. Our approach extracts motion trajectories from input videos, uses a retrieval-based pre-training mechanism to enhance symbolic regression, and discovers equations of motion to forecast physically accurate future trajectories. These trajectories then guide video generation without requiring fine-tuning of existing models. Evaluated on scenarios in Classical Mechanics, including spring-mass, pendulums, and projectile motions, our method successfully recovers ground-truth analytical equations and improves the physical alignment of generated videos over baseline methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06830) | **Categories:** cs.CV, cs.AI

---

### [4] [Token Bottleneck: One Token to Remember Dynamics](https://arxiv.org/abs/2507.06543)
*Taekyung Kim, Dongyoon Han, Byeongho Heo, Jeongeun Park, Sangdoo Yun*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Deriving compact and temporally aware visual representations from dynamic scenes is essential for successful execution of sequential scene understanding tasks such as visual tracking and robotic manipulation. In this paper, we introduce Token Bottleneck (ToBo), a simple yet intuitive self-supervised learning pipeline that squeezes a scene into a bottleneck token and predicts the subsequent scene using minimal patches as hints. The ToBo pipeline facilitates the learning of sequential scene representations by conservatively encoding the reference scene into a compact bottleneck token during the squeeze step. In the expansion step, we guide the model to capture temporal dynamics by predicting the target scene using the bottleneck token along with few target patches as hints. This design encourages the vision backbone to embed temporal dependencies, thereby enabling understanding of dynamic transitions across scenes. Extensive experiments in diverse sequential tasks, including video label propagation and robot manipulation in simulated environments demonstrate the superiority of ToBo over baselines. Moreover, deploying our pre-trained model on physical robots confirms its robustness and effectiveness in real-world environments. We further validate the scalability of ToBo across different model scales.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06543) | **Categories:** cs.CV

---

### [5] [MOST: Motion Diffusion Model for Rare Text via Temporal Clip Banzhaf Interaction](https://arxiv.org/abs/2507.06590)
*Yin Wang, Mu li, Zhiying Leng, Frederick W. B. Li, Xiaohui Liang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce MOST, a novel motion diffusion model via temporal clip Banzhaf interaction, aimed at addressing the persistent challenge of generating human motion from rare language prompts. While previous approaches struggle with coarse-grained matching and overlook important semantic cues due to motion redundancy, our key insight lies in leveraging fine-grained clip relationships to mitigate these issues. MOST's retrieval stage presents the first formulation of its kind - temporal clip Banzhaf interaction - which precisely quantifies textual-motion coherence at the clip level. This facilitates direct, fine-grained text-to-motion clip matching and eliminates prevalent redundancy. In the generation stage, a motion prompt module effectively utilizes retrieved motion clips to produce semantically consistent movements. Extensive evaluations confirm that MOST achieves state-of-the-art text-to-motion retrieval and generation performance by comprehensively addressing previous challenges, as demonstrated through quantitative and qualitative results highlighting its effectiveness, especially for rare prompts.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06590) | **Categories:** cs.CV

---

### [6] [A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding](https://arxiv.org/abs/2507.06719)
*Zhenyang Liu, Sixiao Zheng, Siyu Chen, Cairong Zhao, Longfei Liang, Xiangyang Xue, Yanwei Fu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06719) | **Categories:** cs.CV, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](https://arxiv.org/abs/2507.06952)
*Keyon Vafa, Peter G. Chang, Ashesh Rambachan, Sendhil Mullainathan*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundation models are premised on the idea that sequence prediction can uncover deeper domain understanding, much like how Kepler's predictions of planetary motion later led to the discovery of Newtonian mechanics. However, evaluating whether these models truly capture deeper structure remains a challenge. We develop a technique for evaluating foundation models that examines how they adapt to synthetic datasets generated from some postulated world model. Our technique measures whether the foundation model's inductive bias aligns with the world model, and so we refer to it as an inductive bias probe. Across multiple domains, we find that foundation models can excel at their training tasks yet fail to develop inductive biases towards the underlying world model when adapted to new tasks. We particularly find that foundation models trained on orbital trajectories consistently fail to apply Newtonian mechanics when adapted to new physics tasks. Further analysis reveals that these models behave as if they develop task-specific heuristics that fail to generalize.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06952) | **Categories:** cs.LG, cs.AI

---

### [2] [Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models](https://arxiv.org/abs/2507.06466)
*Aaron Dharna, Cong Lu, Jeff Clune*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-agent interactions have long fueled innovation, from natural predator-prey dynamics to the space race. Self-play (SP) algorithms try to harness these dynamics by pitting agents against ever-improving opponents, thereby creating an implicit curriculum toward learning high-quality solutions. However, SP often fails to produce diverse solutions and can get stuck in locally optimal behaviors. We introduce Foundation-Model Self-Play (FMSP), a new direction that leverages the code-generation capabilities and vast knowledge of foundation models (FMs) to overcome these challenges by leaping across local optima in policy space. We propose a family of approaches: (1) \textbf{Vanilla Foundation-Model Self-Play (vFMSP)} continually refines agent policies via competitive self-play; (2) \textbf{Novelty-Search Self-Play (NSSP)} builds a diverse population of strategies, ignoring performance; and (3) the most promising variant, \textbf{Quality-Diveristy Self-Play (QDSP)}, creates a diverse set of high-quality policies by combining the diversity of NSSP and refinement of vFMSP. We evaluate FMSPs in Car Tag, a continuous-control pursuer-evader setting, and in Gandalf, a simple AI safety simulation in which an attacker tries to jailbreak an LLM's defenses. In Car Tag, FMSPs explore a wide variety of reinforcement learning, tree search, and heuristic-based methods, to name just a few. In terms of discovered policy quality, \ouralgo and vFMSP surpass strong human-designed strategies. In Gandalf, FMSPs can successfully automatically red-team an LLM, breaking through and jailbreaking six different, progressively stronger levels of defense. Furthermore, FMSPs can automatically proceed to patch the discovered vulnerabilities. Overall, FMSPs represent a promising new research frontier of improving self-play with foundation models, opening fresh paths toward more creative and open-ended strategy discovery

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06466) | **Categories:** cs.LG, cs.AI

---

### [3] [MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models](https://arxiv.org/abs/2507.06502)
*Yiwen Liu, Chenyu Zhang, Junjie Song, Siqi Chen, Sun Yin, Zihan Wang, Lingming Zeng, Yuji Cao, Junming Jiao*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: As a prominent data modality task, time series forecasting plays a pivotal role in diverse applications. With the remarkable advancements in Large Language Models (LLMs), the adoption of LLMs as the foundational architecture for time series modeling has gained significant attention. Although existing models achieve some success, they rarely both model time and frequency characteristics in a pretraining-finetuning paradigm leading to suboptimal performance in predictions of complex time series, which requires both modeling periodicity and prior pattern knowledge of signals. We propose MoFE-Time, an innovative time series forecasting model that integrates time and frequency domain features within a Mixture of Experts (MoE) network. Moreover, we use the pretraining-finetuning paradigm as our training framework to effectively transfer prior pattern knowledge across pretraining and finetuning datasets with different periodicity distributions. Our method introduces both frequency and time cells as experts after attention modules and leverages the MoE routing mechanism to construct multidimensional sparse representations of input signals. In experiments on six public benchmarks, MoFE-Time has achieved new state-of-the-art performance, reducing MSE and MAE by 6.95% and 6.02% compared to the representative methods Time-MoE. Beyond the existing evaluation benchmarks, we have developed a proprietary dataset, NEV-sales, derived from real-world business scenarios. Our method achieves outstanding results on this dataset, underscoring the effectiveness of the MoFE-Time model in practical commercial applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06502) | **Categories:** cs.LG, cs.AI

---


## 机器人学 (Robotics) [cs.RO]
### [1] [SkyVLN: Vision-and-Language Navigation and NMPC Control for UAVs in Urban Environments](https://arxiv.org/abs/2507.06564)
*Tianshun Li, Tianyi Huai, Zhen Li, Yichun Gao, Haoang Li, Xinhu Zheng*

Main category: cs.RO

TL;DR: SkyVLN 是一个集成了视觉语言导航和非线性模型预测控制的新框架，旨在提高无人机在复杂城市环境中的自主性。


<details>
  <summary>Details</summary>
Motivation: 为了提高无人机在复杂城市环境中的自主性。

Method: 集成了视觉语言导航 (VLN) 与非线性模型预测控制 (NMPC) 的新框架。

Result: SkyVLN 显著提高了导航成功率和效率，尤其是在新的和未见过的环境中。

Conclusion: SkyVLN 显著提高了导航成功率和效率，尤其是在新的和未见过的环境中。

Abstract: 无人机（UAV）凭借其移动性和适应性，已成为各个领域的多功能工具。本文介绍了一种名为 SkyVLN 的新框架，该框架集成了视觉语言导航（VLN）与非线性模型预测控制（NMPC），旨在提高无人机在复杂城市环境中的自主性。与传统的导航方法不同，SkyVLN 利用大型语言模型（LLM）来解释自然语言指令和视觉观察，从而使无人机能够在动态 3D 空间中以更高的精度和鲁棒性进行导航。我们提出了一个多模态导航代理，该代理配备了细粒度的空间语言器和历史路径记忆机制。这些组件使无人机能够消除空间背景的歧义，处理模糊的指令，并在必要时进行回溯。该框架还包含一个 NMPC 模块，用于动态避障，从而确保精确的轨迹跟踪和碰撞预防。为了验证我们的方法，我们使用 AirSim 开发了一个高保真 3D 城市模拟环境，该环境具有逼真的图像和动态城市元素。大量的实验表明，SkyVLN 显着提高了导航成功率和效率，尤其是在新的和未见过的环境中。

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06564) | **Categories:** cs.RO, cs.AI, cs.SY, eess.SY

---

### [2] [LOVON: Legged Open-Vocabulary Object Navigator](https://arxiv.org/abs/2507.06747)
*Daojie Peng, Jiahang Cao, Qiang Zhang, Jun Ma*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Object navigation in open-world environments remains a formidable and pervasive challenge for robotic systems, particularly when it comes to executing long-horizon tasks that require both open-world object detection and high-level task planning. Traditional methods often struggle to integrate these components effectively, and this limits their capability to deal with complex, long-range navigation missions. In this paper, we propose LOVON, a novel framework that integrates large language models (LLMs) for hierarchical task planning with open-vocabulary visual detection models, tailored for effective long-range object navigation in dynamic, unstructured environments. To tackle real-world challenges including visual jittering, blind zones, and temporary target loss, we design dedicated solutions such as Laplacian Variance Filtering for visual stabilization. We also develop a functional execution logic for the robot that guarantees LOVON's capabilities in autonomous navigation, task adaptation, and robust task completion. Extensive evaluations demonstrate the successful completion of long-sequence tasks involving real-time detection, search, and navigation toward open-vocabulary dynamic targets. Furthermore, real-world experiments across different legged robots (Unitree Go2, B2, and H1-2) showcase the compatibility and appealing plug-and-play feature of LOVON.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06747) | **Categories:** cs.RO, cs.CV

---

### [3] [Learning to Evaluate Autonomous Behaviour in Human-Robot Interaction](https://arxiv.org/abs/2507.06404)
*Matteo Tiezzi, Tommaso Apicella, Carlos Cardenas-Perez, Giovanni Fregonese, Stefano Dafarra, Pietro Morerio, Daniele Pucci, Alessio Del Bue*

Main category: cs.RO

TL;DR: 提出了一种基于深度学习的神经元Meta评估器（NeME）来评估人形机器人模仿学习的轨迹性能，无需人工干预。


<details>
  <summary>Details</summary>
Motivation: 自主人形机器人的性能评估和比较具有挑战性，因为成功率指标难以重现，并且无法捕捉机器人运动轨迹的复杂性，这在人机交互和协作（HRIC）中至关重要。

Method: 提出了神经元Meta评估器（NeME），一种用于分类机器人关节轨迹动作的深度学习模型。

Result: 实验结果表明，该方法比基线方法更符合机器人的成功率。

Conclusion: 该研究表明，提出的NeME方法在评估机器人控制策略方面更准确，且无需人工干预，为复杂人机交互任务中多模态模仿学习方法的性能比较提供了一种可重复、系统和有见地的手段。

Abstract: 评估和比较自主人形机器人的性能具有挑战性，因为成功率指标难以重现，并且无法捕捉机器人运动轨迹的复杂性，这在人机交互和协作（HRIC）中至关重要。为了解决这些挑战，我们提出了一个通用的评估框架，通过关注轨迹性能来衡量模仿学习（IL）方法的质量。我们设计了神经元Meta评估器（NeME），一种用于分类机器人关节轨迹动作的深度学习模型。NeME作为一个meta评估器，可以比较机器人控制策略的性能，无需人工干预即可进行策略评估。我们使用远程操作数据在人形机器人ergoCub上验证了我们的框架，并比较了为可用平台量身定制的IL方法。实验结果表明，我们的方法比基线方法更符合机器人的成功率，为比较复杂人机交互任务中多模态模仿学习方法的性能提供了一种可重复、系统和有见地的手段。

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06404) | **Categories:** cs.RO, cs.CV, cs.LG

---

### [4] [Growing Trees with an Agent: Accelerating RRTs with Learned, Multi-Step Episodic Exploration](https://arxiv.org/abs/2507.06605)
*Xinyu Wu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Classical sampling-based motion planners like the RRTs suffer from inefficiencies, particularly in cluttered or high-dimensional spaces, due to their reliance on undirected, random sampling. This paper introduces the Episodic RRT, a novel hybrid planning framework that replaces the primitive of a random point with a learned, multi-step "exploratory episode" generated by a Deep Reinforcement Learning agent. By making the DRL agent the engine of exploration, ERRT transforms the search process from a diffuse, volumetric expansion into a directed, branch-like growth. This paradigm shift yields key advantages: it counters the curse of dimensionality with focused exploration, minimizes expensive collision checks by proactively proposing locally valid paths, and improves connectivity by generating inherently connected path segments. We demonstrate through extensive empirical evaluation across 2D, 3D, and 6D environments that ERRT and its variants consistently and significantly outperform their classical counterparts. In a challenging 6D robotic arm scenario, ERRT achieves a 98% success rate compared to 19% for RRT, is up to 107x faster, reduces collision checks by over 99.6%, and finds initial paths that are nearly 50% shorter. Furthermore, its asymptotically optimal variant, ERRT*, demonstrates vastly superior anytime performance, refining solutions to near-optimality up to 29x faster than standard RRT* in 3D environments. Code: https://xinyuwuu.github.io/Episodic_RRT/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06605) | **Categories:** cs.RO

---

### [5] [Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](https://arxiv.org/abs/2507.06710)
*Zhenyang Liu, Yikai Wang, Kuanning Wang, Longfei Liang, Xiangyang Xue, Yanwei Fu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Visual imitation learning is effective for robots to learn versatile tasks. However, many existing methods rely on behavior cloning with supervised historical trajectories, limiting their 3D spatial and 4D spatiotemporal awareness. Consequently, these methods struggle to capture the 3D structures and 4D spatiotemporal relationships necessary for real-world deployment. In this work, we propose 4D Diffusion Policy (DP4), a novel visual imitation learning method that incorporates spatiotemporal awareness into diffusion-based policies. Unlike traditional approaches that rely on trajectory cloning, DP4 leverages a dynamic Gaussian world model to guide the learning of 3D spatial and 4D spatiotemporal perceptions from interactive environments. Our method constructs the current 3D scene from a single-view RGB-D observation and predicts the future 3D scene, optimizing trajectory generation by explicitly modeling both spatial and temporal dependencies. Extensive experiments across 17 simulation tasks with 173 variants and 3 real-world robotic tasks demonstrate that the 4D Diffusion Policy (DP4) outperforms baseline methods, improving the average simulation task success rate by 16.4% (Adroit), 14% (DexArt), and 6.45% (RLBench), and the average real-world robotic task success rate by 8.6%.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06710) | **Categories:** cs.RO

---


## eess.SY [eess.SY]
### [1] [VisioPath: Vision-Language Enhanced Model Predictive Control for Safe Autonomous Navigation in Mixed Traffic](https://arxiv.org/abs/2507.06441)
*Shanting Wang, Panagiotis Typaldos, Chenjun Li, Andreas A. Malikopoulos*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this paper, we introduce VisioPath, a novel framework combining vision-language models (VLMs) with model predictive control (MPC) to enable safe autonomous driving in dynamic traffic environments. The proposed approach leverages a bird's-eye view video processing pipeline and zero-shot VLM capabilities to obtain structured information about surrounding vehicles, including their positions, dimensions, and velocities. Using this rich perception output, we construct elliptical collision-avoidance potential fields around other traffic participants, which are seamlessly integrated into a finite-horizon optimal control problem for trajectory planning. The resulting trajectory optimization is solved via differential dynamic programming with an adaptive regularization scheme and is embedded in an event-triggered MPC loop. To ensure collision-free motion, a safety verification layer is incorporated in the framework that provides an assessment of potential unsafe trajectories. Extensive simulations in Simulation of Urban Mobility (SUMO) demonstrate that VisioPath outperforms conventional MPC baselines across multiple metrics. By combining modern AI-driven perception with the rigorous foundation of optimal control, VisioPath represents a significant step forward in safe trajectory planning for complex traffic systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06441) | **Categories:** eess.SY, cs.RO, cs.SY

---


## q-bio.QM [q-bio.QM]
### [1] [Self-supervised learning predicts plant growth trajectories from multi-modal industrial greenhouse data](https://arxiv.org/abs/2507.06336)
*Adam J Riesselman, Evan M Cofer, Therese LaRue, Wim Meeussen*

Main category: q-bio.QM

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Quantifying organism-level phenotypes, such as growth dynamics and biomass accumulation, is fundamental to understanding agronomic traits and optimizing crop production. However, quality growing data of plants at scale is difficult to generate. Here we use a mobile robotic platform to capture high-resolution environmental sensing and phenotyping measurements of a large-scale hydroponic leafy greens system. We describe a self-supervised modeling approach to build a map from observed growing data to the entire plant growth trajectory. We demonstrate our approach by forecasting future plant height and harvest mass of crops in this system. This approach represents a significant advance in combining robotic automation and machine learning, as well as providing actionable insights for agronomic research and operational efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.06336) | **Categories:** q-bio.QM, cs.LG, cs.RO

---


## stat.AP [stat.AP]
### [1] [When Context Is Not Enough: Modeling Unexplained Variability in Car-Following Behavior](https://arxiv.org/abs/2507.07012)
*Chengyuan Zhang, Zhengbing He, Cathy Wu, Lijun Sun*

Main category: stat.AP

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modeling car-following behavior is fundamental to microscopic traffic simulation, yet traditional deterministic models often fail to capture the full extent of variability and unpredictability in human driving. While many modern approaches incorporate context-aware inputs (e.g., spacing, speed, relative speed), they frequently overlook structured stochasticity that arises from latent driver intentions, perception errors, and memory effects -- factors that are not directly observable from context alone. To fill the gap, this study introduces an interpretable stochastic modeling framework that captures not only context-dependent dynamics but also residual variability beyond what context can explain. Leveraging deep neural networks integrated with nonstationary Gaussian processes (GPs), our model employs a scenario-adaptive Gibbs kernel to learn dynamic temporal correlations in acceleration decisions, where the strength and duration of correlations between acceleration decisions evolve with the driving context. This formulation enables a principled, data-driven quantification of uncertainty in acceleration, speed, and spacing, grounded in both observable context and latent behavioral variability. Comprehensive experiments on the naturalistic vehicle trajectory dataset collected from the German highway, i.e., the HighD dataset, demonstrate that the proposed stochastic simulation method within this framework surpasses conventional methods in both predictive performance and interpretable uncertainty quantification. The integration of interpretability and accuracy makes this framework a promising tool for traffic analysis and safety-critical applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2507.07012) | **Categories:** stat.AP, cs.LG, cs.RO

---
