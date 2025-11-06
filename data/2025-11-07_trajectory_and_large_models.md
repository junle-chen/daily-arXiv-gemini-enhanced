# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-07

## 目录

- [astro-ph.EP (1)](#astro-ph-ep)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (6)](#cs-ro)
- [cs.SE (1)](#cs-se)

## astro-ph.EP [astro-ph.EP]
### [1] [Optimizing Earth-Moon Transfer and Cislunar Navigation: Integrating Low-Energy Trajectories, AI Techniques and GNSS-R Technologies](https://arxiv.org/abs/2511.03173)
*Arsalan Muhammad, Wasiu Akande Ahmed, Omada Friday Ojonugwa, Paul Puspendu Biswas*

Main category: astro-ph.EP

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rapid growth of cislunar activities, including lunar landings, the Lunar Gateway, and in-space refueling stations, requires advances in cost-efficient trajectory design and reliable integration of navigation and remote sensing. Traditional Earth-Moon transfers suffer from rigid launch windows and high propellant demands, while Earth-based GNSS systems provide little to no coverage beyond geostationary orbit. This limits autonomy and environmental awareness in cislunar space. This review compares four major transfer strategies by evaluating velocity requirements, flight durations, and fuel efficiency, and by identifying their suitability for both crewed and robotic missions. The emerging role of artificial intelligence and machine learning is highlighted: convolutional neural networks support automated crater recognition and digital terrain model generation, while deep reinforcement learning enables adaptive trajectory refinement during descent and landing to reduce risk and decision latency. The study also examines how GNSS-Reflectometry and advanced Positioning, Navigation, and Timing architectures can extend navigation capabilities beyond current limits. GNSS-R can act as a bistatic radar for mapping lunar ice, soil properties, and surface topography, while PNT systems support autonomous rendezvous, Lagrange point station-keeping, and coordinated satellite swarm operations. Combining these developments establishes a scalable framework for sustainable cislunar exploration and long-term human and robotic presence.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03173) | **Categories:** astro-ph.EP, cs.AI, cs.LG, cs.RO

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Multi-Object Tracking Retrieval with LLaVA-Video: A Training-Free Solution to MOT25-StAG Challenge](https://arxiv.org/abs/2511.03332)
*Yi Yang, Yiming Xu, Timo Kaiser, Hao Cheng, Bodo Rosenhahn, Michael Ying Yang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this report, we present our solution to the MOT25-Spatiotemporal Action Grounding (MOT25-StAG) Challenge. The aim of this challenge is to accurately localize and track multiple objects that match specific and free-form language queries, using video data of complex real-world scenes as input. We model the underlying task as a video retrieval problem and present a two-stage, zero-shot approach, combining the advantages of the SOTA tracking model FastTracker and Multi-modal Large Language Model LLaVA-Video. On the MOT25-StAG test set, our method achieves m-HIoU and HOTA scores of 20.68 and 10.73 respectively, which won second place in the challenge.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03332) | **Categories:** cs.CV

---

### [2] [SurgAnt-ViVQA: Learning to Anticipate Surgical Events through GRU-Driven Temporal Cross-Attention](https://arxiv.org/abs/2511.03178)
*Shreyas C. Dhake, Jiayuan Huang, Runlong He, Danyal Z. Khan, Evangelos B. Mazomenos, Sophia Bano, Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarak I. Hoque*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Anticipating forthcoming surgical events is vital for real-time assistance in endonasal transsphenoidal pituitary surgery, where visibility is limited and workflow changes rapidly. Most visual question answering (VQA) systems reason on isolated frames with static vision language alignment, providing little support for forecasting next steps or instrument needs. Existing surgical VQA datasets likewise center on the current scene rather than the near future. We introduce PitVQA-Anticipation, the first VQA dataset designed for forward looking surgical reasoning. It comprises 33.5 hours of operative video and 734,769 question answer pairs built from temporally grouped clips and expert annotations across four tasks: predicting the future phase, next step, upcoming instrument, and remaining duration. We further propose SurgAnt-ViVQA, a video language model that adapts a large language model using a GRU Gated Temporal Cross-Attention module. A bidirectional GRU encodes frame to frame dynamics, while an adaptive gate injects visual context into the language stream at the token level. Parameter efficient fine tuning customizes the language backbone to the surgical domain. SurgAnt-ViVQA tested upon on PitVQA-Anticipation and EndoVis datasets, surpassing strong image and video based baselines. Ablations show that temporal recurrence and gated fusion drive most of the gains. A frame budget study indicates a trade-off: 8 frames maximize fluency, whereas 32 frames slightly reduce BLEU but improve numeric time estimation. By pairing a temporally aware encoder with fine grained gated cross-attention, SurgAnt-ViVQA advances surgical VQA from retrospective description to proactive anticipation. PitVQA-Anticipation offers a comprehensive benchmark for this setting and highlights the importance of targeted temporal modeling for reliable, future aware surgical assistance.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03178) | **Categories:** cs.CV

---

### [3] [SurgViVQA: Temporally-Grounded Video Question Answering for Surgical Scene Understanding](https://arxiv.org/abs/2511.03325)
*Mauro Orazio Drago, Luca Carlini, Pelinsu Celebi Balyemez, Dennis Pierantozzi, Chiara Lena, Cesare Hassan, Danail Stoyanov, Elena De Momi, Sophia Bano, Mobarak I. Hoque*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Video Question Answering (VideoQA) in the surgical domain aims to enhance intraoperative understanding by enabling AI models to reason over temporally coherent events rather than isolated frames. Current approaches are limited to static image features, and available datasets often lack temporal annotations, ignoring the dynamics critical for accurate procedural interpretation. We propose SurgViVQA, a surgical VideoQA model that extends visual reasoning from static images to dynamic surgical scenes. It uses a Masked Video--Text Encoder to fuse video and question features, capturing temporal cues such as motion and tool--tissue interactions, which a fine-tuned large language model (LLM) then decodes into coherent answers. To evaluate its performance, we curated REAL-Colon-VQA, a colonoscopic video dataset that includes motion-related questions and diagnostic attributes, as well as out-of-template questions with rephrased or semantically altered formulations to assess model robustness. Experimental validation on REAL-Colon-VQA and the public EndoVis18-VQA dataset shows that SurgViVQA outperforms existing image-based VQA benchmark models, particularly in keyword accuracy, improving over PitVQA by +11\% on REAL-Colon-VQA and +9\% on EndoVis18-VQA. A perturbation study on the questions further confirms improved generalizability and robustness to variations in question phrasing. SurgViVQA and the REAL-Colon-VQA dataset provide a framework for temporally-aware understanding in surgical VideoQA, enabling AI models to interpret dynamic procedural contexts more effectively. Code and dataset available at https://github.com/madratak/SurgViVQA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03325) | **Categories:** cs.CV

---

### [4] [Disentangled Concepts Speak Louder Than Words:Explainable Video Action Recognition](https://arxiv.org/abs/2511.03725)
*Jongseo Lee, Wooil Lee, Gyeong-Moon Park, Seong Tae Kim, Jinwoo Choi*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective explanations of video action recognition models should disentangle how movements unfold over time from the surrounding spatial context. However, existing methods based on saliency produce entangled explanations, making it unclear whether predictions rely on motion or spatial context. Language-based approaches offer structure but often fail to explain motions due to their tacit nature -- intuitively understood but difficult to verbalize. To address these challenges, we propose Disentangled Action aNd Context concept-based Explainable (DANCE) video action recognition, a framework that predicts actions through disentangled concept types: motion dynamics, objects, and scenes. We define motion dynamics concepts as human pose sequences. We employ a large language model to automatically extract object and scene concepts. Built on an ante-hoc concept bottleneck design, DANCE enforces prediction through these concepts. Experiments on four datasets -- KTH, Penn Action, HAA500, and UCF-101 -- demonstrate that DANCE significantly improves explanation clarity with competitive performance. We validate the superior interpretability of DANCE through a user study. Experimental results also show that DANCE is beneficial for model debugging, editing, and failure analysis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03725) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276)
*Jinjie Ni, Qian Liu, Longxu Dou, Chao Du, Zili Wang, Hang Yan, Tianyu Pang, Michael Qizhe Shieh*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Under strictly controlled pre-training settings, we observe a Crossover: when unique data is limited, diffusion language models (DLMs) consistently surpass autoregressive (AR) models by training for more epochs. The crossover shifts later with more or higher-quality data, earlier with larger models, and persists across dense and sparse architectures. We attribute the gains to three compounding factors: (1) any-order modeling, (2) super-dense compute from iterative bidirectional denoising, and (3) built-in Monte Carlo augmentation; input or parameter noise improves AR under data constraint but cannot close the gap. At scale, a 1.7B DLM trained with a ~1.5T-token compute budget on 10B unique Python tokens overtakes an AR coder trained with strictly matched settings. In addition, a 1B-parameter DLM achieves > 56% accuracy on HellaSwag and > 33% on MMLU using only 1B tokens, without any special tricks, just by repeating standard pre-training data. We also show that rising validation cross-entropy does not imply degraded downstream performance in this regime.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03276) | **Categories:** cs.LG

---

### [2] [A Modular, Data-Free Pipeline for Multi-Label Intention Recognition in Transportation Agentic AI Applications](https://arxiv.org/abs/2511.03363)
*Xiaocai Zhang, Hur Lim, Ke Wang, Zhe Xiao, Jing Wang, Kelvin Lee, Xiuju Fu, Zheng Qin*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this study, a modular, data-free pipeline for multi-label intention recognition is proposed for agentic AI applications in transportation. Unlike traditional intent recognition systems that depend on large, annotated corpora and often struggle with fine-grained, multi-label discrimination, our approach eliminates the need for costly data collection while enhancing the accuracy of multi-label intention understanding. Specifically, the overall pipeline, named DMTC, consists of three steps: 1) using prompt engineering to guide large language models (LLMs) to generate diverse synthetic queries in different transport scenarios; 2) encoding each textual query with a Sentence-T5 model to obtain compact semantic embeddings; 3) training a lightweight classifier using a novel online focal-contrastive (OFC) loss that emphasizes hard samples and maximizes inter-class separability. The applicability of the proposed pipeline is demonstrated in an agentic AI application in the maritime transportation context. Extensive experiments show that DMTC achieves a Hamming loss of 5.35% and an AUC of 95.92%, outperforming state-of-the-art multi-label classifiers and recent end-to-end SOTA LLM-based baselines. Further analysis reveals that Sentence-T5 embeddings improve subset accuracy by at least 3.29% over alternative encoders, and integrating the OFC loss yields an additional 0.98% gain compared to standard contrastive objectives. In conclusion, our system seamlessly routes user queries to task-specific modules (e.g., ETA information, traffic risk evaluation, and other typical scenarios in the transportation domain), laying the groundwork for fully autonomous, intention-aware agents without costly manual labelling.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03363) | **Categories:** cs.LG

---

### [3] [From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation](https://arxiv.org/abs/2511.03128)
*Najrin Sultana, Md Rafi Ur Rashid, Kang Gu, Shagufta Mehnaz*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at https://github.com/Shukti042/AdversarialExample.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03128) | **Categories:** cs.LG, cs.CL

---


## 机器人学 (Robotics) [cs.RO]
### [1] [ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications](https://arxiv.org/abs/2511.03497)
*Lei Fu, Sahar Salimpour, Leonardo Militano, Harry Edelman, Jorge Peña Queralta, Giovanni Toffetti*

Main category: cs.RO

TL;DR: 本文介绍了一个用于分析ROS和ROS 2 bags的MCP服务器，通过LLMs和VLMs实现用自然语言分析、可视化和处理机器人数据。


<details>
  <summary>Details</summary>
Motivation: 缺乏对Agentic Embodied AI交叉领域的研究，需要工具来支持机器人数据的自然语言分析。

Method: 构建了一个MCP服务器，用于分析ROS和ROS 2 bags，并提供了一个轻量级的UI，用于使用不同的LLMs对工具进行基准测试。

Result: 实验结果表明，不同LLM/VLM模型的工具调用能力存在很大差异，Kimi K2和Claude Sonnet 4表现出明显更优越的性能。工具描述模式、参数数量和可用工具数量等多种因素会影响成功率。

Conclusion: 工具调用能力存在很大差异，多种因素影响成功率。

Abstract: 本文介绍了一个用于分析ROS和ROS 2 bags的MCP服务器，该服务器允许通过LLMs和VLMs使用自然语言分析、可视化和处理机器人数据。我们描述了使用机器人领域知识构建的特定工具，我们的初始版本侧重于移动机器人，并原生支持轨迹、激光扫描数据、变换或时间序列数据的分析。此外，还提供了一个与标准ROS 2 CLI工具（“ros2 bag list”或“ros2 bag info”）的接口，以及使用主题子集过滤bags或在时间上修剪bags的功能。结合MCP服务器，我们提供了一个轻量级UI，允许使用不同的LLM（专有的（Anthropic、OpenAI）和开源的（通过Groq））对工具进行基准测试。我们的实验结果包括对八种不同的最先进的LLM/VLM模型（包括专有的和开源的、大型的和小型）的工具调用能力分析。我们的实验表明，工具调用能力存在很大差异，Kimi K2和Claude Sonnet 4表现出明显更优越的性能。我们还得出结论，从工具描述模式到参数数量，以及模型可用的工具数量，都有多种因素影响成功率。该代码以宽松的许可证在https://github.com/binabik-ai/mcp-rosbags上提供。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03497) | **Categories:** cs.RO, cs.AI, cs.SE

---

### [2] [SENT Map -- Semantically Enhanced Topological Maps with Foundation Models](https://arxiv.org/abs/2511.03165)
*Raj Surya Rajendran Kathirvel, Zach A Chavis, Stephen J. Guy, Karthik Desingh*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We introduce SENT-Map, a semantically enhanced topological map for representing indoor environments, designed to support autonomous navigation and manipulation by leveraging advancements in foundational models (FMs). Through representing the environment in a JSON text format, we enable semantic information to be added and edited in a format that both humans and FMs understand, while grounding the robot to existing nodes during planning to avoid infeasible states during deployment. Our proposed framework employs a two stage approach, first mapping the environment alongside an operator with a Vision-FM, then using the SENT-Map representation alongside a natural-language query within an FM for planning. Our experimental results show that semantic-enhancement enables even small locally-deployable FMs to successfully plan over indoor environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03165) | **Categories:** cs.RO

---

### [3] [Learning-based Cooperative Robotic Paper Wrapping: A Unified Control Policy with Residual Force Control](https://arxiv.org/abs/2511.03181)
*Rewida Ali, Cristian C. Beltran-Hernandez, Weiwei Wan, Kensuke Harada*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human-robot cooperation is essential in environments such as warehouses and retail stores, where workers frequently handle deformable objects like paper, bags, and fabrics. Coordinating robotic actions with human assistance remains difficult due to the unpredictable dynamics of deformable materials and the need for adaptive force control. To explore this challenge, we focus on the task of gift wrapping, which exemplifies a long-horizon manipulation problem involving precise folding, controlled creasing, and secure fixation of paper. Success is achieved when the robot completes the sequence to produce a neatly wrapped package with clean folds and no tears.   We propose a learning-based framework that integrates a high-level task planner powered by a large language model (LLM) with a low-level hybrid imitation learning (IL) and reinforcement learning (RL) policy. At its core is a Sub-task Aware Robotic Transformer (START) that learns a unified policy from human demonstrations. The key novelty lies in capturing long-range temporal dependencies across the full wrapping sequence within a single model. Unlike vanilla Action Chunking with Transformer (ACT), typically applied to short tasks, our method introduces sub-task IDs that provide explicit temporal grounding. This enables robust performance across the entire wrapping process and supports flexible execution, as the policy learns sub-goals rather than merely replicating motion sequences.   Our framework achieves a 97% success rate on real-world wrapping tasks. We show that the unified transformer-based policy reduces the need for specialized models, allows controlled human supervision, and effectively bridges high-level intent with the fine-grained force control required for deformable object manipulation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03181) | **Categories:** cs.RO, cs.LG

---

### [4] [GUIDES: Guidance Using Instructor-Distilled Embeddings for Pre-trained Robot Policy Enhancement](https://arxiv.org/abs/2511.03400)
*Minquan Gao, Xinyi Li, Qing Yan, Xiaojian Sun, Xiaopan Zhang, Chien-Ming Huang, Jiachen Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Pre-trained robot policies serve as the foundation of many validated robotic systems, which encapsulate extensive embodied knowledge. However, they often lack the semantic awareness characteristic of foundation models, and replacing them entirely is impractical in many situations due to high costs and the loss of accumulated knowledge. To address this gap, we introduce GUIDES, a lightweight framework that augments pre-trained policies with semantic guidance from foundation models without requiring architectural redesign. GUIDES employs a fine-tuned vision-language model (Instructor) to generate contextual instructions, which are encoded by an auxiliary module into guidance embeddings. These embeddings are injected into the policy's latent space, allowing the legacy model to adapt to this new semantic input through brief, targeted fine-tuning. For inference-time robustness, a large language model-based Reflector monitors the Instructor's confidence and, when confidence is low, initiates a reasoning loop that analyzes execution history, retrieves relevant examples, and augments the VLM's context to refine subsequent actions. Extensive validation in the RoboCasa simulation environment across diverse policy architectures shows consistent and substantial improvements in task success rates. Real-world deployment on a UR5 robot further demonstrates that GUIDES enhances motion precision for critical sub-tasks such as grasping. Overall, GUIDES offers a practical and resource-efficient pathway to upgrade, rather than replace, validated robot policies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03400) | **Categories:** cs.RO

---

### [5] [Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning](https://arxiv.org/abs/2511.03591)
*Qingyi Chen, Ruiqi Ni, Jun Kim, Ahmed H. Qureshi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Safe multi-agent motion planning (MAMP) under task-induced constraints is a critical challenge in robotics. Many real-world scenarios require robots to navigate dynamic environments while adhering to manifold constraints imposed by tasks. For example, service robots must carry cups upright while avoiding collisions with humans or other robots. Despite recent advances in decentralized MAMP for high-dimensional systems, incorporating manifold constraints remains difficult. To address this, we propose a manifold-constrained Hamilton-Jacobi reachability (HJR) learning framework for decentralized MAMP. Our method solves HJR problems under manifold constraints to capture task-aware safety conditions, which are then integrated into a decentralized trajectory optimization planner. This enables robots to generate motion plans that are both safe and task-feasible without requiring assumptions about other agents' policies. Our approach generalizes across diverse manifold-constrained tasks and scales effectively to high-dimensional multi-agent manipulation problems. Experiments show that our method outperforms existing constrained motion planners and operates at speeds suitable for real-world applications. Video demonstrations are available at https://youtu.be/RYcEHMnPTH8 .

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03591) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [6] [Flying Robotics Art: ROS-based Drone Draws the Record-Breaking Mural](https://arxiv.org/abs/2511.03651)
*Andrei A. Korigodskii, Oleg D. Kalachev, Artem E. Vasiunik, Matvei V. Urvantsev, Georgii E. Bondar*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents the innovative design and successful deployment of a pioneering autonomous unmanned aerial system developed for executing the world's largest mural painted by a drone. Addressing the dual challenges of maintaining artistic precision and operational reliability under adverse outdoor conditions such as wind and direct sunlight, our work introduces a robust system capable of navigating and painting outdoors with unprecedented accuracy. Key to our approach is a novel navigation system that combines an infrared (IR) motion capture camera and LiDAR technology, enabling precise location tracking tailored specifically for largescale artistic applications. We employ a unique control architecture that uses different regulation in tangential and normal directions relative to the planned path, enabling precise trajectory tracking and stable line rendering. We also present algorithms for trajectory planning and path optimization, allowing for complex curve drawing and area filling. The system includes a custom-designed paint spraying mechanism, specifically engineered to function effectively amidst the turbulent airflow generated by the drone's propellers, which also protects the drone's critical components from paint-related damage, ensuring longevity and consistent performance. Experimental results demonstrate the system's robustness and precision in varied conditions, showcasing its potential for autonomous large-scale art creation and expanding the functional applications of robotics in creative fields.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.03651) | **Categories:** cs.RO, cs.CV, cs.SY, eess.SY, I.2.9; J.5

---


## cs.SE [cs.SE]
### [1] [LM-Fix: Lightweight Bit-Flip Detection and Rapid Recovery Framework for Language Models](https://arxiv.org/abs/2511.02866)
*Ahmad Tahmasivand, Noureldin Zahran, Saba Al-Sayouri, Mohammed Fouda, Khaled N. Khasawneh*

Main category: cs.SE

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents LM-Fix, a lightweight detection and rapid recovery framework for faults in large language models (LLMs). Existing integrity approaches are often heavy or slow for modern LLMs. LM-Fix runs a short test-vector pass and uses hash-guided checks to detect bit-flip faults, then repairs them locally without a full reload. Across multiple models, it detects over 94% of single-bit flips at TVL=200 and nearly 100% of multi-bit flips with approximately 1% to 7.7% runtime overhead; recovery is more than 100x faster than reloading. These results show a practical, low-overhead solution to keep LLMs reliable in production

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.02866) | **Categories:** cs.SE, cs.AI, cs.AR, cs.CR

---
