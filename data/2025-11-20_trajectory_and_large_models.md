# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-20

## 目录

- [人工智能 (Artificial Intelligence) (6)](#cs-ai)
- [计算机视觉 (Computer Vision) (8)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (7)](#cs-ro)
- [eess.SY (1)](#eess-sy)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Run, Ruminate, and Regulate: A Dual-process Thinking System for Vision-and-Language Navigation](https://arxiv.org/abs/2511.14131)
*Yu Zhong, Zihao Zhang, Rui Zhang, Lingdong Huang, Haihan Gao, Shuo Wang, Da Li, Ruijian Han, Jiaming Guo, Shaohui Peng, Di Huang, Yunji Chen*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-and-Language Navigation (VLN) requires an agent to dynamically explore complex 3D environments following human instructions. Recent research underscores the potential of harnessing large language models (LLMs) for VLN, given their commonsense knowledge and general reasoning capabilities. Despite their strengths, a substantial gap in task completion performance persists between LLM-based approaches and domain experts, as LLMs inherently struggle to comprehend real-world spatial correlations precisely. Additionally, introducing LLMs is accompanied with substantial computational cost and inference latency. To address these issues, we propose a novel dual-process thinking framework dubbed R3, integrating LLMs' generalization capabilities with VLN-specific expertise in a zero-shot manner. The framework comprises three core modules: Runner, Ruminator, and Regulator. The Runner is a lightweight transformer-based expert model that ensures efficient and accurate navigation under regular circumstances. The Ruminator employs a powerful multimodal LLM as the backbone and adopts chain-of-thought (CoT) prompting to elicit structured reasoning. The Regulator monitors the navigation progress and controls the appropriate thinking mode according to three criteria, integrating Runner and Ruminator harmoniously. Experimental results illustrate that R3 significantly outperforms other state-of-the-art methods, exceeding 3.28% and 3.30% in SPL and RGSPL respectively on the REVERIE benchmark. This pronounced enhancement highlights the effectiveness of our method in handling challenging VLN tasks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14131) | **Categories:** cs.AI

---

### [2] [Scene Graph-Guided Generative AI Framework for Synthesizing and Evaluating Industrial Hazard Scenarios](https://arxiv.org/abs/2511.13970)
*Sanjay Acharjee, Abir Khan Ratul, Diego Patino, Md Nazmus Sakib*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Training vision models to detect workplace hazards accurately requires realistic images of unsafe conditions that could lead to accidents. However, acquiring such datasets is difficult because capturing accident-triggering scenarios as they occur is nearly impossible. To overcome this limitation, this study presents a novel scene graph-guided generative AI framework that synthesizes photorealistic images of hazardous scenarios grounded in historical Occupational Safety and Health Administration (OSHA) accident reports. OSHA narratives are analyzed using GPT-4o to extract structured hazard reasoning, which is converted into object-level scene graphs capturing spatial and contextual relationships essential for understanding risk. These graphs guide a text-to-image diffusion model to generate compositionally accurate hazard scenes. To evaluate the realism and semantic fidelity of the generated data, a visual question answering (VQA) framework is introduced. Across four state-of-the-art generative models, the proposed VQA Graph Score outperforms CLIP and BLIP metrics based on entropy-based validation, confirming its higher discriminative sensitivity.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13970) | **Categories:** cs.AI, cs.CV

---

### [3] [HFL-FlowLLM: Large Language Models for Network Traffic Flow Classification in Heterogeneous Federated Learning](https://arxiv.org/abs/2511.14199)
*Jiazhuo Tian, Yachao Yuan*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In modern communication networks driven by 5G and the Internet of Things (IoT), effective network traffic flow classification is crucial for Quality of Service (QoS) management and security. Traditional centralized machine learning struggles with the distributed data and privacy concerns in these heterogeneous environments, while existing federated learning approaches suffer from high costs and poor generalization. To address these challenges, we propose HFL-FlowLLM, which to our knowledge is the first framework to apply large language models to network traffic flow classification in heterogeneous federated learning. Compared to state-of-the-art heterogeneous federated learning methods for network traffic flow classification, the proposed approach improves the average F1 score by approximately 13%, demonstrating compelling performance and strong robustness. When compared to existing large language models federated learning frameworks, as the number of clients participating in each training round increases, the proposed method achieves up to a 5% improvement in average F1 score while reducing the training costs by about 87%. These findings prove the potential and practical value of HFL-FlowLLM in modern communication networks security.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14199) | **Categories:** cs.AI

---

### [4] [DevPiolt: Operation Recommendation for IoT Devices at Xiaomi Home](https://arxiv.org/abs/2511.14227)
*Yuxiang Wang, Siwen Wang, Haowei Han, Ao Wang, Boya Liu, Yong Zhao, Chengbo Wu, Bin Zhu, Bin Qin, Xiaokai Zhou, Xiao Yan, Jiawei Jiang, Bo Du*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Operation recommendation for IoT devices refers to generating personalized device operations for users based on their context, such as historical operations, environment information, and device status. This task is crucial for enhancing user satisfaction and corporate profits. Existing recommendation models struggle with complex operation logic, diverse user preferences, and sensitive to suboptimal suggestions, limiting their applicability to IoT device operations. To address these issues, we propose DevPiolt, a LLM-based recommendation model for IoT device operations. Specifically, we first equip the LLM with fundamental domain knowledge of IoT operations via continual pre-training and multi-task fine-tuning. Then, we employ direct preference optimization to align the fine-tuned LLM with specific user preferences. Finally, we design a confidence-based exposure control mechanism to avoid negative user experiences from low-quality recommendations. Extensive experiments show that DevPiolt significantly outperforms baselines on all datasets, with an average improvement of 69.5% across all metrics. DevPiolt has been practically deployed in Xiaomi Home app for one quarter, providing daily operation recommendations to 255,000 users. Online experiment results indicate a 21.6% increase in unique visitor device coverage and a 29.1% increase in page view acceptance rates.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14227) | **Categories:** cs.AI, cs.LG

---

### [5] [Enhancing Regional Airbnb Trend Forecasting Using LLM-Based Embeddings of Accessibility and Human Mobility](https://arxiv.org/abs/2511.14248)
*Hongju Lee, Youngjun Park, Jisun An, Dongman Lee*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The expansion of short-term rental platforms, such as Airbnb, has significantly disrupted local housing markets, often leading to increased rental prices and housing affordability issues. Accurately forecasting regional Airbnb market trends can thus offer critical insights for policymakers and urban planners aiming to mitigate these impacts. This study proposes a novel time-series forecasting framework to predict three key Airbnb indicators -- Revenue, Reservation Days, and Number of Reservations -- at the regional level. Using a sliding-window approach, the model forecasts trends 1 to 3 months ahead. Unlike prior studies that focus on individual listings at fixed time points, our approach constructs regional representations by integrating listing features with external contextual factors such as urban accessibility and human mobility. We convert structured tabular data into prompt-based inputs for a Large Language Model (LLM), producing comprehensive regional embeddings. These embeddings are then fed into advanced time-series models (RNN, LSTM, Transformer) to better capture complex spatio-temporal dynamics. Experiments on Seoul's Airbnb dataset show that our method reduces both average RMSE and MAE by approximately 48% compared to conventional baselines, including traditional statistical and machine learning models. Our framework not only improves forecasting accuracy but also offers practical insights for detecting oversupplied regions and supporting data-driven urban policy decisions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14248) | **Categories:** cs.AI

---

### [6] [SkillGen: Learning Domain Skills for In-Context Sequential Decision Making](https://arxiv.org/abs/2511.14670)
*Ruomeng Ding, Wei Cheng, Minglai Shao, Chen Zhao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) are increasingly applied to sequential decision-making through in-context learning (ICL), yet their effectiveness is highly sensitive to prompt quality. Effective prompts should meet three principles: focus on decision-critical information, provide step-level granularity, and minimize reliance on expert annotations through label efficiency. However, existing ICL methods often fail to satisfy all three criteria simultaneously. Motivated by these challenges, we introduce SkillGen, a skill-based ICL framework for structured sequential reasoning. It constructs an action-centric, domain-level graph from sampled trajectories, identifies high-utility actions via temporal-difference credit assignment, and retrieves step-wise skills to generate fine-grained, context-aware prompts. We further present a theoretical analysis showing that focusing on high-utility segments supports task identifiability and informs more effective ICL prompt design. Experiments on ALFWorld, BabyAI, and ScienceWorld, using both open-source and proprietary LLMs, show that SkillGen achieves consistent gains, improving progress rate by 5.9%-16.5% on average across models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14670) | **Categories:** cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Multi-view Phase-aware Pedestrian-Vehicle Incident Reasoning Framework with Vision-Language Models](https://arxiv.org/abs/2511.14120)
*Hao Zhen, Yunxiang Yang, Jidong J. Yang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Pedestrian-vehicle incidents remain a critical urban safety challenge, with pedestrians accounting for over 20% of global traffic fatalities. Although existing video-based systems can detect when incidents occur, they provide little insight into how these events unfold across the distinct cognitive phases of pedestrian behavior. Recent vision-language models (VLMs) have shown strong potential for video understanding, but they remain limited in that they typically process videos in isolation, without explicit temporal structuring or multi-view integration. This paper introduces Multi-view Phase-aware Pedestrian-Vehicle Incident Reasoning (MP-PVIR), a unified framework that systematically processes multi-view video streams into structured diagnostic reports through four stages: (1) event-triggered multi-view video acquisition, (2) pedestrian behavior phase segmentation, (3) phase-specific multi-view reasoning, and (4) hierarchical synthesis and diagnostic reasoning. The framework operationalizes behavioral theory by automatically segmenting incidents into cognitive phases, performing synchronized multi-view analysis within each phase, and synthesizing results into causal chains with targeted prevention strategies. Particularly, two specialized VLMs underpin the MP-PVIR pipeline: TG-VLM for behavioral phase segmentation (mIoU = 0.4881) and PhaVR-VLM for phase-aware multi-view analysis, achieving a captioning score of 33.063 and up to 64.70% accuracy on question answering. Finally, a designated large language model is used to generate comprehensive reports detailing scene understanding, behavior interpretation, causal reasoning, and prevention recommendations. Evaluation on the Woven Traffic Safety dataset shows that MP-PVIR effectively translates multi-view video data into actionable insights, advancing AI-driven traffic safety analytics for vehicle-infrastructure cooperative systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14120) | **Categories:** cs.CV, cs.AI

---

### [2] [VLMs Guided Interpretable Decision Making for Autonomous Driving](https://arxiv.org/abs/2511.13881)
*Xin Hu, Taotao Jing, Renran Tian, Zhengming Ding*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advancements in autonomous driving (AD) have explored the use of vision-language models (VLMs) within visual question answering (VQA) frameworks for direct driving decision-making. However, these approaches often depend on handcrafted prompts and suffer from inconsistent performance, limiting their robustness and generalization in real-world scenarios. In this work, we evaluate state-of-the-art open-source VLMs on high-level decision-making tasks using ego-view visual inputs and identify critical limitations in their ability to deliver reliable, context-aware decisions. Motivated by these observations, we propose a new approach that shifts the role of VLMs from direct decision generators to semantic enhancers. Specifically, we leverage their strong general scene understanding to enrich existing vision-based benchmarks with structured, linguistically rich scene descriptions. Building on this enriched representation, we introduce a multi-modal interactive architecture that fuses visual and linguistic features for more accurate decision-making and interpretable textual explanations. Furthermore, we design a post-hoc refinement module that utilizes VLMs to enhance prediction reliability. Extensive experiments on two autonomous driving benchmarks demonstrate that our approach achieves state-of-the-art performance, offering a promising direction for integrating VLMs into reliable and interpretable AD systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13881) | **Categories:** cs.CV

---

### [3] [Enhancing End-to-End Autonomous Driving with Risk Semantic Distillaion from VLM](https://arxiv.org/abs/2511.14499)
*Jack Qin, Zhitao Wang, Yinan Zheng, Keyu Chen, Yang Zhou, Yuanxin Zhong, Siyuan Cheng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The autonomous driving (AD) system has exhibited remarkable performance in complex driving scenarios. However, generalization is still a key limitation for the current system, which refers to the ability to handle unseen scenarios or unfamiliar sensor configurations.Related works have explored the use of Vision-Language Models (VLMs) to address few-shot or zero-shot tasks. While promising, these methods introduce a new challenge: the emergence of a hybrid AD system, where two distinct systems are used to plan a trajectory, leading to potential inconsistencies. Alternative research directions have explored Vision-Language-Action (VLA) frameworks that generate control actions from VLM directly. However, these end-to-end solutions demonstrate prohibitive computational demands. To overcome these challenges, we introduce Risk Semantic Distillation (RSD), a novel framework that leverages VLMs to enhance the training of End-to-End (E2E) AD backbones. By providing risk attention for key objects, RSD addresses the issue of generalization. Specifically, we introduce RiskHead, a plug-in module that distills causal risk estimates from Vision-Language Models into Bird's-Eye-View (BEV) features, yielding interpretable risk-attention maps.This approach allows BEV features to learn richer and more nuanced risk attention representations, which directly enhance the model's ability to handle spatial boundaries and risky objects.By focusing on risk attention, RSD aligns better with human-like driving behavior, which is essential to navigate in complex and dynamic environments. Our experiments on the Bench2Drive benchmark demonstrate the effectiveness of RSD in managing complex and unpredictable driving conditions. Due to the enhanced BEV representations enabled by RSD, we observed a significant improvement in both perception and planning capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14499) | **Categories:** cs.CV, cs.RO

---

### [4] [A Trajectory-free Crash Detection Framework with Generative Approach and Segment Map Diffusion](https://arxiv.org/abs/2511.13795)
*Weiying Shen, Hao Yu, Yu Dong, Pan Liu, Yu Han, Xin Wen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Real-time crash detection is essential for developing proactive safety management strategy and enhancing overall traffic efficiency. To address the limitations associated with trajectory acquisition and vehicle tracking, road segment maps recording the individual-level traffic dynamic data were directly served in crash detection. A novel two-stage trajectory-free crash detection framework, was present to generate the rational future road segment map and identify crashes. The first-stage diffusion-based segment map generation model, Mapfusion, conducts a noisy-to-normal process that progressively adds noise to the road segment map until the map is corrupted to pure Gaussian noise. The denoising process is guided by sequential embedding components capturing the temporal dynamics of segment map sequences. Furthermore, the generation model is designed to incorporate background context through ControlNet to enhance generation control. Crash detection is achieved by comparing the monitored segment map with the generations from diffusion model in second stage. Trained on non-crash vehicle motion data, Mapfusion successfully generates realistic road segment evolution maps based on learned motion patterns and remains robust across different sampling intervals. Experiments on real-world crashes indicate the effectiveness of the proposed two-stage method in accurately detecting crashes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13795) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [5] [SAE-MCVT: A Real-Time and Scalable Multi-Camera Vehicle Tracking Framework Powered by Edge Computing](https://arxiv.org/abs/2511.13904)
*Yuqiang Lin, Sam Lockyer, Florian Stanek, Markus Zarbock, Adrian Evans, Wenbin Li, Nic Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In modern Intelligent Transportation Systems (ITS), cameras are a key component due to their ability to provide valuable information for multiple stakeholders. A central task is Multi-Camera Vehicle Tracking (MCVT), which generates vehicle trajectories and enables applications such as anomaly detection, traffic density estimation, and suspect vehicle tracking. However, most existing studies on MCVT emphasize accuracy while overlooking real-time performance and scalability. These two aspects are essential for real-world deployment and become increasingly challenging in city-scale applications as the number of cameras grows. To address this issue, we propose SAE-MCVT, the first scalable real-time MCVT framework. The system includes several edge devices that interact with one central workstation separately. On the edge side, live RTSP video streams are serialized and processed through modules including object detection, object tracking, geo-mapping, and feature extraction. Only lightweight metadata -- vehicle locations and deep appearance features -- are transmitted to the central workstation. On the central side, cross-camera association is calculated under the constraint of spatial-temporal relations between adjacent cameras, which are learned through a self-supervised camera link model. Experiments on the RoundaboutHD dataset show that SAE-MCVT maintains real-time operation on 2K 15 FPS video streams and achieves an IDF1 score of 61.2. To the best of our knowledge, this is the first scalable real-time MCVT framework suitable for city-scale deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13904) | **Categories:** cs.CV

---

### [6] [Mind the Gap: Evaluating LLM Understanding of Human-Taught Road Safety Principles](https://arxiv.org/abs/2511.13909)
*Chalamalasetti Kranti*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Following road safety norms is non-negotiable not only for humans but also for the AI systems that govern autonomous vehicles. In this work, we evaluate how well multi-modal large language models (LLMs) understand road safety concepts, specifically through schematic and illustrative representations. We curate a pilot dataset of images depicting traffic signs and road-safety norms sourced from school text books and use it to evaluate models capabilities in a zero-shot setting. Our preliminary results show that these models struggle with safety reasoning and reveal gaps between human learning and model interpretation. We further provide an analysis of these performance gaps for future research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13909) | **Categories:** cs.CV

---

### [7] [Learning Skill-Attributes for Transferable Assessment in Video](https://arxiv.org/abs/2511.13993)
*Kumar Ashutosh, Kristen Grauman*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Skill assessment from video entails rating the quality of a person's physical performance and explaining what could be done better. Today's models specialize for an individual sport, and suffer from the high cost and scarcity of expert-level supervision across the long tail of sports. Towards closing that gap, we explore transferable video representations for skill assessment. Our CrossTrainer approach discovers skill-attributes, such as balance, control, and hand positioning -- whose meaning transcends the boundaries of any given sport, then trains a multimodal language model to generate actionable feedback for a novel video, e.g., "lift hands more to generate more power" as well as its proficiency level, e.g., early expert. We validate the new model on multiple datasets for both cross-sport (transfer) and intra-sport (in-domain) settings, where it achieves gains up to 60% relative to the state of the art. By abstracting out the shared behaviors indicative of human skill, the proposed video representation generalizes substantially better than an array of existing techniques, enriching today's multimodal large language models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13993) | **Categories:** cs.CV

---

### [8] [Text-Driven Reasoning Video Editing via Reinforcement Learning on Digital Twin Representations](https://arxiv.org/abs/2511.14100)
*Yiqing Shen, Chenjia Li, Mathias Unberath*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Text-driven video editing enables users to modify video content only using text queries. While existing methods can modify video content if explicit descriptions of editing targets with precise spatial locations and temporal boundaries are provided, these requirements become impractical when users attempt to conceptualize edits through implicit queries referencing semantic properties or object relationships. We introduce reasoning video editing, a task where video editing models must interpret implicit queries through multi-hop reasoning to infer editing targets before executing modifications, and a first model attempting to solve this complex task, RIVER (Reasoning-based Implicit Video Editor). RIVER decouples reasoning from generation through digital twin representations of video content that preserve spatial relationships, temporal trajectories, and semantic attributes. A large language model then processes this representation jointly with the implicit query, performing multi-hop reasoning to determine modifications, then outputs structured instructions that guide a diffusion-based editor to execute pixel-level changes. RIVER training uses reinforcement learning with rewards that evaluate reasoning accuracy and generation quality. Finally, we introduce RVEBenchmark, a benchmark of 100 videos with 519 implicit queries spanning three levels and categories of reasoning complexity specifically for reasoning video editing. RIVER demonstrates best performance on the proposed RVEBenchmark and also achieves state-of-the-art performance on two additional video editing benchmarks (VegGIE and FiVE), where it surpasses six baseline methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14100) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Robustness of LLM-enabled vehicle trajectory prediction under data security threats](https://arxiv.org/abs/2511.13753)
*Feilong Wang, Fuqiang Liu*

Main category: cs.LG

TL;DR: 该研究揭示了基于大型语言模型（LLM）的车辆轨迹预测模型在面对对抗性攻击时的脆弱性。


<details>
  <summary>Details</summary>
Motivation: 探讨基于LLM的车辆轨迹预测模型在安全关键驾驶系统中的鲁棒性问题，填补了现有研究的空白。

Method: 提出了一种单特征差分进化攻击方法，在黑盒设置下扰动LLM输入提示中周围车辆的单个运动学特征。

Result: 实验表明，即使是微小的、物理上合理的扰动也可能显著干扰模型输出，表明基于LLM的预测器容易受到对抗性操纵。

Conclusion: 研究结果首次揭示了LLM驱动的自动驾驶车辆模型在车辆交互中的对抗性漏洞，并强调了未来基于LLM的智能交通系统需要面向鲁棒性的设计。

Abstract: 将大型语言模型（LLM）集成到自动驾驶系统中，通过将复杂的驾驶环境转化为语言可理解的表示，为推理和决策开辟了新的可能性。最近的研究表明，通过收集和转换来自周围车辆的数据，微调的LLM可以准确地预测车辆轨迹和变道意图。然而，尽管人们越来越关注LLM的可信度，但这种基于LLM的预测模型对于安全关键驾驶系统的鲁棒性仍未得到探索。本研究通过对LLM赋能的车辆轨迹预测进行系统的脆弱性分析来解决这一差距。我们提出了一种单特征差分进化攻击，在黑盒设置下扰动LLM输入提示中周围车辆的单个运动学特征。在highD数据集上的实验表明，即使是微小的、物理上合理的扰动也可能显著扰乱模型输出，突显了基于LLM的预测器对抗性操纵的敏感性。进一步的分析揭示了准确性和鲁棒性之间的权衡，检查了失败机制，并探讨了潜在的缓解解决方案。该研究结果首次深入了解了车辆交互背景下LLM驱动的自动驾驶车辆模型的对抗性漏洞，并强调了未来基于LLM的智能交通系统需要面向鲁棒性的设计。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13753) | **Categories:** cs.LG, cs.AI, cs.CR

---

### [2] [Unified Multimodal Vessel Trajectory Prediction with Explainable Navigation Intention](https://arxiv.org/abs/2511.14265)
*Rui Zhang, Chao Li, Kezhong Liu, Chen Wang, Bolong Zheng, Hongbo Jiang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vessel trajectory prediction is fundamental to intelligent maritime systems. Within this domain, short-term prediction of rapid behavioral changes in complex maritime environments has established multimodal trajectory prediction (MTP) as a promising research area. However, existing vessel MTP methods suffer from limited scenario applicability and insufficient explainability. To address these challenges, we propose a unified MTP framework incorporating explainable navigation intentions, which we classify into sustained and transient categories. Our method constructs sustained intention trees from historical trajectories and models dynamic transient intentions using a Conditional Variational Autoencoder (CVAE), while using a non-local attention mechanism to maintain global scenario consistency. Experiments on real Automatic Identification System (AIS) datasets demonstrates our method's broad applicability across diverse scenarios, achieving significant improvements in both ADE and FDE. Furthermore, our method improves explainability by explicitly revealing the navigational intentions underlying each predicted trajectory.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14265) | **Categories:** cs.LG

---

### [3] [Blurred Encoding for Trajectory Representation Learning](https://arxiv.org/abs/2511.13741)
*Silin Zhou, Yao Chen, Shuo Shang, Lisi Chen, Bingsheng He, Ryosuke Shibasaki*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory representation learning (TRL) maps trajectories to vector embeddings and facilitates tasks such as trajectory classification and similarity search. State-of-the-art (SOTA) TRL methods transform raw GPS trajectories to grid or road trajectories to capture high-level travel semantics, i.e., regions and roads. However, they lose fine-grained spatial-temporal details as multiple GPS points are grouped into a single grid cell or road segment. To tackle this problem, we propose the BLUrred Encoding method, dubbed BLUE, which gradually reduces the precision of GPS coordinates to create hierarchical patches with multiple levels. The low-level patches are small and preserve fine-grained spatial-temporal details, while the high-level patches are large and capture overall travel patterns. To complement different patch levels with each other, our BLUE is an encoder-decoder model with a pyramid structure. At each patch level, a Transformer is used to learn the trajectory embedding at the current level, while pooling prepares inputs for the higher level in the encoder, and up-resolution provides guidance for the lower level in the decoder. BLUE is trained using the trajectory reconstruction task with the MSE loss. We compare BLUE with 8 SOTA TRL methods for 3 downstream tasks, the results show that BLUE consistently achieves higher accuracy than all baselines, outperforming the best-performing baselines by an average of 30.90%. Our code is available at https://github.com/slzhou-xy/BLUE.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13741) | **Categories:** cs.LG

---

### [4] [PROF: An LLM-based Reward Code Preference Optimization Framework for Offline Imitation Learning](https://arxiv.org/abs/2511.13765)
*Shengjie Sun, Jiafei Lyu, Runze Liu, Mengbei Yan, Bo Liu, Deheng Ye, Xiu Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Offline imitation learning (offline IL) enables training effective policies without requiring explicit reward annotations. Recent approaches attempt to estimate rewards for unlabeled datasets using a small set of expert demonstrations. However, these methods often assume that the similarity between a trajectory and an expert demonstration is positively correlated with the reward, which oversimplifies the underlying reward structure. We propose PROF, a novel framework that leverages large language models (LLMs) to generate and improve executable reward function codes from natural language descriptions and a single expert trajectory. We propose Reward Preference Ranking (RPR), a novel reward function quality assessment and ranking strategy without requiring environment interactions or RL training. RPR calculates the dominance scores of the reward functions, where higher scores indicate better alignment with expert preferences. By alternating between RPR and text-based gradient optimization, PROF fully automates the selection and refinement of optimal reward functions for downstream policy learning. Empirical results on D4RL demonstrate that PROF surpasses or matches recent strong baselines across numerous datasets and domains, highlighting the effectiveness of our approach.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13765) | **Categories:** cs.LG, cs.AI

---

### [5] [Beat the long tail: Distribution-Aware Speculative Decoding for RL Training](https://arxiv.org/abs/2511.13841)
*Zelei Shao, Vikranth Srivatsa, Sanjana Srivastava, Qingyang Wu, Alpay Ariyak, Xiaoxia Wu, Ameen Patel, Jue Wang, Percy Liang, Tri Dao, Ce Zhang, Yiying Zhang, Ben Athiwaratkun, Chenfeng Xu, Junxiong Wang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reinforcement learning(RL) post-training has become essential for aligning large language models (LLMs), yet its efficiency is increasingly constrained by the rollout phase, where long trajectories are generated token by token. We identify a major bottleneck:the long-tail distribution of rollout lengths, where a small fraction of long generations dominates wall clock time and a complementary opportunity; the availability of historical rollouts that reveal stable prompt level patterns across training epochs. Motivated by these observations, we propose DAS, a Distribution Aware Speculative decoding framework that accelerates RL rollouts without altering model outputs. DAS integrates two key ideas: an adaptive, nonparametric drafter built from recent rollouts using an incrementally maintained suffix tree, and a length aware speculation policy that allocates more aggressive draft budgets to long trajectories that dominate makespan. This design exploits rollout history to sustain acceptance while balancing base and token level costs during decoding. Experiments on math and code reasoning tasks show that DAS reduces rollout time up to 50% while preserving identical training curves, demonstrating that distribution-aware speculative decoding can significantly accelerate RL post training without compromising learning quality.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13841) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding](https://arxiv.org/abs/2511.13961)
*Jiarui Li, Alessandro Zanardi, Runyu Zhang, Gioele Zardini*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.13961) | **Categories:** cs.RO

---

### [2] [AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models](https://arxiv.org/abs/2511.14148)
*Yuhua Jiang, Shuang Cheng, Yan Ding, Feifei Gao, Biqing Qi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-language-action (VLA) models have recently emerged as a powerful paradigm for building generalist robots. However, traditional VLA models that generate actions through flow matching (FM) typically rely on rigid and uniform time schedules, i.e., synchronous FM (SFM). Without action context awareness and asynchronous self-correction, SFM becomes unstable in long-horizon tasks, where a single action error can cascade into failure. In this work, we propose asynchronous flow matching VLA (AsyncVLA), a novel framework that introduces temporal flexibility in asynchronous FM (AFM) and enables self-correction in action generation. AsyncVLA breaks from the vanilla SFM in VLA models by generating the action tokens in a non-uniform time schedule with action context awareness. Besides, our method introduces the confidence rater to extract confidence of the initially generated actions, enabling the model to selectively refine inaccurate action tokens before execution. Moreover, we propose a unified training procedure for SFM and AFM that endows a single model with both modes, improving KV-cache utilization. Extensive experiments on robotic manipulation benchmarks demonstrate that AsyncVLA is data-efficient and exhibits self-correction ability. AsyncVLA achieves state-of-the-art results across general embodied evaluations due to its asynchronous generation in AFM. Our code is available at https://github.com/YuhuaJiang2002/AsyncVLA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14148) | **Categories:** cs.RO, cs.AI, cs.LG

---

### [3] [Towards Deploying VLA without Fine-Tuning: Plug-and-Play Inference-Time VLA Policy Steering via Embodied Evolutionary Diffusion](https://arxiv.org/abs/2511.14178)
*Zhuo Li, Junjia Liu, Zhipeng Dong, Tao Teng, Quentin Rouxel, Darwin Caldwell, Fei Chen*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language-Action (VLA) models have demonstrated significant potential in real-world robotic manipulation. However, pre-trained VLA policies still suffer from substantial performance degradation during downstream deployment. Although fine-tuning can mitigate this issue, its reliance on costly demonstration collection and intensive computation makes it impractical in real-world settings. In this work, we introduce VLA-Pilot, a plug-and-play inference-time policy steering method for zero-shot deployment of pre-trained VLA without any additional fine-tuning or data collection. We evaluate VLA-Pilot on six real-world downstream manipulation tasks across two distinct robotic embodiments, encompassing both in-distribution and out-of-distribution scenarios. Experimental results demonstrate that VLA-Pilot substantially boosts the success rates of off-the-shelf pre-trained VLA policies, enabling robust zero-shot generalization to diverse tasks and embodiments. Experimental videos and code are available at: https://rip4kobe.github.io/vla-pilot/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14178) | **Categories:** cs.RO, cs.AI

---

### [4] [Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning](https://arxiv.org/abs/2511.14396)
*Xiuxiu Qi, Yu Yang, Jiannong Cao, Luyao Bai, Chongshan Fan, Chengtai Cao, Hongpeng Wang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14396) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [5] [Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language](https://arxiv.org/abs/2511.14565)
*Minyoung Hwang, Alexandra Forsey-Smerek, Nathaniel Dennler, Andreea Bobu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robots can adapt to user preferences by learning reward functions from demonstrations, but with limited data, reward models often overfit to spurious correlations and fail to generalize. This happens because demonstrations show robots how to do a task but not what matters for that task, causing the model to focus on irrelevant state details. Natural language can more directly specify what the robot should focus on, and, in principle, disambiguate between many reward functions consistent with the demonstrations. However, existing language-conditioned reward learning methods typically treat instructions as simple conditioning signals, without fully exploiting their potential to resolve ambiguity. Moreover, real instructions are often ambiguous themselves, so naive conditioning is unreliable. Our key insight is that these two input types carry complementary information: demonstrations show how to act, while language specifies what is important. We propose Masked Inverse Reinforcement Learning (Masked IRL), a framework that uses large language models (LLMs) to combine the strengths of both input types. Masked IRL infers state-relevance masks from language instructions and enforces invariance to irrelevant state components. When instructions are ambiguous, it uses LLM reasoning to clarify them in the context of the demonstrations. In simulation and on a real robot, Masked IRL outperforms prior language-conditioned IRL methods by up to 15% while using up to 4.7 times less data, demonstrating improved sample-efficiency, generalization, and robustness to ambiguous language. Project page: https://MIT-CLEAR-Lab.github.io/Masked-IRL and Code: https://github.com/MIT-CLEAR-Lab/Masked-IRL

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14565) | **Categories:** cs.RO, cs.AI

---

### [6] [Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks](https://arxiv.org/abs/2511.14592)
*Xianhui Meng, Yuchen Zhang, Zhijian Huang, Zheng Lu, Ziling Ji, Yaoyao Yin, Hongyuan Zhang, Guangfeng Jiang, Yandan Lin, Long Chen, Hangjun Ye, Li Zhang, Jun Liu, Xiaoshuai Hao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision-Language Models (VLMs) show great promise for autonomous driving, but their suitability for safety-critical scenarios is largely unexplored, raising safety concerns. This issue arises from the lack of comprehensive benchmarks that assess both external environmental risks and in-cabin driving behavior safety simultaneously. To bridge this critical gap, we introduce DSBench, the first comprehensive Driving Safety Benchmark designed to assess a VLM's awareness of various safety risks in a unified manner. DSBench encompasses two major categories: external environmental risks and in-cabin driving behavior safety, divided into 10 key categories and a total of 28 sub-categories. This comprehensive evaluation covers a wide range of scenarios, ensuring a thorough assessment of VLMs' performance in safety-critical contexts. Extensive evaluations across various mainstream open-source and closed-source VLMs reveal significant performance degradation under complex safety-critical situations, highlighting urgent safety concerns. To address this, we constructed a large dataset of 98K instances focused on in-cabin and external safety scenarios, showing that fine-tuning on this dataset significantly enhances the safety performance of existing VLMs and paves the way for advancing autonomous driving technology. The benchmark toolkit, code, and model checkpoints will be publicly accessible.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14592) | **Categories:** cs.RO, cs.AI

---

### [7] [NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-based Preference Rewards](https://arxiv.org/abs/2511.14659)
*Chia-Yu Hung, Navonil Majumder, Haoyuan Deng, Liu Renhang, Yankang Ang, Amir Zadeh, Chuan Li, Dorien Herremans, Ziwei Wang, Soujanya Poria*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Vision--language--action (VLA) models have recently shown promising performance on a variety of embodied tasks, yet they still fall short in reliability and generalization, especially when deployed across different embodiments or real-world environments. In this work, we introduce NORA-1.5, a VLA model built from the pre-trained NORA backbone by adding to it a flow-matching-based action expert. This architectural enhancement alone yields substantial performance gains, enabling NORA-1.5 to outperform NORA and several state-of-the-art VLA models across both simulated and real-world benchmarks. To further improve robustness and task success, we develop a set of reward models for post-training VLA policies. Our rewards combine (i) an action-conditioned world model (WM) that evaluates whether generated actions lead toward the desired goal, and (ii) a deviation-from-ground-truth heuristic that distinguishes good actions from poor ones. Using these reward signals, we construct preference datasets and adapt NORA-1.5 to target embodiments through direct preference optimization (DPO). Extensive evaluations show that reward-driven post-training consistently improves performance in both simulation and real-robot settings, demonstrating significant VLA model-reliability gains through simple yet effective reward models. Our findings highlight NORA-1.5 and reward-guided post-training as a viable path toward more dependable embodied agents suitable for real-world deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.14659) | **Categories:** cs.RO, cs.AI

---


## eess.SY [eess.SY]
### [1] [Who Moved My Distribution? Conformal Prediction for Interactive Multi-Agent Systems](https://arxiv.org/abs/2511.11567)
*Allen Emmanuel Binny, Anushri Dixit*

Main category: eess.SY

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Uncertainty-aware prediction is essential for safe motion planning, especially when using learned models to forecast the behavior of surrounding agents. Conformal prediction is a statistical tool often used to produce uncertainty-aware prediction regions for machine learning models. Most existing frameworks utilizing conformal prediction-based uncertainty predictions assume that the surrounding agents are non-interactive. This is because in closed-loop, as uncertainty-aware agents change their behavior to account for prediction uncertainty, the surrounding agents respond to this change, leading to a distribution shift which we call endogenous distribution shift. To address this challenge, we introduce an iterative conformal prediction framework that systematically adapts the uncertainty-aware ego-agent controller to the endogenous distribution shift. The proposed method provides probabilistic safety guarantees while adapting to the evolving behavior of reactive, non-ego agents. We establish a model for the endogenous distribution shift and provide the conditions for the iterative conformal prediction pipeline to converge under such a distribution shift. We validate our framework in simulation for 2- and 3- agent interaction scenarios, demonstrating collision avoidance without resulting in overly conservative behavior and an overall improvement in success rates of up to 9.6% compared to other conformal prediction-based baselines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.11567) | **Categories:** eess.SY, cs.RO

---
