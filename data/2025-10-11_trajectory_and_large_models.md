# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-11

## 目录

- [人工智能 (Artificial Intelligence) (6)](#cs-ai)
- [计算机视觉 (Computer Vision) (4)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (12)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [LinguaSim: Interactive Multi-Vehicle Testing Scenario Generation via Natural Language Instruction Based on Large Language Models](https://arxiv.org/abs/2510.08046)
*Qingyuan Shi, Qingwen Meng, Hao Cheng, Qing Xu, Jianqiang Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The generation of testing and training scenarios for autonomous vehicles has drawn significant attention. While Large Language Models (LLMs) have enabled new scenario generation methods, current methods struggle to balance command adherence accuracy with the realism of real-world driving environments. To reduce scenario description complexity, these methods often compromise realism by limiting scenarios to 2D, or open-loop simulations where background vehicles follow predefined, non-interactive behaviors. We propose LinguaSim, an LLM-based framework that converts natural language into realistic, interactive 3D scenarios, ensuring both dynamic vehicle interactions and faithful alignment between the input descriptions and the generated scenarios. A feedback calibration module further refines the generation precision, improving fidelity to user intent. By bridging the gap between natural language and closed-loop, interactive simulations, LinguaSim constrains adversarial vehicle behaviors using both the scenario description and the autonomous driving model guiding them. This framework facilitates the creation of high-fidelity scenarios that enhance safety testing and training. Experiments show LinguaSim can generate scenarios with varying criticality aligned with different natural language descriptions (ACT: 0.072 s for dangerous vs. 3.532 s for safe descriptions; comfortability: 0.654 vs. 0.764), and its refinement module effectively reduces excessive aggressiveness in LinguaSim's initial outputs, lowering the crash rate from 46.9% to 6.3% to better match user intentions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.08046) | **Categories:** cs.AI

---

### [2] [CompassLLM: A Multi-Agent Approach toward Geo-Spatial Reasoning for Popular Path Query](https://arxiv.org/abs/2510.07516)
*Md. Nazmul Islam Ananto, Shamit Fatin, Mohammed Eunus Ali, Md Rizwan Parvez*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The popular path query - identifying the most frequented routes between locations from historical trajectory data - has important applications in urban planning, navigation optimization, and travel recommendations. While traditional algorithms and machine learning approaches have achieved success in this domain, they typically require model training, parameter tuning, and retraining when accommodating data updates. As Large Language Models (LLMs) demonstrate increasing capabilities in spatial and graph-based reasoning, there is growing interest in exploring how these models can be applied to geo-spatial problems.   We introduce CompassLLM, a novel multi-agent framework that intelligently leverages the reasoning capabilities of LLMs into the geo-spatial domain to solve the popular path query. CompassLLM employs its agents in a two-stage pipeline: the SEARCH stage that identifies popular paths, and a GENERATE stage that synthesizes novel paths in the absence of an existing one in the historical trajectory data. Experiments on real and synthetic datasets show that CompassLLM demonstrates superior accuracy in SEARCH and competitive performance in GENERATE while being cost-effective.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07516) | **Categories:** cs.AI, cs.CL

---

### [3] [An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation](https://arxiv.org/abs/2510.07825)
*Yuping Zhou, Siqi Lai, Jindong Han, Hao Liu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The rise of Internet of Vehicles (IoV) technologies is transforming traffic management from isolated control to a collective, multi-vehicle process. At the heart of this shift is multi-vehicle dynamic navigation, which requires simultaneously routing large fleets under evolving traffic conditions. Existing path search algorithms and reinforcement learning methods struggle to scale to city-wide networks, often failing to capture the nonlinear, stochastic, and coupled dynamics of urban traffic. To address these challenges, we propose CityNav, a hierarchical, LLM-powered framework for large-scale multi-vehicle navigation. CityNav integrates a global traffic allocation agent, which coordinates strategic traffic flow distribution across regions, with local navigation agents that generate locally adaptive routes aligned with global directives. To enable effective cooperation, we introduce a cooperative reasoning optimization mechanism, in which agents are jointly trained with a dual-reward structure: individual rewards promote per-vehicle efficiency, while shared rewards encourage network-wide coordination and congestion reduction. Extensive experiments on four real-world road networks of varying scales (up to 1.6 million roads and 430,000 intersections) and traffic datasets demonstrate that CityNav consistently outperforms nine classical path search and RL-based baselines in city-scale travel efficiency and congestion mitigation. Our results highlight the potential of LLMs to enable scalable, adaptive, and cooperative city-wide traffic navigation, providing a foundation for intelligent, large-scale vehicle routing in complex urban environments. Our project is available at https://github.com/usail-hkust/CityNav.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07825) | **Categories:** cs.AI

---

### [4] [Less is More: Strategic Expert Selection Outperforms Ensemble Complexity in Traffic Forecasting](https://arxiv.org/abs/2510.07426)
*Walid Guettala, Yufan Zhao, László Gulyás*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Traffic forecasting is fundamental to intelligent transportation systems, enabling congestion mitigation and emission reduction in increasingly complex urban environments. While recent graph neural network approaches have advanced spatial temporal modeling, existing mixture of experts frameworks like Time Enhanced Spatio Temporal Attention Model (TESTAM) lack explicit incorporation of physical road network topology, limiting their spatial capabilities. We present TESTAM+, an enhanced spatio temporal forecasting framework that introduces a novel SpatioSemantic Expert integrating physical road topology with data driven feature similarity through hybrid graph construction. TESTAM+ achieves significant improvements over TESTAM: 1.3% MAE reduction on METR LA (3.10 vs. 3.14) and 4.1% improvement on PEMS BAY (1.65 vs. 1.72). Through comprehensive ablation studies, we discover that strategic expert selection fundamentally outperforms naive ensemble aggregation. Individual experts demonstrate remarkable effectiveness: the Adaptive Expert achieves 1.63 MAE on PEMS BAY, outperforming the original three expert TESTAM (1.72 MAE), while the SpatioSemantic Expert matches this performance with identical 1.63 MAE. The optimal Identity + Adaptive configuration achieves an 11.5% MAE reduction compared to state of the art MegaCRN on METR LA (2.99 vs. 3.38), while reducing inference latency by 53.1% compared to the full four expert TESTAM+. Our findings reveal that fewer, strategically designed experts outperform complex multi expert ensembles, establishing new state of the art performance with superior computational efficiency for real time deployment.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07426) | **Categories:** cs.AI, 68T07, 68T05, 62M10, 90B20, I.2.6; I.5.4

---

### [5] [TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering](https://arxiv.org/abs/2510.07432)
*Penghang Liu, Elizabeth Fons, Svitlana Vyetrenko, Daniel Borrajo, Vamsi Potluru, Manuela Veloso*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) have shown strong abilities in reasoning and problem solving, but recent studies reveal that they still struggle with time series reasoning tasks, where outputs are often affected by hallucination or knowledge leakage. In this work we propose TS-Agent, a time series reasoning agent that leverages LLMs strictly for what they excel at, i.e., gathering evidence and synthesizing it into conclusions through step-by-step reasoning, while delegating the extraction of statistical and structural information to time series analytical tools. Instead of mapping time series into text tokens, images, or embeddings, our agent interacts with raw numeric sequences through atomic operators, records outputs in an explicit evidence log, and iteratively refines its reasoning under the guidance of a self-critic and a final quality gate. This design avoids multi-modal alignment training, preserves the native form of time series, ensures interpretability and verifiability, and mitigates knowledge leakage or hallucination. Empirically, we evaluate the agent on established benchmarks. Our experiments show that TS-Agent achieves performance comparable to state-of-the-art LLMs on understanding benchmarks, and delivers significant improvements on reasoning tasks, where existing models often rely on memorization and fail in zero-shot settings.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07432) | **Categories:** cs.AI

---

### [6] [Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models](https://arxiv.org/abs/2510.07858)
*Zhiqing Cui, Binwu Wang, Qingxiang Liu, Yeqiang Wang, Zhengyang Zhou, Yuxuan Liang, Yang Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLM) have emerged as a promising avenue for time series forecasting, offering the potential to integrate multimodal data. However, existing LLM-based approaches face notable limitations-such as marginalized role in model architectures, reliance on coarse statistical text prompts, and lack of interpretability. In this work, we introduce Augur, a fully LLM driven time series forecasting framework that exploits LLM causal reasoning to discover and use directed causal associations among covariates. Augur uses a two stage teacher student architecture where a powerful teacher LLM infers a directed causal graph from time series using heuristic search together with pairwise causality testing. A lightweight student agent then refines the graph and fine tune on high confidence causal associations that are encoded as rich textual prompts to perform forecasting. This design improves predictive accuracy while yielding transparent, traceable reasoning about variable interactions. Extensive experiments on real-world datasets with 25 baselines demonstrate that Augur achieves competitive performance and robust zero-shot generalization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07858) | **Categories:** cs.AI, cs.LG, 62M10, I.2.7

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](https://arxiv.org/abs/2510.07791)
*Qinghongbing Xie, Zhaoyuan Xia, Feng Zhu, Lijun Gong, Ziyue Li, Rui Zhao, Long Zeng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for Autonomous Driving, Embodied AI and General Artificial Intelligence. Existing spatial-temporal benchmarks mainly focus on egocentric perspective reasoning with images/video context, or geographic perspective reasoning with graphics context (eg. a map), thus fail to assess VLMs' geographic spatial-temporal intelligence with both images/video and graphics context, which is important for areas like traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench demonstrate that even the best proprietary model, Gemini-2.5-Pro (34.9%), significantly lags behind human performance (78.61%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three primary deficiencies of current models for geo-temporal reasoning. (1) VLMs' reasoning is impaired by an imbalanced utilization of spatial-temporal context. (2) VLMs are weak in temporal forecasting, which leads to worse performance on temporal-emphasized tasks than on spatial-emphasized tasks. (3) VLMs lack the proficiency to comprehend or align the map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07791) | **Categories:** cs.CV

---

### [2] [ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving](https://arxiv.org/abs/2510.08562)
*Zhiyu Zheng, Shaoyu Chen, Haoran Yin, Xinbang Zhang, Jialv Zou, Xinggang Wang, Qian Zhang, Lefei Zhang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of causal inference, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes the learning task to predict the residual deviation from a deterministic inertial reference. The inertial reference serves as a counterfactual, forcing the model to move beyond simple pattern recognition and instead identify the underlying causal factors (e.g., traffic rules, obstacles) that necessitate deviations from a default, inertially-guided path. To deal with the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. It re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. Extensive experiments validate the effectiveness of our framework. On the NAVSIM benchmark, ResAD achieves a state-of-the-art PDMS of 88.6 using a vanilla diffusion policy with only two denoising steps, demonstrating that our approach significantly simplifies the learning task and improves model performance. The code will be released to facilitate further research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.08562) | **Categories:** cs.CV, cs.RO

---

### [3] [TRAVL: A Recipe for Making Video-Language Models Better Judges of Physics Implausibility](https://arxiv.org/abs/2510.07550)
*Saman Motamed, Minghao Chen, Luc Van Gool, Iro Laina*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite impressive visual fidelity, modern video generative models frequently produce sequences that violate intuitive physical laws, such as objects floating, teleporting, or morphing in ways that defy causality. While humans can easily detect such implausibilities, there remains no robust method for quantitatively assessing physical realism in video. In this work, we explore whether Video-Language Models (VLMs) can be trained to serve as reliable judges of physical plausibility. We find that existing VLMs struggle to identify physics violations, exposing fundamental limitations in their temporal and causal reasoning. To address this, we introduce TRAVL, a fine-tuning recipe that combines a balanced training dataset with a trajectory-aware attention module to improve motion encoding and discrimination in VLMs. To evaluate physical reasoning more rigorously, we propose ImplausiBench, a benchmark of 300 videos (150 real, 150 generated) that removes linguistic biases and isolates visual-temporal understanding. Performance is reported both with gold-standard human judgments and stricter LLM-as-judge metrics. Together, TRAVL and ImplausiBench offer a unified framework for probing and improving physical plausibility in multimodal models, shedding light on a challenging and underexplored aspect of visual-temporal understanding.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07550) | **Categories:** cs.CV, cs.AI

---

### [4] [CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving](https://arxiv.org/abs/2510.07944)
*Tianrui Zhang, Yichen Liu, Zilin Guo, Yuxin Guo, Jingcheng Ni, Chenjing Ding, Dan Xu, Lewei Lu, Zehuan Wu*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07944) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [MLLM4TS: Leveraging Vision and Multimodal Language Models for General Time-Series Analysis](https://arxiv.org/abs/2510.07513)
*Qinghua Liu, Sam Heshmati, Zheda Mai, Zubin Abraham, John Paparrizos, Liu Ren*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Effective analysis of time series data presents significant challenges due to the complex temporal dependencies and cross-channel interactions in multivariate data. Inspired by the way human analysts visually inspect time series to uncover hidden patterns, we ask: can incorporating visual representations enhance automated time-series analysis? Recent advances in multimodal large language models have demonstrated impressive generalization and visual understanding capability, yet their application to time series remains constrained by the modality gap between continuous numerical data and discrete natural language. To bridge this gap, we introduce MLLM4TS, a novel framework that leverages multimodal large language models for general time-series analysis by integrating a dedicated vision branch. Each time-series channel is rendered as a horizontally stacked color-coded line plot in one composite image to capture spatial dependencies across channels, and a temporal-aware visual patch alignment strategy then aligns visual patches with their corresponding time segments. MLLM4TS fuses fine-grained temporal details from the numerical data with global contextual information derived from the visual representation, providing a unified foundation for multimodal time-series analysis. Extensive experiments on standard benchmarks demonstrate the effectiveness of MLLM4TS across both predictive tasks (e.g., classification) and generative tasks (e.g., anomaly detection and forecasting). These results underscore the potential of integrating visual modalities with pretrained language models to achieve robust and generalizable time-series analysis.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07513) | **Categories:** cs.LG, cs.AI, cs.CV, cs.DB

---

### [2] [GeoGen: A Two-stage Coarse-to-Fine Framework for Fine-grained Synthetic Location-based Social Network Trajectory Generation](https://arxiv.org/abs/2510.07735)
*Rongchao Xu, Kunlin Cai, Lin Jiang, Dahai Yu, Zhiqing Hong, Yuan Tian, Guang Wang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Location-Based Social Network (LBSN) check-in trajectory data are important for many practical applications, like POI recommendation, advertising, and pandemic intervention. However, the high collection costs and ever-increasing privacy concerns prevent us from accessing large-scale LBSN trajectory data. The recent advances in synthetic data generation provide us with a new opportunity to achieve this, which utilizes generative AI to generate synthetic data that preserves the characteristics of real data while ensuring privacy protection. However, generating synthetic LBSN check-in trajectories remains challenging due to their spatially discrete, temporally irregular nature and the complex spatio-temporal patterns caused by sparse activities and uncertain human mobility. To address this challenge, we propose GeoGen, a two-stage coarse-to-fine framework for large-scale LBSN check-in trajectory generation. In the first stage, we reconstruct spatially continuous, temporally regular latent movement sequences from the original LBSN check-in trajectories and then design a Sparsity-aware Spatio-temporal Diffusion model (S$^2$TDiff) with an efficient denosing network to learn their underlying behavioral patterns. In the second stage, we design Coarse2FineNet, a Transformer-based Seq2Seq architecture equipped with a dynamic context fusion mechanism in the encoder and a multi-task hybrid-head decoder, which generates fine-grained LBSN trajectories based on coarse-grained latent movement sequences by modeling semantic relevance and behavioral uncertainty. Extensive experiments on four real-world datasets show that GeoGen excels state-of-the-art models for both fidelity and utility evaluation, e.g., it increases over 69% and 55% in distance and radius metrics on the FS-TKY dataset.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07735) | **Categories:** cs.LG

---

### [3] [Weak Form Learning for Mean-Field Partial Differential Equations: an Application to Insect Movement](https://arxiv.org/abs/2510.07786)
*Seth Minor, Bret D. Elderd, Benjamin Van Allen, David M. Bortz, Vanja Dukic*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Insect species subject to infection, predation, and anisotropic environmental conditions may exhibit preferential movement patterns. Given the innate stochasticity of exogenous factors driving these patterns over short timescales, individual insect trajectories typically obey overdamped stochastic dynamics. In practice, data-driven modeling approaches designed to learn the underlying Fokker-Planck equations from observed insect distributions serve as ideal tools for understanding and predicting such behavior. Understanding dispersal dynamics of crop and silvicultural pests can lead to a better forecasting of outbreak intensity and location, which can result in better pest management. In this work, we extend weak-form equation learning techniques, coupled with kernel density estimation, to learn effective models for lepidopteran larval population movement from highly sparse experimental data. Galerkin methods such as the Weak form Sparse Identification of Nonlinear Dynamics (WSINDy) algorithm have recently proven useful for learning governing equations in several scientific contexts. We demonstrate the utility of the method on a sparse dataset of position measurements of fall armyworms (Spodoptera frugiperda) obtained in simulated agricultural conditions with varied plant resources and infection status.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07786) | **Categories:** cs.LG, math.DS, q-bio.PE, 60J70, 62FXX, 92-08

---


## 机器人学 (Robotics) [cs.RO]
### [1] [NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions](https://arxiv.org/abs/2510.08173)
*Haolin Yang, Yuxing Long, Zhuoyuan Yu, Zihan Yang, Minghan Wang, Jiapeng Xu, Yihan Wang, Ziyan Yu, Wenzhe Cai, Lei Kang, Hao Dong*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Instruction-following navigation is a key step toward embodied intelligence. Prior benchmarks mainly focus on semantic understanding but overlook systematically evaluating navigation agents' spatial perception and reasoning capabilities. In this work, we introduce the NavSpace benchmark, which contains six task categories and 1,228 trajectory-instruction pairs designed to probe the spatial intelligence of navigation agents. On this benchmark, we comprehensively evaluate 22 navigation agents, including state-of-the-art navigation models and multimodal large language models. The evaluation results lift the veil on spatial intelligence in embodied navigation. Furthermore, we propose SNav, a new spatially intelligent navigation model. SNav outperforms existing navigation agents on NavSpace and real robot tests, establishing a strong baseline for future work.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.08173) | **Categories:** cs.RO, cs.AI, cs.CL, cs.CV

---

### [2] [FLEET: Formal Language-Grounded Scheduling for Heterogeneous Robot Teams](https://arxiv.org/abs/2510.07417)
*Corban Rivera, Grayson Byrd, Meghan Booker, Bethany Kemp, Allison Gaines, Emma Holmes, James Uplinger, Celso M de Melo, David Handelman*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Coordinating heterogeneous robot teams from free-form natural-language instructions is hard. Language-only planners struggle with long-horizon coordination and hallucination, while purely formal methods require closed-world models. We present FLEET, a hybrid decentralized framework that turns language into optimized multi-robot schedules. An LLM front-end produces (i) a task graph with durations and precedence and (ii) a capability-aware robot--task fitness matrix; a formal back-end solves a makespan-minimization problem while the underlying robots execute their free-form subtasks with agentic closed-loop control. Across multiple free-form language-guided autonomy coordination benchmarks, FLEET improves success over state of the art generative planners on two-agent teams across heterogeneous tasks. Ablations show that mixed integer linear programming (MILP) primarily improves temporal structure, while LLM-derived fitness is decisive for capability-coupled tasks; together they deliver the highest overall performance. We demonstrate the translation to real world challenges with hardware trials using a pair of quadruped robots with disjoint capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07417) | **Categories:** cs.RO

---

### [3] [VeMo: A Lightweight Data-Driven Approach to Model Vehicle Dynamics](https://arxiv.org/abs/2510.07447)
*Girolamo Oddo, Roberto Nuca, Matteo Parsani*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Developing a dynamic model for a high-performance vehicle is a complex problem that requires extensive structural information about the system under analysis. This information is often unavailable to those who did not design the vehicle and represents a typical issue in autonomous driving applications, which are frequently developed on top of existing vehicles; therefore, vehicle models are developed under conditions of information scarcity. This paper proposes a lightweight encoder-decoder model based on Gate Recurrent Unit layers to correlate the vehicle's future state with its past states, measured onboard, and control actions the driver performs. The results demonstrate that the model achieves a maximum mean relative error below 2.6% in extreme dynamic conditions. It also shows good robustness when subject to noisy input data across the interested frequency components. Furthermore, being entirely data-driven and free from physical constraints, the model exhibits physical consistency in the output signals, such as longitudinal and lateral accelerations, yaw rate, and the vehicle's longitudinal velocity.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07447) | **Categories:** cs.RO, cs.LG, math.DS

---

### [4] [GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control](https://arxiv.org/abs/2510.07625)
*Alexander Du, Emre Adabag, Gabriel Bravo, Brian Plancher*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While Model Predictive Control (MPC) delivers strong performance across robotics applications, solving the underlying (batches of) nonlinear trajectory optimization (TO) problems online remains computationally demanding. Existing GPU-accelerated approaches typically (i) parallelize a single solve to meet real-time deadlines, (ii) scale to very large batches at slower-than-real-time rates, or (iii) achieve speed by restricting model generality (e.g., point-mass dynamics or a single linearization). This leaves a large gap in solver performance for many state-of-the-art MPC applications that require real-time batches of tens to low-hundreds of solves. As such, we present GATO, an open source, GPU-accelerated, batched TO solver co-designed across algorithm, software, and computational hardware to deliver real-time throughput for these moderate batch size regimes. Our approach leverages a combination of block-, warp-, and thread-level parallelism within and across solves for ultra-high performance. We demonstrate the effectiveness of our approach through a combination of: simulated benchmarks showing speedups of 18-21x over CPU baselines and 1.4-16x over GPU baselines as batch size increases; case studies highlighting improved disturbance rejection and convergence behavior; and finally a validation on hardware using an industrial manipulator. We open source GATO to support reproducibility and adoption.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07625) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [5] [EB-MBD: Emerging-Barrier Model-Based Diffusion for Safe Trajectory Optimization in Highly Constrained Environments](https://arxiv.org/abs/2510.07700)
*Raghav Mishra, Ian R. Manchester*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose enforcing constraints on Model-Based Diffusion by introducing emerging barrier functions inspired by interior point methods. We show that constraints on Model-Based Diffusion can lead to catastrophic performance degradation, even on simple 2D systems due to sample inefficiency in the Monte Carlo approximation of the score function. We introduce Emerging-Barrier Model-Based Diffusion (EB-MBD) which uses progressively introduced barrier constraints to avoid these problems, significantly improving solution quality, without the need for computationally expensive operations such as projections. We analyze the sampling liveliness of samples each iteration to inform barrier parameter scheduling choice. We demonstrate results for 2D collision avoidance and a 3D underwater manipulator system and show that our method achieves lower cost solutions than Model-Based Diffusion, and requires orders of magnitude less computation time than projection based methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07700) | **Categories:** cs.RO, cs.SY, eess.SY

---

### [6] [Trajectory Conditioned Cross-embodiment Skill Transfer](https://arxiv.org/abs/2510.07773)
*YuHang Tang, Yixuan Lou, Pengfei Han, Haoming Song, Xinyi Ye, Dong Wang, Bin Zhao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning manipulation skills from human demonstration videos presents a promising yet challenging problem, primarily due to the significant embodiment gap between human body and robot manipulators. Existing methods rely on paired datasets or hand-crafted rewards, which limit scalability and generalization. We propose TrajSkill, a framework for Trajectory Conditioned Cross-embodiment Skill Transfer, enabling robots to acquire manipulation skills directly from human demonstration videos. Our key insight is to represent human motions as sparse optical flow trajectories, which serve as embodiment-agnostic motion cues by removing morphological variations while preserving essential dynamics. Conditioned on these trajectories together with visual and textual inputs, TrajSkill jointly synthesizes temporally consistent robot manipulation videos and translates them into executable actions, thereby achieving cross-embodiment skill transfer. Extensive experiments are conducted, and the results on simulation data (MetaWorld) show that TrajSkill reduces FVD by 39.6\% and KVD by 36.6\% compared with the state-of-the-art, and improves cross-embodiment success rate by up to 16.7\%. Real-robot experiments in kitchen manipulation tasks further validate the effectiveness of our approach, demonstrating practical human-to-robot skill transfer across embodiments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07773) | **Categories:** cs.RO, cs.AI

---

### [7] [GM3: A General Physical Model for Micro-Mobility Vehicles](https://arxiv.org/abs/2510.07807)
*Grace Cai, Nithin Parepally, Laura Zheng, Ming C. Lin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Modeling the dynamics of micro-mobility vehicles (MMV) is becoming increasingly important for training autonomous vehicle systems and building urban traffic simulations. However, mainstream tools rely on variants of the Kinematic Bicycle Model (KBM) or mode-specific physics that miss tire slip, load transfer, and rider/vehicle lean. To our knowledge, no unified, physics-based model captures these dynamics across the full range of common MMVs and wheel layouts. We propose the "Generalized Micro-mobility Model" (GM3), a tire-level formulation based on the tire brush representation that supports arbitrary wheel configurations, including single/double track and multi-wheel platforms. We introduce an interactive model-agnostic simulation framework that decouples vehicle/layout specification from dynamics to compare the GM3 with the KBM and other models, consisting of fixed step RK4 integration, human-in-the-loop and scripted control, real-time trajectory traces and logging for analysis. We also empirically validate the GM3 on the Stanford Drone Dataset's deathCircle (roundabout) scene for biker, skater, and cart classes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07807) | **Categories:** cs.RO

---

### [8] [Team Xiaomi EV-AD VLA: Learning to Navigate Socially Through Proactive Risk Perception -- Technical Report for IROS 2025 RoboSense Challenge Social Navigation Track](https://arxiv.org/abs/2510.07871)
*Erjia Xiao, Lingfeng Zhang, Yingbo Tang, Hao Cheng, Renjing Xu, Wenbo Ding, Lei Zhou, Long Chen, Hangjun Ye, Xiaoshuai Hao*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this report, we describe the technical details of our submission to the IROS 2025 RoboSense Challenge Social Navigation Track. This track focuses on developing RGBD-based perception and navigation systems that enable autonomous agents to navigate safely, efficiently, and socially compliantly in dynamic human-populated indoor environments. The challenge requires agents to operate from an egocentric perspective using only onboard sensors including RGB-D observations and odometry, without access to global maps or privileged information, while maintaining social norm compliance such as safe distances and collision avoidance. Building upon the Falcon model, we introduce a Proactive Risk Perception Module to enhance social navigation performance. Our approach augments Falcon with collision risk understanding that learns to predict distance-based collision risk scores for surrounding humans, which enables the agent to develop more robust spatial awareness and proactive collision avoidance behaviors. The evaluation on the Social-HM3D benchmark demonstrates that our method improves the agent's ability to maintain personal space compliance while navigating toward goals in crowded indoor scenes with dynamic human agents, achieving 2nd place among 16 participating teams in the challenge.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07871) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [9] [Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots](https://arxiv.org/abs/2510.07882)
*Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In recent years, Multimodal Large Language Models (MLLMs) have demonstrated the ability to serve as high-level planners, enabling robots to follow complex human instructions. However, their effectiveness, especially in long-horizon tasks involving dual-arm humanoid robots, remains limited. This limitation arises from two main challenges: (i) the absence of simulation platforms that systematically support task evaluation and data collection for humanoid robots, and (ii) the insufficient embodiment awareness of current MLLMs, which hinders reasoning about dual-arm selection logic and body positions during planning. To address these issues, we present DualTHOR, a new dual-arm humanoid simulator, with continuous transition and a contingency mechanism. Building on this platform, we propose Proprio-MLLM, a model that enhances embodiment awareness by incorporating proprioceptive information with motion-based position embedding and a cross-spatial encoder. Experiments show that, while existing MLLMs struggle in this environment, Proprio-MLLM achieves an average improvement of 19.75% in planning performance. Our work provides both an essential simulation platform and an effective model to advance embodied intelligence in humanoid robotics. The code is available at https://anonymous.4open.science/r/DualTHOR-5F3B.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.07882) | **Categories:** cs.RO

---

### [10] [Towards Reliable LLM-based Robot Planning via Combined Uncertainty Estimation](https://arxiv.org/abs/2510.08044)
*Shiyuan Yin, Chenjia Bai, Zihao Zhang, Junwei Jin, Xinxin Zhang, Chi Zhang, Xuelong Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models (LLMs) demonstrate advanced reasoning abilities, enabling robots to understand natural language instructions and generate high-level plans with appropriate grounding. However, LLM hallucinations present a significant challenge, often leading to overconfident yet potentially misaligned or unsafe plans. While researchers have explored uncertainty estimation to improve the reliability of LLM-based planning, existing studies have not sufficiently differentiated between epistemic and intrinsic uncertainty, limiting the effectiveness of uncertainty estimation. In this paper, we present Combined Uncertainty estimation for Reliable Embodied planning (CURE), which decomposes the uncertainty into epistemic and intrinsic uncertainty, each estimated separately. Furthermore, epistemic uncertainty is subdivided into task clarity and task familiarity for more accurate evaluation. The overall uncertainty assessments are obtained using random network distillation and multi-layer perceptron regression heads driven by LLM features. We validated our approach in two distinct experimental settings: kitchen manipulation and tabletop rearrangement experiments. The results show that, compared to existing methods, our approach yields uncertainty estimates that are more closely aligned with the actual execution outcomes.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.08044) | **Categories:** cs.RO, cs.AI

---

### [11] [NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos](https://arxiv.org/abs/2510.08568)
*Hongyu Li, Lingfeng Sun, Yafei Hu, Duy Ta, Jennifer Barry, George Konidaris, Jiahui Fu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Enabling robots to execute novel manipulation tasks zero-shot is a central goal in robotics. Most existing methods assume in-distribution tasks or rely on fine-tuning with embodiment-matched data, limiting transfer across platforms. We present NovaFlow, an autonomous manipulation framework that converts a task description into an actionable plan for a target robot without any demonstrations. Given a task description, NovaFlow synthesizes a video using a video generation model and distills it into 3D actionable object flow using off-the-shelf perception modules. From the object flow, it computes relative poses for rigid objects and realizes them as robot actions via grasp proposals and trajectory optimization. For deformable objects, this flow serves as a tracking objective for model-based planning with a particle-based dynamics model. By decoupling task understanding from low-level control, NovaFlow naturally transfers across embodiments. We validate on rigid, articulated, and deformable object manipulation tasks using a table-top Franka arm and a Spot quadrupedal mobile robot, and achieve effective zero-shot execution without demonstrations or embodiment-specific training. Project website: https://novaflow.lhy.xyz/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.08568) | **Categories:** cs.RO, cs.AI, cs.CV

---

### [12] [BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation](https://arxiv.org/abs/2510.08572)
*Rocktim Jyoti Das, Harsh Singh, Diana Turmakhan, Muhammad Abdullah Sohail, Mingfei Han, Preslav Nakov, Fabio Pizzati, Ivan Laptev*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.08572) | **Categories:** cs.RO, cs.AI, cs.LG

---
