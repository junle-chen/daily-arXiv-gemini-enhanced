# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-11-22

## 目录

- [人工智能 (Artificial Intelligence) (2)](#cs-ai)
- [计算机视觉 (Computer Vision) (7)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Detecting Sleeper Agents in Large Language Models via Semantic Drift Analysis](https://arxiv.org/abs/2511.15992)
*Shahin Zanbaghi, Ryan Rostampour, Farhan Abid, Salim Al Jarmakani*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) can be backdoored to exhibit malicious behavior under specific deployment conditions while appearing safe during training a phenomenon known as "sleeper agents." Recent work by Hubinger et al. demonstrated that these backdoors persist through safety training, yet no practical detection methods exist. We present a novel dual-method detection system combining semantic drift analysis with canary baseline comparison to identify backdoored LLMs in real-time. Our approach uses Sentence-BERT embeddings to measure semantic deviation from safe baselines, complemented by injected canary questions that monitor response consistency. Evaluated on the official Cadenza-Labs dolphin-llama3-8B sleeper agent model, our system achieves 92.5% accuracy with 100% precision (zero false positives) and 85% recall. The combined detection method operates in real-time (<1s per query), requires no model modification, and provides the first practical solution to LLM backdoor detection. Our work addresses a critical security gap in AI deployment and demonstrates that embedding-based detection can effectively identify deceptive model behavior without sacrificing deployment efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15992) | **Categories:** cs.AI

---

### [2] [FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos](https://arxiv.org/abs/2511.16183)
*Jeremie Ochin, Raphael Chekroun, Bogdan Stanciulescu, Sotiris Manitsaris*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Soccer video understanding has motivated the creation of datasets for tasks such as temporal action localization, spatiotemporal action detection (STAD), or multiobject tracking (MOT). The annotation of structured sequences of events (who does what, when, and where) used for soccer analytics requires a holistic approach that integrates both STAD and MOT. However, current action recognition methods remain insufficient for constructing reliable play-by-play data and are typically used to assist rather than fully automate annotation. Parallel research has advanced tactical modeling, trajectory forecasting, and performance analysis, all grounded in game-state and play-by-play data. This motivates leveraging tactical knowledge as a prior to support computer-vision-based predictions, enabling more automated and reliable extraction of play-by-play data. We introduce Footovision Play-by-Play Action Spotting in Soccer Dataset (FOOTPASS), the first benchmark for play-by-play action spotting over entire soccer matches in a multi-modal, multi-agent tactical context. It enables the development of methods for player-centric action spotting that exploit both outputs from computer-vision tasks (e.g., tracking, identification) and prior knowledge of soccer, including its tactical regularities over long time horizons, to generate reliable play-by-play data streams. These streams form an essential input for data-driven sports analytics.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16183) | **Categories:** cs.AI, cs.CV

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [LiSTAR: Ray-Centric World Models for 4D LiDAR Sequences in Autonomous Driving](https://arxiv.org/abs/2511.16049)
*Pei Liu, Songtao Wang, Lang Zhang, Xingyue Peng, Yuandong Lyu, Jiaxin Deng, Songxin Lu, Weiliang Ma, Xueyang Zhang, Yifei Zhan, XianPeng Lang, Jun Ma*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Synthesizing high-fidelity and controllable 4D LiDAR data is crucial for creating scalable simulation environments for autonomous driving. This task is inherently challenging due to the sensor's unique spherical geometry, the temporal sparsity of point clouds, and the complexity of dynamic scenes. To address these challenges, we present LiSTAR, a novel generative world model that operates directly on the sensor's native geometry. LiSTAR introduces a Hybrid-Cylindrical-Spherical (HCS) representation to preserve data fidelity by mitigating quantization artifacts common in Cartesian grids. To capture complex dynamics from sparse temporal data, it utilizes a Spatio-Temporal Attention with Ray-Centric Transformer (START) that explicitly models feature evolution along individual sensor rays for robust temporal coherence. Furthermore, for controllable synthesis, we propose a novel 4D point cloud-aligned voxel layout for conditioning and a corresponding discrete Masked Generative START (MaskSTART) framework, which learns a compact, tokenized representation of the scene, enabling efficient, high-resolution, and layout-guided compositional generation. Comprehensive experiments validate LiSTAR's state-of-the-art performance across 4D LiDAR reconstruction, prediction, and conditional generation, with substantial quantitative gains: reducing generation MMD by a massive 76%, improving reconstruction IoU by 32%, and lowering prediction L1 Med by 50%. This level of performance provides a powerful new foundation for creating realistic and controllable autonomous systems simulations. Project link: https://ocean-luna.github.io/LiSTAR.gitub.io.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16049) | **Categories:** cs.CV

---

### [2] [Reasoning Guided Embeddings: Leveraging MLLM Reasoning for Improved Multimodal Retrieval](https://arxiv.org/abs/2511.16150)
*Chunxu Liu, Jiyuan Yang, Ruopeng Gao, Yuhan Zhu, Feng Zhu, Rui Zhao, Limin Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multimodal embeddings are widely used in downstream tasks such as multimodal retrieval, enabling alignment of interleaved modalities in a shared representation space. While recent studies show that Multimodal Large Language Models (MLLMs) can serve as strong embedding extractors, existing approaches treat embedding extraction as a direct encoding step, overlooking the fact that MLLMs possess the generative capability for reasoning that could be leveraged to enhance representation quality. In this work, we explore how to explicitly incorporate reasoning into the embedding process. To this end, we propose Reasoning Guided Embeddings (RGE), which preserves the generative rationale process of MLLMs and couples it with contrastive training. Our method first enables the model to perform structured rationale generation conditioned on the instruction, and then extracts representations after reasoning has unfolded. This simple design enhances the context-conditional inference signals within the embedding, leading to improved multimodal representation quality. Experiments on the MMEB benchmark show that reasoning-guided conditioning improves multimodal retrieval performance by 4.9% over the non-reasoning baseline, confirming that explicit reasoning can effectively enhance embedding quality.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16150) | **Categories:** cs.CV

---

### [3] [Video2Layout: Recall and Reconstruct Metric-Grounded Cognitive Map for Spatial Reasoning](https://arxiv.org/abs/2511.16160)
*Yibin Huang, Wang Xu, Wanyue Zhang, Helu Zhi, Jingjing Huang, Yangbin Xu, Yangang Sun, Conghui Zhu, Tiejun Zhao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Spatial intelligence is a critical frontier for Multimodal Large Language Models (MLLMs), empowering them to comprehend the physical world. Drawing inspiration from human perception mechanisms, existing studies attempt to construct a coherent spatial understanding via grid-based cognitive maps from multi-frame visual inputs. However, current grid-based map methods rely on discretized raster representations, which limit the model's ability in fine-grained spatial reasoning. To overcome this limitation, we propose Video2Layout, a framework for reconstructing metric-grounded spatial layouts from video. The framework employs continuous object boundary coordinates to quantify inter-object physical distances and object size. This empowers the model with quantitative spatial computation capabilities, effectively alleviating the inherent ambiguity when describing spatial relationships in natural language. Specifically, our method comprises two core stages. First, in supervised fine-tuning stage, we construct a high-quality dataset from the AI2THOR simulator, which enables the model to learn the mapping from visual inputs to precise boundary coordinates. Subsequently, a reinforcement fine-tuning stage further enhances the model's real-world generalization capabilities. To systematically evaluate the correlation between cognitive map accuracy and image quantity, as well as how the quantity of image inputs affects spatial reasoning accuracy, we introduce QVS-Bench, a diagnostic benchmark designed to analyze the relevant mechanisms. Evaluated on QVS-Bench and mainstream spatial reasoning benchmarks, our model, V2LO-7B achieves an average improvement of 4.92% over the model trained on grid maps, validating the superiority of our method. Our code is available at https://github.com/ybrrraway/Video2Layout.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16160) | **Categories:** cs.CV

---

### [4] [EvoVLA: Self-Evolving Vision-Language-Action Model](https://arxiv.org/abs/2511.16166)
*Zeting Liu, Zida Yang, Zeyu Zhang, Hao Tang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Long-horizon robotic manipulation remains challenging for Vision-Language-Action (VLA) models despite recent progress in zero-shot generalization and simulation-to-real-world transfer. Current VLA models suffer from stage hallucination, where agents exploit coarse evaluation signals to shortcut multi-step tasks, reporting high progress without truly completing them. We present EvoVLA, a self-supervised VLA framework that addresses this issue through three complementary components: Stage-Aligned Reward (SAR), which uses triplet contrastive learning with Gemini-generated hard negatives to prevent visual shortcuts; Pose-Based Object Exploration (POE), which grounds curiosity in relative object-gripper pose instead of raw pixels; and Long-Horizon Memory, which uses selective context retention and gated fusion to stabilize intrinsic shaping during extended rollouts. Extensive evaluations on Discoverse-L, a long-horizon manipulation benchmark with three multi-stage tasks, show that EvoVLA improves average task success by 10.2 percentage points over the strongest baseline (OpenVLA-OFT), reaching 69.2 percent. EvoVLA also achieves one-and-a-half times better sample efficiency and reduces stage hallucination from 38.5 percent to 14.8 percent. Real-world deployment on physical robots reaches an average success rate of 54.6 percent across four manipulation tasks, outperforming OpenVLA-OFT by 11 points, demonstrating effective sim-to-real transfer and strong generalization. Code: https://github.com/AIGeeksGroup/EvoVLA. Website: https://aigeeksgroup.github.io/EvoVLA.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16166) | **Categories:** cs.CV

---

### [5] [Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight](https://arxiv.org/abs/2511.16175)
*Yi Yang, Xueqi Li, Yiyang Chen, Jin Song, Yihan Wang, Zipeng Xiao, Jiadi Su, You Qiaoben, Pengfei Liu, Zhijie Deng*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in Vision-Language-Action (VLA) models demonstrate that visual signals can effectively complement sparse action supervisions. However, letting VLA directly predict high-dimensional visual states can distribute model capacity and incur prohibitive training cost, while compressing visual states into more compact supervisory signals inevitably incurs information bottlenecks. Moreover, existing methods often suffer from poor comprehension and reasoning capabilities due to the neglect of language supervision. This paper introduces Mantis, a novel framework featuring a Disentangled Visual Foresight (DVF) to tackle these issues. Specifically, Mantis decouples visual foresight prediction from the backbone with the combination of meta queries and a diffusion Transformer (DiT) head. With the current visual state provided to the DiT via a residual connection, a simple next-state prediction objective enables the meta queries to automatically capture the latent actions that delineate the visual trajectory, and hence boost the learning of explicit actions. The disentanglement reduces the burden of the VLA backbone, enabling it to maintain comprehension and reasoning capabilities through language supervision. Empirically, pretrained on human manipulation videos, robot demonstrations, and image-text pairs, Mantis achieves a 96.7% success rate on LIBERO benchmark after fine-tuning, surpassing powerful baselines while exhibiting high convergence speed. Real-world evaluations show that Mantis outperforms $π_{0.5}$, a leading open-source VLA model, particularly in instruction-following capability, generalization to unseen instructions, and reasoning ability. Code and weights are released to support the open-source community.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16175) | **Categories:** cs.CV, cs.AI

---

### [6] [SwiTrack: Tri-State Switch for Cross-Modal Object Tracking](https://arxiv.org/abs/2511.16227)
*Boyue Xu, Ruichao Hou, Tongwei Ren, Dongming Zhou, Gangshan Wu, Jinde Cao*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Cross-modal object tracking (CMOT) is an emerging task that maintains target consistency while the video stream switches between different modalities, with only one modality available in each frame, mostly focusing on RGB-Near Infrared (RGB-NIR) tracking. Existing methods typically connect parallel RGB and NIR branches to a shared backbone, which limits the comprehensive extraction of distinctive modality-specific features and fails to address the issue of object drift, especially in the presence of unreliable inputs. In this paper, we propose SwiTrack, a novel state-switching framework that redefines CMOT through the deployment of three specialized streams. Specifically, RGB frames are processed by the visual encoder, while NIR frames undergo refinement via a NIR gated adapter coupled with the visual encoder to progressively calibrate shared latent space features, thereby yielding more robust cross-modal representations. For invalid modalities, a consistency trajectory prediction module leverages spatio-temporal cues to estimate target movement, ensuring robust tracking and mitigating drift. Additionally, we incorporate dynamic template reconstruction to iteratively update template features and employ a similarity alignment loss to reinforce feature consistency. Experimental results on the latest benchmarks demonstrate that our tracker achieves state-of-the-art performance, boosting precision rate and success rate gains by 7.2\% and 4.3\%, respectively, while maintaining real-time tracking at 65 frames per second. Code and models are available at https://github.com/xuboyue1999/SwiTrack.git.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16227) | **Categories:** cs.CV

---

### [7] [Mem-MLP: Real-Time 3D Human Motion Generation from Sparse Inputs](https://arxiv.org/abs/2511.16264)
*Sinan Mutlu, Georgios F. Angelis, Savas Ozkan, Paul Wisbey, Anastasios Drosou, Mete Ozay*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Realistic and smooth full-body tracking is crucial for immersive AR/VR applications. Existing systems primarily track head and hands via Head Mounted Devices (HMDs) and controllers, making the 3D full-body reconstruction in-complete. One potential approach is to generate the full-body motions from sparse inputs collected from limited sensors using a Neural Network (NN) model. In this paper, we propose a novel method based on a multi-layer perceptron (MLP) backbone that is enhanced with residual connections and a novel NN-component called Memory-Block. In particular, Memory-Block represents missing sensor data with trainable code-vectors, which are combined with the sparse signals from previous time instances to improve the temporal consistency. Furthermore, we formulate our solution as a multi-task learning problem, allowing our MLP-backbone to learn robust representations that boost accuracy. Our experiments show that our method outperforms state-of-the-art baselines by substantially reducing prediction errors. Moreover, it achieves 72 FPS on mobile HMDs that ultimately improves the accuracy-running time tradeoff.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16264) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Pathlet Variational Auto-Encoder for Robust Trajectory Generation](https://arxiv.org/abs/2511.16105)
*Yuanbo Tang, Yan Tang, Zixuan Zhang, Zihui Zhao, Yang Li*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory generation has recently drawn growing interest in privacy-preserving urban mobility studies and location-based service applications. Although many studies have used deep learning or generative AI methods to model trajectories and have achieved promising results, the robustness and interpretability of such models are largely unexplored. This limits the application of trajectory generation algorithms on noisy real-world data and their trustworthiness in downstream tasks. To address this issue, we exploit the regular structure in urban trajectories and propose a deep generative model based on the pathlet representation, which encode trajectories with binary vectors associated with a learned dictionary of trajectory segments. Specifically, we introduce a probabilistic graphical model to describe the trajectory generation process, which includes a Variational Autoencoder (VAE) component and a linear decoder component. During training, the model can simultaneously learn the latent embedding of pathlet representations and the pathlet dictionary that captures mobility patterns in the trajectory dataset. The conditional version of our model can also be used to generate customized trajectories based on temporal and spatial constraints.   Our model can effectively learn data distribution even using noisy data, achieving relative improvements of $35.4\%$ and $26.3\%$ over strong baselines on two real-world trajectory datasets. Moreover, the generated trajectories can be conveniently utilized for multiple downstream tasks, including trajectory prediction and data denoising. Lastly, the framework design offers a significant efficiency advantage, saving $64.8\%$ of the time and $56.5\%$ of GPU memory compared to previous approaches.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16105) | **Categories:** cs.LG

---

### [2] [Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning](https://arxiv.org/abs/2511.16043)
*Peng Xia, Kaide Zeng, Jiaqi Liu, Can Qin, Fang Wu, Yiyang Zhou, Caiming Xiong, Huaxiu Yao*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Model (LLM) Agents, often trained with Reinforcement Learning (RL), are constrained by a dependency on human-curated data, limiting scalability and tethering AI to human knowledge. Existing self-evolution frameworks offer an alternative but are typically restricted by the model's inherent capabilities and single-round interactions, hindering the development of complex curricula involving tool use or dynamic reasoning. We introduce Agent0, a fully autonomous framework that evolves high-performing agents without external data through multi-step co-evolution and seamless tool integration. Agent0 establishes a symbiotic competition between two agents initialized from the same base LLM: a curriculum agent that proposes increasingly challenging frontier tasks, and an executor agent that learns to solve them. We integrate external tools to enhance the executor's problem-solving capacity; this improvement, in turn, pressures the curriculum agent to construct more complex, tool-aware tasks. Through this iterative process, Agent0 establishes a self-reinforcing cycle that continuously produces high-quality curricula. Empirically, Agent0 substantially boosts reasoning capabilities, improving the Qwen3-8B-Base model by 18% on mathematical reasoning and 24% on general reasoning benchmarks. Code is available at https://github.com/aiming-lab/Agent0.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16043) | **Categories:** cs.LG

---

### [3] [GeoPTH: A Lightweight Approach to Category-Based Trajectory Retrieval via Geometric Prototype Trajectory Hashing](https://arxiv.org/abs/2511.16258)
*Yang Xu, Zuliang Yang, Kai Ming Ting*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory similarity retrieval is an important part of spatiotemporal data mining, however, existing methods have the following limitations: traditional metrics are computationally expensive, while learning-based methods suffer from substantial training costs and potential instability. This paper addresses these problems by proposing \textbf{Geo}metric \textbf{P}rototype \textbf{T}rajectory \textbf{H}ashing (GeoPTH), a novel, lightweight, and non-learning framework for efficient category-based trajectory retrieval. GeoPTH constructs data-dependent hash functions by using representative trajectory prototypes, i.e., small point sets preserving geometric characteristics, as anchors. The hashing process is efficient, which involves mapping a new trajectory to its closest prototype via a robust, \textit{Hausdorff} metric. Extensive experiments show that GeoPTH's retrieval accuracy is highly competitive with both traditional metrics and state-of-the-art learning methods, and it significantly outperforms binary codes generated through simple binarization of the learned embeddings. Critically, GeoPTH consistently outperforms all competitors in terms of efficiency. Our work demonstrates that a lightweight, prototype-centric approach offers a practical and powerful alternative, achieving an exceptional retrieval performance and computational efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16258) | **Categories:** cs.LG

---

### [4] [Beyond Generative AI: World Models for Clinical Prediction, Counterfactuals, and Planning](https://arxiv.org/abs/2511.16333)
*Mohammad Areeb Qazi, Maryam Nadeem, Mohammad Yaqub*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Healthcare requires AI that is predictive, reliable, and data-efficient. However, recent generative models lack physical foundation and temporal reasoning required for clinical decision support. As scaling language models show diminishing returns for grounded clinical reasoning, world models are gaining traction because they learn multimodal, temporally coherent, and action-conditioned representations that reflect the physical and causal structure of care. This paper reviews World Models for healthcare systems that learn predictive dynamics to enable multistep rollouts, counterfactual evaluation and planning. We survey recent work across three domains: (i) medical imaging and diagnostics (e.g., longitudinal tumor simulation, projection-transition modeling, and Joint Embedding Predictive Architecture i.e., JEPA-style predictive representation learning), (ii) disease progression modeling from electronic health records (generative event forecasting at scale), and (iii) robotic surgery and surgical planning (action-conditioned guidance and control). We also introduce a capability rubric: L1 temporal prediction, L2 action-conditioned prediction, L3 counterfactual rollouts for decision support, and L4 planning/control. Most reviewed systems achieve L1--L2, with fewer instances of L3 and rare L4. We identify cross-cutting gaps that limit clinical reliability; under-specified action spaces and safety constraints, weak interventional validation, incomplete multimodal state construction, and limited trajectory-level uncertainty calibration. This review outlines a research agenda for clinically robust prediction-first world models that integrate generative backbones (transformers, diffusion, VAE) with causal/mechanical foundation for safe decision support in healthcare.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16333) | **Categories:** cs.LG

---

### [5] [FreqFlow: Long-term forecasting using lightweight flow matching](https://arxiv.org/abs/2511.16426)
*Seyed Mohamad Moghadas, Bruno Cornelis, Adrian Munteanu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multivariate time-series (MTS) forecasting is fundamental to applications ranging from urban mobility and resource management to climate modeling. While recent generative models based on denoising diffusion have advanced state-of-the-art performance in capturing complex data distributions, they suffer from significant computational overhead due to iterative stochastic sampling procedures that limit real-time deployment. Moreover, these models can be brittle when handling high-dimensional, non-stationary, and multi-scale periodic patterns characteristic of real-world sensor networks. We introduce FreqFlow, a novel framework that leverages conditional flow matching in the frequency domain for deterministic MTS forecasting. Unlike conventional approaches that operate in the time domain, FreqFlow transforms the forecasting problem into the spectral domain, where it learns to model amplitude and phase shifts through a single complex-valued linear layer. This frequency-domain formulation enables the model to efficiently capture temporal dynamics via complex multiplication, corresponding to scaling and temporal translations. The resulting architecture is exceptionally lightweight with only 89k parameters - an order of magnitude smaller than competing diffusion-based models-while enabling single-pass deterministic sampling through ordinary differential equation (ODE) integration. Our approach decomposes MTS signals into trend, seasonal, and residual components, with the flow matching mechanism specifically designed for residual learning to enhance long-term forecasting accuracy. Extensive experiments on real-world traffic speed, volume, and flow datasets demonstrate that FreqFlow achieves state-of-the-art forecasting performance, on average 7\% RMSE improvements, while being significantly faster and more parameter-efficient than existing methods

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16426) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Semantic Glitch: Agency and Artistry in an Autonomous Pixel Cloud](https://arxiv.org/abs/2511.16048)
*Qing Zhang, Jing Huang, Mingyang Xu, Jun Rekimoto*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: While mainstream robotics pursues metric precision and flawless performance, this paper explores the creative potential of a deliberately "lo-fi" approach. We present the "Semantic Glitch," a soft flying robotic art installation whose physical form, a 3D pixel style cloud, is a "physical glitch" derived from digital archaeology. We detail a novel autonomous pipeline that rejects conventional sensors like LiDAR and SLAM, relying solely on the qualitative, semantic understanding of a Multimodal Large Language Model to navigate. By authoring a bio-inspired personality for the robot through a natural language prompt, we create a "narrative mind" that complements the "weak," historically, loaded body. Our analysis begins with a 13-minute autonomous flight log, and a follow-up study statistically validates the framework's robustness for authoring quantifiably distinct personas. The combined analysis reveals emergent behaviors, from landmark-based navigation to a compelling "plan to execution" gap, and a character whose unpredictable, plausible behavior stems from a lack of precise proprioception. This demonstrates a lo-fi framework for creating imperfect companions whose success is measured in character over efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16048) | **Categories:** cs.RO, cs.AI, cs.HC

---

### [2] [MiMo-Embodied: X-Embodied Foundation Model Technical Report](https://arxiv.org/abs/2511.16518)
*Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, Long Chen*

Main category: cs.RO

TL;DR: MiMo-Embodied 是首个跨具身基础模型，在自动驾驶和具身智能领域均达到最先进的性能。


<details>
  <summary>Details</summary>
Motivation: 解决自动驾驶和具身智能领域缺乏通用基础模型的问题。

Method: 通过多阶段学习、精选数据构建和 CoT/RL 微调，整合自动驾驶和具身智能。

Result: MiMo-Embodied 在 17 个具身 AI 基准测试和 12 个自动驾驶基准测试中均显著优于现有模型。

Conclusion: 自动驾驶和具身智能这两个领域表现出强大的正向迁移和相互促进作用。

Abstract: 我们开源了 MiMo-Embodied，这是首个跨具身基础模型，它成功地整合了自动驾驶和具身智能，并在两个领域都达到了最先进的性能。MiMo-Embodied 在任务规划、可供性预测和空间理解等 17 个具身 AI 基准测试中，以及在环境感知、状态预测和驾驶规划等 12 个自动驾驶基准测试中，都创造了新的记录。在这些任务中，MiMo-Embodied 显著优于现有的开源、闭源和专用基线模型。我们的结果表明，通过多阶段学习、精选数据构建和 CoT/RL 微调，这两个领域表现出强大的正向迁移和相互促进作用。我们提供了模型设计和训练方法的详细分析，以促进进一步的研究。代码和模型可在 https://github.com/XiaomiMiMo/MiMo-Embodied 获取。

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16518) | **Categories:** cs.RO, cs.CL, cs.CV

---

### [3] [I've Changed My Mind: Robots Adapting to Changing Human Goals during Collaboration](https://arxiv.org/abs/2511.15914)
*Debasmita Ghose, Oz Gitelson, Ryan Jin, Grace Abawe, Marynel Vazquez, Brian Scassellati*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: For effective human-robot collaboration, a robot must align its actions with human goals, even as they change mid-task. Prior approaches often assume fixed goals, reducing goal prediction to a one-time inference. However, in real-world scenarios, humans frequently shift goals, making it challenging for robots to adapt without explicit communication. We propose a method for detecting goal changes by tracking multiple candidate action sequences and verifying their plausibility against a policy bank. Upon detecting a change, the robot refines its belief in relevant past actions and constructs Receding Horizon Planning (RHP) trees to actively select actions that assist the human while encouraging Differentiating Actions to reveal their updated goal. We evaluate our approach in a collaborative cooking environment with up to 30 unique recipes and compare it to three comparable human goal prediction algorithms. Our method outperforms all baselines, quickly converging to the correct goal after a switch, reducing task completion time, and improving collaboration efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15914) | **Categories:** cs.RO

---

### [4] [PushingBots: Collaborative Pushing via Neural Accelerated Combinatorial Hybrid Optimization](https://arxiv.org/abs/2511.15995)
*Zili Tang, Ying Zhang, Meng Guo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Many robots are not equipped with a manipulator and many objects are not suitable for prehensile manipulation (such as large boxes and cylinders). In these cases, pushing is a simple yet effective non-prehensile skill for robots to interact with and further change the environment. Existing work often assumes a set of predefined pushing modes and fixed-shape objects. This work tackles the general problem of controlling a robotic fleet to push collaboratively numerous arbitrary objects to respective destinations, within complex environments of cluttered and movable obstacles. It incorporates several characteristic challenges for multi-robot systems such as online task coordination under large uncertainties of cost and duration, and for contact-rich tasks such as hybrid switching among different contact modes, and under-actuation due to constrained contact forces. The proposed method is based on combinatorial hybrid optimization over dynamic task assignments and hybrid execution via sequences of pushing modes and associated forces. It consists of three main components: (I) the decomposition, ordering and rolling assignment of pushing subtasks to robot subgroups; (II) the keyframe guided hybrid search to optimize the sequence of parameterized pushing modes for each subtask; (III) the hybrid control to execute these modes and transit among them. Last but not least, a diffusion-based accelerator is adopted to predict the keyframes and pushing modes that should be prioritized during hybrid search; and further improve planning efficiency. The framework is complete under mild assumptions. Its efficiency and effectiveness under different numbers of robots and general-shaped objects are validated extensively in simulations and hardware experiments, as well as generalizations to heterogeneous robots, planar assembly and 6D pushing.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.15995) | **Categories:** cs.RO

---

### [5] [PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](https://arxiv.org/abs/2511.16200)
*Kewei Chen, Yayu Long, Mingsheng Shang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Multi-robot systems in complex physical collaborations face a "shared brain dilemma": transmitting high-dimensional multimedia data (e.g., video streams at ~30MB/s) creates severe bandwidth bottlenecks and decision-making latency. To address this, we propose PIPHEN, an innovative distributed physical cognition-control framework. Its core idea is to replace "raw data communication" with "semantic communication" by performing "semantic distillation" at the robot edge, reconstructing high-dimensional perceptual data into compact, structured physical representations. This idea is primarily realized through two key components: (1) a novel Physical Interaction Prediction Network (PIPN), derived from large model knowledge distillation, to generate this representation; and (2) a Hamiltonian Energy Network (HEN) controller, based on energy conservation, to precisely translate this representation into coordinated actions. Experiments show that, compared to baseline methods, PIPHEN can compress the information representation to less than 5% of the original data volume and reduce collaborative decision-making latency from 315ms to 76ms, while significantly improving task success rates. This work provides a fundamentally efficient paradigm for resolving the "shared brain dilemma" in resource-constrained multi-robot systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16200) | **Categories:** cs.RO

---

### [6] [DynaMimicGen: A Data Generation Framework for Robot Learning of Dynamic Tasks](https://arxiv.org/abs/2511.16223)
*Vincenzo Pomponi, Paolo Franceschi, Stefano Baraldo, Loris Roveda, Oliver Avram, Luca Maria Gambardella, Anna Valente*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning robust manipulation policies typically requires large and diverse datasets, the collection of which is time-consuming, labor-intensive, and often impractical for dynamic environments. In this work, we introduce DynaMimicGen (D-MG), a scalable dataset generation framework that enables policy training from minimal human supervision while uniquely supporting dynamic task settings. Given only a few human demonstrations, D-MG first segments the demonstrations into meaningful sub-tasks, then leverages Dynamic Movement Primitives (DMPs) to adapt and generalize the demonstrated behaviors to novel and dynamically changing environments. Improving prior methods that rely on static assumptions or simplistic trajectory interpolation, D-MG produces smooth, realistic, and task-consistent Cartesian trajectories that adapt in real time to changes in object poses, robot states, or scene geometry during task execution. Our method supports different scenarios - including scene layouts, object instances, and robot configurations - making it suitable for both static and highly dynamic manipulation tasks. We show that robot agents trained via imitation learning on D-MG-generated data achieve strong performance across long-horizon and contact-rich benchmarks, including tasks like cube stacking and placing mugs in drawers, even under unpredictable environment changes. By eliminating the need for extensive human demonstrations and enabling generalization in dynamic settings, D-MG offers a powerful and efficient alternative to manual data collection, paving the way toward scalable, autonomous robot learning.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16223) | **Categories:** cs.RO

---

### [7] [FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](https://arxiv.org/abs/2511.16233)
*Kewei Chen, Yayu Long, Shuai Li, Mingsheng Shang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The powerful generalization of Vision-Language-Action (VLA) models is bottlenecked by their heavy reliance on massive, redundant, and unevenly valued datasets, hindering their widespread application. Existing model-centric optimization paths, such as model compression (which often leads to performance degradation) or policy distillation (whose products are model-dependent and lack generality), fail to fundamentally address this data-level challenge. To this end, this paper introduces FT-NCFM, a fundamentally different, data-centric generative data distillation framework. Our framework employs a self-contained Fact-Tracing (FT) engine that combines causal attribution with programmatic contrastive verification to assess the intrinsic value of samples. Guided by these assessments, an adversarial NCFM process synthesizes a model-agnostic, information-dense, and reusable data asset. Experimental results on several mainstream VLA benchmarks show that models trained on just 5% of our distilled coreset achieve a success rate of 85-90% compared with training on the full dataset, while reducing training time by over 80%. Our work demonstrates that intelligent data distillation is a highly promising new path for building efficient, high-performance VLA models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16233) | **Categories:** cs.RO

---

### [8] [Flow-Aided Flight Through Dynamic Clutters From Point To Motion](https://arxiv.org/abs/2511.16372)
*Bowen Xu, Zexuan Yan, Minghao Lu, Xiyu Fan, Yi Luo, Youshen Lin, Zhiqiang Chen, Yeke Chen, Qiyuan Qiao, Peng Lu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Challenges in traversing dynamic clutters lie mainly in the efficient perception of the environmental dynamics and the generation of evasive behaviors considering obstacle movement. Previous solutions have made progress in explicitly modeling the dynamic obstacle motion for avoidance, but this key dependency of decision-making is time-consuming and unreliable in highly dynamic scenarios with occlusions. On the contrary, without introducing object detection, tracking, and prediction, we empower the reinforcement learning (RL) with single LiDAR sensing to realize an autonomous flight system directly from point to motion. For exteroception, a depth sensing distance map achieving fixed-shape, low-resolution, and detail-safe is encoded from raw point clouds, and an environment change sensing point flow is adopted as motion features extracted from multi-frame observations. These two are integrated into a lightweight and easy-to-learn representation of complex dynamic environments. For action generation, the behavior of avoiding dynamic threats in advance is implicitly driven by the proposed change-aware sensing representation, where the policy optimization is indicated by the relative motion modulated distance field. With the deployment-friendly sensing simulation and dynamics model-free acceleration control, the proposed system shows a superior success rate and adaptability to alternatives, and the policy derived from the simulator can drive a real-world quadrotor with safe maneuvers.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16372) | **Categories:** cs.RO

---

### [9] [LAOF: Robust Latent Action Learning with Optical Flow Constraints](https://arxiv.org/abs/2511.16407)
*Xizhou Bu, Jiexi Lyu, Fulei Sun, Ruichen Yang, Zhiqiang Ma, Wei Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning latent actions from large-scale videos is crucial for the pre-training of scalable embodied foundation models, yet existing methods often struggle with action-irrelevant distractors. Although incorporating action supervision can alleviate these distractions, its effectiveness is restricted by the scarcity of available action labels. Optical flow represents pixel-level motion between consecutive frames, naturally suppressing background elements and emphasizing moving objects. Motivated by this, we propose robust Latent Action learning with Optical Flow constraints, called LAOF, a pseudo-supervised framework that leverages the agent's optical flow as an action-driven signal to learn latent action representations robust to distractors. Experimental results show that the latent representations learned by LAOF outperform existing methods on downstream imitation learning and reinforcement learning tasks. This superior performance arises from optical flow constraints, which substantially stabilize training and improve the quality of latent representations under extremely label-scarce conditions, while remaining effective as the proportion of action labels increases to 10 percent. Importantly, even without action supervision, LAOF matches or surpasses action-supervised methods trained with 1 percent of action labels.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16407) | **Categories:** cs.RO

---

### [10] [InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy](https://arxiv.org/abs/2511.16651)
*Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, Jiangmiao Pang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent works explore how real and synthetic data contribute to Vision-Language-Action (VLA) models' generalization. While current VLA models have shown the strong effectiveness of large-scale real-robot pre-training, synthetic data has not previously demonstrated comparable capability at scale. This paper provides the first evidence that synthetic data alone can match the performance of the strongest $π$-dataset in pre-training a VLA model, revealing the substantial value of large-scale simulation. The resulting model also exhibits surprisingly zero-shot sim-to-real transfer on several challenging tasks. Our synthetic dataset, InternData-A1, contains over 630k trajectories and 7,433 hours across 4 embodiments, 18 skills, 70 tasks, and 227 scenes, covering rigid, articulated, deformable, and fluid-object manipulation. It is generated through a highly autonomous, fully decoupled, and compositional simulation pipeline that enables long-horizon skill composition, flexible task assembly, and heterogeneous embodiments with minimal manual tuning. Using the same architecture as $π_0$, we pre-train a model entirely on InternData-A1 and find that it matches the official $π_0$ across 49 simulation tasks, 5 real-world tasks, and 4 long-horizon dexterous tasks. We release the dataset and will open-source the generation pipeline to broaden access to large-scale robotic data and to lower the barrier to scalable data creation for embodied AI research.

</details>

[**[PDF]**](https://arxiv.org/pdf/2511.16651) | **Categories:** cs.RO

---
