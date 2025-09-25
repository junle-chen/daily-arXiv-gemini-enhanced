# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-09-26

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算语言学 (Computation and Language) (1)](#cs-cl)
- [计算机视觉 (Computer Vision) (3)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (10)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Embodied AI: From LLMs to World Models](https://arxiv.org/abs/2509.20021)
*Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20021) | **Categories:** cs.AI, cs.CL, cs.RO

---

### [2] [Steerable Adversarial Scenario Generation through Test-Time Preference Alignment](https://arxiv.org/abs/2509.20102)
*Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Sun, Haotian Shi, Wei Ma, Jian Sun*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Adversarial scenario generation is a cost-effective approach for safety assessment of autonomous driving systems. However, existing methods are often constrained to a single, fixed trade-off between competing objectives such as adversariality and realism. This yields behavior-specific models that cannot be steered at inference time, lacking the efficiency and flexibility to generate tailored scenarios for diverse training and testing requirements. In view of this, we reframe the task of adversarial scenario generation as a multi-objective preference alignment problem and introduce a new framework named \textbf{S}teerable \textbf{A}dversarial scenario \textbf{GE}nerator (SAGE). SAGE enables fine-grained test-time control over the trade-off between adversariality and realism without any retraining. We first propose hierarchical group-based preference optimization, a data-efficient offline alignment method that learns to balance competing objectives by decoupling hard feasibility constraints from soft preferences. Instead of training a fixed model, SAGE fine-tunes two experts on opposing preferences and constructs a continuous spectrum of policies at inference time by linearly interpolating their weights. We provide theoretical justification for this framework through the lens of linear mode connectivity. Extensive experiments demonstrate that SAGE not only generates scenarios with a superior balance of adversariality and realism but also enables more effective closed-loop training of driving policies. Project page: https://tongnie.github.io/SAGE/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20102) | **Categories:** cs.AI

---

### [3] [Design Insights and Comparative Evaluation of a Hardware-Based Cooperative Perception Architecture for Lane Change Prediction](https://arxiv.org/abs/2509.20218)
*Mohamed Manzour, Catherine M. Elias, Omar M. Shehata, Rubén Izquierdo, Miguel Ángel Sotelo*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Research on lane change prediction has gained attention in the last few years. Most existing works in this area have been conducted in simulation environments or with pre-recorded datasets, these works often rely on simplified assumptions about sensing, communication, and traffic behavior that do not always hold in practice. Real-world deployments of lane-change prediction systems are relatively rare, and when they are reported, the practical challenges, limitations, and lessons learned are often under-documented. This study explores cooperative lane-change prediction through a real hardware deployment in mixed traffic and shares the insights that emerged during implementation and testing. We highlight the practical challenges we faced, including bottlenecks, reliability issues, and operational constraints that shaped the behavior of the system. By documenting these experiences, the study provides guidance for others working on similar pipelines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20218) | **Categories:** cs.AI, cs.AR, cs.CV, cs.LG

---


## 计算语言学 (Computation and Language) [cs.CL]
### [1] [Readme_AI: Dynamic Context Construction for Large Language Models](https://arxiv.org/abs/2509.19322)
*Millie Vyas, Timothy Blattner, Alden Dima*

Main category: cs.CL

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite being trained on significant amounts of data, Large Language Models (LLMs) can provide inaccurate or unreliable information in the context of a user's specific query. Given query-specific context significantly improves the usefulness of its responses. In this paper, we present a specification that can be used to dynamically build context for data sources. The data source owner creates the file containing metadata for LLMs to use when reasoning about dataset-related queries. To demonstrate our proposed specification, we created a prototype Readme_AI Model Context Protocol (MCP) server that retrieves the metadata from the data source and uses it to dynamically build context. Some features that make this specification dynamic are the extensible types that represent crawling web-pages, fetching data from data repositories, downloading and parsing publications, and general text. The context is formatted and grouped using user-specified tags that provide clear contextual information for the LLM to reason about the content. We demonstrate the capabilities of this early prototype by asking the LLM about the NIST-developed Hedgehog library, for which common LLMs often provides inaccurate and irrelevant responses containing hallucinations. With Readme_AI, the LLM receives enough context that it is now able to reason about the library and its use, and even generate code interpolated from examples that were included in the Readme_AI file provided by Hedgehog's developer. Our primary contribution is a extensible protocol for dynamically grounding LLMs in specialized, owner-provided data, enhancing responses from LLMs and reducing hallucinations. The source code for the Readme_AI tool is posted here: https://github.com/usnistgov/readme_ai .

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19322) | **Categories:** cs.CL, cs.AI

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning](https://arxiv.org/abs/2509.19552)
*Manyi Yao, Bingbing Zhuang, Sparsh Garg, Amit Roy-Chowdhury, Christian Shelton, Manmohan Chandraker, Abhishek Aich*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19552) | **Categories:** cs.CV

---

### [2] [OmniScene: Attention-Augmented Multimodal 4D Scene Understanding for Autonomous Driving](https://arxiv.org/abs/2509.19973)
*Pei Liu, Hongliang Lu, Haichao Liu, Haipeng Liu, Xin Liu, Ruoyu Yao, Shengbo Eben Li, Jun Ma*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human vision is capable of transforming two-dimensional observations into an egocentric three-dimensional scene understanding, which underpins the ability to translate complex scenes and exhibit adaptive behaviors. This capability, however, remains lacking in current autonomous driving systems, where mainstream approaches primarily rely on depth-based 3D reconstruction rather than true scene understanding. To address this limitation, we propose a novel human-like framework called OmniScene. First, we introduce the OmniScene Vision-Language Model (OmniVLM), a vision-language framework that integrates multi-view and temporal perception for holistic 4D scene understanding. Then, harnessing a teacher-student OmniVLM architecture and knowledge distillation, we embed textual representations into 3D instance features for semantic supervision, enriching feature learning, and explicitly capturing human-like attentional semantics. These feature representations are further aligned with human driving behaviors, forming a more human-like perception-understanding-action architecture. In addition, we propose a Hierarchical Fusion Strategy (HFS) to address imbalances in modality contributions during multimodal integration. Our approach adaptively calibrates the relative significance of geometric and semantic features at multiple abstraction levels, enabling the synergistic use of complementary cues from visual and textual modalities. This learnable dynamic fusion enables a more nuanced and effective exploitation of heterogeneous information. We evaluate OmniScene comprehensively on the nuScenes dataset, benchmarking it against over ten state-of-the-art models across various tasks. Our approach consistently achieves superior results, establishing new benchmarks in perception, prediction, planning, and visual question answering.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19973) | **Categories:** cs.CV

---

### [3] [SynchroRaMa : Lip-Synchronized and Emotion-Aware Talking Face Generation via Multi-Modal Emotion Embedding](https://arxiv.org/abs/2509.19965)
*Phyo Thet Yee, Dimitrios Kollias, Sudeepta Mishra, Abhinav Dhall*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Audio-driven talking face generation has received growing interest, particularly for applications requiring expressive and natural human-avatar interaction. However, most existing emotion-aware methods rely on a single modality (either audio or image) for emotion embedding, limiting their ability to capture nuanced affective cues. Additionally, most methods condition on a single reference image, restricting the model's ability to represent dynamic changes in actions or attributes across time. To address these issues, we introduce SynchroRaMa, a novel framework that integrates a multi-modal emotion embedding by combining emotional signals from text (via sentiment analysis) and audio (via speech-based emotion recognition and audio-derived valence-arousal features), enabling the generation of talking face videos with richer and more authentic emotional expressiveness and fidelity. To ensure natural head motion and accurate lip synchronization, SynchroRaMa includes an audio-to-motion (A2M) module that generates motion frames aligned with the input audio. Finally, SynchroRaMa incorporates scene descriptions generated by Large Language Model (LLM) as additional textual input, enabling it to capture dynamic actions and high-level semantic attributes. Conditioning the model on both visual and textual cues enhances temporal consistency and visual realism. Quantitative and qualitative experiments on benchmark datasets demonstrate that SynchroRaMa outperforms the state-of-the-art, achieving improvements in image quality, expression preservation, and motion realism. A user study further confirms that SynchroRaMa achieves higher subjective ratings than competing methods in overall naturalness, motion diversity, and video smoothness. Our project page is available at <https://novicemm.github.io/synchrorama>.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19965) | **Categories:** cs.CV

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Wavelet Fourier Diffuser: Frequency-Aware Diffusion Model for Reinforcement Learning](https://arxiv.org/abs/2509.19305)
*Yifu Luo, Yongzhe Chang, Xueqian Wang*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion probability models have shown significant promise in offline reinforcement learning by directly modeling trajectory sequences. However, existing approaches primarily focus on time-domain features while overlooking frequency-domain features, leading to frequency shift and degraded performance according to our observation. In this paper, we investigate the RL problem from a new perspective of the frequency domain. We first observe that time-domain-only approaches inadvertently introduce shifts in the low-frequency components of the frequency domain, which results in trajectory instability and degraded performance. To address this issue, we propose Wavelet Fourier Diffuser (WFDiffuser), a novel diffusion-based RL framework that integrates Discrete Wavelet Transform to decompose trajectories into low- and high-frequency components. To further enhance diffusion modeling for each component, WFDiffuser employs Short-Time Fourier Transform and cross attention mechanisms to extract frequency-domain features and facilitate cross-frequency interaction. Extensive experiment results on the D4RL benchmark demonstrate that WFDiffuser effectively mitigates frequency shift, leading to smoother, more stable trajectories and improved decision-making performance over existing methods.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19305) | **Categories:** cs.LG, cs.AI, eess.SP

---

### [2] [DAWM: Diffusion Action World Models for Offline Reinforcement Learning via Action-Inferred Transitions](https://arxiv.org/abs/2509.19538)
*Zongyue Li, Xiao Han, Yusong Li, Niklas Strauss, Matthias Schubert*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion-based world models have demonstrated strong capabilities in synthesizing realistic long-horizon trajectories for offline reinforcement learning (RL). However, many existing methods do not directly generate actions alongside states and rewards, limiting their compatibility with standard value-based offline RL algorithms that rely on one-step temporal difference (TD) learning. While prior work has explored joint modeling of states, rewards, and actions to address this issue, such formulations often lead to increased training complexity and reduced performance in practice. We propose \textbf{DAWM}, a diffusion-based world model that generates future state-reward trajectories conditioned on the current state, action, and return-to-go, paired with an inverse dynamics model (IDM) for efficient action inference. This modular design produces complete synthetic transitions suitable for one-step TD-based offline RL, enabling effective and computationally efficient training. Empirically, we show that conservative offline RL algorithms such as TD3BC and IQL benefit significantly from training on these augmented trajectories, consistently outperforming prior diffusion-based baselines across multiple tasks in the D4RL benchmark.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19538) | **Categories:** cs.LG, cs.AI

---

### [3] [RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving](https://arxiv.org/abs/2509.19789)
*Carlo Bosio, Greg Woelki, Noureldin Hendy, Nicholas Roy, Byungsoo Kim*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19789) | **Categories:** cs.LG, cs.AI, cs.RO

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving](https://arxiv.org/abs/2509.20109)
*Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, Xianpeng Lang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20109) | **Categories:** cs.RO, cs.AI, cs.CL

---

### [2] [OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation](https://arxiv.org/abs/2509.19480)
*Noriaki Hirose, Catherine Glossop, Dhruv Shah, Sergey Levine*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Humans can flexibly interpret and compose different goal specifications, such as language instructions, spatial coordinates, or visual references, when navigating to a destination. In contrast, most existing robotic navigation policies are trained on a single modality, limiting their adaptability to real-world scenarios where different forms of goal specification are natural and complementary. In this work, we present a training framework for robotic foundation models that enables omni-modal goal conditioning for vision-based navigation. Our approach leverages a high-capacity vision-language-action (VLA) backbone and trains with three primary goal modalities: 2D poses, egocentric images, and natural language, as well as their combinations, through a randomized modality fusion strategy. This design not only expands the pool of usable datasets but also encourages the policy to develop richer geometric, semantic, and visual representations. The resulting model, OmniVLA, achieves strong generalization to unseen environments, robustness to scarce modalities, and the ability to follow novel natural language instructions. We demonstrate that OmniVLA outperforms specialist baselines across modalities and offers a flexible foundation for fine-tuning to new modalities and tasks. We believe OmniVLA provides a step toward broadly generalizable and flexible navigation policies, and a scalable path for building omni-modal robotic foundation models. We present videos showcasing OmniVLA performance and will release its checkpoints and training code on our project page.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19480) | **Categories:** cs.RO, cs.LG

---

### [3] [Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments](https://arxiv.org/abs/2509.19851)
*Benjamin Bogenberger, Oliver Harrison, Orrin Dahanaggamaarachchi, Lukas Brunke, Jingxing Qian, Siqi Zhou, Angela P. Schoellig*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Robots deployed in real-world environments, such as homes, must not only navigate safely but also understand their surroundings and adapt to environment changes. To perform tasks efficiently, they must build and maintain a semantic map that accurately reflects the current state of the environment. Existing research on semantic exploration largely focuses on static scenes without persistent object-level instance tracking. A consistent map is, however, crucial for real-world robotic applications where objects in the environment can be removed, reintroduced, or shifted over time. In this work, to close this gap, we propose an open-vocabulary, semantic exploration system for semi-static environments. Our system maintains a consistent map by building a probabilistic model of object instance stationarity, systematically tracking semi-static changes, and actively exploring areas that have not been visited for a prolonged period of time. In addition to active map maintenance, our approach leverages the map's semantic richness with LLM-based reasoning for open-vocabulary object-goal navigation. This enables the robot to search more efficiently by prioritizing contextually relevant areas. We evaluate our approach across multiple real-world semi-static environments. Our system detects 95% of map changes on average, improving efficiency by more than 29% as compared to random and patrol baselines. Overall, our approach achieves a mapping precision within 2% of a fully rebuilt map while requiring substantially less exploration and further completes object goal navigation tasks about 14% faster than the next-best tested strategy (coverage patrolling). A video of our work can be found at http://tiny.cc/sem-explor-semi-static .

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19851) | **Categories:** cs.RO

---

### [4] [LLM Trainer: Automated Robotic Data Generating via Demonstration Augmentation using LLMs](https://arxiv.org/abs/2509.20070)
*Abraham George, Amir Barati Farimani*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We present LLM Trainer, a fully automated pipeline that leverages the world knowledge of Large Language Models (LLMs) to transform a small number of human demonstrations (as few as one) into a large robot dataset for imitation learning. Our approach decomposes demonstration generation into two steps: (1) offline demonstration annotation that extracts keyframes, salient objects, and pose-object relations; and (2) online keypose retargeting that adapts those keyframes to a new scene, given an initial observation. Using these modified keypoints, our system warps the original demonstration to generate a new trajectory, which is then executed, and the resulting demo, if successful, is saved. Because the annotation is reusable across scenes, we use Thompson sampling to optimize the annotation, significantly improving generation success rate. We evaluate our method on a range of tasks, and find that our data annotation method consistently outperforms expert-engineered baselines. We further show an ensemble policy that combines the optimized LLM feed-forward plan with a learned feedback imitation learning controller. Finally, we demonstrate hardware feasibility on a Franka Emika Panda robot. For additional materials and demonstration videos, please see the project website: https://sites.google.com/andrew.cmu.edu/llm-trainer

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20070) | **Categories:** cs.RO

---

### [5] [Supercomputing for High-speed Avoidance and Reactive Planning in Robots](https://arxiv.org/abs/2509.19486)
*Kieran S. Lachmansingh, José R. González-Estrada, Ryan E. Grant, Matthew K. X. J. Pan*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents SHARP (Supercomputing for High-speed Avoidance and Reactive Planning), a proof-of-concept study demonstrating how high-performance computing (HPC) can enable millisecond-scale responsiveness in robotic control. While modern robots face increasing demands for reactivity in human--robot shared workspaces, onboard processors are constrained by size, power, and cost. Offloading to HPC offers massive parallelism for trajectory planning, but its feasibility for real-time robotics remains uncertain due to network latency and jitter. We evaluate SHARP in a stress-test scenario where a 7-DOF manipulator must dodge high-speed foam projectiles. Using a parallelized multi-goal A* search implemented with MPI on both local and remote HPC clusters, the system achieves mean planning latencies of 22.9 ms (local) and 30.0 ms (remote, ~300 km away), with avoidance success rates of 84% and 88%, respectively. These results show that when round-trip latency remains within the tens-of-milliseconds regime, HPC-side computation is no longer the bottleneck, enabling avoidance well below human reaction times. The SHARP results motivate hybrid control architectures: low-level reflexes remain onboard for safety, while bursty, high-throughput planning tasks are offloaded to HPC for scalability. By reporting per-stage timing and success rates, this study provides a reproducible template for assessing real-time feasibility of HPC-driven robotics. Collectively, SHARP reframes HPC offloading as a viable pathway toward dependable, reactive robots in dynamic environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19486) | **Categories:** cs.RO, cs.DC

---

### [6] [Trajectory Planning Using Safe Ellipsoidal Corridors as Projections of Orthogonal Trust Regions](https://arxiv.org/abs/2509.19734)
*Akshay Jaitly, Jon Arrizabalaga, Guanrui Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Planning collision free trajectories in complex environments remains a core challenge in robotics. Existing corridor based planners which rely on decomposition of the free space into collision free subsets scale poorly with environmental complexity and require explicit allocations of time windows to trajectory segments. We introduce a new trajectory parameterization that represents trajectories in a nonconvex collision free corridor as being in a convex cartesian product of balls. This parameterization allows us to decouple problem size from geometric complexity of the solution and naturally avoids explicit time allocation by allowing trajectories to evolve continuously inside ellipsoidal corridors. Building on this representation, we formulate the Orthogonal Trust Region Problem (Orth-TRP), a specialized convex program with separable block constraints, and develop a solver that exploits this parallel structure and the unique structure of each parallel subproblem for efficient optimization. Experiments on a quadrotor trajectory planning benchmark show that our approach produces smoother trajectories and lower runtimes than state-of-the-art corridor based planners, especially in highly complicated environments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19734) | **Categories:** cs.RO

---

### [7] [DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations](https://arxiv.org/abs/2509.19804)
*Sowoo Lee, Dongyun Kang, Jaehyun Park, Hae-Won Park*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces DynaFlow, a novel framework that embeds a differentiable simulator directly into a flow matching model. By generating trajectories in the action space and mapping them to dynamically feasible state trajectories via the simulator, DynaFlow ensures all outputs are physically consistent by construction. This end-to-end differentiable architecture enables training on state-only demonstrations, allowing the model to simultaneously generate physically consistent state trajectories while inferring the underlying action sequences required to produce them. We demonstrate the effectiveness of our approach through quantitative evaluations and showcase its real-world applicability by deploying the generated actions onto a physical Go1 quadruped robot. The robot successfully reproduces diverse gait present in the dataset, executes long-horizon motions in open-loop control and translates infeasible kinematic demonstrations into dynamically executable, stylistic behaviors. These hardware experiments validate that DynaFlow produces deployable, highly effective motions on real-world hardware from state-only demonstrations, effectively bridging the gap between kinematic data and real-world execution.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19804) | **Categories:** cs.RO

---

### [8] [Robot Trajectron V2: A Probabilistic Shared Control Framework for Navigation](https://arxiv.org/abs/2509.19954)
*Pinhao Song, Yurui Du, Ophelie Saussus, Sofie De Schrijver, Irene Caprara, Peter Janssen, Renaud Detry*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose a probabilistic shared-control solution for navigation, called Robot Trajectron V2 (RT-V2), that enables accurate intent prediction and safe, effective assistance in human-robot interaction. RT-V2 jointly models a user's long-term behavioral patterns and their noisy, low-dimensional control signals by combining a prior intent model with a posterior update that accounts for real-time user input and environmental context. The prior captures the multimodal and history-dependent nature of user intent using recurrent neural networks and conditional variational autoencoders, while the posterior integrates this with uncertain user commands to infer desired actions. We conduct extensive experiments to validate RT-V2 across synthetic benchmarks, human-computer interaction studies with keyboard input, and brain-machine interface experiments with non-human primates. Results show that RT-V2 outperforms the state of the art in intent estimation, provides safe and efficient navigation support, and adequately balances user autonomy with assistive intervention. By unifying probabilistic modeling, reinforcement learning, and safe optimization, RT-V2 offers a principled and generalizable approach to shared control for diverse assistive technologies.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.19954) | **Categories:** cs.RO

---

### [9] [Lidar-based Tracking of Traffic Participants with Sensor Nodes in Existing Urban Infrastructure](https://arxiv.org/abs/2509.20009)
*Simon Schäfer, Bassam Alrifaee, Ehsan Hashemi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper presents a lidar-only state estimation and tracking framework, along with a roadside sensing unit for integration with existing urban infrastructure. Urban deployments demand scalable, real-time tracking solutions, yet traditional remote sensing remains costly and computationally intensive, especially under perceptually degraded conditions. Our sensor node couples a single lidar with an edge computing unit and runs a computationally efficient, GPU-free observer that simultaneously estimates object state, class, dimensions, and existence probability. The pipeline performs: (i) state updates via an extended Kalman filter, (ii) dimension estimation using a 1D grid-map/Bayesian update, (iii) class updates via a lookup table driven by the most probable footprint, and (iv) existence estimation from track age and bounding-box consistency. Experiments in dynamic urban-like scenes with diverse traffic participants demonstrate real-time performance and high precision: The complete end-to-end pipeline finishes within \SI{100}{\milli\second} for \SI{99.88}{\%} of messages, with an excellent detection rate. Robustness is further confirmed under simulated wind and sensor vibration. These results indicate that reliable, real-time roadside tracking is feasible on CPU-only edge hardware, enabling scalable, privacy-friendly deployments within existing city infrastructure. The framework integrates with existing poles, traffic lights, and buildings, reducing deployment costs and simplifying large-scale urban rollouts and maintenance efforts.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20009) | **Categories:** cs.RO

---

### [10] [C-3TO: Continuous 3D Trajectory Optimization on Neural Euclidean Signed Distance Fields](https://arxiv.org/abs/2509.20084)
*Guillermo Gil, Jose Antonio Cobano, Luis Merino, Fernando Caballero*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper introduces a novel framework for continuous 3D trajectory optimization in cluttered environments, leveraging online neural Euclidean Signed Distance Fields (ESDFs). Unlike prior approaches that rely on discretized ESDF grids with interpolation, our method directly optimizes smooth trajectories represented by fifth-order polynomials over a continuous neural ESDF, ensuring precise gradient information throughout the entire trajectory. The framework integrates a two-stage nonlinear optimization pipeline that balances efficiency, safety and smoothness. Experimental results demonstrate that C-3TO produces collision-aware and dynamically feasible trajectories. Moreover, its flexibility in defining local window sizes and optimization parameters enables straightforward adaptation to diverse user's needs without compromising performance. By combining continuous trajectory parameterization with a continuously updated neural ESDF, C-3TO establishes a robust and generalizable foundation for safe and efficient local replanning in aerial robotics.

</details>

[**[PDF]**](https://arxiv.org/pdf/2509.20084) | **Categories:** cs.RO

---
