# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-29

## 目录

- [人工智能 (Artificial Intelligence) (4)](#cs-ai)
- [计算机视觉 (Computer Vision) (2)](#cs-cv)
- [机器学习 (Machine Learning) (3)](#cs-lg)
- [机器人学 (Robotics) (12)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [LIFT: Interpretable truck driving risk prediction with literature-informed fine-tuned LLMs](https://arxiv.org/abs/2510.22333)
*Xiao Hu, Yuansheng Lian, Ke Zhang, Yunxuan Li, Yuelong Su, Meng Li*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This study proposes an interpretable prediction framework with literature-informed fine-tuned (LIFT) LLMs for truck driving risk prediction. The framework integrates an LLM-driven Inference Core that predicts and explains truck driving risk, a Literature Processing Pipeline that filters and summarizes domain-specific literature into a literature knowledge base, and a Result Evaluator that evaluates the prediction performance as well as the interpretability of the LIFT LLM. After fine-tuning on a real-world truck driving risk dataset, the LIFT LLM achieved accurate risk prediction, outperforming benchmark models by 26.7% in recall and 10.1% in F1-score. Furthermore, guided by the literature knowledge base automatically constructed from 299 domain papers, the LIFT LLM produced variable importance ranking consistent with that derived from the benchmark model, while demonstrating robustness in interpretation results to various data sampling conditions. The LIFT LLM also identified potential risky scenarios by detecting key combination of variables in truck driving risk, which were verified by PERMANOVA tests. Finally, we demonstrated the contribution of the literature knowledge base and the fine-tuning process in the interpretability of the LIFT LLM, and discussed the potential of the LIFT LLM in data-driven knowledge discovery.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22333) | **Categories:** cs.AI, cs.LG

---

### [2] [Performance Trade-offs of Optimizing Small Language Models for E-Commerce](https://arxiv.org/abs/2510.21970)
*Josip Tomo Licardo, Nikola Tankovic*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language Models (LLMs) offer state-of-the-art performance in natural language understanding and generation tasks. However, the deployment of leading commercial models for specialized tasks, such as e-commerce, is often hindered by high computational costs, latency, and operational expenses. This paper investigates the viability of smaller, open-weight models as a resource-efficient alternative. We present a methodology for optimizing a one-billion-parameter Llama 3.2 model for multilingual e-commerce intent recognition. The model was fine-tuned using Quantized Low-Rank Adaptation (QLoRA) on a synthetically generated dataset designed to mimic real-world user queries. Subsequently, we applied post-training quantization techniques, creating GPU-optimized (GPTQ) and CPU-optimized (GGUF) versions. Our results demonstrate that the specialized 1B model achieves 99% accuracy, matching the performance of the significantly larger GPT-4.1 model. A detailed performance analysis revealed critical, hardware-dependent trade-offs: while 4-bit GPTQ reduced VRAM usage by 41%, it paradoxically slowed inference by 82% on an older GPU architecture (NVIDIA T4) due to dequantization overhead. Conversely, GGUF formats on a CPU achieved a speedup of up to 18x in inference throughput and a reduction of over 90% in RAM consumption compared to the FP16 baseline. We conclude that small, properly optimized open-weight models are not just a viable but a more suitable alternative for domain-specific applications, offering state-of-the-art accuracy at a fraction of the computational cost.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21970) | **Categories:** cs.AI, cs.CL

---

### [3] [A Multi-Component AI Framework for Computational Psychology: From Robust Predictive Modeling to Deployed Generative Dialogue](https://arxiv.org/abs/2510.21720)
*Anant Pareek*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: The confluence of Artificial Intelligence and Computational Psychology presents an opportunity to model, understand, and interact with complex human psychological states through computational means. This paper presents a comprehensive, multi-faceted framework designed to bridge the gap between isolated predictive modeling and an interactive system for psychological analysis. The methodology encompasses a rigorous, end-to-end development lifecycle. First, foundational performance benchmarks were established on four diverse psychological datasets using classical machine learning techniques. Second, state-of-the-art transformer models were fine-tuned, a process that necessitated the development of effective solutions to overcome critical engineering challenges, including the resolution of numerical instability in regression tasks and the creation of a systematic workflow for conducting large-scale training under severe resource constraints. Third, a generative large language model (LLM) was fine-tuned using parameter-efficient techniques to function as an interactive "Personality Brain." Finally, the entire suite of predictive and generative models was architected and deployed as a robust, scalable microservices ecosystem. Key findings include the successful stabilization of transformer-based regression models for affective computing, showing meaningful predictive performance where standard approaches failed, and the development of a replicable methodology for democratizing large-scale AI research. The significance of this work lies in its holistic approach, demonstrating a complete research-to-deployment pipeline that integrates predictive analysis with generative dialogue, thereby providing a practical model for future research in computational psychology and human-AI interaction.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21720) | **Categories:** cs.AI, cs.HC, cs.LG

---

### [4] [Embracing Trustworthy Brain-Agent Collaboration as Paradigm Extension for Intelligent Assistive Technologies](https://arxiv.org/abs/2510.22095)
*Yankai Chen, Xinni Zhang, Yifei Zhang, Yangning Li, Henry Peng Zou, Chunyu Miao, Weizhi Zhang, Xue Liu, Philip S. Yu*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Brain-Computer Interfaces (BCIs) offer a direct communication pathway between the human brain and external devices, holding significant promise for individuals with severe neurological impairments. However, their widespread adoption is hindered by critical limitations, such as low information transfer rates and extensive user-specific calibration. To overcome these challenges, recent research has explored the integration of Large Language Models (LLMs), extending the focus from simple command decoding to understanding complex cognitive states. Despite these advancements, deploying agentic AI faces technical hurdles and ethical concerns. Due to the lack of comprehensive discussion on this emerging direction, this position paper argues that the field is poised for a paradigm extension from BCI to Brain-Agent Collaboration (BAC). We emphasize reframing agents as active and collaborative partners for intelligent assistance rather than passive brain signal data processors, demanding a focus on ethical data handling, model reliability, and a robust human-agent collaboration framework to ensure these systems are safe, trustworthy, and effective.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22095) | **Categories:** cs.AI, cs.CL

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Gestura: A LVLM-Powered System Bridging Motion and Semantics for Real-Time Free-Form Gesture Understanding](https://arxiv.org/abs/2510.21814)
*Zhuoming Li, Aitong Liu, Mengxi Jia, Tengxiang Zhang, Dell Zhang, Xuelong Li*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Free-form gesture understanding is highly appealing for human-computer interaction, as it liberates users from the constraints of predefined gesture categories. However, the sole existing solution GestureGPT suffers from limited recognition accuracy and slow response times. In this paper, we propose Gestura, an end-to-end system for free-form gesture understanding. Gestura harnesses a pre-trained Large Vision-Language Model (LVLM) to align the highly dynamic and diverse patterns of free-form gestures with high-level semantic concepts. To better capture subtle hand movements across different styles, we introduce a Landmark Processing Module that compensate for LVLMs' lack of fine-grained domain knowledge by embedding anatomical hand priors. Further, a Chain-of-Thought (CoT) reasoning strategy enables step-by-step semantic inference, transforming shallow knowledge into deep semantic understanding and significantly enhancing the model's ability to interpret ambiguous or unconventional gestures. Together, these components allow Gestura to achieve robust and adaptable free-form gesture comprehension. Additionally, we have developed the first open-source dataset for free-form gesture intention reasoning and understanding with over 300,000 annotated QA pairs.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21814) | **Categories:** cs.CV, cs.AI

---

### [2] [EventFormer: A Node-graph Hierarchical Attention Transformer for Action-centric Video Event Prediction](https://arxiv.org/abs/2510.21786)
*Qile Su, Shoutai Zhu, Shuai Zhang, Baoyu Liang, Chao Tong*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Script event induction, which aims to predict the subsequent event based on the context, is a challenging task in NLP, achieving remarkable success in practical applications. However, human events are mostly recorded and presented in the form of videos rather than scripts, yet there is a lack of related research in the realm of vision. To address this problem, we introduce AVEP (Action-centric Video Event Prediction), a task that distinguishes itself from existing video prediction tasks through its incorporation of more complex logic and richer semantic information. We present a large structured dataset, which consists of about $35K$ annotated videos and more than $178K$ video clips of event, built upon existing video event datasets to support this task. The dataset offers more fine-grained annotations, where the atomic unit is represented as a multimodal event argument node, providing better structured representations of video events. Due to the complexity of event structures, traditional visual models that take patches or frames as input are not well-suited for AVEP. We propose EventFormer, a node-graph hierarchical attention based video event prediction model, which can capture both the relationships between events and their arguments and the coreferencial relationships between arguments. We conducted experiments using several SOTA video prediction models as well as LVLMs on AVEP, demonstrating both the complexity of the task and the value of the dataset. Our approach outperforms all these video prediction models. We will release the dataset and code for replicating the experiments and annotations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21786) | **Categories:** cs.CV, cs.AI, cs.MM, I.2.10

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](https://arxiv.org/abs/2510.21788)
*Larkin Liu, Jalal Etesami*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We explore the use of expert-guided bandit learning, which we refer to as online mixture-of-experts (OMoE). In this setting, given a context, a candidate committee of experts must determine how to aggregate their outputs to achieve optimal results in terms of aggregate accuracy. We propose two algorithms to address this problem. The first algorithm combines aggregate voting with UCB-driven successive elimination, efficiently pruning suboptimal exploration actions. The second algorithm employs an online weighted-majority-voting mechanism, leveraging the respective voting power of each expert proportional to their predictive power. We derive theoretical guarantees for the regret properties in the bandit setting under ideal circumstances, and empirical results are provided accordingly. As a modern study on applications, these methods are applied to the online fine-tuning of a set of expert large language models (LLMs), where after each response, the generative LLM dynamically reweighs its set of experts and/or selects the optimal committee of experts to generate the most accurate response. Our results introduce new methodologies and no-regret guarantees for combining multiple experts to improve on the performance of the an aggregate model overall.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21788) | **Categories:** cs.LG, cs.AI, I.2; G.3

---

### [2] [MARS-M: When Variance Reduction Meets Matrices](https://arxiv.org/abs/2510.21800)
*Yifeng Liu, Angela Yuan, Quanquan Gu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Matrix-based preconditioned optimizers, such as Muon, have recently been shown to be more efficient than scalar-based optimizers for training large-scale neural networks, including large language models (LLMs). On the other hand, recent benchmarks on optimizers for LLM pre-training have demonstrated that variance-reduction techniques such as MARS can achieve substantial speedups over standard optimizers that do not employ variance reduction. In this paper, to achieve the best of both worlds, we introduce MARS-M, a new optimizer that integrates the variance reduction technique in MARS with Muon. Under standard regularity conditions, we prove that Muon-M converges to a first-order stationary point at a rate of $\tilde{\mathcal{O}}(T^{-1/3})$, which improves upon $\tilde{\mathcal{O}}(T^{-1/4})$ rate attained by Muon. Our empirical results on language modeling and computer vision tasks demonstrate that MARS-M consistently yields lower losses and improved performance across various downstream benchmarks. The implementation of MARS-M is available at https://github.com/AGI-Arena/MARS/MARS_M.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21800) | **Categories:** cs.LG, math.OC, stat.ML

---

### [3] [Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks](https://arxiv.org/abs/2510.21910)
*Mahavir Dabas, Tran Huynh, Nikhil Reddy Billa, Jiachen T. Wang, Peng Gao, Charith Peris, Yao Ma, Rahul Gupta, Ming Jin, Prateek Mittal, Ruoxi Jia*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large language models remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Defending against novel jailbreaks represents a critical challenge in AI safety. Adversarial training -- designed to make models robust against worst-case perturbations -- has been the dominant paradigm for adversarial robustness. However, due to optimization challenges and difficulties in defining realistic threat models, adversarial training methods often fail on newly developed jailbreaks in practice. This paper proposes a new paradigm for improving robustness against unseen jailbreaks, centered on the Adversarial D\'ej\`a Vu hypothesis: novel jailbreaks are not fundamentally new, but largely recombinations of adversarial skills from previous attacks. We study this hypothesis through a large-scale analysis of 32 attack papers published over two years. Using an automated pipeline, we extract and compress adversarial skills into a sparse dictionary of primitives, with LLMs generating human-readable descriptions. Our analysis reveals that unseen attacks can be effectively explained as sparse compositions of earlier skills, with explanatory power increasing monotonically as skill coverage grows. Guided by this insight, we introduce Adversarial Skill Compositional Training (ASCoT), which trains on diverse compositions of skill primitives rather than isolated attack instances. ASCoT substantially improves robustness to unseen attacks, including multi-turn jailbreaks, while maintaining low over-refusal rates. We also demonstrate that expanding adversarial skill coverage, not just data scale, is key to defending against novel attacks. \textcolor{red}{\textbf{Warning: This paper contains content that may be harmful or offensive in nature.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21910) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Next-Generation LLM for UAV: From Natural Language to Autonomous Flight](https://arxiv.org/abs/2510.21739)
*Liangqi Yuan, Chuhao Deng, Dong-Jun Han, Inseok Hwang, Sabine Brunswicker, Christopher G. Brinton*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: With the rapid advancement of Large Language Models (LLMs), their capabilities in various automation domains, particularly Unmanned Aerial Vehicle (UAV) operations, have garnered increasing attention. Current research remains predominantly constrained to small-scale UAV applications, with most studies focusing on isolated components such as path planning for toy drones, while lacking comprehensive investigation of medium- and long-range UAV systems in real-world operational contexts. Larger UAV platforms introduce distinct challenges, including stringent requirements for airport-based take-off and landing procedures, adherence to complex regulatory frameworks, and specialized operational capabilities with elevated mission expectations. This position paper presents the Next-Generation LLM for UAV (NeLV) system -- a comprehensive demonstration and automation roadmap for integrating LLMs into multi-scale UAV operations. The NeLV system processes natural language instructions to orchestrate short-, medium-, and long-range UAV missions through five key technical components: (i) LLM-as-Parser for instruction interpretation, (ii) Route Planner for Points of Interest (POI) determination, (iii) Path Planner for waypoint generation, (iv) Control Platform for executable trajectory implementation, and (v) UAV monitoring. We demonstrate the system's feasibility through three representative use cases spanning different operational scales: multi-UAV patrol, multi-POI delivery, and multi-hop relocation. Beyond the current implementation, we establish a five-level automation taxonomy that charts the evolution from current LLM-as-Parser capabilities (Level 1) to fully autonomous LLM-as-Autopilot systems (Level 5), identifying technical prerequisites and research challenges at each stage.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21739) | **Categories:** cs.RO, cs.AI, cs.CL, cs.SY, eess.SY

---

### [2] [Improving the performance of AI-powered Affordable Robotics for Assistive Tasks](https://arxiv.org/abs/2510.21771)
*Dharunish Yugeswardeenoo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: By 2050, the global demand for assistive care is expected to reach 3.5 billion people, far outpacing the availability of human caregivers. Existing robotic solutions remain expensive and require technical expertise, limiting accessibility. This work introduces a low-cost robotic arm for assistive tasks such as feeding, cleaning spills, and fetching medicine. The system uses imitation learning from demonstration videos, requiring no task-specific programming or manual labeling. The robot consists of six servo motors, dual cameras, and 3D-printed grippers. Data collection via teleoperation with a leader arm yielded 50,000 video frames across the three tasks. A novel Phased Action Chunking Transformer (PACT) captures temporal dependencies and segments motion dynamics, while a Temporal Ensemble (TE) method refines trajectories to improve accuracy and smoothness. Evaluated across five model sizes and four architectures, with ten hours of real-world testing, the system achieved over 90% task accuracy, up to 40% higher than baselines. PACT enabled a 5x model size reduction while maintaining 75% accuracy. Saliency analysis showed reliance on key visual cues, and phase token gradients peaked at critical trajectory moments, indicating effective temporal reasoning. Future work will explore bimanual manipulation and mobility for expanded assistive capabilities.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21771) | **Categories:** cs.RO

---

### [3] [BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles](https://arxiv.org/abs/2510.22370)
*Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this paper, we propose Bootstrapped Language-Image Pretraining-driven Fused State Representation in Proximal Policy Optimization (BLIP-FusePPO), a novel multimodal reinforcement learning (RL) framework for autonomous lane-keeping (LK), in which semantic embeddings generated by a vision-language model (VLM) are directly fused with geometric states, LiDAR observations, and Proportional-Integral-Derivative-based (PID) control feedback within the agent observation space. The proposed method lets the agent learn driving rules that are aware of their surroundings and easy to understand by combining high-level scene understanding from the VLM with low-level control and spatial signals. Our architecture brings together semantic, geometric, and control-aware representations to make policy learning more robust. A hybrid reward function that includes semantic alignment, LK accuracy, obstacle avoidance, and speed regulation helps learning to be more efficient and generalizable. Our method is different from the approaches that only use semantic models to shape rewards. Instead, it directly embeds semantic features into the state representation. This cuts down on expensive runtime inference and makes sure that semantic guidance is always available. The simulation results show that the proposed model is better at LK stability and adaptability than the best vision-based and multimodal RL baselines in a wide range of difficult driving situations. We make our code publicly available.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22370) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG, cs.SE

---

### [4] [A phase-aware AI car-following model for electric vehicles with adaptive cruise control: Development and validation using real-world data](https://arxiv.org/abs/2510.21735)
*Yuhui Liu, Shian Wang, Ansel Panicker, Kate Embry, Ayana Asanova, Tianyi Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Internal combustion engine (ICE) vehicles and electric vehicles (EVs) exhibit distinct vehicle dynamics. EVs provide rapid acceleration, with electric motors producing peak power across a wider speed range, and achieve swift deceleration through regenerative braking. While existing microscopic models effectively capture the driving behavior of ICE vehicles, a modeling framework that accurately describes the unique car-following dynamics of EVs is lacking. Developing such a model is essential given the increasing presence of EVs in traffic, yet creating an easy-to-use and accurate analytical model remains challenging.   To address these gaps, this study develops and validates a Phase-Aware AI (PAAI) car-following model specifically for EVs. The proposed model enhances traditional physics-based frameworks with an AI component that recognizes and adapts to different driving phases, such as rapid acceleration and regenerative braking. Using real-world trajectory data from vehicles equipped with adaptive cruise control (ACC), we conduct comprehensive simulations to validate the model's performance. The numerical results demonstrate that the PAAI model significantly improves prediction accuracy over traditional car-following models, providing an effective tool for accurately representing EV behavior in traffic simulations.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21735) | **Categories:** cs.RO, cs.AI, cs.SY, eess.SY

---

### [5] [Learn2Drive: A neural network-based framework for socially compliant automated vehicle control](https://arxiv.org/abs/2510.21736)
*Yuhui Liu, Samannita Halder, Shian Wang, Tianyi Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This study introduces a novel control framework for adaptive cruise control (ACC) in automated driving, leveraging Long Short-Term Memory (LSTM) networks and physics-informed constraints. As automated vehicles (AVs) adopt advanced features like ACC, transportation systems are becoming increasingly intelligent and efficient. However, existing AV control strategies primarily focus on optimizing the performance of individual vehicles or platoons, often neglecting their interactions with human-driven vehicles (HVs) and the broader impact on traffic flow. This oversight can exacerbate congestion and reduce overall system efficiency. To address this critical research gap, we propose a neural network-based, socially compliant AV control framework that incorporates social value orientation (SVO). This framework enables AVs to account for their influence on HVs and traffic dynamics. By leveraging AVs as mobile traffic regulators, the proposed approach promotes adaptive driving behaviors that reduce congestion, improve traffic efficiency, and lower energy consumption. Within this framework, we define utility functions for both AVs and HVs, which are optimized based on the SVO of each AV to balance its own control objectives with broader traffic flow considerations. Numerical results demonstrate the effectiveness of the proposed method in adapting to varying traffic conditions, thereby enhancing system-wide efficiency. Specifically, when the AV's control mode shifts from prioritizing energy consumption to optimizing traffic flow efficiency, vehicles in the following platoon experience at least a 58.99% increase in individual energy consumption alongside at least a 38.39% improvement in individual average speed, indicating significant enhancements in traffic dynamics.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21736) | **Categories:** cs.RO, cs.AI, cs.LG, cs.MA, cs.SY, eess.SY

---

### [6] [Avi: Action from Volumetric Inference](https://arxiv.org/abs/2510.21746)
*Harris Song, Long Le*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: We propose Avi, a novel 3D Vision-Language-Action (VLA) architecture that reframes robotic action generation as a problem of 3D perception and spatial reasoning, rather than low-level policy learning. While existing VLA models primarily operate on 2D visual inputs and are trained end-to-end on task-specific action policies, Avi leverages 3D point clouds and language-grounded scene understanding to compute actions through classical geometric transformations. Most notably, Avi does not train on previous action tokens, rather, we build upon a 3D Multi-modal Large Language Model (MLLM) to generate the next point cloud and explicitly calculate the actions through classical transformations. This approach enables generalizable behaviors that are robust to occlusions, camera pose variations, and changes in viewpoint. By treating the robotic decision-making process as a structured reasoning task over 3D representations, Avi bridges the gap between high-level language instructions and low-level actuation without requiring opaque policy learning. Our preliminary results highlight the potential of 3D vision-language reasoning as a foundation for scalable, robust robotic systems. Check it out at https://avi-3drobot.github.io/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21746) | **Categories:** cs.RO

---

### [7] [Real-time Mixed-Integer Quadratic Programming for Driving Behavior-Inspired Speed Bump Optimal Trajectory Planning](https://arxiv.org/abs/2510.21751)
*Van Nam Dinh, Van Vy Phan, Thai Son Dang, Van Du Phan, The Anh Mai, Van Chuong Le, Sy Phuong Ho, Dinh Tu Duong, Hung Cuong Ta*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper proposes a novel methodology for trajectory planning in autonomous vehicles (AVs), addressing the complex challenge of negotiating speed bumps within a unified Mixed-Integer Quadratic Programming (MIQP) framework. By leveraging Model Predictive Control (MPC), we develop trajectories that optimize both the traversal of speed bumps and overall passenger comfort. A key contribution of this work is the formulation of speed bump handling constraints that closely emulate human driving behavior, seamlessly integrating these with broader road navigation requirements. Through extensive simulations in varied urban driving environments, we demonstrate the efficacy of our approach, highlighting its ability to ensure smooth speed transitions over speed bumps while maintaining computational efficiency suitable for real-time deployment. The method's capability to handle both static road features and dynamic constraints, alongside expert human driving, represents a significant step forward in trajectory planning for urban

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21751) | **Categories:** cs.RO

---

### [8] [A Physics-Informed Neural Network Approach for UAV Path Planning in Dynamic Environments](https://arxiv.org/abs/2510.21874)
*Shuning Zhang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Unmanned aerial vehicles (UAVs) operating in dynamic wind fields must generate safe and energy-efficient trajectories under physical and environmental constraints. Traditional planners, such as A* and kinodynamic RRT*, often yield suboptimal or non-smooth paths due to discretization and sampling limitations. This paper presents a physics-informed neural network (PINN) framework that embeds UAV dynamics, wind disturbances, and obstacle avoidance directly into the learning process. Without requiring supervised data, the PINN learns dynamically feasible and collision-free trajectories by minimizing physical residuals and risk-aware objectives. Comparative simulations show that the proposed method outperforms A* and Kino-RRT* in control energy, smoothness, and safety margin, while maintaining similar flight efficiency. The results highlight the potential of physics-informed learning to unify model-based and data-driven planning, providing a scalable and physically consistent framework for UAV trajectory optimization.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.21874) | **Categories:** cs.RO, cs.AI

---

### [9] [EasyUUV: An LLM-Enhanced Universal and Lightweight Sim-to-Real Reinforcement Learning Framework for UUV Attitude Control](https://arxiv.org/abs/2510.22126)
*Guanwen Xie, Jingzehua Xu, Jiwei Tang, Yubo Huang, Shuai Zhang, Xiaofan Li*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Despite recent advances in Unmanned Underwater Vehicle (UUV) attitude control, existing methods still struggle with generalizability, robustness to real-world disturbances, and efficient deployment. To address the above challenges, this paper presents EasyUUV, a Large Language Model (LLM)-enhanced, universal, and lightweight simulation-to-reality reinforcement learning (RL) framework for robust attitude control of UUVs. EasyUUV combines parallelized RL training with a hybrid control architecture, where a learned policy outputs high-level attitude corrections executed by an adaptive S-Surface controller. A multimodal LLM is further integrated to adaptively tune controller parameters at runtime using visual and textual feedback, enabling training-free adaptation to unmodeled dynamics. Also, we have developed a low-cost 6-DoF UUV platform and applied an RL policy trained through efficient parallelized simulation. Extensive simulation and real-world experiments validate the effectiveness and outstanding performance of EasyUUV in achieving robust and adaptive UUV attitude control across diverse underwater conditions. The source code is available from the following website: https://360zmem.github.io/easyuuv/

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22126) | **Categories:** cs.RO

---

### [10] [ACG: Action Coherence Guidance for Flow-based VLA models](https://arxiv.org/abs/2510.22201)
*Minho Park, Kinam Kim, Junha Hyung, Hyojin Jang, Hoiyeong Jin, Jooyeol Yun, Hojoon Lee, Jaegul Choo*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Diffusion and flow matching models have emerged as powerful robot policies, enabling Vision-Language-Action (VLA) models to generalize across diverse scenes and instructions. Yet, when trained via imitation learning, their high generative capacity makes them sensitive to noise in human demonstrations: jerks, pauses, and jitter which reduce action coherence. Reduced action coherence causes instability and trajectory drift during deployment, failures that are catastrophic in fine-grained manipulation where precision is crucial. In this paper, we present Action Coherence Guidance (ACG) for VLA models, a training-free test-time guidance algorithm that improves action coherence and thereby yields performance gains. Evaluated on RoboCasa, DexMimicGen, and real-world SO-101 tasks, ACG consistently improves action coherence and boosts success rates across diverse manipulation tasks. Code and project page are available at https://github.com/DAVIAN-Robotics/ACG and https://DAVIAN-Robotics.github.io/ACG , respectively.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22201) | **Categories:** cs.RO

---

### [11] [Bridging Perception and Reasoning: Dual-Pipeline Neuro-Symbolic Landing for UAVs in Cluttered Environments](https://arxiv.org/abs/2510.22204)
*Weixian Qian, Sebastian Schroder, Yao Deng, Jiaohong Yao, Linfeng Liang, Xiao Cheng, Richard Han, Xi Zheng*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous landing in unstructured (cluttered, uneven, and map-poor) environments is a core requirement for Unmanned Aerial Vehicles (UAVs), yet purely vision-based or deep learning models often falter under covariate shift and provide limited interpretability. We propose NeuroSymLand, a neuro-symbolic framework that tightly couples two complementary pipelines: (i) an offline pipeline, where Large Language Models (LLMs) and human-in-the-loop refinement synthesize Scallop code from diverse landing scenarios, distilling generalizable and verifiable symbolic knowledge; and (ii) an online pipeline, where a compact foundation-based semantic segmentation model generates probabilistic Scallop facts that are composed into semantic scene graphs for real-time deductive reasoning. This design combines the perceptual strengths of lightweight foundation models with the interpretability and verifiability of symbolic reasoning. Node attributes (e.g., flatness, area) and edge relations (adjacency, containment, proximity) are computed with geometric routines rather than learned, avoiding the data dependence and latency of train-time graph builders. The resulting Scallop program encodes landing principles (avoid water and obstacles; prefer large, flat, accessible regions) and yields calibrated safety scores with ranked Regions of Interest (ROIs) and human-readable justifications. Extensive evaluations across datasets, diverse simulation maps, and real UAV hardware show that NeuroSymLand achieves higher accuracy, stronger robustness to covariate shift, and superior efficiency compared with state-of-the-art baselines, while advancing UAV safety and reliability in emergency response, surveillance, and delivery missions.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22204) | **Categories:** cs.RO, cs.AI

---

### [12] [Learning Neural Observer-Predictor Models for Limb-level Sampling-based Locomotion Planning](https://arxiv.org/abs/2510.22789)
*Abhijeet M. Kulkarni, Ioannis Poulakakis, Guoquan Huang*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Accurate full-body motion prediction is essential for the safe, autonomous navigation of legged robots, enabling critical capabilities like limb-level collision checking in cluttered environments. Simplified kinematic models often fail to capture the complex, closed-loop dynamics of the robot and its low-level controller, limiting their predictions to simple planar motion. To address this, we present a learning-based observer-predictor framework that accurately predicts this motion. Our method features a neural observer with provable UUB guarantees that provides a reliable latent state estimate from a history of proprioceptive measurements. This stable estimate initializes a computationally efficient predictor, designed for the rapid, parallel evaluation of thousands of potential trajectories required by modern sampling-based planners. We validated the system by integrating our neural predictor into an MPPI-based planner on a Vision 60 quadruped. Hardware experiments successfully demonstrated effective, limb-aware motion planning in a challenging, narrow passage and over small objects, highlighting our system's ability to provide a robust foundation for high-performance, collision-aware planning on dynamic robotic platforms.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.22789) | **Categories:** cs.RO

---
