# 每日 ArXiv 轨迹预测与大模型摘要速递: 2025-10-03

## 目录

- [人工智能 (Artificial Intelligence) (3)](#cs-ai)
- [计算机视觉 (Computer Vision) (5)](#cs-cv)
- [机器学习 (Machine Learning) (5)](#cs-lg)
- [机器人学 (Robotics) (8)](#cs-ro)

## 人工智能 (Artificial Intelligence) [cs.AI]
### [1] [Towards Self-Evolving Benchmarks: Synthesizing Agent Trajectories via Test-Time Exploration under Validate-by-Reproduce Paradigm](https://arxiv.org/abs/2510.00415)
*Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, Jing Shao*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Recent advances in large language models (LLMs) and agent system designs have empowered agents with unprecedented levels of capability. However, existing agent benchmarks are showing a trend of rapid ceiling-hitting by newly developed agents, making it difficult to meet the demands for evaluating agent abilities. To address this problem, we propose the Trajectory-based Validated-by-Reproducing Agent-benchmark Complexity Evolution (TRACE) framework. This framework takes an original task from an existing benchmark and encourages agents to freely explore and evolve it into a new task with higher difficulty while recording validatable agent trajectories. The framework proceeds in three stages: (1) evolutionary proposal mining, which provides task evolution proposals through preliminary exploration and divergent thinking; (2) problem formation and free exploration, where proposals are conceptualized into feasible problem candidates and the agents then explore them freely while recording their execution trajectories; and (3) multi-level validation, which ensures that the evolved tasks are accompanied by validatable and reproducible trajectories. Experiments on the GAIA benchmark demonstrate that the TRACE framework consistently enhances task complexity while improving the reliability of correctness through validatable execution trajectories. This work marks a paradigm shift from static, manually curated benchmarks to dynamic, self-evolving evaluation systems, providing a sustainable and challenging runway for agent development.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00415) | **Categories:** cs.AI

---

### [2] [Collaborative-Distilled Diffusion Models (CDDM) for Accelerated and Lightweight Trajectory Prediction](https://arxiv.org/abs/2510.00627)
*Bingzhang Wang, Kehua Chen, Yinhai Wang*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Trajectory prediction is a fundamental task in Autonomous Vehicles (AVs) and Intelligent Transportation Systems (ITS), supporting efficient motion planning and real-time traffic safety management. Diffusion models have recently demonstrated strong performance in probabilistic trajectory prediction, but their large model size and slow sampling process hinder real-world deployment. This paper proposes Collaborative-Distilled Diffusion Models (CDDM), a novel method for real-time and lightweight trajectory prediction. Built upon Collaborative Progressive Distillation (CPD), CDDM progressively transfers knowledge from a high-capacity teacher diffusion model to a lightweight student model, jointly reducing both the number of sampling steps and the model size across distillation iterations. A dual-signal regularized distillation loss is further introduced to incorporate guidance from both the teacher and ground-truth data, mitigating potential overfitting and ensuring robust performance. Extensive experiments on the ETH-UCY pedestrian benchmark and the nuScenes vehicle benchmark demonstrate that CDDM achieves state-of-the-art prediction accuracy. The well-distilled CDDM retains 96.2% and 95.5% of the baseline model's ADE and FDE performance on pedestrian trajectories, while requiring only 231K parameters and 4 or 2 sampling steps, corresponding to 161x compression, 31x acceleration, and 9 ms latency. Qualitative results further show that CDDM generates diverse and accurate trajectories under dynamic agent behaviors and complex social interactions. By bridging high-performing generative models with practical deployment constraints, CDDM enables resource-efficient probabilistic prediction for AVs and ITS. Code is available at https://github.com/bingzhangw/CDDM.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00627) | **Categories:** cs.AI

---

### [3] [Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI](https://arxiv.org/abs/2510.00167)
*Diego Ortiz Barbosa, Mohit Agrawal, Yash Malegaonkar, Luis Burbano, Axel Andersson, György Dán, Henrik Sandberg, Alvaro A. Cardenas*

Main category: cs.AI

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00167) | **Categories:** cs.AI, cs.CR, cs.RO

---


## 计算机视觉 (Computer Vision) [cs.CV]
### [1] [Less is More: Lean yet Powerful Vision-Language Model for Autonomous Driving](https://arxiv.org/abs/2510.00060)
*Sheng Yang, Tong Zhan, Guancheng Chen, Yanfeng Lu, Jian Wang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In this work, we reconceptualize autonomous driving as a generalized language and formulate the trajectory planning task as next waypoint prediction. We introduce Max-V1, a novel framework for one-stage end-to-end autonomous driving. Our framework presents a single-pass generation paradigm that aligns with the inherent sequentiality of driving. This approach leverages the generative capacity of the VLM (Vision-Language Model) to enable end-to-end trajectory prediction directly from front-view camera input. The efficacy of this method is underpinned by a principled supervision strategy derived from statistical modeling. This provides a well-defined learning objective, which makes the framework highly amenable to master complex driving policies through imitation learning from large-scale expert demonstrations. Empirically, our method achieves the state-of-the-art performance on the nuScenes dataset, delivers an overall improvement of over 30% compared to prior baselines. Furthermore, it exhibits superior generalization performance on cross-domain datasets acquired from diverse vehicles, demonstrating notable potential for cross-vehicle robustness and adaptability. Due to these empirical strengths, this work introduces a model enabling fundamental driving behaviors, laying the foundation for the development of more capable self-driving agents. Code will be available upon publication.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00060) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [2] [EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations](https://arxiv.org/abs/2510.00405)
*Jiayi Liu, Jiaming Zhou, Ke Ye, Kun-Yu Lin, Allan Wang, Junwei Liang*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Reliable trajectory prediction from an ego-centric perspective is crucial for robotic navigation in human-centric environments. However, existing methods typically assume idealized observation histories, failing to account for the perceptual artifacts inherent in first-person vision, such as occlusions, ID switches, and tracking drift. This discrepancy between training assumptions and deployment reality severely limits model robustness. To bridge this gap, we introduce EgoTraj-Bench, the first real-world benchmark that grounds noisy, first-person visual histories in clean, bird's-eye-view future trajectories, enabling robust learning under realistic perceptual constraints. Building on this benchmark, we propose BiFlow, a dual-stream flow matching model that concurrently denoises historical observations and forecasts future motion by leveraging a shared latent representation. To better model agent intent, BiFlow incorporates our EgoAnchor mechanism, which conditions the prediction decoder on distilled historical features via feature modulation. Extensive experiments show that BiFlow achieves state-of-the-art performance, reducing minADE and minFDE by 10-15% on average and demonstrating superior robustness. We anticipate that our benchmark and model will provide a critical foundation for developing trajectory forecasting systems truly resilient to the challenges of real-world, ego-centric perception.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00405) | **Categories:** cs.CV, cs.AI, cs.RO

---

### [3] [KeySG: Hierarchical Keyframe-Based 3D Scene Graphs](https://arxiv.org/abs/2510.01049)
*Abdelrhman Werby, Dennis Rotondi, Fabio Scaparro, Kai O. Arras*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: In recent years, 3D scene graphs have emerged as a powerful world representation, offering both geometric accuracy and semantic richness. Combining 3D scene graphs with large language models enables robots to reason, plan, and navigate in complex human-centered environments. However, current approaches for constructing 3D scene graphs are semantically limited to a predefined set of relationships, and their serialization in large environments can easily exceed an LLM's context window. We introduce KeySG, a framework that represents 3D scenes as a hierarchical graph consisting of floors, rooms, objects, and functional elements, where nodes are augmented with multi-modal information extracted from keyframes selected to optimize geometric and visual coverage. The keyframes allow us to efficiently leverage VLM to extract scene information, alleviating the need to explicitly model relationship edges between objects, enabling more general, task-agnostic reasoning and planning. Our approach can process complex and ambiguous queries while mitigating the scalability issues associated with large scene graphs by utilizing a hierarchical retrieval-augmented generation (RAG) pipeline to extract relevant context from the graph. Evaluated across four distinct benchmarks -- including 3D object segmentation and complex query retrieval -- KeySG outperforms prior approaches on most metrics, demonstrating its superior semantic richness and efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.01049) | **Categories:** cs.CV, cs.RO

---

### [4] [PAL-UI: Planning with Active Look-back for Vision-Based GUI Agents](https://arxiv.org/abs/2510.00413)
*Zikang Liu, Junyi Li, Wayne Xin Zhao, Dawei Gao, Yaliang Li, Ji-rong Wen*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) promise human-like interaction with software applications, yet long-horizon tasks remain challenging due to memory limitations. Existing approaches either truncate history or rely on simple textual summaries, which risk losing critical information when past visual details become necessary for future decisions. In this paper, we propose \textbf{PAL-UI} (\textbf{P}lanning with \textbf{A}ctive \textbf{L}ook-back), a novel framework that enables GUI agents to adaptively retrieve past observations when required. PAL-UI combines a dual-level summarization agent, capturing both observation-level cues and action-level outcomes, with a dedicated retrieval tool that allows the agent to recall specific historical screenshots during planning. We curate a step-level instruction dataset of 8.6K samples from mobile GUI navigation trajectories and train \textbf{PAL-UI-3B} and \textbf{PAL-UI-7B} models based on Qwen2.5-VL. Extensive experiments demonstrate that PAL-UI significantly outperforms baseline models and prior methods in mobile GUI navigation tasks, even under data-efficient settings. Moreover, PAL-UI exhibits strong cross-domain generalization, achieving notable improvements in web navigation without additional training. Our work highlights the potential of active memory retrieval for long-horizon planning capabilities of vision-based GUI agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00413) | **Categories:** cs.CV

---

### [5] [Strategic Fusion of Vision Language Models: Shapley-Credited Context-Aware Dawid-Skene for Multi-Label Tasks in Autonomous Driving](https://arxiv.org/abs/2510.01126)
*Yuxiang Feng, Keyang Zhang, Hassane Ouchouid, Ashwil Kaniamparambil, Ioannis Souflas, Panagiotis Angeloudis*

Main category: cs.CV

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large vision-language models (VLMs) are increasingly used in autonomous-vehicle (AV) stacks, but hallucination limits their reliability in safety-critical pipelines. We present Shapley-credited Context-Aware Dawid-Skene with Agreement, a game-theoretic fusion method for multi-label understanding of ego-view dashcam video. It learns per-model, per-label, context-conditioned reliabilities from labelled history and, at inference, converts each model's report into an agreement-guardrailed log-likelihood ratio that is combined with a contextual prior and a public reputation state updated via Shapley-based team credit. The result is calibrated, thresholdable posteriors that (i) amplify agreement among reliable models, (ii) preserve uniquely correct single-model signals, and (iii) adapt to drift. To specialise general VLMs, we curate 1,000 real-world dashcam clips with structured annotations (scene description, manoeuvre recommendation, rationale) via an automatic pipeline that fuses HDD ground truth, vehicle kinematics, and YOLOv11 + BoT-SORT tracking, guided by a three-step chain-of-thought prompt; three heterogeneous VLMs are then fine-tuned with LoRA. We evaluate with Hamming distance, Micro-Macro-F1, and average per-video latency. Empirically, the proposed method achieves a 23% reduction in Hamming distance, 55% improvement in Macro-F1, and 47% improvement in Micro-F1 when comparing with the best single model, supporting VLM fusion as a calibrated, interpretable, and robust decision-support component for AV pipelines.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.01126) | **Categories:** cs.CV, cs.RO

---


## 机器学习 (Machine Learning) [cs.LG]
### [1] [LoRAFusion: Efficient LoRA Fine-Tuning for LLMs](https://arxiv.org/abs/2510.00206)
*Zhanda Zhu, Qidong Su, Yaoyao Ding, Kevin Song, Shang Wang, Gennady Pekhimenko*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Low-Rank Adaptation (LoRA) has become the leading Parameter-Efficient Fine-Tuning (PEFT) method for Large Language Models (LLMs), as it significantly reduces GPU memory usage while maintaining competitive fine-tuned model quality on downstream tasks. Despite these benefits, we identify two key inefficiencies in existing LoRA fine-tuning systems. First, they incur substantial runtime overhead due to redundant memory accesses on large activation tensors. Second, they miss the opportunity to concurrently fine-tune multiple independent LoRA adapters that share the same base model on the same set of GPUs. This leads to missed performance gains such as reduced pipeline bubbles, better communication overlap, and improved GPU load balance.   To address these issues, we introduce LoRAFusion, an efficient LoRA fine-tuning system for LLMs. At the kernel level, we propose a graph-splitting method that fuses memory-bound operations. This design eliminates unnecessary memory accesses and preserves the performance of compute-bound GEMMs without incurring the cost of recomputation or synchronization. At the scheduling level, LoRAFusion introduces an adaptive batching algorithm for multi-job fine-tuning. It first splits LoRA adapters into groups to intentionally stagger batch execution across jobs, and then solves a bin-packing problem within each group to generate balanced, dependency-aware microbatches. LoRAFusion achieves up to $1.96\times$ ($1.47\times$ on average) end-to-end speedup compared to Megatron-LM, and up to $1.46\times$ ($1.29\times$ on average) improvement over mLoRA, the state-of-the-art multi-LoRA fine-tuning system. Our fused kernel achieves up to $1.39\times$ ($1.27\times$ on average) kernel performance improvement and can directly serve as a plug-and-play replacement in existing LoRA systems. We open-source LoRAFusion at https://github.com/CentML/lorafusion.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00206) | **Categories:** cs.LG, cs.AI, cs.DC

---

### [2] [Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey](https://arxiv.org/abs/2510.00078)
*Sicong Liu, Weiye Wu, Xiangrui Xu, Teng Li, Bowen Pang, Bin Guo, Zhiwen Yu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundation models have reshaped AI by unifying fragmented architectures into scalable backbones with multimodal reasoning and contextual adaptation. In parallel, the long-standing notion of AI agents, defined by the sensing-decision-action loop, is entering a new paradigm: with FMs as their cognitive core, agents transcend rule-based behaviors to achieve autonomy, generalization, and self-reflection. This dual shift is reinforced by real-world demands such as autonomous driving, robotics, virtual assistants, and GUI agents, as well as ecosystem advances in embedded hardware, edge computing, mobile deployment platforms, and communication protocols that together enable large-scale deployment. Yet this convergence collides with reality: while applications demand long-term adaptability and real-time interaction, mobile and edge deployments remain constrained by memory, energy, bandwidth, and latency. This creates a fundamental tension between the growing complexity of FMs and the limited resources of deployment environments. This survey provides the first systematic characterization of adaptive, resource-efficient agentic AI systems. We summarize enabling techniques into elastic inference, test-time adaptation, dynamic multimodal integration, and agentic AI applications, and identify open challenges in balancing accuracy-latency-communication trade-offs and sustaining robustness under distribution shifts. We further highlight future opportunities in algorithm-system co-design, cognitive adaptation, and collaborative edge deployment. By mapping FM structures, cognition, and hardware resources, this work establishes a unified perspective toward scalable, adaptive, and resource-efficient agentic AI. We believe this survey can help readers to understand the connections between enabling technologies while promoting further discussions on the fusion of agentic intelligence and intelligent agents.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00078) | **Categories:** cs.LG, cs.AI, cs.DC

---

### [3] [Large Language Models Inference Engines based on Spiking Neural Networks](https://arxiv.org/abs/2510.00133)
*Adarsha Balaji, Sandeep Madireddy*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Foundational models based on the transformer architecture are currently the state-of-the-art in general language modeling, as well as in scientific areas such as material science and climate. However, training and deploying these models is computationally challenging as the time and space complexity has a quadratic relation to the input sequence length. Several efforts exploring efficient computational paradigms and model architectures to address these limitations have been made. In this work, we explore spiking neural networks (SNNs) to design transformer models. A challenge in training large-scale SNNs, using existing surrogate learning methods is inefficient and time-consuming. On the other hand, techniques to convert existing transformer-based models to their SNN equivalent are not scalable, as achieving optimal performance comes at the cost of a large number of spike time-steps, i.e. increased latency. To address this, we propose NeurTransformer, a methodology for designing transformer-based SNN for inference using a supervised fine-tuning approach with existing conversion methods. The proposed methodology works by: (1) replacing the self-attention mechanism with a spike-based self-attention (SSA), (2) converting the feed-forward block of the trained transformer model to its equivalent SNN, and (3) fine-tuning the SSA block using SNN-based surrogate learning algorithms. We benchmark the proposed methodology and demonstrate its accuracy and scalability using three variants of the GPT-2 model of increasing model size. We observe that the converted GPT-2 small models demonstrate a 5-12% loss in cosine similarity and a 9.7% reduction in perplexity. Finally, we demonstrate the energy efficiency of the SSA block compared to the ASA block and show between 64.71% and 85.28% reductions in estimated energy consumption when implementing the self-attention mechanism on a digital hardware.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00133) | **Categories:** cs.LG

---

### [4] [Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis](https://arxiv.org/abs/2510.00373)
*Carlo Bosio, Matteo Guarrera, Alberto Sangiovanni-Vincentelli, Mark W. Mueller*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Large Language models (LLMs) have shown promise as generators of symbolic control policies, producing interpretable program-like representations through iterative search. However, these models are not capable of separating the functional structure of a policy from the numerical values it is parametrized by, thus making the search process slow and inefficient. We propose a hybrid approach that decouples structural synthesis from parameter optimization by introducing an additional optimization layer for local parameter search. In our method, the numerical parameters of LLM-generated programs are extracted and optimized numerically to maximize task performance. With this integration, an LLM iterates over the functional structure of programs, while a separate optimization loop is used to find a locally optimal set of parameters accompanying candidate programs. We evaluate our method on a set of control tasks, showing that it achieves higher returns and improved sample efficiency compared to purely LLM-guided search. We show that combining symbolic program synthesis with numerical optimization yields interpretable yet high-performing policies, bridging the gap between language-model-guided design and classical control tuning. Our code is available at https://sites.google.com/berkeley.edu/colmo.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00373) | **Categories:** cs.LG, cs.AI, cs.NE, cs.SY, eess.SY

---

### [5] [Composer: A Search Framework for Hybrid Neural Architecture Design](https://arxiv.org/abs/2510.00379)
*Bilge Acun, Prasoon Sinha, Newsha Ardalani, Sangmin Bae, Alicia Golden, Chien-Yu Lin, Meghana Madhyastha, Fei Sun, Neeraja J. Yadwadkar, Carole-Jean Wu*

Main category: cs.LG

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Hybrid model architectures that combine computational primitives (e.g., Attention, MLP) in different ratios have shown promising performance beyond Transformers. Some studies have shown that different interleavings of primitives can affect model quality as well. However, prior works explore the hybrid model architecture design space manually. Due to the large design space and training costs, discovering hybrid models that combine key computational primitives for pre-training is challenging. In this work, we take a principled approach in designing a modular hybrid model architecture search framework -- Composer. Composer explores model architectures at a small scale and extrapolates the top-performing model architectures to a larger scale using our proposed scaling strategies. Using Composer, we discover new hybrid LLM architectures that outperform Llama 3.2. Compared to Llama 3.2 and previous state-of-the-art baselines, the new model architectures consistently reduce validation loss at parameter scales of 350M-3B and improve evaluation accuracy on the downstream tasks by up to 2.8-8.3% (1.1-3.1% on average) while improving both training and inference efficiency.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00379) | **Categories:** cs.LG

---


## 机器人学 (Robotics) [cs.RO]
### [1] [Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting](https://arxiv.org/abs/2510.00401)
*Shounak Sural, Charles Kekeh, Wenliang Liu, Federico Pecora, Mouhacine Benosman*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Long-horizon motion forecasting for multiple autonomous robots is challenging due to non-linear agent interactions, compounding prediction errors, and continuous-time evolution of dynamics. Learned dynamics of such a system can be useful in various applications such as travel time prediction, prediction-guided planning and generative simulation. In this work, we aim to develop an efficient trajectory forecasting model conditioned on multi-agent goals. Motivated by the recent success of physics-guided deep learning for partially known dynamical systems, we develop a model based on neural Controlled Differential Equations (CDEs) for long-horizon motion forecasting. Unlike discrete-time methods such as RNNs and transformers, neural CDEs operate in continuous time, allowing us to combine physics-informed constraints and biases to jointly model multi-robot dynamics. Our approach, named PINCoDE (Physics-Informed Neural Controlled Differential Equations), learns differential equation parameters that can be used to predict the trajectories of a multi-agent system starting from an initial condition. PINCoDE is conditioned on future goals and enforces physics constraints for robot motion over extended periods of time. We adopt a strategy that scales our model from 10 robots to 100 robots without the need for additional model parameters, while producing predictions with an average ADE below 0.5 m for a 1-minute horizon. Furthermore, progressive training with curriculum learning for our PINCoDE model results in a 2.7X reduction of forecasted pose error over 4 minute horizons compared to analytical models.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00401) | **Categories:** cs.RO, cs.AI, cs.LG, cs.MA

---

### [2] [Learning Human Reaching Optimality Principles from Minimal Observation Inverse Reinforcement Learning](https://arxiv.org/abs/2510.00329)
*Sarmad Mehrdad, Maxime Sabbah, Vincent Bonnet, Ludovic Righetti*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: This paper investigates the application of Minimal Observation Inverse Reinforcement Learning (MO-IRL) to model and predict human arm-reaching movements with time-varying cost weights. Using a planar two-link biomechanical model and high-resolution motion-capture data from subjects performing a pointing task, we segment each trajectory into multiple phases and learn phase-specific combinations of seven candidate cost functions. MO-IRL iteratively refines cost weights by scaling observed and generated trajectories in the maximum entropy IRL formulation, greatly reducing the number of required demonstrations and convergence time compared to classical IRL approaches. Training on ten trials per posture yields average joint-angle Root Mean Squared Errors (RMSE) of 6.4 deg and 5.6 deg for six- and eight-segment weight divisions, respectively, versus 10.4 deg using a single static weight. Cross-validation on remaining trials and, for the first time, inter-subject validation on an unseen subject's 20 trials, demonstrates comparable predictive accuracy, around 8 deg RMSE, indicating robust generalization. Learned weights emphasize joint acceleration minimization during movement onset and termination, aligning with smoothness principles observed in biological motion. These results suggest that MO-IRL can efficiently uncover dynamic, subject-independent cost structures underlying human motor control, with potential applications for humanoid robots.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00329) | **Categories:** cs.RO

---

### [3] [Integrating Offline Pre-Training with Online Fine-Tuning: A Reinforcement Learning Approach for Robot Social Navigation](https://arxiv.org/abs/2510.00466)
*Run Su, Hao Fu, Shuai Zhou, Yingao Fu*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Offline reinforcement learning (RL) has emerged as a promising framework for addressing robot social navigation challenges. However, inherent uncertainties in pedestrian behavior and limited environmental interaction during training often lead to suboptimal exploration and distributional shifts between offline training and online deployment. To overcome these limitations, this paper proposes a novel offline-to-online fine-tuning RL algorithm for robot social navigation by integrating Return-to-Go (RTG) prediction into a causal Transformer architecture. Our algorithm features a spatiotem-poral fusion model designed to precisely estimate RTG values in real-time by jointly encoding temporal pedestrian motion patterns and spatial crowd dynamics. This RTG prediction framework mitigates distribution shift by aligning offline policy training with online environmental interactions. Furthermore, a hybrid offline-online experience sampling mechanism is built to stabilize policy updates during fine-tuning, ensuring balanced integration of pre-trained knowledge and real-time adaptation. Extensive experiments in simulated social navigation environments demonstrate that our method achieves a higher success rate and lower collision rate compared to state-of-the-art baselines. These results underscore the efficacy of our algorithm in enhancing navigation policy robustness and adaptability. This work paves the way for more reliable and adaptive robotic navigation systems in real-world applications.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00466) | **Categories:** cs.RO, cs.AI

---

### [4] [From Human Hands to Robot Arms: Manipulation Skills Transfer via Trajectory Alignment](https://arxiv.org/abs/2510.00491)
*Han Zhou, Jinjin Cao, Liyuan Ma, Xueji Fang, Guo-jun Qi*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Learning diverse manipulation skills for real-world robots is severely bottlenecked by the reliance on costly and hard-to-scale teleoperated demonstrations. While human videos offer a scalable alternative, effectively transferring manipulation knowledge is fundamentally hindered by the significant morphological gap between human and robotic embodiments. To address this challenge and facilitate skill transfer from human to robot, we introduce Traj2Action,a novel framework that bridges this embodiment gap by using the 3D trajectory of the operational endpoint as a unified intermediate representation, and then transfers the manipulation knowledge embedded in this trajectory to the robot's actions. Our policy first learns to generate a coarse trajectory, which forms an high-level motion plan by leveraging both human and robot data. This plan then conditions the synthesis of precise, robot-specific actions (e.g., orientation and gripper state) within a co-denoising framework. Extensive real-world experiments on a Franka robot demonstrate that Traj2Action boosts the performance by up to 27% and 22.25% over $\pi_0$ baseline on short- and long-horizon real-world tasks, and achieves significant gains as human data scales in robot policy learning. Our project website, featuring code and video demonstrations, is available at https://anonymous.4open.science/w/Traj2Action-4A45/.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00491) | **Categories:** cs.RO, cs.AI

---

### [5] [Hybrid Training for Vision-Language-Action Models](https://arxiv.org/abs/2510.00600)
*Pietro Mazzaglia, Cansu Sancaktar, Markus Peschl, Daniel Dijkman*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Using Large Language Models to produce intermediate thoughts, a.k.a. Chain-of-thought (CoT), before providing an answer has been a successful recipe for solving complex language tasks. In robotics, similar embodied CoT strategies, generating thoughts before actions, have also been shown to lead to improved performance when using Vision-Language-Action models (VLAs). As these techniques increase the length of the model's generated outputs to include the thoughts, the inference time is negatively affected. Delaying an agent's actions in real-world executions, as in robotic manipulation settings, strongly affects the usability of a method, as tasks require long sequences of actions. However, is the generation of long chains-of-thought a strong prerequisite for achieving performance improvements? In this work, we explore the idea of Hybrid Training (HyT), a framework that enables VLAs to learn from thoughts and benefit from the associated performance gains, while enabling the possibility to leave out CoT generation during inference. Furthermore, by learning to conditionally predict a diverse set of outputs, HyT supports flexibility at inference time, enabling the model to either predict actions directly, generate thoughts or follow instructions. We evaluate the proposed method in a series of simulated benchmarks and real-world experiments.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00600) | **Categories:** cs.RO, cs.AI, cs.CV, cs.LG

---

### [6] [What Did I Learn? Operational Competence Assessment for AI-Based Trajectory Planners](https://arxiv.org/abs/2510.00619)
*Michiel Braat, Maren Buermann, Marijke van Weperen, Jan-Pieter Paardekooper*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Automated driving functions increasingly rely on machine learning for tasks like perception and trajectory planning, requiring large, relevant datasets. The performance of these algorithms depends on how closely the training data matches the task. To ensure reliable functioning, it is crucial to know what is included in the dataset to assess the trained model's operational risk. We aim to enhance the safe use of machine learning in automated driving by developing a method to recognize situations that an automated vehicle has not been sufficiently trained on. This method also improves explainability by describing the dataset at a human-understandable level. We propose modeling driving data as knowledge graphs, representing driving scenes with entities and their relationships. These graphs are queried for specific sub-scene configurations to check their occurrence in the dataset. We estimate a vehicle's competence in a driving scene by considering the coverage and complexity of sub-scene configurations in the training set. Higher complexity scenes require greater coverage for high competence. We apply this method to the NuPlan dataset, modeling it with knowledge graphs and analyzing the coverage of specific driving scenes. This approach helps monitor the competence of machine learning models trained on the dataset, which is essential for trustworthy AI to be deployed in automated driving.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00619) | **Categories:** cs.RO, cs.AI

---

### [7] [Trajectory Based Observer Design: A Framework for Lightweight Sensor Fusion](https://arxiv.org/abs/2510.00630)
*Federico Oliva, Tom Shaked, Daniele Carnevale, Amir Degani*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Efficient observer design and accurate sensor fusion are key in state estimation. This work proposes an optimization-based methodology, termed Trajectory Based Optimization Design (TBOD), allowing the user to easily design observers for general nonlinear systems and multi-sensor setups. Starting from parametrized observer dynamics, the proposed method considers a finite set of pre-recorded measurement trajectories from the nominal plant and exploits them to tune the observer parameters through numerical optimization. This research hinges on the classic observer's theory and Moving Horizon Estimators methodology. Optimization is exploited to ease the observer's design, providing the user with a lightweight, general-purpose sensor fusion methodology. TBOD's main characteristics are the capability to handle general sensors efficiently and in a modular way and, most importantly, its straightforward tuning procedure. The TBOD's performance is tested on a terrestrial rover localization problem, combining IMU and ranging sensors provided by Ultra Wide Band antennas, and validated through a motion-capture system. Comparison with an Extended Kalman Filter is also provided, matching its position estimation accuracy and significantly improving in the orientation.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00630) | **Categories:** cs.RO

---

### [8] [HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy](https://arxiv.org/abs/2510.00695)
*Myungkyu Koo, Daewon Choi, Taeyoung Kim, Kyungmin Lee, Changyeon Kim, Youngyo Seo, Jinwoo Shin*

Main category: cs.RO

TL;DR: 达到API配额限制，请明天再试


<details>
  <summary>Details</summary>
Motivation: Error: API quota exceeded

Method: Error: API quota exceeded

Result: Error: API quota exceeded

Conclusion: 请联系管理员或等待明天API配额重置。

Abstract: Inherently, robotic manipulation tasks are history-dependent: leveraging past context could be beneficial. However, most existing Vision-Language-Action models (VLAs) have been designed without considering this aspect, i.e., they rely solely on the current observation, ignoring preceding context. In this paper, we propose HAMLET, a scalable framework to adapt VLAs to attend to the historical context during action prediction. Specifically, we introduce moment tokens that compactly encode perceptual information at each timestep. Their representations are initialized with time-contrastive learning, allowing them to better capture temporally distinctive aspects. Next, we employ a lightweight memory module that integrates the moment tokens across past timesteps into memory features, which are then leveraged for action prediction. Through empirical evaluation, we show that HAMLET successfully transforms a state-of-the-art VLA into a history-aware policy, especially demonstrating significant improvements on long-horizon tasks that require historical context. In particular, on top of GR00T N1.5, HAMLET achieves an average success rate of 76.4% on history-dependent real-world tasks, surpassing the baseline performance by 47.2%. Furthermore, HAMLET pushes prior art performance from 64.1% to 66.4% on RoboCasa Kitchen (100-demo setup) and from 95.6% to 97.7% on LIBERO, highlighting its effectiveness even under generic robot-manipulation benchmarks.

</details>

[**[PDF]**](https://arxiv.org/pdf/2510.00695) | **Categories:** cs.RO, cs.CV

---
