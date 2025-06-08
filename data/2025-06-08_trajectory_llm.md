# 每日 ArXiv 摘要速递: 2025-06-08

### [Learning dissection trajectories from expert surgical videos via imitation learning with equivariant diffusion](https://arxiv.org/abs/2506.04716)

**一句话总结:** 提出了一种用于内镜黏膜下剥离术（ESD）解剖轨迹预测的具有等变表示的隐式扩散策略模仿学习（iDPOE）方法。

**Authors:** Hongyu Wang, Yonghao Long, Yueyao Chen, Hon-Chi Yip, Markus Scheppach, Philip Wai-Yan Chiu, Yeung Yam, Helen Mei-Ling Meng, Qi Dou
**Categories:** `cs.CV`

[**[PDF]**](https://arxiv.org/pdf/2506.04716)

#### 中文摘要 (Abstract in Chinese)

> 内镜黏膜下剥离术 (ESD) 是一种成熟的切除上皮病变的技术。预测 ESD 视频中的解剖轨迹具有增强手术技能培训和简化学习过程的巨大潜力，但这个领域尚未被充分探索。虽然模仿学习在从专家演示中获取技能方面显示出前景，但在处理不确定的未来运动、学习几何对称性和泛化到不同的手术场景方面仍然存在挑战。为了解决这些问题，我们引入了一种新方法：具有等变表示的隐式扩散策略模仿学习 (iDPOE)。我们的方法通过联合状态动作分布来模拟专家行为，捕获解剖轨迹的随机性，并实现跨各种内窥镜视图的鲁棒视觉表征学习。通过将扩散模型融入策略学习中，iDPOE 确保了高效的训练和采样，从而实现了更准确的预测和更好的泛化。此外，我们通过将等变性嵌入到学习过程中，增强了模型泛化到几何对称性的能力。为了解决状态不匹配问题，我们开发了一种前向过程引导的动作推理策略，用于条件采样。使用包含近 2000 个剪辑的 ESD 视频数据集，实验结果表明，我们的方法在轨迹预测方面优于最先进的方法（包括显式和隐式方法）。据我们所知，这是模仿学习在解剖轨迹预测手术技能发展中的首次应用。
