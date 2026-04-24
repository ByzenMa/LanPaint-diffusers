# 论文初稿（基于 LanPaint 的改进）

## 暂定标题
**LanPaint-DSC: Dynamic Step Size Langevin Dynamics with ControlNet-Enhanced Z-Image for Training-Free Diffusion Inpainting**

---

## 1. 研究背景与动机
LanPaint 通过训练自由（training-free）的条件采样策略，在扩散修复任务中实现了较好的保真度与效率。然而，在实际应用中，固定步长的 Langevin dynamics 在不同噪声阶段可能存在：

1. 早期探索不足或震荡；
2. 中后期收敛速度与细节恢复不稳定；
3. 面对结构约束（例如边缘、轮廓）时，局部一致性仍有改进空间。

为此，本文在 LanPaint 基础上提出两个互补改进：

- **改进点1：动态步长 Langevin dynamics（Dynamic Step Size）**；
- **改进点2：在 lanpaint-z-image 分支中融入 ControlNet 结构条件**。

---

## 2. 方法

### 2.1 基线方法
本文以 **LanPaint** 为基础框架，保持其 training-free 采样范式不变，仅在采样动力学与条件控制两个模块进行增量设计。

### 2.2 改进点1：动态步长 Langevin dynamics
在 `LanPaint/src/LanPaint/lanpaint.py` 中，对 `langevin_dynamics` 引入 `use_dynamic_step_size` 开关；当开启时，步长不再固定，而由阶段相关函数动态给定。本文实现三类策略：

1. `step_size_dynamics`：通用动态步长函数；
2. `step_size_linear_dynamics`：线性变化策略；
3. `step_size_cosine_dynammics`：余弦调度策略。

其核心思想是在不同时间步自适应调节更新幅度：

- 早期保持较强探索能力；
- 中期平衡探索与收敛；
- 后期减小步长以稳定细节与边界。

### 2.3 改进点2：lanpaint-z-image + ControlNet
在 z-image 路径中引入 ControlNet 条件分支，使采样过程在保持语义编辑能力的同时，利用额外结构先验（如边缘/引导图）提升几何一致性与局部可控性。

该改进与动态步长可叠加：

- 动态步长负责优化采样轨迹与收敛行为；
- ControlNet 负责增强结构约束；
- 二者结合提升“可控 + 细节 + 稳定”的整体表现。

---

## 3. 实验设置

### 3.1 实验分组建议
建议至少包含以下对照：

- **Baseline**：原始 LanPaint（固定步长，无 ControlNet）；
- **+DynamicStep**：仅启用动态步长（含三种函数对比）；
- **+ControlNet**：仅引入 z-image + ControlNet；
- **+DynamicStep+ControlNet**：两项改进同时启用。

### 3.2 结果来源
动态步长实验结果位于：

- `results_dynamic_step_size`

可在正文中以可视化样例与定量指标共同报告。

### 3.3 评价指标建议
- 图像质量：PSNR / SSIM / LPIPS；
- 感知与语义一致性：CLIPScore（可选）；
- 结构一致性：边缘保持率或关键区域误差（可选）；
- 采样效率：迭代步数、平均推理时长。

---

## 4. 主要贡献（可直接用于摘要/引言）

1. 在 LanPaint 框架中提出并实现了 **动态步长 Langevin dynamics**，并提供三种可替换调度函数（通用、线性、余弦）。
2. 将 **ControlNet** 融合进 lanpaint-z-image 路径，提高结构引导下的修复可控性。
3. 通过模块化组合验证两项改进的互补性，在视觉质量、结构一致性与稳定性上取得提升（以实验结果为准）。

---

## 5. 论文写作骨架（建议）

- **Abstract**：问题、方法（Dynamic Step + ControlNet）、结果与贡献。
- **1 Introduction**：LanPaint 优势与局限；本文两个改进点及贡献。
- **2 Related Work**：Training-free inpainting、Langevin sampling 调度、ControlNet 条件控制。
- **3 Method**：
  - 3.1 Baseline LanPaint 回顾；
  - 3.2 Dynamic Step Size 设计与三种函数；
  - 3.3 ControlNet 融合方案；
  - 3.4 复杂度与实现细节。
- **4 Experiments**：数据、设置、对照组、定量与可视化。
- **5 Ablation Study**：
  - 三种步长函数比较；
  - 是否启用 dynamic step；
  - 是否启用 ControlNet；
  - 组合收益。
- **6 Conclusion**：总结与未来工作（例如自动策略选择、多条件控制扩展）。

---

## 6. 可直接复用的摘要草稿（中文）

本文基于 LanPaint 提出一种面向训练自由扩散修复的增强框架。首先，我们在 Langevin dynamics 中引入动态步长机制，并设计了通用动态、线性动态与余弦动态三类步长函数，以适配不同采样阶段的探索与收敛需求。其次，我们在 lanpaint-z-image 路径中融合 ControlNet 条件分支，以增强结构约束下的修复可控性。实验表明，动态步长策略可改善采样稳定性与细节恢复效果，而与 ControlNet 结合后可进一步提升结构一致性与视觉质量。上述结果验证了“采样动力学优化 + 条件控制增强”的互补性，为训练自由扩散修复提供了可扩展的改进方向。

---

## 7. 后续补充清单

- 补全严格的数学定义（步长函数形式、符号系统）；
- 明确实验数据集与评测协议；
- 增加失败案例与局限性分析；
- 增加与最新 inpainting/control 方法对比；
- 在附录给出关键超参与复现实验命令。
