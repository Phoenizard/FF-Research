# 深层重训：冻结浅层，深层改用端到端梯度

来源：tmp/deep_bp（2026-11-17），未在仓库代码上重跑。CIFAR-100，CNN32-depth，种子 1。

## 问题

TTGS-32 的浅层固定不变时，深层卷积能否被训练出比 TTGS 自己更好的参数？

## 做法

- 起点：results/cifar100/32-depth-v2/TTGS/seed1/last.pt。L1–L20 保留 TTGS 权重，全程不更新（5 组训练后均核对未变）。
- L21–L32 与 Fusion 头：重置为同一种子的初始权重（"重置"），或保留 TTGS 权重（"续训"）。
- 前向与 TTGS 相同（avg Pool、通道 LayerNorm、跳连流与随机深度、Dropout）；L21–L32 的类别树与 goodness 不使用。
- 训练信号：L21–L32 与 Fusion 头一起，用 Fusion 交叉熵端到端训练；梯度在 L21–L32 之间贯通（含跳连流），在 L20 与 L21 之间截断。
- 每轮在官方 10,000 张测试图片上评估，取最后一轮。

## 结果（最后一轮 Fusion，%）

| 设置 | 轮数 | 训练 | 测试 | 训练耗时（h） |
|---|---|---|---|---|
| TTGS-32 自身（对照） | 400 | 81.55 | 64.64 | — |
| 重置，dropout 0.1 | 400 | 97.98 | 63.51 | 2.20 |
| 重置，dropout 0.3 | 400 | 95.36 | 64.03 | 2.28 |
| 重置，dropout 0.5 | 100 | 74.41 | 64.27 | 0.56 |
| 续训，dropout 0.3 | 100 | 89.41 | 65.33 | 0.56 |
| 续训，只训 Fusion 头 | 100 | 79.58 | 65.25 | 0.48 |

## 结论

- 端到端梯度能把深层在训练集上拟合得更高（97.98 对 81.55），但测试不超过 TTGS 自己的深层（最高 64.27 对 64.64）。在 TTGS 的浅层之上，深层卷积参数没有被训练得更好。
- 续训的增益（65.33）与只训 Fusion 头（65.25）相当，增益来自读出头，不来自卷积。

## 复现参数

checkpoint 只在 tmp/deep_bp 中，以下参数足以重跑。

- 共同：构建模型前 torch.manual_seed(1)；重置项为 convs.20–31 与 fusion 的 weight、bias，共 26 个张量，取种子 1 的初始值；AdamW，lr $2\times10^{-4}$，CosineAnnealingLR（T_max = 该组轮数，eta_min $10^{-5}$），weight decay $10^{-3}$，betas (0.9, 0.999)，eps $10^{-8}$；batch 128，loader 种子 1，4 workers，标准增强（RandomCrop 32/4、水平翻转 0.5）；测试 batch 500；TTGS 设置不变（skip_beta_sq 0.25、stochastic_depth_max 0.5，L1–L20 的 Dropout 0.1）；RTX 5070。
- 各组：

| 目录 | L21–L32 输出的 Dropout | 初值 | 训练的参数 | 轮数 | 峰值显存（MiB） |
|---|---|---|---|---|---|
| results_dropout01 | 0.1 | 重置 | L21–L32 + Fusion | 400 | 1554.1 |
| results_dropout03 | 0.3 | 重置 | L21–L32 + Fusion | 400 | 1554.1 |
| results_dropout05_100 | 0.5 | 重置 | L21–L32 + Fusion | 100 | 1554.1 |
| results_keep_dropout03 | 0.3 | 保留 | L21–L32 + Fusion | 100 | 1554.1 |
| results_keep_headonly | 0.3 | 保留 | 只有 Fusion | 100 | 1354.7 |

## 文件

- deep_retrain.py：实验代码（tmp/deep_bp/run.py 原样，文内路径未改）。
- results*/config.json、history.json、result.json：各组配置、逐轮记录与最终结果。
