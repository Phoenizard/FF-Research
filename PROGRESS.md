## 阶段1: 训练结果

### 实验设置

| | CIFAR10 | CIFAR100 | TinyIN |
|---|---|---|---|
| seed | 1 | 1 | 1 |
| epochs | 400 | 400 | 400 |
| batch_size | 128 | 128 | 256 |

## CIFAR10

训练时间不含评估；Best Pred 按评估集选层，仅作参考。

### 8 层

| | Fusion（%） | Best Pred（%） | 秒/轮 | 设备 |
|---|---|---|---|---|
| BiCovG | 90.35 | 90.33 | 14.37 | RTX 5070 |
| BiCovG (Random Readout) | 88.05 | 87.82 | 10.47 | RTX 5070 |
| ASGE | 87.97 | 87.92 | 8.23 | RTX 5070 |
| DTG-FF | 89.80 | 89.46 | 8.39 | RTX 5070 |
| CwC | 83.89 | 85.40 | 7.22 | RTX 5070 |
| DeeperForward | 86.85 | 84.70 | 6.85 | RTX 5070 |
| HCL-FF | 85.93 | 86.05 | 10.65 | RTX 5070 |
| TTG | 89.96 | 89.98 | 8.09 | RTX 5070 |
| TTGS | 89.98 | 89.97 | 8.15 | RTX 5070 |
| Global BP | 90.41 | — | 11.59 | RTX 5070 |

## CIFAR100

训练时间不含评估，显存为训练峰值；Best Pred 按评估集选层，仅作参考。

### 8 层

| | Fusion（%） | 种子 1～3 均值 ± 标准差 | Best Pred（%） | 秒/轮 | 显存（MiB） | 设备 |
|---|---|---|---|---|---|---|
| BiCovG | 66.09 | — | 64.70 | 10.78 | 516.3 | RTX 5070 |
| BiCovG (Random Readout) | 54.36 | — | 52.98 | 10.63 | 474.7 | RTX 5070 |
| ASGE | 57.64 | — | 55.89 | 13.12* | 414.2 | RTX 5070 |
| DTG-FF | 60.13 | 60.75 ± 0.60 | 59.48 | 8.20 | 414.1 | RTX 5070 |
| CwC | 53.07 | — | 58.52 | 7.27 | 411.0 | RTX 5070 |
| DeeperForward | 57.32 | — | 51.81 | 7.14 | 435.2 | RTX 5070 |
| HCL-FF | 60.52 | 60.23 ± 0.29 | 53.83 | 11.51 | 408.9 | RTX 5070 |
| TTG（外部树） | 64.26 | 64.23 ± 0.06 | 63.70 | 8.39 | 414.4 | RTX 5070 |
| TTG | 64.26 | — | 64.01 | 8.40 | 414.8 | RTX 5070 |
| TTGS | 64.27 | — | 64.12 | 8.21 | 464.6 | RTX 5070 |
| Global BP | 63.34 | — | — | 11.73 | 582.8 | RTX 5070 |

\* 耗时异常，同设备 Smoke Test 稳态约 8.55 s/轮，未复测。

### 16 层

16 层骨干由 8 层配置每项重复一次。

| | Fusion（%） | Best Pred（%） | 秒/轮 | 显存（MiB） | 设备 |
|---|---|---|---|---|---|
| ASGE | 53.73 | 52.44 | 17.85 | 523.1 | RTX 5070 |
| DTG-FF | 57.26 | 56.78 | 18.08 | 523.7 | RTX 5070 |
| CwC | 52.04 | 56.66 | 16.09 | 567.0 | RTX 5070 |
| HCL-FF | 58.42 | 49.94 | 24.60 | 536.4 | RTX 5070 |
| DeeperForward | 53.54 | 45.19 | 15.41 | 541.8 | RTX 5070 |
| TTG（外部树） | 63.89 | 63.96 | 18.09 | 523.7 | RTX 5070 |
| TTG | 63.77 | 63.52 | 18.07 | 524.0 | RTX 5070 |
| TTGS | 65.46 | 65.12 | 18.58 | 579.2 | RTX 5070 |
| Global BP | 63.88 | — | 26.52 | 1200.8 | RTX 5070 |

## TinyIN

训练时间不含评估，显存为训练峰值；Best Pred 按评估集选层，仅作参考。

### 8 层

| | Fusion（%） | Best Pred（%） | 秒/轮 | 显存（MiB） | 设备 |
|---|---|---|---|---|---|
| BiCovG | 51.29 | 49.81 | 72.89 | 5822.0 | RTX 4090 |
| BiCovG (Random Readout) | 39.78 | 39.96 | 72.61 | 5757.5 | RTX 4090 |
| ASGE | 43.30 | 41.38 | 53.52 | 4936.8 | RTX 4090 |
| DTG-FF | 42.67 | 42.16 | 49.30 | 4939.5 | RTX 4090 |
| CwC | 30.51 | 41.03 | 44.15 | 4933.7 | RTX 4090 |
| DeeperForward | 39.34 | 37.11 | 41.34 | 4936.4 | RTX 4090 |
| HCL-FF | 42.75 | 21.87 | 56.46 | 4156.4 | RTX 4090 |
| TTG | 49.34 | 50.73 | 41.81 | 4938.1 | A100-SXM4-40GB |
| TTGS | 49.25 | 49.87 | 41.88 | 5738.2 | A100-SXM4-40GB |
| Global BP | 49.26 | — | 68.71 | 5948.7 | RTX 4090 |

### 16 层

16 层骨干由 8 层配置每项重复一次。

| | Fusion（%） | Best Pred（%） | 秒/轮 | 显存（MiB） | 设备 |
|---|---|---|---|---|---|
| ASGE | 41.26 | 39.48 | 91.50 | 5062.6 | A100-SXM4-40GB |
| DTG-FF | 37.22 | 35.98 | 85.34 | 5062.3 | A100-SXM4-40GB |
| CwC | 21.39 | 32.16 | 78.90 | 5853.3 | A100-SXM4-40GB |
| HCL-FF | 42.34 | 21.98 | 135.61 | 4298.6 | L40S |
| DeeperForward | 34.70 | 32.73 | 71.50 | 5062.1 | A100-SXM4-40GB |
| TTG | 49.66 | 49.80 | 86.86 | 5062.9 | A100-SXM4-40GB |
| TTGS | 50.52 | 51.00 | 93.22 | 5867.6 | A100-SXM4-40GB |
| Global BP | 51.78 | — | 114.08 | 12550.2 | A100-SXM4-40GB |

## 阶段2：研究路线

### 深层正确率衰减线路

| 可能原因 | 判别方法 | 成本 |
|---|---|---|
| A. 骨干本身的上限（分辨率、容量） | BP-16 的 Fusion 是否明显高于 BP-8；BP-16 各层的线性探针是否也在 L12 后走平 | 一次 BP-16（需授权） |
| B. 深层只是复制流，没加新信息 | 用现有 TTGS（C100-16）checkpoint 离线测：$u_l$ 和 $s_{l-1}$ 的 CKA、$u_l$ 的增量能带来多少探针准确率、有效秩 | 不训练 |
| C. 深层学到了新信息但过拟合 | 尾部训练准确率升、测试准确率平；$s_l$ 和 $u_l$ 的探针在训练集和测试集上的差距 | 不训练 |

#### 判定结果（C100，种子 1）

判据在 BP-16 出结果前写定。$F_W$ 为第 381～400 轮 Fusion 均值，取自 history.json；探针为对最终 checkpoint 逐层 GAP 特征的线性探针（全批量 L-BFGS，L2 $10^{-4}$），由 `diagnosis/depth/depth_abc.py` 离线测得。阈值均为 1.0。

| 假设 | 量 | 值 | 结论 |
|---|---|---|---|
| A | $\Delta F_{BP} = F_W(\text{BP-16}) - F_W(\text{BP-8})$ | 64.13 − 63.30 = 0.83 | 成立 |
| | $G_{BP}$ = BP-16 第 13～16 层最佳探针 − 第 1～12 层最佳探针 | 60.28 − 60.93 = −0.65 | |
| B/C | $\Delta_{tr}$ = 探针训练 $s_{16} - s_{12}$ | 97.84 − 98.26 = −0.42 | B 成立 |
| | $\Delta_{te}$ = 探针测试 $s_{16} - s_{12}$ | 60.96 − 62.16 = −1.20 | |

逐层探针（测试 %）：

| 层 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|
| BP-16 $f_l$ | 55.9 | 58.1 | 60.9 | 60.6 | 60.3 | 58.1 | 54.7 | 49.9 |
| TTG-16 $f_l$ | 58.8 | 61.0 | 61.3 | 61.8 | 61.8 | 61.6 | 60.9 | 59.9 |
| TTGS-16 $f_l$ | 58.9 | 61.6 | 63.3 | 62.5 | 63.4 | 63.7 | 63.3 | 63.3 |
| TTGS-16 $s_l$ | 54.1 | 57.1 | 57.4 | 62.2 | 61.8 | 61.4 | 61.2 | 61.0 |

结论：
1. A 成立：BP 加深到 16 层也几乎没有收益，探针在第 11 层见顶。第 13～16 层分辨率为 2×2，第 12 层以后的上限来自骨干，不来自 FF。
2. B 成立：TTGS 尾部的传递流没有加入新信息。尾部各层自身特征 $f_l$ 最高 63.7，比 $s_{12}$ 高约 1.5，但混入流后被冲淡。
3. BP-16 尾部训练准确率保持约 100%，测试从 60.6 降到 49.9；TTGS 尾部测试稳定在约 63%。
4. C100-16 骨干不适合证明“加深继续带来收益”，深层主张转到 ResNet 深度序列上验证。

#### 主张

1. FF 加深后的衰减是训练端的问题，不是过拟合。
2. 在 TTG 上，零参数、保持方差的跳连流消除训练端衰减；再加随机深度（TTGS），16 层超过 8 层。
3. 衰减消除后，FF 的深度收益能跟上 BP 的深度收益（待 ResNet 深度序列验证）。

先在 CIFAR-100 上完成，再在 TinyIN 上印证。

#### 主张 1：衰减类型（C100，种子 1）

由 `diagnosis/depth/degradation.py` 从 history.json 计算，判据在任何 16 层运行前写定。各量为第 381～400 轮均值：$\Delta F$ = 16 层 Fusion − 8 层 Fusion，$\Delta F_{tr}$ 为训练集 Fusion 之差；Drop16 = 测试局部准确率峰值 − 第 16 层，TDrop16 为训练集的同一量。E1：Drop16 ≥ 2.0；E2：$\Delta F$ ≤ −1.0；N：非 E 且 $\Delta F$ ≥ +1.0；S：其余。原因：C1 欠拟合型（训练端也降），C2 过拟合型，C3 不能区分。

| 方法 | $\Delta F$ | $\Delta F_{tr}$ | Drop16 | TDrop16 | 类别 | 原因 |
|---|---|---|---|---|---|---|
| ASGE | −3.91 | −8.82 | 3.44 | 13.92 | E1+E2 | C1 |
| DTG-FF | −2.58 | −7.10 | 2.16 | 6.83 | E1+E2 | C1 |
| CwC | −1.42 | −0.68 | 1.07 | 4.12 | E2 | C3 |
| DeeperForward | −4.63 | −7.22 | 41.66 | 39.63 | E1+E2 | C1 |
| HCL-FF | −2.12 | −3.59 | 34.86 | 56.07 | E1+E2 | C1 |
| TTG | −0.52 | −3.08 | 2.96 | 3.12 | E1 | C1 |
| TTGS | +1.41 | −0.92 | 0.65 | 2.46 | N | — |

结论：
1. 除 TTGS 外的 6 个方法都存在衰减，5 个为欠拟合型，CwC 不能区分，没有过拟合型。主张 1 成立（单种子）。
2. DeeperForward、HCL-FF 在 8 层时 Drop 已很大（38.88、49.67），局部读出末层崩塌不是加深造成的；对它们只看 $\Delta F$、$\Delta F_{tr}$。

#### 主张 2：TTG 消融链（C100-16，种子 1）

| 组 | 内容 | F16 | $\Delta F$（对 TTG-8） | Drop16 | TDrop16 | 类别 |
|---|---|---|---|---|---|---|
| TTG | 基线 | 63.66 | −0.52 | 2.96 | 3.12 | E1 C1 |
| TTG-Skip | + 跳连流 | 64.34 | +0.16 | 1.14 | 0.00 | S |
| TTGS-Expected | + 跳连流 + 随机深度（训练也只用期望系数） | 64.68 | +0.51 | 0.96 | 0.00 | S |
| TTGS | + 跳连流 + 逐样本随机深度 | 65.58 | +1.41 | 0.65 | 2.46 | N |

结论：
1. 跳连流消除训练端衰减：TDrop16 由 3.12 降到 0，第 9～16 层训练准确率单调升到 84.2；测试端 Drop16 由 2.96 降到 1.14。
2. 只有加逐样本随机深度的 TTGS 为 N 类；TTGS 与 TTGS-Expected 相差 0.90 < 1.0，单种子下不能断定随机采样本身有效。
3. TTGS 的 TDrop16 = 2.46 推测来自训练模式下记录的训练准确率（深层被随机跳过）；TTGS-Expected 训练时不随机跳过，TDrop16 为 0。

### ResNet架构通用性

在深层正确率衰减线路完成后进行。

TODO: CIFAR ResNet 深度序列（ResNet-20/32/56）三个 stage 的分辨率固定为 32/16/8，加深只在 stage 内部加 block，没有 2×2 的问题。这里 BP 随深度确实会变好，所以“FF 加深是否跟着变好”是一个干净的问题。深层主张最有力的证据就在这里。

### 理论证明

**记号**：第 $l$ 层 goodness $g\in\mathbb R_{\ge0}^{D}$，固定读出 $W\in\mathbb R^{D\times K}$，偏置为 0，
$$
z=\tfrac1T W^\top g,\qquad p=\operatorname{softmax}(z),\qquad \mathcal L=-\log p_y,\qquad \nabla_g\mathcal L=\tfrac1T W(p-e_y).
$$

**证明目标**

1. 中心化：零列和读出去除“整体抬高能量”的方向。
2. 树码：logit 按类别树分解；梯度只落在节点二分决策上；码的几何就是树的相似度。
3. 跳连流：尺度不随深度膨胀；评估系数的含义；梯度仍在层内。

#### 1. 共模不变

**Theorem 1.** 若 $\mathbf 1_D^\top W=0$，则对任意 $c\in\mathbb R$，$\mathcal L(g+c\mathbf 1_D)=\mathcal L(g)$，且 $\mathbf 1_D^\top\nabla_g\mathcal L=0$。若 $m=W^\top\mathbf 1_D\neq0$，则 $\mathbf 1_D^\top\nabla_g\mathcal L=\tfrac1T m^\top(p-e_y)$，一般不为 0。

*Proof.* $z(g+c\mathbf 1)=z(g)+\tfrac cT W^\top\mathbf 1=z(g)$。$\mathbf 1^\top\nabla_g\mathcal L=\tfrac1T(W^\top\mathbf 1)^\top(p-e_y)$，分别代入两种情形。∎

树码与构树前的中心化随机码满足 $\mathbf 1^\top W=0$（见 Lemma）；DTG-FF 的随机码不满足。

#### 2. 树码

**树码定义.** 内部节点 $i=1,\dots,K-1$，左右子树叶集合 $A_i,B_i$，$\operatorname{sub}(i)=A_i\cup B_i$；$\sigma_i(k)=+1$（$k\in A_i$）、$-1$（$k\in B_i$）、$0$（其余）。互不相交的单元集合 $L_i,R_i$，$|L_i|=|R_i|=m$：
$$
W_{r,k}=\begin{cases}c_k\sigma_i(k), & r\in L_i\\ -c_k\sigma_i(k), & r\in R_i\\ 0,&\text{未分配的单元}\end{cases}
\qquad c_k^2=\frac{D}{2mK\,\mathrm{depth}_k}.
$$

**Lemma.** $\mathbf 1^\top W=0$，且 $\|W_{:,k}\|^2=D/K$。

*Proof.* 每个节点对第 $k$ 列的贡献为 $mc_k\sigma_i(k)-mc_k\sigma_i(k)=0$。第 $k$ 列在 $2m\,\mathrm{depth}_k$ 个单元上取 $\pm c_k$，平方和为 $2m\,\mathrm{depth}_k c_k^2=D/K$。∎

**Theorem 2（logit 的树分解）.** 令节点裕量 $\Delta_i=\sum_{r\in L_i}g_r-\sum_{r\in R_i}g_r$，则
$$
z_k=\frac{c_k}{T}\sum_{i\in\operatorname{anc}(k)}\sigma_i(k)\,\Delta_i .
$$

*Proof.* $z_k=\tfrac1T\sum_r W_{r,k}g_r$；只有 $\sigma_i(k)\neq0$，即 $i\in\operatorname{anc}(k)$ 的单元非零，节点 $i$ 贡献 $c_k\sigma_i(k)\big(\sum_{L_i}g_r-\sum_{R_i}g_r\big)$。∎

**Corollary 2.1（梯度落在节点二分上）.** 对 $r\in L_i$（取 $+$）或 $r\in R_i$（取 $-$）：
$$
\frac{\partial\mathcal L}{\partial g_r}=\pm\frac1T\sum_{k\in\operatorname{sub}(i)}c_k\sigma_i(k)\,\big(p_k-\mathbf 1[k=y]\big).
$$
(i) 同一节点同一侧的单元梯度相同；(ii) 若 $y\notin\operatorname{sub}(i)$，则 $|\partial\mathcal L/\partial g_r|\le\tfrac1T\max_k c_k\cdot P(\operatorname{sub}(i))$；(iii) 若 $c_k$ 在 $\operatorname{sub}(i)$ 内相同（记为 $c$），则
$$
\frac{\partial\mathcal L}{\partial g_r}=\pm\frac cT\Big[\big(P(A_i)-P(B_i)\big)-\big(\mathbf 1[y\in A_i]-\mathbf 1[y\in B_i]\big)\Big],
$$
即节点 $i$ 上“$A_i$ 对 $B_i$”二分决策的误差。

*Proof.* 将 $W$ 的第 $r$ 行代入 $\nabla_g\mathcal L=\tfrac1T W(p-e_y)$。(ii) 中 $\mathbf 1[k=y]=0$，$|\sum c_k\sigma_i(k)p_k|\le\max c_k\sum_{k\in\operatorname{sub}(i)}p_k$。(iii) 按 $A_i,B_i$ 拆分求和。∎

**Theorem 3（码几何 = 树相似度）.** 对 $k\neq j$，记 $\delta_{kj}$ 为根到 $\operatorname{LCA}(k,j)$ 路径上的节点数（含两端），则
$$
\cos(W_{:,k},W_{:,j})=\frac{\delta_{kj}-2}{\sqrt{\mathrm{depth}_k\,\mathrm{depth}_j}} .
$$

*Proof.* 只有公共祖先使两列同时非零，每个公共祖先 $i$ 贡献 $2mc_kc_j\sigma_i(k)\sigma_i(j)$。LCA 以上的 $\delta_{kj}-1$ 个节点中两类同侧，各贡献 $+1$；在 LCA 处两类异侧，贡献 $-1$。故 $\langle W_{:,k},W_{:,j}\rangle=2mc_kc_j(\delta_{kj}-2)$，再除以两列范数 $\sqrt{2m\,\mathrm{depth}_k}\,c_k\cdot\sqrt{2m\,\mathrm{depth}_j}\,c_j$。∎

**Corollary 3.1（树感知的裕量上限）.** 若 $\|g\|_2\le G$，则
$$
z_k-z_j\le\frac GT\|W_{:,k}-W_{:,j}\|=\frac GT\sqrt{\frac{2D}{K}\big(1-\cos(W_{:,k},W_{:,j})\big)} .
$$
相同能量预算下，LCA 越深（越近）的两类，可达到的 logit 差越小。

*Proof.* Cauchy–Schwarz；$\|W_{:,k}-W_{:,j}\|^2=\tfrac{2D}{K}(1-\cos)$（由 Lemma）。∎

#### 3. 跳连流

记 $u_l=\operatorname{LN}(\operatorname{Pool}_l(F_l))$（无仿射），逐空间位置 $\nu_l=\tfrac1C\|u_l\|^2=\tfrac{\sigma^2}{\sigma^2+\epsilon}\le1$；$a^2+\beta^2=1$。

**Theorem 4（尺度不随深度膨胀）.** 同形状段从 $j_0$ 重置（$s_{j_0}=u_{j_0}$），则对任意 $\xi$ 的取值，
$$
s_l=\sum_{j=j_0}^{l}w_{l,j}u_j,\qquad \sum_{j}w_{l,j}^2=1,
\qquad
\tfrac1C\|s_l\|^2=\sum_j w_{l,j}^2\nu_j+2\sum_{i<j}w_{l,i}w_{l,j}\rho_{ij},
$$
其中 $\rho_{ij}=\tfrac1C\langle u_i,u_j\rangle$。若各 $u_j$ 两两不相关（$\rho_{ij}=0$），则 $\tfrac1C\|s_l\|^2\le1$ 对任意深度成立；同条件下单位权重残差 $s_l=\sum_j u_j$ 为 $\sum_j\nu_j$，随段长线性增长。

*Proof.* $\xi_l=0$ 时 $s_l=s_{l-1}$，权重不变。$\xi_l=1$ 时，若混合前 $\sum w^2=1$，混合后为 $a^2\sum w^2+\beta^2=1$。由 $s_{j_0}=u_{j_0}$ 归纳即得 $\sum w^2=1$。展开 $\|\sum_j w_ju_j\|^2$ 得范数式。∎

**Proposition 5（评估系数）.** 令 $d=d_l$，$a_e=d+(1-d)a$，$b_e=(1-d)\beta$。
(i) $\mathbb E_{\xi_l}[s_l\mid s_{l-1},u_l]=a_e s_{l-1}+b_e u_l$，即评估公式是单步条件期望；
(ii) $a_e^2+b_e^2=1-2d(1-d)(1-a)\le1$，当且仅当 $d\in\{0,1\}$ 时取等；
(iii) 新层占比 $b_e^2/(a_e^2+b_e^2)$ 在 $d\in[0,1)$ 上严格递减。

*Proof.* (i) $\xi_l\sim\mathrm{Bernoulli}(1-d)$，$\mathbb E=(1-d)(as_{l-1}+\beta u_l)+ds_{l-1}$。(ii) $d^2+2d(1-d)a+(1-d)^2=1-2d(1-d)(1-a)$，且 $\beta>0\Rightarrow a<1$。(iii) 令 $t=d/(1-d)$，占比 $=\beta^2/(t^2+2at+1)$，分母随 $t\ge0$ 递增，$t$ 随 $d$ 递增。∎

限制：$u_l$ 经卷积和 ReLU 依赖前面各层的 $\xi$，多步时评估流不等于训练流的精确期望，(i) 只对单步成立。

**Proposition 6（梯度局部性）.** TTGS 中 $\partial\mathcal L_l/\partial\theta_j=0$（$j\neq l$）。

*Proof.* $X_l=\operatorname{Dropout}(\operatorname{stopgrad}(s_{l-1}))$，$\theta_{j<l}$ 只经 stopgrad 进入 $\mathcal L_l$，梯度为 0；$\mathcal L_l$ 不依赖 $\theta_{j>l}$。读出 $W$ 固定，温度 $T_l$ 只属于本层。构树是一次性无梯度步骤，不产生梯度。∎