# FF 训练设计

## BiCovG

```text
ReLU 特征
  ├→ 平方 → 多尺度平均池化 ───────────┐
  └→ 可学习 1×1 通道投影 → 平方 → 多尺度平均池化
                                    ↓
                                   拼接
                                    ↓
                            可学习线性 readout
                                    ↓
                             局部交叉熵损失
                                    ↓
                   更新当前卷积、通道投影和 readout
```

当前层特征：
\[
F_l=\operatorname{ReLU}\!\left(\operatorname{Conv}_{\theta_l}
(\operatorname{detach}(X_l))\right)
\]

BiCovG：
\[
g_l=\operatorname{Concat}_{s\in S_l}
\left[
\operatorname{vec}\!\left(\operatorname{AvgPool}_{s}(F_l^2)\right),
\operatorname{vec}\!\left(\operatorname{AvgPool}_{s}((P_lF_l)^2)\right)
\right]
\]

局部分类与训练：
\[
z_l=W_lg_l+b_l,\qquad
\mathcal L_l=\operatorname{CE}(z_l,y)
\]
\[
\text{由 }\mathcal L_l\text{ 更新 }\{\theta_l,P_l,W_l,b_l\}
\]

\(P_l\)：可学习通道投影；\(S_l\)：池化尺度集合；\(\operatorname{AvgPool}_{s}\)：输出为 \(s\times s\) 的自适应平均池化。

## ASGE

```text
ReLU 特征
    ↓
   平方
    ↓
按通道数确定空间划分 → 自适应平均池化
                          ↓
                         展平
                          ↓
                 固定随机线性 readout
                          ↓
                   局部交叉熵损失
                          ↓
                    仅更新当前卷积
```

当前层特征：
\[
F_l=\operatorname{ReLU}\!\left(\operatorname{Conv}_{\theta_l}
(\operatorname{detach}(X_l))\right)
\]

ASGE：
\[
p_l=\min\!\left(\max\!\left(1,\left\lfloor\alpha\frac{C_L}{C_l}\right\rfloor\right),H_l,W_l\right),
\qquad
g_l=\operatorname{vec}\!\left(\operatorname{AvgPool}_{p_l}(F_l^2)\right)
\]

局部分类与训练：
\[
z_l=W_l^{\mathrm{fixed}}g_l+b_l^{\mathrm{fixed}},\qquad
\mathcal L_l=\operatorname{CE}(z_l,y)
\]
\[
\text{由 }\mathcal L_l\text{ 仅更新 }\theta_l
\]

\(C_l,C_L\)：当前层、最后卷积层的通道数；\(H_l,W_l\)：特征图尺寸；\(\alpha\)：划分系数；\(p_l\)：池化输出边长。readout 的权重与偏置随机初始化后固定。

[来源：ASGE 方法](https://arxiv.org/html/2509.12394v1#S3)

### 最终预测：Fusion Pred

局部随机 readout 固定；最终 Fusion 分类器另行训练，只接收 detach 后的特征，不更新卷积。

```text
第 2–L 层传递特征 → detach → 各层 GAP → 可学习线性分类 → 相加
                                                           ↓
                                                     预测交叉熵
                                                           ↓
                                                  仅更新 Fusion 分类器
```

\[
z_{\mathrm{fusion}}=\sum_{l=2}^{L}
\left[A_l\operatorname{GAP}(\operatorname{detach}(X_{l+1}))+b_l\right],
\qquad\mathcal L_{\mathrm{fusion}}=\operatorname{CE}(z_{\mathrm{fusion}},y)
\]

\(X_{l+1}\)：第 \(l\) 层经过 Pool／Dropout／Norm 后的传递特征；\(A_l,b_l\)：Fusion 分类器参数。

[来源：ASGE Prediction Strategies](https://arxiv.org/html/2509.12394v1)、[官方 VGG8 实现](https://github.com/KAI-Laboratory/ASGE/blob/3226665035a15da90c70f3df0a6bc5ea7de326cc/VGG8.py)

## DTG-FF

```text
ReLU 特征（CNN版）
          ↓
         平方
          ↓
   自适应平均池化 → 展平
          ↓
   除以本层可学习温度
          ↓
  固定随机线性 readout
          ↓
    局部交叉熵损失
          ↓
 更新当前卷积和温度参数
```

当前层特征：
\[
F_l=\operatorname{ReLU}\!\left(\operatorname{Conv}_{\theta_l}
(\operatorname{detach}(X_l))\right)
\]

温度缩放 goodness：
\[
g_l=\operatorname{vec}\!\left(\operatorname{AvgPool}_{(h_l,w_l)}(F_l^2)\right),
\qquad
\tilde g_l=\frac{g_l}{T_l}
\]
\[
T_l=T_{\min}+(T_{\max}-T_{\min})\sigma(\alpha_l)
\]

局部分类与训练：
\[
z_l=W_l^{\mathrm{fixed}}\tilde g_l+b_l^{\mathrm{fixed}},\qquad
\mathcal L_l=\operatorname{CE}(z_l,y)
\]
\[
\text{由 }\mathcal L_l\text{ 更新 }\{\theta_l,\alpha_l\}
\]

\((h_l,w_l)\)：池化输出尺寸；\(\alpha_l\)：本层可学习标量；\(\sigma\)：Sigmoid。CNN 版本取 \(T_{\min}=0.5,T_{\max}=2.0\)；readout 的权重与偏置随机初始化后固定。

### 温度训练

每层初始化 \(\alpha_l=0\)，因此 \(T_l=0.5+1.5\sigma(0)=1.25\)。温度与当前卷积使用同一个局部交叉熵损失训练，无需额外标签或损失。

```text
goodness ÷ T → 固定 readout → 局部交叉熵
                           ↓
                      局部反向传播
                     ↙           ↘
                 更新卷积     更新 α → 改变 T
```

温度参数的梯度：
\[
\frac{\partial\mathcal L_l}{\partial\alpha_l}
=\frac{\partial\mathcal L_l}{\partial T_l}
\cdot1.5\,\sigma(\alpha_l)\bigl(1-\sigma(\alpha_l)\bigr)
\]

由局部优化器更新；以梯度下降形式表示：
\[
\alpha_l\leftarrow\alpha_l-\eta\frac{\partial\mathcal L_l}{\partial\alpha_l}
\]

\(\eta\)：学习率。更新 \(\alpha_l\) 后重新计算 \(T_l\)，Sigmoid 将温度约束在 \(0.5\) 到 \(2.0\) 之间。

[来源：DTG-FF 方法（CNN版）及附录 A.4](https://arxiv.org/html/2606.06539v1#S4)

## CwC

```text
ReLU 特征 → 池化（按层配置）→ BatchNorm
                              ↓
                    通道均分为 K 组（一类一组）
                              ↓
                      平方 → 组内平均
                              ↓
                    K 维 goodness 直接作为 logits
                              ↓
                        局部交叉熵损失
                              ↓
                      更新当前学习单元
```

当前层特征：
\[
F_l=\operatorname{ReLU}\!\left(\operatorname{Conv}_{\theta_l}
(\operatorname{detach}(X_l))\right),\qquad
A_l=\operatorname{BN}_{\phi_l}\!\left(\operatorname{Pool}_l(F_l)\right)
\]

通道组平方 goodness：
\[
g_{l,k}=\frac{1}{m_lH_lW_l}
\sum_{c\in\mathcal G_{l,k}}\sum_{h=1}^{H_l}\sum_{w=1}^{W_l}
A_{l,c,h,w}^{\,2},\qquad m_l=\frac{C_l}{K}
\]

局部分类与训练：
\[
z_l=(g_{l,1},\ldots,g_{l,K}),\qquad
\mathcal L_l=\operatorname{CE}(z_l,y)
\]
\[
\text{由 }\mathcal L_l\text{ 更新当前卷积 }\theta_l
\text{ 及启用时的 BN 仿射参数 }\phi_l
\]

\(K\)：类别数；\(\mathcal G_{l,k}\)：固定对应第 \(k\) 类的通道组；\(C_l,H_l,W_l\)：\(A_l\) 的尺寸。无池化的层取恒等映射；无需线性 readout。

共同骨干须满足 \(C_l\bmod K=0\)。这里的 goodness 通道分组不改变 Conv 的 groups；原文另配套 CFSE 分组卷积结构，此处仅列局部训练设计。

[来源：CwC 方法](https://ojs.aaai.org/index.php/AAAI/article/view/29369)

## DeeperForward

```text
ReLU 特征 → 通道均分为 K 组（一类一组）
              ├→ 组内平均（不平方）
              │          ↓
              │  K 维 goodness 直接作为 logits
              │          ↓
              │    局部交叉熵损失
              │          ↓
              │    仅更新当前卷积
              └→ GroupNorm（同样 K 组）
                         ↓
                 平均池化 → detach → 下一层
```

当前层特征：
\[
F_l=\operatorname{ReLU}\!\left(\operatorname{Conv}_{\theta_l}
(\operatorname{detach}(X_l))\right)
\]

通道组均值 goodness：
\[
g_{l,k}=\frac{1}{m_lH_lW_l}
\sum_{c\in\mathcal G_{l,k}}\sum_{h=1}^{H_l}\sum_{w=1}^{W_l}
F_{l,c,h,w},\qquad m_l=\frac{C_l}{K}
\]

局部分类与训练：
\[
z_l=(g_{l,1},\ldots,g_{l,K}),\qquad
\mathcal L_l=\operatorname{CE}(z_l,y),\qquad
\text{由 }\mathcal L_l\text{ 仅更新 }\theta_l
\]

传递特征时去除组内均值：
\[
Z_{l,c,h,w}=\frac{F_{l,c,h,w}-g_{l,k}}{\sqrt{v_{l,k}+\epsilon}},
\qquad c\in\mathcal G_{l,k}
\]

\(v_{l,k}\)：当前样本第 \(k\) 组内的方差；\(\epsilon\)：数值稳定项。goodness 在归一化前计算；归一化按样本、按组进行，不使用 batch 统计量。下采样使用平均池化，无需下采样时省略。

同样要求 \(C_l\bmod K=0\)，无需线性 readout；与 CwC 的核心区别是“不平方的均值 goodness + 按类别组归一化”。

[来源：DeeperForward §3.1–3.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/7dd309df03d37643b96f5048b44da798-Paper-Conference.pdf)

## HCL-FF

DeeperForward 的均值 goodness，加上“分层监督”和“归一化后的监督对比学习”。

```text
卷积 → ReLU
         ├→ 通道分组 → 组内均值 goodness
         │                      ↓
         │             按当前层的类别层级聚合
         │                      ↓
         │                分层交叉熵损失
         │
         └→ GroupNorm
                ├→ 全局平均池化 → 可学习投影 → 监督对比损失
                └→ detach → 下一层
```

分层监督：浅层区分粗类别，深层逐渐区分细类别。监督对比：在归一化后的特征上，让同类样本接近、异类样本分离，始终使用细类别标签。

\[
g_{l,k}=\operatorname{Mean}(F_{l,k}),\qquad
\hat g_{l,j}=\frac{1}{|S_{l,j}|}\sum_{k\in S_{l,j}}g_{l,k}
\]
\[
Z_l=\operatorname{GroupNorm}(F_l;K),\qquad
h_l=P_l\operatorname{GAP}(Z_l)
\]
\[
\boxed{\mathcal L_l
=\operatorname{CE}(\hat g_l,\hat y_l)
+\lambda\,\mathcal L_{\mathrm{SupCon}}(h_l,y)}
\]

\(F_l\)：当前卷积的 ReLU 输出；\(F_{l,k}\)：第 \(k\) 个细类别通道组；\(S_{l,j}\)：当前层某个粗类别包含的细类别集合；\(\hat y_l\)：对应的粗类别标签；\(P_l\)：可学习投影。

两个损失合并后进行本层局部 BP：当前卷积接受两个损失的梯度，投影只接受对比损失的梯度。投影是辅助训练分支，不在通向下一层的主路径上；下一层接收归一化后的特征，而非投影结果。

[来源：HCL-FF §3.1–3.4](https://arxiv.org/html/2605.24797v1#S3)

### 平台适配

把 HCL-FF 的 FF 训练规则接到共同 CNN 骨干上，而不是照搬作者整套网络。

- Conv 的深度、通道数和卷积参数遵守共同配置，原文的残差骨干不自动带入。
- 共同骨干每层通道数须满足 \(C_l\bmod K=0\)。
- 保留分层 goodness 与归一化后对比学习；对比分支在 detach 之前取特征。
- 按共同网络深度安排粗到细监督，并明确类别层级如何构造。

在机制上可适配共同平台；适配后的准确率需要重新实验，不能直接沿用原文结果。

### 矩阵示例

下面用一个**人为构造的小例子**。设有 4 个细类别：

```text
动物：猫、狗
交通工具：汽车、卡车
```

当前层有 **8 个通道**，每两个通道对应一个细类别；为简化，每个通道只有一个空间位置。

#### 1. 分层分类损失：浅层先学分大类

假设输入是一张猫的图片，当前层 ReLU 输出为：

\[
F=
\begin{bmatrix}
5&3\\
4&2\\
3&1\\
2&0
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow\text{猫的通道组}\\
\leftarrow\text{狗的通道组}\\
\leftarrow\text{汽车的通道组}\\
\leftarrow\text{卡车的通道组}
\end{matrix}
\]

这里每行代表两个通道，**不是空间上的一行像素**。

先取每个通道组的均值，得到细类别 goodness：

\[
g=
\begin{bmatrix}
4&3&2&1
\end{bmatrix}
\]

**浅层：只要求分清“动物／交通工具”**

把细类别 goodness 再按粗类别平均：

\[
\hat g=
\begin{bmatrix}
(4+3)/2 &(2+1)/2
\end{bmatrix}
=
\begin{bmatrix}
3.5&1.5
\end{bmatrix}
\]

猫属于动物，因此：

\[
p(\text{动物})
=\frac{e^{3.5}}{e^{3.5}+e^{1.5}}
\approx0.881
\]

\[
\boxed{\mathcal L_{\text{浅层分类}}
=-\log p(\text{动物})\approx0.127}
\]

**这时猫组高还是狗组高，不是该分类损失关心的重点；动物总体高于交通工具即可。**

**深层：要求分清具体类别**

若同样的 goodness 出现在采用细类别监督的深层，就直接计算：

\[
\boxed{
\mathcal L_{\text{深层分类}}
=-\log\frac{e^4}{e^4+e^3+e^2+e^1}
\approx0.440
}
\]

这时不仅要压过汽车和卡车，**猫也必须压过狗**。这就是“从粗到细”。[原文分层监督](https://arxiv.org/html/2605.24797v1#S3.S2)

#### 2. 对比损失：训练归一化后剩下的特征

上面猫图片的每个通道组，经去均值、除标准差后，忽略 \(\epsilon\)：

\[
[5,3]\rightarrow[1,-1],\qquad
[4,2]\rightarrow[1,-1]
\]

其他两组也一样，因此：

\[
Z_{\text{猫}_1}
=[1,-1,\;1,-1,\;1,-1,\;1,-1]
\]

**原来的组均值 \(4,3,2,1\) 被去掉了；留下的是组内相对变化。**

现在考虑一个含四张图片的 batch。假设归一化后的特征为：

\[
Z=
\begin{bmatrix}
1&-1& 1&-1& 1&-1& 1&-1\\
1&-1&-1& 1& 1&-1&-1& 1\\
-1&1& 1&-1&-1&1& 1&-1\\
-1&1&-1& 1&-1&1&-1& 1
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow\text{猫}_1\\
\leftarrow\text{猫}_2\\
\leftarrow\text{汽车}_1\\
\leftarrow\text{汽车}_2
\end{matrix}
\]

因为空间尺寸是 \(1\times1\)，全局平均池化不改变这些向量。

**可学习投影**

用一个 \(2\times8\) 的矩阵，将 8 维特征投影成 2 维：

\[
P=
\begin{bmatrix}
1&0&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0
\end{bmatrix}
,\qquad H=ZP^\top
\]

得到：

\[
H=
\begin{bmatrix}
1&1\\
1&-1\\
-1&1\\
-1&-1
\end{bmatrix}
\quad
\begin{matrix}
\text{猫}_1\\
\text{猫}_2\\
\text{汽车}_1\\
\text{汽车}_2
\end{matrix}
\]

目前两张猫图片的向量互相垂直，**同类还没有聚在一起**。

**计算对比损失**

以猫₁为锚点：

- 猫₂：同类，作为正样本。
- 汽车₁、汽车₂：异类，作为负样本。

使用余弦相似度，令温度 \(\tau=1\)：

\[
s(\text{猫}_1,\text{猫}_2)=0,\quad
s(\text{猫}_1,\text{汽车}_1)=0,\quad
s(\text{猫}_1,\text{汽车}_2)=-1
\]

于是：

\[
\boxed{
\mathcal L_{\text{对比},\text{猫}_1}
=-\log\frac{e^0}{e^0+e^0+e^{-1}}
\approx0.862
}
\]

它希望：**提高猫₁与猫₂的相似度，降低猫₁与汽车的相似度。**其他样本也分别作为锚点计算。[原文监督对比目标](https://arxiv.org/html/2605.24797v1#S3.S3)

**投影“学到了”什么？**

例如，将投影矩阵变为：

\[
P'=
\begin{bmatrix}
1&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0
\end{bmatrix}
\]

此时：

\[
H'=
\begin{bmatrix}
1&0\\
1&0\\
-1&0\\
-1&0
\end{bmatrix}
\]

两张猫完全同向，两张汽车完全同向，猫与汽车反向：

\[
\boxed{
\mathcal L'_{\text{对比},\text{猫}_1}
=-\log\frac{e^1}{e^1+e^{-1}+e^{-1}}
\approx0.240
}
\]

**损失降低了：投影学会保留区分类别的方向，忽略这个例子中无助于分类的方向。**这里的 \(P'\) 只是展示一种降低损失的变化，不是声称一次梯度更新就得到它。

#### 3. 实际训练不是只改变投影

\[
\mathcal L_l
=\mathcal L_{\text{分层分类},l}
+\lambda\mathcal L_{\text{对比},l}
\]

```text
分层分类损失 ──────────────→ 更新当前卷积
对比损失 → 投影 → Norm ───→ 更新当前卷积
             └───────────→ 更新投影参数
```

**投影帮助构造训练信号；这个信号也穿过 Norm 改善当前卷积。下一层接收的仍然是 Norm 后的特征，而不是投影结果。**
