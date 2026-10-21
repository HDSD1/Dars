## C. 基于任务需求感知注意力的 DRL 框架

由于问题 $\mathcal{P}1$ 同时包含连续轨迹控制、连续卸载比例、离散用户关联以及时变能量约束，原问题属于高维混合非凸优化问题，难以直接获得全局最优解。本文采用分层求解：外层 DRL 只输出 UAV 飞行动作；中层在覆盖和容量约束下完成用户关联；下层在给定关联后联合优化卸载比例、CPU 算力和带宽。

本文的核心设计在于：将传统扁平拼接的用户状态表示替换为**任务需求感知的注意力表示**（Task-Demand Attention）。在随机任务到达模型下，每个时隙仅部分用户产生计算任务，且各用户的任务量存在显著差异。传统 MLP Actor-Critic 将所有用户特征直接拼接输入，隐含地将所有用户等同处理，难以识别真正影响系统性能的高任务密度用户。为此，本文在 Actor 和 Critic 中嵌入用户注意力模块，从逐用户状态中自适应提取任务需求表示，使 UAV 策略在轨迹规划时能够关注活跃且任务量较大的用户，从而更有效地适应任务分布的时空变化。

### 1) MDP 建模

将系统建模为马尔可夫决策过程 $\langle \mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma\rangle$。

**状态空间。**
时隙 $t$ 的系统状态由 UAV 状态、用户状态、无线充电信息和时间信息组成：

$$
\mathbf{s}(t)=
\Big[
\underbrace{B_k(t)/E_{\max},\mathbf{q}_k(t)}_{\text{UAV 状态}},
\underbrace{\mathbf{w}_i(t),D_i(t),\xi_i(t)}_{\text{用户状态}},
\underbrace{\tilde{\boldsymbol{\ell}}_k(t)}_{\text{WCT 信息}},
t/T
\Big].
\tag{28}
$$

其中，$B_k(t)$ 为 UAV $k$ 的剩余电量，$\mathbf{q}_k(t)\in\mathbb{R}^2$ 为 UAV 水平位置，$\mathbf{w}_i(t)\in\mathbb{R}^2$ 为用户位置，$D_i(t)$ 为任务数据量，$\xi_i(t)\in\{0,1\}$ 为任务活跃指示变量。$\tilde{\boldsymbol{\ell}}_k(t)$ 表示 UAV $k$ 到最近 WCT 的归一化距离和方向信息。状态维度为 $\dim(\mathcal{S})=6K+4M+1$。在默认设置 $K=2,M=30$ 下，$\dim(\mathcal{S})=133$。

**动作空间。**
Actor 仅输出每架 UAV 的飞行方向和速度：

$$
\mathbf{a}(t)=
\Big[
\underbrace{c_k(t),s_k(t),a_{v,k}(t)}_{\text{UAV 飞行动作}}
\Big]_{k=1}^{K}
\in[-1,1]^{3K}.
$$

其中，$(c_k,s_k)$ 为方向余弦和正弦分量，速度映射为
$v_k=\frac{1}{2}(a_{v,k}+1)V_{\max}$。卸载比例不再由 Actor 输出；用户关联后，环境根据联合优化结果计算 $x_i$。因此动作维度从 $3K+M$ 降为 $3K$，默认 $K=2$ 时为 6。
**奖励函数。**
本文以服务延迟缩减比（SLRR）作为主要优化目标。单时隙奖励定义为

$$
r(t)=M\bar{\rho}_{\mathrm{act}}(t)
-\lambda_{\mathrm{low}}\Pi^{\mathrm{low}}(t),
\tag{30}
$$

其中

$$
\bar{\rho}_{\mathrm{act}}(t)
=
\frac{1}{|\mathcal{M}_{\mathrm{act}}(t)|}
\sum_{i\in\mathcal{M}_{\mathrm{act}}(t)}
\rho_i(t)
\tag{31}
$$

为活跃用户平均 SLRR。当 $|\mathcal{M}_{\mathrm{act}}(t)| = 0$（无活跃用户）时，定义 $\bar{\rho}_{\mathrm{act}}(t) = 0$。乘以 $M$ 是为了保持奖励量级稳定，避免不同时隙活跃用户数量变化导致 Q 值尺度剧烈波动。低电量惩罚项定义为

$$
\Pi^{\mathrm{low}}(t)=
\sum_{k=1}^{K}
\left[
\max\left(0,\varepsilon-\frac{B_k(t)}{E_{\max}}\right)
\right]^2,
\tag{32}
$$

其中 $\varepsilon$ 为低电量阈值。需要指出的是，本文并未将能效作为独立优化目标；能量模型主要用于刻画 UAV 的可行性约束和长期服务持续性。当任一 UAV 电量耗尽时，当前 episode 提前终止，策略失去后续全部时隙的 SLRR 奖励，从而构成隐式的长时域机会成本，驱动智能体自主学习适时返回充电。

**优化目标。**
策略 $\pi_\theta$ 的目标为最大化期望折扣累积回报：

$$
\max_{\theta}
\mathbb{E}_{\pi_\theta}
\left[
\sum_{t=0}^{T-1}
\gamma^t r(t)
\right].
\tag{33}
$$

### 2) 任务需求感知注意力表示

传统 Actor-Critic 网络通常将所有用户特征直接拼接为一个 $4M$ 维的高维向量输入 MLP。该方式存在两个固有缺陷：(i) **缺乏集合建模的不变性**——扁平拼接依赖于任意的用户排序，输入排列的改变将导致不同的网络输出，而用户编号本身不具备物理含义；(ii) **信息稀释**——在伯努利随机任务到达模型下，每个时隙仅部分用户产生任务（$\xi_i=1$），当 $M$ 较大时，少数活跃用户的关键特征被大量不活跃用户的零值特征淹没，策略网络难以识别真正影响系统性能的高需求用户。

为解决上述问题，本文将状态向量分解为 UAV 级特征和逐用户特征：

$$
\mathbf{s}(t)\rightarrow
\left(
\mathbf{z}_{\mathrm{UAV}}(t),
\mathbf{Z}_{\mathrm{UE}}(t)
\right),
\tag{34}
$$

其中 $\mathbf{z}_{\mathrm{UAV}}(t)\in\mathbb{R}^{6K+1}$ 包含 UAV 电量、位置、WCT 信息和时间进度；$\mathbf{Z}_{\mathrm{UE}}(t) = [\mathbf{u}_1(t),\ldots,\mathbf{u}_M(t)]^\top \in\mathbb{R}^{M\times 4}$ 为逐用户特征矩阵，每个用户特征为

$$
\mathbf{u}_i(t)=
[w_{i,x}(t),w_{i,y}(t),D_i(t),\xi_i(t)].
\tag{35}
$$

随后，将逐用户特征经线性投影映射至 $d_e$ 维嵌入空间：

$$
\mathbf{E}_0(t)=\mathbf{Z}_{\mathrm{UE}}(t)\mathbf{W}_p,
\quad \mathbf{W}_p \in \mathbb{R}^{4 \times d_e}.
\tag{36}
$$

多头自注意力（MHA）模块进一步建模用户之间的任务需求关联。首先计算查询、键、值矩阵：

$$
\mathbf{Q}=\mathbf{E}_0\mathbf{W}_Q,\quad
\mathbf{K}=\mathbf{E}_0\mathbf{W}_K,\quad
\mathbf{V}=\mathbf{E}_0\mathbf{W}_V,
\tag{37a}
$$

其中 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_e \times d_e}$ 为可学习的投影矩阵。注意力输出为

$$
\mathbf{E}(t)=
\mathrm{softmax}\!\left(
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_e/n_h}}
\right)\mathbf{V}.
\tag{37b}
$$

上述计算在 $n_h$ 个注意力头上独立进行后拼接，使模型能够从多个子空间捕获用户间的任务需求关联。注意力权重由 Scaled Dot-Product 机制自动学习，使网络根据用户特征（任务量 $D_i$、活跃状态 $\xi_i$、空间位置 $\mathbf{w}_i$）动态调整对不同用户的关注程度。经残差连接与 LayerNorm 后，对所有用户嵌入进行均值池化，得到全局用户需求表示：

$$
\bar{\mathbf{e}}(t)=
\frac{1}{M}
\sum_{i=1}^{M}
\mathrm{LayerNorm}
\left(
\mathbf{E}_i(t)+\mathbf{E}_{0,i}(t)
\right).
\tag{38}
$$

最终，将 UAV 级特征与用户需求表示拼接：

$$
\mathbf{h}(t)=
[\mathbf{z}_{\mathrm{UAV}}(t),\bar{\mathbf{e}}(t)]
\in \mathbb{R}^{6K+1+d_e}.
\tag{39}
$$

$\mathbf{h}(t)$ 被同时输入 Actor 和 Twin Critic 网络（两层全连接，隐藏层维度 $400 \times 300$），用于生成轨迹决策并评估对应 Q 值。该注意力表示具有**置换等变性**——用户排列顺序的改变不影响聚合结果，并且能够根据任务需求的时空分布变化自适应调整关注权重。

### 3) 分层决策与联合卸载/资源分配

每个时隙依次执行：

1. **外层 DRL：** Attention-SAC 根据用户需求表示输出 UAV 飞行动作 $(c_k,s_k,v_k)$。
2. **中层关联：** 按覆盖约束构造可行用户-UAV 对，用边际 SLRR 贪心和有界 move/swap 选择关联集合。可行集合是用户唯一关联与 UAV 容量两个划分拟阵的交。目标单调，但两种资源耦合下没有子模性证明，因此不声明 $1/3$ 近似保证。
3. **下层联合求解：** 对 UAV $k$ 的每个关联用户，定义全本地时延 $A_i=C_i/f_{\mathrm{UE}}$、传输系数 $a_{i,k}=D_i/\log_2(1+\mathrm{SNR}_{i,k})$、计算量 $c_i=C_i$。对给定 $(b_{i,k},f_{i,k})$，令 $E_{i,k}=a_{i,k}/b_{i,k}+c_i/f_{i,k}$，则最优卸载比例为
   $$
   x_i^*=\frac{A_i}{A_i+E_{i,k}},\qquad
   \rho_i^*=\frac{A_i}{A_i+E_{i,k}}.
   $$
   因而内层最大化 $\sum_i A_i/(A_i+a_{i,k}/b_{i,k}+c_i/f_{i,k})$，满足 $\sum_i b_{i,k}\le W$ 和 $\sum_i f_{i,k}\le F_{\mathrm{UAV}}$。目标是凹函数，KKT 条件充分。令
   $$
   g_i=\sqrt{a_{i,k}\lambda}+\sqrt{c_i\mu},\quad
   y_i=\left[\frac{1}{\sqrt{A_i}}-\frac{g_i}{A_i}\right]^+,
   $$
   则
   $$
   b_{i,k}=y_i\sqrt{\frac{a_{i,k}}{\lambda}},\qquad
   f_{i,k}=y_i\sqrt{\frac{c_i}{\mu}}.
   $$
   固定带宽乘子 $\lambda$ 后，将 $q=\sqrt{\mu}$ 作为未知量。按阈值 $\tau_i=(\sqrt{A_i}-\sqrt{a_{i,k}\lambda})^+/\sqrt{c_i}$ 排序并搜索 CPU 活跃前缀 $I$，其乘子有显式解 $q=\frac{\sum_{i\in I}\sqrt{c_i}(\sqrt{A_i}-\sqrt{a_{i,k}\lambda})/A_i}{F_{\mathrm{UAV}}+\sum_{i\in I}c_i/A_i}$；随后仅对 $\lambda$ 二分。单 UAV 复杂度为 $O(n_k\log n_k\log(1/\epsilon))$；所有 UAV 的保守复杂度为 $O(M\log M\log(1/\epsilon))$。执行前安全层仅在电量或边缘服务时限触发时投影卸载比例。

该设计把策略输出维度从 $3K+M$ 降到 $3K$。联合求解替代旧的独立 CPU/带宽平方根分配，避免将卸载比例留给 Actor 逐用户学习。注意力池化仍用于移动策略的用户需求表示。
### 4) 训练机制

本文采用 SAC（Soft Actor-Critic）作为基础强化学习框架。SAC 通过最大熵目标在累积奖励最大化和策略随机性之间取得平衡，其目标函数为

$$
J(\pi)=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T-1}
\gamma^t
\left(
r(t)+\alpha\mathcal{H}(\pi(\cdot|\mathbf{s}(t)))
\right)
\right],
\tag{41}
$$

其中 $\alpha$ 通过对偶梯度下降自动调节，使策略在训练早期保持充分探索，后期自动收敛至确定性决策。

为提升训练稳定性和样本效率，本文进一步采用以下机制：(i) **优先经验回放（PER）**，按 TD 误差大小优先采样高学习价值的转移 [Schaul et al., 2016]；(ii) **N-step return**（$n=3$），提升 TD 目标的信噪比；(iii) **Twin Critic 与 Huber Loss**，缓解 Q 值过估计并增强对离群 TD 误差的鲁棒性；(iv) **IS 加权 Actor 更新**，修正 PER 非均匀采样对策略梯度的偏差。需要指出的是，上述机制主要用于改善样本效率和训练鲁棒性，**本文的核心设计在于任务需求感知注意力表示及其与分层资源分配框架的结合**。

### Algorithm 1: Task-Demand Attention Enhanced DRL Framework

**Input:** 环境参数 $K, M, N, T, \tau$；超参数 $\gamma, \alpha_{\text{per}}, n$
**Output:** 训练完毕的策略网络 $\pi_\theta$

| 行号 | 步骤 |
|:---:|:---|
| 1 | 初始化 Attention-SAC Actor $\pi_\theta$、Twin Critic $Q_{\phi_1}, Q_{\phi_2}$ 及目标网络；初始化温度 $\log\alpha$、PER 缓冲区 $\mathcal{B}$、N-step 缓冲区 |
| 2 | **for** episode $e = 1$ to $E_{\max}$ **do** |
| 3 | &emsp;重置环境，所有 UAV 满电 $B_k(0) = E_{\max}$，获取初始状态 $\mathbf{s}(0)$ |
| 4 | &emsp;**for** 时隙 $t = 0$ to $T-1$ **do** |
| 5 | &emsp;&emsp;**（注意力编码）** 分解 $\mathbf{s}(t) \to (\mathbf{z}_{\text{UAV}}, \mathbf{Z}_{\text{UE}})$；MHA 聚合得 $\mathbf{h}(t)$（式 34-39） |
| 6 | &emsp;&emsp;**（外层 DRL）** 若预热阶段则随机采样，否则 $\mathbf{a}(t) \sim \pi_\theta(\cdot|\mathbf{h}(t))$ |
| 7 | &emsp;&emsp;将飞行动作映射至速度；更新 UAV 位置（式 1-2） |
| 8 | &emsp;&emsp;**（中层关联）** 按最大正 SLRR 边际增益贪心建立关联，再执行有界 move/swap 改进，得 $\mathcal{S}_k(t)$ |
| 9 | &emsp;&emsp;**（下层联合优化）** 计算最优卸载比例 $x_i^*$、频率 $f_{i,k}^*$ 和带宽 $b_{i,k}^*$，按活跃集直接求 CPU 乘子并对带宽乘子二分 |
| 10 | &emsp;&emsp;计算延迟、充电能量、能耗；更新电池；计算奖励 $r(t)$ |
| 11 | &emsp;&emsp;**if** $\exists k: B_k(t+1) < 0$ **then** done $\leftarrow$ True |
| 12 | &emsp;&emsp;存储至 N-step 缓冲区；累积 $n$ 步后存入 PER 缓冲区 $\mathcal{B}$ |
| 13 | &emsp;&emsp;**if** 非预热阶段 **then** PER 采样、更新 Critic/Actor/温度/目标网络 |
| 14 | &emsp;&emsp;**if** done **then break** |
| 15 | &emsp;**end for** |
| 16 | **end for**；**return** $\pi_\theta$ |

**复杂度分析。** 记覆盖内候选对数为 $E\le MK$，贪心加入次数为 $r\le\min(M,KN_{\max})$，局部改进轮数上限为 $L$。每个 UAV 子集的效用评估至多遍历 $M$ 个用户，且相同子集结果会缓存；因此关联搜索的保守复杂度为 $O((rE+L(M^2+MK))M\log M\log(1/\epsilon))$。实际开销取决于覆盖重叠度、候选数和局部搜索轮数，应通过运行时间测量报告。其余每时隙开销包括 MHA 编码 $O(M^2d_e)$ 和 Actor 前向推理；联合资源求解已经计入关联子集评估开销。训练阶段还需 Twin Critic 反向传播与 PER 的 SumTree 更新。
