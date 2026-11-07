# DARS 中层关联替换实验：执行方案

## 结论与实验边界

本实验已实现 Greedy、Matching、Swap 三种关联模块的配对比较，不训练模型。
只由原 DARS-Greedy 推进环境，另两种方法旁路计算。因此结果是冻结 DARS 状态分布下的即时关联模块比较，不是完整替换后重训练的系统排名，也不是全局最优性验证。

默认模型与此前原始时延目标比较一致：`original_main_seed42_gpu_retry_20261029_234000/per_sac_cvx.pth`。
路径由此前评估的 `manifest.json` 读取，同时读取训练时的 source_snapshot。运行前后核对模型参数摘要和 checkpoint 文件 SHA256。默认不使用 seed44 模型。若换模型，需要同时指定对应的 --model、--model-manifest、--source-dir、--env-config，代码会验证核心源文件与训练清单哈希一致。

## 1. Matching 的实现

参考 Sun et al., INFOCOM 2024, Algorithm 2：多对一请求—接受—拒绝匹配。
DOI: https://doi.org/10.1109/INFOCOM52122.2024.10621095
作者稿: https://arxiv.org/html/2404.04597v1
相关期刊版本: TJCCT, IEEE TMC 24(4):3130–3147, 2025, https://doi.org/10.1109/TMC.2024.3505155

保留其匹配机制，去掉原文价格协商，改用 DARS 的名义 SLRR 收益作为双方固定偏好。
每个用户在覆盖内的 UAV 上，以名义 CPU F_UAV/N_MAX、带宽 W/N_MAX 和 Actor 原始卸载比例估计本地/边缘完成时延，分数为 1-max(Tloc,Tedge_est)/Tbase。
只申请收益为正的可覆盖 UAV，用户按收益降序申请。UAV 按该分数保留至多 N_MAX 个用户，拒绝较低分用户；被拒用户申请下一个 UAV。分数在本次匹配内固定，平局按用户/UAV ID 打破。
没有可接受选项的用户本地执行。最终 CPU、带宽及安全修正仍按原 DARS 计算；名义资源只用于排序，不是实际资源分配。

这是适配后的 Matching，不是完整 TJCCT 复现。也不把原文特定效用模型下的理论结论转移到本实验。

## 2. Swap 的实现

参考 Wang et al., IEEE IoTJ 12(12):19782–19800, 2025, Algorithms 1–3。
正式题名: AAV-Assisted Joint Mobile Edge Computing and Data Collection via Matching-Enabled Deep Reinforcement Learning。
DOI: https://doi.org/10.1109/JIOT.2025.3542025
作者稿（题名开头为 UAV-assisted）: https://arxiv.org/html/2502.07388v1

采用文献两阶段匹配思路，保留 MEC 分支，移除数据采集分支：
1. 先运行当前距离关联，得到各 UAV 的关联人数 n_k^dist。
2. 用固定估计速率 W/max(1,n_k^dist)*log2(1+SNR_ik) 构建偏好，再进行多对一请求—接受匹配。固定估计速率是本实验适配，未照搬原文完整物理层。
3. 依次考察单用户转移、本地执行和双用户交换；未关联用户、空闲容量通过虚拟本地/空位选项处理。这是相对原交换邻域的明确扩展。
4. 每个候选重新执行关联掩码、闭式资源分配、卸载安全缩放、资源重算和最终能量检查。
5. 以 J=sum_active max(Tloc,Tedge)/Tbase 为目标；候选可行且 J 下降超过 1e-8 时接受第一个改进，重新扫描邻域。
6. 默认最多 20 轮、1000 次完整候选评价（包含初始方案）；无改进的完整扫描才标记收敛。预算耗尽单独报告，不宣称局部最优。

Swap 不保证优于当前 Greedy，因为初始方案是速率匹配；只保证接受的交换相对其自身初始方案改善。本实验允许报告负的相对 Greedy 增益，不截断。

## 3. 共享约束与固定条件

固定同一个时隙的状态、过滤后的候选 UAV 位置、Actor 原始卸载提案。不能把原 Greedy 下修正后的卸载比例当作其他方法输入。
K=2、M=30、N_MAX=10、覆盖半径、任务、信道和每 UAV 能量预算均来自训练源代码和当前环境。
三种方法都检查活跃用户、覆盖范围、每用户至多一架 UAV、每 UAV 容量上限。所有活跃用户均进入指标，未关联用户按本地执行计入。
只统计运动过滤成功、原动作已执行、存在活跃任务的状态。不按边缘时延分支或收益大小筛样。拒绝动作另写 rejected.jsonl。
每个统计状态均验证 Greedy 的关联、卸载比例、CPU、带宽、计算能耗和时延重建与实际环境一致；不一致立即失败。旁路计算前后还验证真实环境未被修改。

## 4. 今天运行的命令

在 PowerShell 中进入仓库：

```powershell
Set-Location -LiteralPath 'D:\Codes\UAV-DDPG-main\DDPG - 副本\Dars-final'
```

一键顺序运行基础100、高负载50、未见顺序50个评估种子：

```powershell
& .\experiments\run_association_suite.ps1
```

脚本默认使用已验证可导入 PyTorch 的 `D:\PythonEnvs\uav_mec_gpu\python.exe`，强制 CPU 评估。系统默认 Anaconda Python 存在 OpenMP 冲突，不建议用它替换。
脚本在当前窗口显示进度并保存日志；不要关闭窗口。不会后台启动训练。

若只先跑基础场景：

```powershell
& 'D:\PythonEnvs\uav_mec_gpu\python.exe' -u .\experiments\compare_association.py `
  --scenario base --episodes 100 --seed-start 310000 `
  --output-dir .\results\association_base_formal
```

可选低电量扩展（不在默认三场景套件内）：

```powershell
& 'D:\PythonEnvs\uav_mec_gpu\python.exe' -u .\experiments\compare_association.py `
  --scenario low_battery --episodes 50 --seed-start 340000 `
  --output-dir .\results\association_lowbattery_formal
```

已有输出目录禁止覆盖；再次运行请更换 --output-dir 或套件 -Tag。
默认三场景使用独立评估种子，不用于训练，脚本会检测与训练/验证种子重叠。
未见顺序的训练配置必须是预先划分的训练/测试模式，不能为了这个实验把测试顺序加入训练。

仅检查接口的短跑（不能用作论文结果）：

```powershell
& 'D:\PythonEnvs\uav_mec_gpu\python.exe' -u .\experiments\compare_association.py `
  --episodes 1 --max-steps 3 --timing-repeats 1 `
  --output-dir .\results\association_smoke_new
```

## 5. 指标与时间口径

主表：平均 SLRR、相对 Greedy 的平均 SLRR 绝对增益、总模块耗时均值/P95。
另报告目标相对改善百分比（允许负值）、优/平/劣比例、关联/卸载用户数、安全缩放比例、计算能耗、候选评价次数、接受改进次数、搜索预算耗尽比例、约束可行比例。

没有“正确率”或“全局最优性差距”，因为两个匹配方法都不是精确最优标签。
SLRR 增益是绝对差；乘100才是百分点，不是相对提升百分比。
时延目标 J 的相对改善=(J_Greedy-J_method)/J_Greedy*100%。

总模块计时包含偏好构造、关联搜索及所有候选的资源分配、安全检查与时延评价；不含 Actor 推理、运动过滤、快照复制或磁盘IO。因此不可称整个 DARS 加速比。
先预热，每状态独立完整运行每方法3次，轮换方法顺序，以每状态中位时间再汇总均值/P95。没有跨方法结果缓存。
95%置信区间采用按评估种子分组 bootstrap，保持主指标的状态加权口径。

训练可以继续，但会争用 CPU 影响毫秒计时。正式时间表最好在没有其他高负载任务时运行本评估，不要直接用训练并发时的数值声称算法耗时优势。
若 swap 的预算耗尽比例较高，用新的输出目录提高 --swap-evaluations/--swap-rounds 再评估；必须披露改变后的预算。

## 6. 输出与论文位置

每个场景保存 manifest.json、state_results.jsonl、episode_results.jsonl、summary.json、summary.md；拒绝动作有 rejected.jsonl，套件还有逐场景日志。
state_results.jsonl 保留原动作、任务、活跃标志、位置、能量预算，以及各方法的关联/比例/资源/时延/搜索次数和每次计时。
模型摘要一致只是冻结证据，不是算法最优性证据。

论文不改主算法、训练更新或 Algorithm 1；在实验部分的对比方法介绍后增加关联模块比较说明，再在结果部分放三方法表。
可复制文字见 `reviewer_association_methods.tex`，参考文献见 `reviewer_association_refs.bib`。
其文字是实验协议及方法说明，不含尚未完成的实验结果，也不声称 Greedy 更好。

## 7. 本次交付验证

- 11 项专项测试通过，覆盖请求拒绝、容量、无覆盖/无任务、原始提案恢复、实际环境重建、真实安全缩放、零/小能量预算、交换单调性、预算退出和无副作用。
- 已使用冻结 seed42 checkpoint 完成基础3时隙、高负载5时隙以及低电量1回合的接口验证。低电量验证运行15个接受时隙后出现安全拒绝，脚本正常记录并结束。
- 上述都是开发验证，不能当作正式论文统计。完整100/50/50种子套件尚未启动。
- PowerShell脚本通过语法检查；新增Python文件通过编译检查。

LaTeX片段与BibTeX已生成；本次MiKTeX编译因当前环境无法访问其注册表/用户配置目录而未完成，不能声称已通过编译。
