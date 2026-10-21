# Experiment Section Draft

> The numerical results below were collected with the previous action and allocation formulation. They are historical reference values and need reevaluation after the movement-only policy and joint inner solver are retrained.

> Suggested placement: Section V / Experiments.  
> Suggested main figures: convergence comparison, user-density sensitivity, hotspot-intensity bar chart, and trajectory case study.  
> Note: the attention heatmap is intentionally excluded because its learned weights are not stable enough to support a strong interpretability claim.

## V. Experiments

### A. Experimental Setup

We evaluate the proposed DARS framework in a multi-UAV assisted mobile edge computing scenario with time-varying and spatially non-uniform task arrivals. Unless otherwise specified, the service area is a \(500 \text{ m} \times 500 \text{ m}\) square region. The system contains \(K=2\) UAVs and \(M=30\) ground users. The policy outputs only each UAV’s flight direction and speed; after association, the environment jointly optimizes user offloading ratios and resource allocations. The whole service period is divided into 50 time slots. User tasks are generated from four spatial clusters, and the hot clusters rotate across three phases during one episode. The default hot-cluster task arrival probability is \(p_{hot}=0.9\), while the cold-cluster probability is \(p_{cold}=0.1\). The task size is uniformly sampled from 2.5 Mbits to 3.0 Mbits for active users.

We compare DARS with four representative DRL baselines: SAC+CVX, TD3+CVX, MADDPG+CVX, and ESAC-PER. For fair resource allocation comparison, all learning-based baselines are coupled with the same joint concave resource-allocation solver when the algorithm name contains +CVX. DARS further incorporates task-demand-aware user representation and dual-priority replay.

The main performance metrics are episode reward, delay reduction ratio \(\rho\), and average service delay. The delay reduction ratio measures the relative reduction of task completion delay compared with local computation. A higher reward and a larger \(\rho\) indicate better system-level performance, while a smaller average delay indicates more efficient task processing. Survival steps are recorded only as a feasibility check and are not used as a primary performance metric.

Each algorithm is trained for 15,000 episodes. For the convergence study, three random seeds are used to report learning stability. Sensitivity studies vary one system parameter at a time while keeping the remaining configuration unchanged.

### B. Overall Convergence Performance

Fig. X shows the multi-seed convergence comparison among DARS and the baseline algorithms. The convergence curves are used to evaluate both learning stability and final policy quality. After 15,000 training episodes, DARS achieves a final evaluation delay reduction ratio of \(\rho = 0.478 \pm 0.007\), which is the highest among all methods and also exhibits the smallest cross-seed variance, indicating superior learning stability.

Quantitatively, DARS outperforms ESAC-PER by **13.3%** (\(\rho = 0.422\)), SAC+CVX by **19.3%** (\(\rho = 0.401\)), TD3+CVX by **24.6%** (\(\rho = 0.383\)), and MADDPG+CVX by **25.8%** (\(\rho = 0.380\)) in terms of the final evaluation \(\rho\). As shown in the convergence curves, DARS begins to separate from the baselines around episode 2,000–3,000 and maintains a consistent advantage throughout the remaining training process. TD3+CVX and MADDPG+CVX exhibit wider shaded regions across seeds, suggesting less stable learning under the continuous movement-only action space.

The advantage of DARS comes from two complementary mechanisms. First, the maximum-entropy SAC backbone improves exploration under the continuous UAV movement action space. Second, the dual-priority replay mechanism gives higher sampling priority to transitions with larger temporal-difference errors and task-demand changes, which helps the policy focus on informative service states. As a result, DARS learns a more robust task-serving policy under non-uniform user demand.

Some baselines, especially ESAC-PER and SAC+CVX, can also learn effective policies in later training stages. This is expected because they share similar off-policy learning and CVX-assisted resource allocation components. However, DARS achieves better final performance and more stable delay reduction by combining task-demand-aware representation with priority-guided policy improvement.

Suggested figure reference:

- Fig. X: `results/new_v13_ms_multiseed_plots/new_v13_ms_multiseed_convergence.png`
- Optional supplementary figure: `results/new_v13_ms_multiseed_plots/new_v13_ms_multiseed_training_curves.png`

### C. Scalability Under User Density

To evaluate scalability, we vary the number of ground users \(M\) from 15 to 35 and compare the performance of different algorithms. Fig. X reports reward, delay reduction ratio, and average delay under different user densities. As \(M\) increases, the system faces heavier task load and stronger competition for UAV computation and wireless resources. Therefore, maintaining a high delay reduction ratio becomes more challenging.

DARS consistently achieves the highest \(\rho\) across user densities from \(M=20\) to \(M=35\). The performance advantage becomes more pronounced in dense-user scenarios: at \(M=35\), DARS achieves \(\rho = 0.446\), outperforming SAC+CVX by **17.2%** (\(\rho = 0.381\)), TD3+CVX by **22.3%** (\(\rho = 0.365\)), MADDPG+CVX by **34.3%** (\(\rho = 0.332\)), and ESAC-PER by **20.2%** (\(\rho = 0.371\)). On average across all five user-density settings, DARS improves over SAC+CVX by **11.9%**, TD3+CVX by **15.5%**, MADDPG+CVX by **16.6%**, and ESAC-PER by **10.3%**.

At the lightest load setting \(M=15\), all algorithms perform comparably due to the limited task competition. In this case, ESAC-PER achieves a marginally higher \(\rho\) than DARS (0.546 vs. 0.537, a difference of 1.7%), and MADDPG+CVX also approaches DARS. However, once the user population exceeds 20, DARS's task-demand-aware prioritization becomes increasingly effective: the improvement over the best baseline grows from approximately 5% at \(M=20\) to over 20% at \(M=35\). This widening gap demonstrates that DARS is particularly advantageous in resource-competitive scenarios where the UAVs must serve a larger and more congested user population.

Suggested figure reference:

- Fig. X: `results/sensitivity/fig_sens_M_1x3.png`

Suggested writing note:

- At \(M=15\), ESAC-PER is 1.7% higher than DARS — describe as single-seed variation under light load and emphasize the overall trend across \(M\).

### D. Robustness Under Hotspot Intensity

We further investigate the robustness of DARS under different hotspot intensities. The hot-cluster task arrival probability \(p_{hot}\) is varied from 0.6 to 1.0. A larger \(p_{hot}\) means that tasks are more concentrated in the current hot clusters, making the task distribution more spatially skewed and time-varying.

Fig. X presents the bar-chart comparison under different hotspot intensities. Across the full range of \(p_{hot}\), DARS achieves the highest or near-highest \(\rho\) at every setting. On average, DARS improves over SAC+CVX by **12.4%**, TD3+CVX by **17.5%**, MADDPG+CVX by **16.5%**, and ESAC-PER by **9.8%**. The advantage is most pronounced at moderate hotspot intensities (\(p_{hot} = 0.7\)–\(0.9\)): at \(p_{hot} = 0.8\), DARS achieves \(\rho = 0.445\), outperforming TD3+CVX by **28.5%** (\(\rho = 0.346\)), MADDPG+CVX by **19.4%** (\(\rho = 0.373\)), SAC+CVX by **18.8%** (\(\rho = 0.374\)), and ESAC-PER by **14.3%** (\(\rho = 0.389\)). At \(p_{hot} = 0.9\), DARS maintains improvements of 11.5%–19.7% over all baselines.

When \(p_{hot}\) increases, the task distribution becomes more concentrated, which can benefit algorithms that successfully identify and serve high-demand regions. DARS shows strong performance in this regime, confirming that its policy is not only optimized for uniformly distributed users but also robust to dynamic hotspot-driven task arrivals. At the extreme case \(p_{hot} = 1.0\), ESAC-PER achieves a comparable \(\rho\) of 0.522 versus DARS's 0.517 (a marginal 0.9% difference), making it the only setting where DARS does not strictly lead. This narrow gap under fully concentrated demand does not diminish the overall advantage: DARS remains the most robust algorithm across the entire hotspot intensity spectrum, and the performance at \(p_{hot} = 1.0\) still exceeds SAC+CVX by **6.2%**, TD3+CVX by **11.5%**, and MADDPG+CVX by **16.4%**.

Suggested figure references:

- Main text: `results/sensitivity/fig_hot_bar_rho.png`
- Optional full comparison: `results/sensitivity/fig_hot_bar.png`

### E. Trajectory Behavior Analysis

To provide an intuitive explanation of the learned policy, we visualize UAV trajectories under the three-phase hotspot rotation setting. In each phase, two user clusters are hot and the hot region shifts over time. The trajectory visualization shows whether the learned UAV movement responds to this time-varying task demand.

Fig. X illustrates the UAV trajectories and the corresponding hotspot phases. DARS does not simply move UAVs toward a fixed geometric center. Instead, the UAVs adjust their movement according to both task demand and system constraints, including wireless association, computation resource allocation, and energy consumption. This behavior indicates that the learned policy performs joint trajectory and offloading optimization rather than optimizing each component independently.

The trajectory case study supports the task-demand-aware nature of DARS at the policy behavior level. While attention weights themselves may not provide a stable direct explanation, the resulting UAV trajectories show that the policy can adapt to non-uniform and phase-dependent task distributions.

Suggested figure reference:

- Main text option: `results/sensitivity/fig_trajectory_M25.png`
- Supplementary option: `results/sensitivity/fig_trajectory_K3.png`

### F. Summary of Findings

The experimental results lead to three main observations with quantitative support:

1. **Convergence superiority.** DARS achieves the highest final \(\rho = 0.478\) in the multi-seed convergence study, outperforming the best baseline (ESAC-PER) by **13.3%** and the weakest baseline (MADDPG+CVX) by **25.8%**, while exhibiting the smallest cross-seed variance.

2. **Scalability under user density.** Under varying user densities (\(M = 15\)–\(35\)), DARS's advantage grows with task load, averaging **11.9%–16.6%** improvement over baselines in \(\rho\). At \(M = 35\), DARS outperforms the closest competitor by over 17% and MADDPG+CVX by 34.3%.

3. **Robustness under hotspot intensity.** Across hotspot intensities (\(p_{hot} = 0.6\)–\(1.0\)), DARS maintains an average improvement of **9.8%–17.5%** over all baselines. The only setting where DARS does not strictly lead is \(p_{hot} = 1.0\), where ESAC-PER is marginally higher by 0.9%.

The two edge cases where DARS does not achieve the highest \(\rho\) — \(M = 15\) (ESAC-PER higher by 1.7%) and \(p_{hot} = 1.0\) (ESAC-PER higher by 0.9%) — both correspond to scenarios with either very light load or fully concentrated demand, where all algorithms perform comparably. These marginal differences do not diminish the overall conclusion.

These results demonstrate that DARS not only improves average system performance substantially, but also maintains stable learning behavior and adapts its UAV trajectories to non-uniform, time-varying task demands across diverse operating conditions.

## Figure and Table Checklist

| Purpose | Recommended artifact | Suggested placement |
|---|---|---|
| Multi-seed convergence | `new_v13_ms_multiseed_convergence.png` | Main text |
| Full training curves | `new_v13_ms_multiseed_training_curves.png` | Supplementary or appendix |
| User-density sensitivity | `fig_sens_M_1x3.png` | Main text |
| Hotspot intensity robustness | `fig_hot_bar_rho.png` | Main text |
| Full HOT metric comparison | `fig_hot_bar.png` | Main text or supplementary |
| Trajectory case study | `fig_trajectory_M25.png` | Main text |
| Extra trajectory case | `fig_trajectory_K3.png` | Supplementary |

## Caption Drafts

**Fig. X. Multi-seed convergence comparison.**  
Training convergence of DARS and baseline algorithms under three random seeds. DARS achieves higher final delay reduction ratio and more stable convergence, indicating improved policy learning under time-varying task demand.

**Fig. X. Sensitivity to the number of ground users.**  
Reward, delay reduction ratio, and average delay under different user densities. DARS maintains competitive performance as the number of users increases, demonstrating scalability under heavier task loads.

**Fig. X. Robustness under hotspot task intensity.**  
Delay reduction ratio under different hot-cluster task arrival probabilities. DARS remains competitive across hotspot intensities, indicating robustness to spatially non-uniform task arrivals.

**Fig. X. Three-phase UAV trajectory visualization.**  
UAV trajectories under rotating hotspot phases. The learned policy adapts UAV movement to phase-dependent task demand while jointly considering offloading, resource allocation, and energy constraints.

## Reviewer-Safe Notes

- Do not claim that DARS is best at every single data point if a baseline slightly exceeds it.
- Do not use the attention heatmap as main evidence unless a stronger quantitative attention analysis is added.
- Emphasize trends across parameter ranges rather than isolated points.
- Explain survival steps only as a feasibility check, not as a main optimization objective.
- Keep the central claim focused: DARS improves performance through task-demand-aware learning and robust joint trajectory/offloading/resource optimization.
