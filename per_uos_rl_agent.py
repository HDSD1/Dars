"""
PER-UOS-RL baseline adapted to the current UAV-MEC-WCT environment.

Reference:
  H. Shi, Y. Tian, H. Li, J. Huang, L. Shi, and Y. Zhou,
  "Task offloading and trajectory scheduling for UAV-enabled MEC networks:
  An MADRL algorithm with prioritized experience replay,"
  Ad Hoc Networks, vol. 154, 2024, Art. no. 103371.
  DOI: 10.1016/j.adhoc.2023.103371.

The original PER-UOS-RL is a multi-agent UAV-assisted offloading strategy
trained with prioritized experience replay and a MATD3-style deterministic
actor-critic framework. This file provides a local, fair-comparison adaptation:
it uses the existing PER + TD3-style actor-critic implementation, without
task-demand attention, and lets the shared environment handle association and
resource allocation consistently with other baselines.

This is not a verbatim reproduction of the authors' released code; it is an
adapted baseline for the state/action/reward definitions in this repository.
"""

from per_ddpg_agent import PERDDPGAgent


class PERUOSRLAgent(PERDDPGAgent):
    """Adapted PER-UOS-RL baseline.

    In this codebase, PERDDPGAgent already implements the closest local
    counterpart of the paper's core learning machinery:
      - deterministic actor for continuous UAV trajectory/offloading actions,
      - twin critics and delayed actor update (MATD3/TD3-style stabilization),
      - prioritized experience replay with importance-sampling correction.

    The subclass gives the baseline an explicit paper-facing identity without
    changing the implementation used for fair comparison.
    """

    display_name = "PER-UOS-RL"
