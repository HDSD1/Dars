"""Read-only, separate-process exogenous trajectory and charging probe."""
import sys
from pathlib import Path
import json
import hashlib
sys.path.insert(0,sys.argv[1])
import numpy as np
import config
from UAV_env import UAVEnv
from experiment_config import EnvironmentConfig
cfg=EnvironmentConfig(**json.loads((Path(sys.argv[1])/'experiments/train_diverse.json').read_text(encoding='utf-8')))
records=[]
for seed in (42,931000,931099):
    env=UAVEnv(use_cvx_alloc=True,experiment_config=cfg,seed=seed)
    state=env.reset(seed=seed); external=[]
    initial_uav=env.loc_uav.tolist()
    centers=env.cluster_centers.tolist()
    for slot in range(50):
        h=hashlib.sha256()
        for a in (env.loc_ue,env.task_list,env.active_mask):h.update(a.tobytes())
        external.append(h.hexdigest());env.current_step+=1;env._update_ue()
    records.append(dict(seed=seed,external=external,initial_uav=initial_uav,centers=centers))
out=dict(K=config.K,M=config.M,state_dim=len(state),action_dim=config.ACTION_DIM,wct_positions=env.wct_positions.tolist(),
         F_UAV=config.F_UAV,BANDWIDTH=config.BANDWIDTH,hot_prob=config.HOTSPOT_HOT_PROB,cold_prob=config.HOTSPOT_COLD_PROB,
         center_charging_w=env._compute_charging(np.array([[250.,250.],[250.,250.]])).tolist(),records=records)
Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
