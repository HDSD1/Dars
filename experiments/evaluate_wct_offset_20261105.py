"""Paired frozen-policy comparison: original versus prespecified offset WCTs."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ALGOS = ['per_sac_cvx', 'esac_per_cvx', 'sac_cvx', 'td3_cvx', 'maddpg_cvx']
NAMES = dict(zip(ALGOS, ['DARS', 'ESAC-PER', 'SAC', 'TD3', 'MADDPG']))
OFFSET = [[250., 100.], [400., 250.], [250., 400.], [100., 250.]]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, default=serial, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def prepare(out):
    previous = ROOT/'results/reviewer_users_m20_m25_m30_seed42_5000_20261102'
    source = previous/'m30/source_snapshot'
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'source_snapshot'
    frozen.mkdir()
    for path in source.glob('*.py'):
        shutil.copy2(path, frozen/path.name)
    shutil.copy2(__file__, out/Path(__file__).name)
    models = {}
    for algo in ALGOS:
        evaluation = read(previous/f'evaluation/m30/{algo}/manifest.json')
        checkpoint = Path(evaluation['checkpoint'])
        manifest = checkpoint.with_suffix('.manifest.json')
        meta = read(manifest)
        assert meta['status']=='completed' and meta['training_seed']==42
        assert meta['training']['episodes']==15000
        assert sha(checkpoint)==evaluation['checkpoint_sha256']
        # The environment and policy definitions are the exact trained versions.
        for name, digest in meta['source_sha256'].items():
            if name not in ('evaluation.py', 'train_unified.py'):
                assert sha(frozen/name)==digest, name
        models[algo] = {'checkpoint':str(checkpoint), 'sha256':sha(checkpoint),
                        'manifest':str(manifest), 'environment':meta['environment']}
    protocol = {
        'created_at':now(), 'status':'prepared', 'models':models,
        'source_hashes':{p.name:sha(p) for p in frozen.glob('*.py')},
        'script_sha256':sha(out/Path(__file__).name),
        'layouts': {'colocated':[[125.,125.],[375.,125.],[125.,375.],[375.,375.]], 'offset':OFFSET},
        'cluster_centers':[[125.,375.],[375.,375.],[375.,125.],[125.,125.]],
        'K':2, 'M':30, 'initial_energy_j':40000, 'battery_capacity_j':40000,
        'horizon':50, 'test_seed_start':931000, 'test_episodes_per_layout_model':100,
        'total_evaluation_episodes':1000, 'training_seed':42,
        'selection':'One geometry selected before outcomes: four equal nonoverlapping charging disks remain inside the500m square; no cluster center is chargeable.',
        'change':'Only WCT horizontal positions. No changes to policies, rewards, tasks, mobility, charging power, capacities, safety or resource allocation.',
        'execution':'One CPU worker/thread; original and shifted layouts re-evaluated on the same device and same seeds. No training.',
        'inference':'Frozen co-located-trained policy transfer to an unseen layout, not retraining. All completed and early-terminated episodes retained.',
        'statistics':'Paired test-episode bootstrap conditional on training seed42. Full-horizon SLRR denominator50; actual energy until termination.',
    }
    write(out/'protocol.json', protocol)
    write(out/'batch_status.json', {'status':'prepared', 'completed_models':0, 'total_models':5, 'jobs':[]})
    return protocol


def worker(out, algo, episodes, smoke):
    protocol = read(out/'protocol.json')
    source = out/'source_snapshot'
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    import config as c
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from train_unified import ALGO_REGISTRY
    from state_normalization import StateNormalization
    from evaluation import EpisodeMetrics, frozen_policy, policy_input, configure_policy_environment, model_digest, runtime_info

    class LayoutEnv(UAVEnv):
        def __init__(self, layout, **kwargs):
            self.layout_positions = np.asarray(protocol['layouts'][layout], dtype=float)
            super().__init__(**kwargs)

        def _init_wct_positions(self):
            return self.layout_positions.copy()

    def external_hash(env):
        h=hashlib.sha256()
        for array in (env.loc_ue,env.task_list,env.active_mask):
            h.update(array.tobytes())
        return h.hexdigest()

    model=protocol['models'][algo]
    assert sha(model['checkpoint'])==model['sha256']
    agent, _, kwargs=ALGO_REGISTRY[algo]()
    agent.load(model['checkpoint'])
    cfg=EnvironmentConfig(**model['environment'])
    assert (c.K,c.M,c.H,c.H_WCT,c.R_COV,c.L_C)==(2,30,100.,70.,150.,100.)
    normalizer=StateNormalization(cfg.battery_capacity)
    digest=model_digest(agent)
    destination=out/('smoke' if smoke else 'evaluation')/algo
    destination.mkdir(parents=True,exist_ok=False)
    geometry={}
    for layout in ('colocated','offset'):
        env=LayoutEnv(layout,experiment_config=cfg,seed=931000,**kwargs)
        env.reset(seed=931000)
        towers=env.wct_positions
        radius=math.sqrt(c.L_C**2-(c.H-c.H_WCT)**2)
        distances=np.linalg.norm(env.cluster_centers[:,None,:]-towers[None,:,:],axis=2)
        separations=np.linalg.norm(towers[:,None,:]-towers[None,:,:],axis=2)
        separations+=np.eye(4)*1e9
        min_margin=float(np.minimum(towers,500-towers).min())
        power=np.r_[env._compute_charging(env.cluster_centers[:2]),env._compute_charging(env.cluster_centers[2:])]
        assert min_margin>radius and separations.min()>2*radius
        if layout=='offset':assert np.all(distances.min(axis=1)>radius) and np.all(power==0)
        else:assert np.all(distances.min(axis=1)==0) and np.allclose(power,200.)
        geometry[layout]={'wct_positions':towers,'cluster_centers':env.cluster_centers,
            'horizontal_charging_radius_m':radius,'nearest_wct_distance_from_cluster_m':distances.min(axis=1),
            'charging_power_at_cluster_centers_w':power,'minimum_station_separation_m':float(separations.min()),
            'minimum_boundary_margin_m':min_margin,'charging_area_m2':4*math.pi*radius**2,
            'centers_within_communication_radius_of_a_wct':(distances.min(axis=1)<=c.R_COV).tolist()}
    write(destination/'geometry.json',geometry)
    manifest={'status':'running','started_at':now(),'pid':os.getpid(),'algorithm':algo,
        'model':model,'runtime':runtime_info(),'environment':cfg.to_dict(),'episodes_per_layout':episodes,
        'scope':'Frozen paired geometry transfer; only WCT coordinates changed; CPU on both layouts.'}
    write(destination/'manifest.json',manifest)
    paired_external={}
    with frozen_policy(agent):
        for layout in ('colocated','offset'):
            folder=destination/layout;folder.mkdir()
            with (folder/'episodes.jsonl').open('x',encoding='utf-8') as output:
                for idx,seed in enumerate(range(931000,931000+episodes)):
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    env=LayoutEnv(layout,experiment_config=cfg,seed=seed,**kwargs)
                    state=env.reset(seed=seed);configure_policy_environment(agent,env)
                    shadow=LayoutEnv(layout,experiment_config=cfg,seed=seed,**kwargs);shadow.reset(seed=seed)
                    expected=[]
                    for _ in range(cfg.horizon):
                        expected.append(external_hash(shadow));shadow.current_step+=1;shadow._update_ue()
                    if layout=='colocated':paired_external[seed]=expected
                    else:assert paired_external[seed]==expected
                    metrics=EpisodeMetrics(env,seed,keep_trace=False)
                    slots=[];trajectory=[env.loc_uav.copy()]
                    charged=0;executed_uav_slots=0
                    for slot in range(cfg.horizon):
                        assert external_hash(env)==expected[slot]
                        active=env.active_mask.copy();tasks=env.task_list.copy();before=env.loc_uav.copy()
                        start=time.perf_counter();action=agent.select_action(policy_input(agent,normalizer,state),noise_std=0.)
                        elapsed=time.perf_counter()-start
                        state,reward,done,info=env.step(action);metrics.record(info,reward,elapsed)
                        executed=bool(info['action_executed'])
                        total=float(np.sum(info['rho'])) if executed else 0.
                        if executed:
                            physical=np.sum(1-info['delays'][active]/(tasks[active]*c.S_CYCLES/c.F_UE))
                            np.testing.assert_allclose(total,physical,rtol=1e-10,atol=1e-8)
                            np.testing.assert_allclose(info['energy_in'],env._compute_charging(env.loc_uav)*c.T_COM,rtol=1e-10,atol=1e-7)
                            charged+=int(np.sum(info['charge_power']>0));executed_uav_slots+=c.K
                            trajectory.append(env.loc_uav.copy())
                        slots.append({'slot':slot,'executed':executed,'total_slrr':total,
                            'positions_before':before,'positions_after':env.loc_uav.copy(),
                            'battery_after_service_j':info['battery_after_service'],
                            'energy_fly_j':info['energy_fly'],'energy_hover_j':info['energy_hover'],
                            'energy_comp_j':info['energy_comp'],'energy_in_j':info['energy_in'],
                            'hotspot_phase':info['hotspot_phase'],'termination_reason':info['termination_reason']})
                        if done:break
                    row=metrics.finish()
                    assert row['energy_violation_steps']==0
                    np.testing.assert_allclose(sum(s['total_slrr'] for s in slots),row['cumulative_total_slrr'],atol=1e-8)
                    row.update(algorithm=algo,layout=layout,training_seed=42,external_state_hashes=expected,
                        charged_uav_slots=charged,executed_uav_slots=executed_uav_slots,slots=slots,
                        uav_trajectory=trajectory,hotspot_schedule=env.hotspot_schedule[:cfg.horizon],
                        hotspot_segments=env.hotspot_segments)
                    output.write(json.dumps(row,default=serial,allow_nan=False)+'\n');output.flush()
                    if (idx+1)%10==0 or idx+1==episodes:
                        progress={'status':'running','algorithm':algo,'layout':layout,'episodes':idx+1,
                                  'target_per_layout':episodes,'updated_at':now()}
                        write(destination/'progress.json',progress)
                        print(json.dumps(progress),flush=True)
    assert digest==model_digest(agent) and sha(model['checkpoint'])==model['sha256']
    manifest.update(status='completed',ended_at=now(),model_unchanged=True,external_states_matched=True)
    write(destination/'manifest.json',manifest)
    write(destination/'progress.json',{'status':'completed','episodes_per_layout':episodes,'updated_at':now()})


def run(out):
    assert read(out/'batch_status.json')['status']=='prepared'
    protocol=read(out/'protocol.json')
    for name,digest in protocol['source_hashes'].items():assert sha(out/'source_snapshot'/name)==digest
    script=out/'evaluate_wct_offset_20261105.py'
    assert sha(script)==protocol['script_sha256']
    status={'status':'running','pid':os.getpid(),'started_at':now(),'completed_models':0,'total_models':5,'jobs':[]}
    write(out/'batch_status.json',status)
    for algo in ALGOS:
        command=[sys.executable,'-B','-u',str(script),'--worker','--algo',algo,'--output',str(out),'--episodes','100']
        item={'algorithm':algo,'status':'running','started_at':now()};status['jobs'].append(item)
        with (out/f'{algo}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=out,
                 stdin=subprocess.DEVNULL,creationflags=0x08004000 if os.name=='nt' else 0)
            item['pid']=process.pid;write(out/'batch_status.json',status)
            code=process.wait()
        item.update(exit_code=code,ended_at=now(),status='completed' if code==0 else 'failed')
        if code:
            status['status']='failed';write(out/'batch_status.json',status)
            raise RuntimeError(f'{algo} failed; inspect preserved log')
        status['completed_models']+=1;write(out/'batch_status.json',status)
        print(f"Completed {algo}: {status['completed_models']}/5",flush=True)
    status.update(status='completed',ended_at=now());write(out/'batch_status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'results/reviewer_wct_offset_seed42_20261105')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--algo',choices=ALGOS,default='per_sac_cvx')
    p.add_argument('--episodes',type=int,default=100)
    args=p.parse_args();out=args.output.resolve()
    if args.prepare_only:
        prepare(out);print('Prepared '+str(out))
    elif args.worker:worker(out,args.algo,args.episodes,args.smoke)
    else:run(out)
