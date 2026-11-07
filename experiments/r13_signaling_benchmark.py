"""R1.3: lossless application serialization on frozen, archived DARS rollouts.

No training, sockets, or manuscript edits. Link-time values are computed later
from IP byte budgets and explicitly assumed effective IP-layer service rates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hotspot_generalization_v2 as hg
import numpy as np
import torch
from DARS import SACActor
from state_normalization import StateNormalization
import r13_wire as wire

BANK = ROOT / 'results/reviewer_hotspot_generalization_v2_iofix_20261031/bank'
PREVIOUS = BANK.parent
MODEL_PATHS = {
    42: ROOT / 'results/original_main_seed42_gpu_retry_20261029_234000/per_sac_cvx.pth',
    43: ROOT / ('results/formal_all_five_seeds43_44_restart_20261031/source_snapshot/results/'
                'formal_all_five_seeds43_44_session2_20261031_seed43/per_sac_cvx.pth'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=hg.jsonable, allow_nan=False) + '\n', encoding='utf-8')


def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=float(x.mean()), p95=float(np.percentile(x, 95)),
                maximum=float(x.max()), minimum=float(x.min())) if len(x) else None


def actor_digest(actor):
    h = hashlib.sha256()
    for key, tensor in sorted(actor.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def select(actor, normalizer, state):
    x = torch.as_tensor(normalizer.normalize(state), dtype=torch.float32).unsqueeze(0)
    return actor.deterministic(x).numpy()[0]


def exact(a, b, label):
    if a is None or b is None:
        assert a is b, label
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b)), label


def decision_equal(a, b):
    for key in ('action_executed', 'failure_phase', 'association', 'offload_ratios',
                'cpu_allocation', 'bandwidth_allocation', 'executed_action',
                'flight_ok', 'service_ok', 'energy_ok'):
        exact(a[key], b[key], key)


def golden_rows(model_seed, case, selected_seeds):
    path = PREVIOUS / f'per_sac_cvx_seed{model_seed}' / f'{case}.episodes.jsonl'
    rows = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['seed'] in selected_seeds:
                rows[row['seed']] = row
            if len(rows) == len(selected_seeds):
                break
    assert set(rows) == set(selected_seeds), f'Missing reference episodes: {path}'
    return rows


def replay_episode(actor, actor_copies, manifest, bank, seed, case, model_seed, golden, sample_dir):
    envs = [hg.ReplayEnv(bank, experiment_config=hg.base_config(), seed=seed,
                         **manifest['env_kwargs']) for _ in range(3)]
    for env in envs:
        env.reset(seed=seed)
    env = envs[0]
    normalizer = StateNormalization(env.battery_capacity)
    rows, hashes = [], hashlib.sha256()
    for t in range(env.slot_num):
        pos, tasks, active = env.loc_ue.copy(), env.task_list.copy(), env.active_mask.copy()
        q, battery = env.loc_uav.copy(), env.e_battery_uav.copy()
        exact(pos, bank['ue'][t], 'bank positions')
        exact(tasks, bank['tasks'][t], 'bank tasks')
        dist = np.linalg.norm(pos[:, None, :] - q[None, :, :], axis=2)
        # Preconfigured reporting groups need no uncounted position beacons or discovery.
        # Reporting groups are independent of the dynamically optimized service association.
        owners = np.arange(hg.config.M) % 2
        report_dist = dist[np.arange(hg.config.M), owners]
        user_packets, local_records = [], [[], []]
        up_ip_by_uav = [0, 0]
        for i in range(hg.config.M):
            record = dict(id=i, position=pos[i].tolist(), task_bits=float(tasks[i]), active=bool(active[i]))
            payload = wire.encode_user_report(record, episode=seed, slot=t, sender=i)
            decoded_packet = wire.decode_user_report(payload)
            header, decoded = decoded_packet['header'], decoded_packet['record']
            assert header['episode'] == seed and header['slot'] == t
            exact(decoded['position'], pos[i], 'UE position wire roundtrip')
            exact(decoded['task_bits'], tasks[i], 'UE task wire roundtrip')
            exact(decoded['active'], active[i], 'UE activity wire roundtrip')
            local_records[int(owners[i])].append(decoded)
            user_packets.append(payload)
            up_ip_by_uav[int(owners[i])] += len(payload) + 28
        peer_packets, peer_decoded = [], []
        for k in range(2):
            own = dict(position=q[k].tolist(), battery_j=float(battery[k]))
            payload = wire.encode_peer(own, local_records[k], episode=seed, slot=t, sender=k)
            decoded_packet = wire.decode_peer(payload)
            header, own_decoded, records = (decoded_packet['header'], decoded_packet['own_state'], decoded_packet['records'])
            assert header['episode'] == seed and header['slot'] == t and header['sender'] == k
            peer_packets.append(payload)
            peer_decoded.append((own_decoded, records))
        # Each replica sees its own local records/state plus only its received peer packet.
        raw = env._get_obs()
        for k, replica in enumerate(envs[1:]):
            other = 1 - k
            own_remote, remote_records = peer_decoded[other]
            collected = local_records[k] + remote_records
            assert sorted(x['id'] for x in collected) == list(range(hg.config.M))
            for record in collected:
                i = record['id']
                replica.loc_ue[i] = record['position']
                replica.task_list[i] = record['task_bits']
                replica.active_mask[i] = record['active']
            replica.loc_uav[k] = q[k]
            replica.e_battery_uav[k] = battery[k]
            replica.loc_uav[other] = own_remote['position']
            replica.e_battery_uav[other] = own_remote['battery_j']
            replica.current_step = t
            exact(replica._get_obs(), raw, 'reconstructed raw global state')
        action = select(actor, normalizer, raw)
        _, reward, done, info = env.step(action)
        for replica, copy_actor in zip(envs[1:], actor_copies):
            replica_action = select(copy_actor, normalizer, replica._get_obs())
            exact(replica_action, action, 'replica Actor output')
            _, other_reward, other_done, other_info = replica.step(replica_action)
            decision_equal(info, other_info)
            exact(reward, other_reward, 'replica reward')
            exact(done, other_done, 'replica termination')
        reference = golden['trace'][t]
        exact(info['action_executed'], reference['executed'], 'archived execution status')
        exact(q, reference['uav_before'], 'archived preflight position')
        exact(env.loc_uav, reference['uav_after'], 'archived postflight position')
        exact(env.e_battery_uav, reference['battery'], 'archived battery')
        if info['action_executed']:
            exact(float(np.sum(info['rho'])), reference['rho_sum'], 'archived SLRR sum')

        serving = np.full(hg.config.M, -1, dtype=int)
        starts = np.zeros(hg.config.M)
        x = np.zeros(hg.config.M)
        bw = np.zeros(hg.config.M)
        if info['action_executed']:
            valid = active & (info['association'] >= 0) & (info['offload_ratios'] > hg.OFFLOAD_EPS)
            serving[valid] = info['association'][valid]
            x[valid] = info['offload_ratios'][valid]
            bw[valid] = info['bandwidth_allocation'][valid]
            for k in range(2):
                offset = 0.0
                for i in np.flatnonzero(serving == k):
                    starts[i] = offset
                    offset += bw[i]
                assert offset <= hg.config.BANDWIDTH + 1e-6
        down_packets, down_ip_by_uav = [], [0, 0]
        for i in range(hg.config.M):
            grant = dict(id=i, serving_uav=int(serving[i]), offload=float(x[i]),
                         bandwidth_hz=float(bw[i]), bandstart_hz=float(starts[i]),
                         status=0 if info['action_executed'] else 1)
            payload = wire.encode_grant(grant, episode=seed, slot=t, sender=int(owners[i]))
            decoded_packet = wire.decode_grant(payload)
            header, restored = decoded_packet['header'], decoded_packet['grant']
            assert header['episode'] == seed and header['slot'] == t
            assert restored == grant
            down_packets.append(payload)
            down_ip_by_uav[int(owners[i])] += len(payload) + 28
        packets = user_packets + peer_packets + down_packets
        for payload in packets:
            hashes.update(len(payload).to_bytes(4, 'big'))
            hashes.update(payload)
        up, peer, down = [sum(map(len, p)) for p in (user_packets, peer_packets, down_packets)]
        assert (up, peer, down) == (1290, 890, 1350), 'schema length cross-check'
        assert len(packets) == 62
        max_ip = max(map(len, packets)) + 28
        assert max_ip <= 1500, 'Reference IPv4 MTU would require fragmentation'
        row = dict(training_seed=model_seed, case=case, test_seed=seed, slot=t,
                   executed=int(info['action_executed']), failure_phase=info['failure_phase'] or '',
                   active_users=int(active.sum()), min_battery_j=float(battery.min()),
                   app_up_bytes=up, app_peer_bytes=peer, app_down_bytes=down,
                   app_total_bytes=up+peer+down, packet_count=len(packets),
                   ip_up_bytes=up+28*len(user_packets), ip_peer_bytes=peer+28*len(peer_packets),
                   ip_down_bytes=down+28*len(down_packets),
                   ip_total_bytes=up+peer+down+28*len(packets), max_ip_datagram_bytes=max_ip,
                   reporting_uav0_users=int((owners == 0).sum()),
                   reporting_uav1_users=int((owners == 1).sum()),
                   ue_reports_outside_service_radius=int((report_dist > hg.config.R_COV).sum()),
                   max_reporting_horizontal_m=float(report_dist.max()),
                   mean_reporting_horizontal_m=float(report_dist.mean()),
                   max_reporting_slant_m=float(np.sqrt(report_dist**2+hg.config.H**2).max()),
                   inter_uav_distance_m=float(np.linalg.norm(q[0]-q[1])),
                   final_decision_checks=2, raw_state_checks=2,
                   uav0_tx_ip_bytes=len(peer_packets[0])+28+down_ip_by_uav[0],
                   uav1_tx_ip_bytes=len(peer_packets[1])+28+down_ip_by_uav[1],
                   uav0_rx_ip_bytes=up_ip_by_uav[0]+len(peer_packets[1])+28,
                   uav1_rx_ip_bytes=up_ip_by_uav[1]+len(peer_packets[0])+28)
        rows.append(row)
        if sample_dir is not None and t == 0:
            sample_dir.mkdir(parents=True, exist_ok=False)
            for name, payload in [('ue0.bin',user_packets[0]), ('peer0.bin',peer_packets[0]),
                                  ('peer1.bin',peer_packets[1]), ('grant0.bin',down_packets[0])]:
                (sample_dir/name).write_bytes(payload)
            dump(sample_dir/'decoded.json', dict(user=wire.decode_user_report(user_packets[0]),
                 peer0=wire.decode_peer(peer_packets[0]), peer1=wire.decode_peer(peer_packets[1]),
                 grant=wire.decode_grant(down_packets[0])))
        if done:
            break
    assert len(rows) == len(golden['trace']), 'Reference trajectory length'
    return rows, dict(training_seed=model_seed, case=case, test_seed=seed,
                     attempts=len(rows), executed_steps=sum(r['executed'] for r in rows),
                     full_horizon=bool(len(rows)==50 and all(r['executed'] for r in rows)),
                     application_transcript_sha256=hashes.hexdigest())


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available(), 'Do not use the training GPU'
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        assert kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    wire.self_test()
    started = time.time()
    index = json.loads((BANK/'index.json').read_text(encoding='utf-8'))
    selected = index['seeds'][:args.episodes_per_case]
    assert len(selected) == args.episodes_per_case
    cases = args.cases or list(hg.CASES)
    assert all(case in hg.CASES for case in cases)
    protected = [ROOT.parent/'paper/IEEE_Journal_Paper_Template_2/bare_jrnl.tex',
                 ROOT.parent/'paper/IEEE_Journal_Paper_Template_2/references.bib']
    protected_hashes = {str(p):sha(p) for p in protected}
    proto = dict(
        version=1, title='R1.3 reference application serialization; no network sockets',
        slot_seconds=float(hg.config.TAU_SLOT), num_users=int(hg.config.M), num_uavs=2,
        topology='Two UAVs; per-UE unicast reports/grants; two aggregated peer messages per slot',
        reporting_assignment='Preconfigured reporting UAV = user ID modulo 2; independent of dynamic service association; no per-slot position beacons needed',
        continuous_encoding='IEEE 754 binary64 network byte order; preserve raw environment precision',
        header_format='!4sBBIHHH', header_bytes=16,
        header_fields=['magic[4]','version:uint8','type:uint8','episode:uint32','slot:uint16','sender:uint16','count:uint16'],
        user_record_format='!HdddB', user_record_bytes=27,
        user_record_fields=['user_id:uint16','px:float64','py:float64','task_bits:float64','active:uint8'],
        peer_own_format='!ddd', peer_own_bytes=24,
        peer_own_fields=['qx:float64','qy:float64','battery_j:float64'],
        grant_format='!HhdddB', grant_record_bytes=29,
        grant_fields=['user_id:uint16','serving_uav:int16(-1=local)','offload:float64',
                      'bandwidth_hz:float64','bandstart_hz:float64','status:uint8(0=execute,1=abort)'],
        spectrum='Within each UAV, allocated bands concatenated by UE ID; starts relative to its band origin',
        rejected_decisions='Keep attempted slot; send explicit abort status to all UEs; no executable local fallback asserted',
        static_shared=['model weights','normalizer','UAV/UE IDs and fixed reporting-group map','WCT locations and charging limits',
                       'channel model','task cycles per bit','CPU/bandwidth/power constants','horizon and clock origin'],
        training_traffic='Zero per-slot gradients/replay/model updates by frozen deployment design',
        control_coverage_assumption='All users can reach their reporting UAV; the 150m task-service radius does not establish this',
        ip_budget=dict(transport='UDP over IPv4; no options/extensions/tunneling',udp_header_bytes=8,
                       ipv4_header_bytes=20,assumed_mtu_bytes=1500,unit='IP bytes, NOT PHY air bytes'),
        excluded=['MAC/PHY framing','association/discovery','link establishment','ACK traffic',
                  'encryption/authentication','retransmissions as separate traffic','channel access/queueing delays',
                  'clock synchronization protocol','radio and inference energy'],
        sensitivity='R_eff is delivered IP-layer service rate; no separate PER multiplier. Serial stage service budget; not end-to-end radio timing.',
        cases=cases, test_seeds=selected, training_seeds=args.models,
        selection='First specified seeds of the pre-existing bank, fixed before run; no outcome selection',
        interpretation='Fixed-length records intentionally give constant bytes; repeated trials validate content and decisions, not uncertainty of byte count',
        sources=[dict(url='https://www.rfc-editor.org/info/rfc768/',supports='UDP base header 8 bytes'),
                 dict(url='https://www.rfc-editor.org/info/rfc791/',supports='IPv4 header without options 20 bytes'),
                 dict(url='https://docs.python.org/3/library/struct.html',supports='portable binary64 packing with network byte order'),
                 dict(url='https://doi.org/10.1109/TSP.2017.2728501',supports='signaling-dimension analysis only, not our field layout')])
    dump(output/'protocol.json', proto)
    provenance = dict(status='running', pid=os.getpid(), started_unix=started, python=sys.version,
                      platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                      device='cpu', threads=1, bank=str(BANK), bank_index_sha256=sha(BANK/'index.json'),
                      evaluator_sha256=sha(hg.__file__), script_sha256=sha(__file__),
                      wire_sha256=sha(wire.__file__), source=str(hg.SOURCE),
                      protected_files_before=protected_hashes, models=[])
    dump(output/'manifest.json', provenance)
    all_rows, episode_rows, model_provenance = [], [], []
    csv_file = (output/'slots.csv').open('x',newline='',encoding='utf-8')
    ep_file = (output/'episodes.jsonl').open('x',encoding='utf-8')
    writer = None
    try:
        for model_seed in args.models:
            checkpoint = MODEL_PATHS[model_seed]
            manifest_path = checkpoint.with_suffix('.manifest.json')
            manifest = hg.validate_manifest(manifest_path, selected)
            before_hash = sha(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            actors = []
            for _ in range(3):
                actor = SACActor(hg.config.STATE_DIM, hg.config.ACTION_DIM).cpu()
                actor.load_state_dict(saved['actor'])
                actor.eval()
                actor.requires_grad_(False)
                actors.append(actor)
            del saved
            before_actor = actor_digest(actors[0])
            with torch.inference_mode():
                for case in cases:
                    reference_rows = golden_rows(model_seed, case, selected)
                    for i, seed in enumerate(selected):
                        entry = index['entries'][case][i]
                        assert entry['seed']==seed and entry['case']==case
                        path = BANK/entry['file']
                        assert sha(path)==entry['sha256']
                        with np.load(path) as z:
                            bank = {name:z[name] for name in z.files}
                        sample_dir = output/'sample_packets'/f'seed{model_seed}_{case}' if i==0 else None
                        rows, episode = replay_episode(actors[0], actors[1:], manifest, bank,
                            seed, case, model_seed, reference_rows[seed], sample_dir)
                        if writer is None:
                            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
                            writer.writeheader()
                        writer.writerows(rows)
                        csv_file.flush()
                        ep_file.write(json.dumps(episode)+'\n')
                        ep_file.flush()
                        all_rows.extend(rows)
                        episode_rows.append(episode)
                        dump(output/'progress.json', dict(status='running', training_seed=model_seed,
                             case=case, completed_episodes=len(episode_rows), total_episodes=len(args.models)*len(cases)*len(selected),
                             attempted_states=len(all_rows), elapsed_seconds=time.time()-started))
                        if time.time()-started > 1200:
                            raise TimeoutError('This standalone benchmark exceeded its 20 minute limit')
                    print(f'seed{model_seed} {case}: {len(selected)} episodes; {len(all_rows)} total states',flush=True)
            assert sha(checkpoint)==before_hash
            assert all(actor_digest(a)==before_actor for a in actors)
            model_provenance.append(dict(training_seed=model_seed, checkpoint=str(checkpoint),
                 checkpoint_sha256_before=before_hash, checkpoint_sha256_after=sha(checkpoint),
                 actor_digest_before=before_actor, actor_digest_after=actor_digest(actors[0]),
                 parameters=sum(p.numel() for p in actors[0].parameters())))
    finally:
        csv_file.close()
        ep_file.close()
    assert {str(p):sha(p) for p in protected} == protected_hashes
    numeric = ['app_up_bytes','app_peer_bytes','app_down_bytes','app_total_bytes','ip_up_bytes',
               'ip_peer_bytes','ip_down_bytes','ip_total_bytes','packet_count','max_ip_datagram_bytes',
               'active_users','min_battery_j','max_reporting_horizontal_m','mean_reporting_horizontal_m',
               'max_reporting_slant_m','inter_uav_distance_m','uav0_tx_ip_bytes','uav1_tx_ip_bytes',
               'uav0_rx_ip_bytes','uav1_rx_ip_bytes']
    group_summaries = []
    for model_seed in args.models:
        for case in cases:
            rows = [r for r in all_rows if r['training_seed']==model_seed and r['case']==case]
            group_summaries.append(dict(training_seed=model_seed,case=case,attempts=len(rows),
                abort_attempts=sum(1-r['executed'] for r in rows),
                outside_service_radius_report_fraction=sum(r['ue_reports_outside_service_radius'] for r in rows)/(len(rows)*hg.config.M),
                metrics={name:stats([r[name] for r in rows]) for name in numeric}))
    summary = dict(status='completed', episodes=len(episode_rows), attempted_states=len(all_rows),
                   executed_states=sum(r['executed'] for r in all_rows),
                   aborted_states=sum(1-r['executed'] for r in all_rows),
                   independent_training_seeds=args.models, scenarios=cases, test_seeds=selected,
                   state_reconstruction_checks=2*len(all_rows), final_decision_checks=2*len(all_rows),
                   archived_trace_checks=len(all_rows), mismatches=0, total_packets=62*len(all_rows),
                   metrics={name:stats([r[name] for r in all_rows]) for name in numeric},
                   outside_service_radius_report_fraction=sum(r['ue_reports_outside_service_radius'] for r in all_rows)/(len(all_rows)*hg.config.M),
                   groups=group_summaries, wall_seconds=time.time()-started,
                   scope='Application bytes measured by actual serialization; UDP/IPv4 headers counted from specifications; no radio or board timing/energy measurement',
                   statistical_scope='Two frozen trained models; neither a three-training-seed result nor a new policy-performance comparison')
    dump(output/'summary.json', summary)
    provenance.update(status='completed',models=model_provenance, ended_unix=time.time(),
                      protected_files_after={str(p):sha(p) for p in protected})
    dump(output/'manifest.json', provenance)
    dump(output/'progress.json',dict(status='completed',episodes=len(episode_rows),attempted_states=len(all_rows)))
    print(json.dumps({k:summary[k] for k in ('status','episodes','attempted_states','aborted_states','mismatches','wall_seconds')}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--episodes-per-case',type=int,default=20)
    p.add_argument('--models',type=int,nargs='+',default=[42,43],choices=[42,43])
    p.add_argument('--cases',nargs='+',choices=list(hg.CASES))
    run(p.parse_args())
