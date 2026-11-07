"""Engineering regressions only; no experiment queue or formal training run."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from UAV_env import UAVEnv
from sequential_environment import SequentialUAVEnv
from conditional_dars import (ConditionalDARSAgent, PhysicalReturns, StageActor,
                              StageTwinCritic, service_observation, split_context,
                              eligible_users, OFF_GLOBAL_DIM, OFF_USER_DIM, CONTEXT_DIM)
from DARS import DEVICE
from config import K, M, GAMMA, T_COM, T_FLY, STATE_DIM, ACTION_DIM
from experiment_config import EnvironmentConfig
from state_normalization import StateNormalization
from evaluation import evaluate_policy, model_digest

torch.set_num_threads(2)


class TwoStageEnvironmentTest(unittest.TestCase):
    def compare(self, old, new, action):
        left = old.step(action); right = new.step(action)
        np.testing.assert_allclose(left[0], right[0], rtol=0, atol=1e-10)
        self.assertEqual(left[1:3], right[1:3])
        for key, value in left[3].items():
            if key == 'decision_seconds': continue
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(value, right[3][key], rtol=0, atol=1e-10, err_msg=key)
            else:
                self.assertEqual(value, right[3][key], key)
        return left[2]

    def test_same_full_rollouts_and_exogenous_states(self):
        for seed in (11, 42):
            cfg = EnvironmentConfig(hotspot_mode='diverse')
            old = UAVEnv(use_cvx_alloc=True, seed=seed, experiment_config=cfg)
            new = SequentialUAVEnv(use_cvx_alloc=True, seed=seed, experiment_config=cfg)
            old.reset(seed); new.reset(seed); rng = np.random.default_rng(seed)
            for _ in range(50):
                if self.compare(old, new, rng.uniform(-1, 1, ACTION_DIM)): break
            self.assertEqual(len(old.trajectory_log), len(new.trajectory_log))

    def test_energy_projection_and_infeasible_movement_match(self):
        for energy in (0., 1800.):
            old = UAVEnv(use_cvx_alloc=True, disable_wct=True, seed=9)
            new = SequentialUAVEnv(use_cvx_alloc=True, disable_wct=True, seed=9)
            for env in (old, new):
                env.reset(9); env.loc_uav[:] = [250, 250]; env.loc_ue[:] = [250, 250]
                env.active_mask[:] = 0; env.active_mask[:4] = 1
                env.e_battery_uav[:] = energy
            self.compare(old, new, np.zeros(ACTION_DIM))

    def test_prepare_is_pure_commit_once_and_invalidates_on_reset(self):
        env = SequentialUAVEnv(use_cvx_alloc=True, seed=42); env.reset(42)
        state = env._get_obs().copy(); step = env.current_step
        rng = [copy.deepcopy(getattr(env, n).bit_generator.state)
               for n in ('init_rng', 'mobility_rng', 'task_rng', 'hotspot_rng')]
        context = env.prepare_service(np.zeros(3*K)); self.assertTrue(context.feasible)
        observation = service_observation(env, context)
        np.testing.assert_array_equal(state, env._get_obs()); self.assertEqual(step, env.current_step)
        for n, before in zip(('init_rng', 'mobility_rng', 'task_rng', 'hotspot_rng'), rng):
            self.assertEqual(before, getattr(env, n).bit_generator.state)
        with self.assertRaises(ValueError): context.association[0] = 0
        env.finish_service(context, np.zeros(M))
        self.assertEqual(step+1, env.current_step)
        np.testing.assert_array_equal(observation, service_observation(env, context))
        with self.assertRaises(ValueError): env.finish_service(context, np.zeros(M))
        context = env.prepare_service(np.zeros(3*K)); env.reset(42)
        with self.assertRaises(ValueError): env.finish_service(context, np.zeros(M))

    def test_context_reports_actual_post_projection_association(self):
        env = SequentialUAVEnv(use_cvx_alloc=True, seed=42); env.reset(42)
        context = env.prepare_service(np.zeros(3*K))
        obs = service_observation(env, context)
        users = obs[OFF_GLOBAL_DIM:].reshape(M, OFF_USER_DIM)
        for i, k in enumerate(context.association):
            self.assertEqual(int(users[i, 4:4+K+1].argmax()), k if k >= 0 else K)
        _, _, _, info = env.finish_service(context, np.zeros(M))
        np.testing.assert_array_equal(context.association, info['association'])


class PhysicalReturnTest(unittest.TestCase):
    def test_physical_discount_and_rejected_movement_flush(self):
        class Replay:
            def __init__(self): self.rows = []
            def store_stage(self, *args): self.rows.append(args)
        replay = Replay(); queue = PhysicalReturns(replay, n_step=3, gamma=.9)
        context = np.ones(CONTEXT_DIM, np.float32); action = np.zeros(M, np.float32)
        for i, reward in enumerate((1., 2., 3.)):
            queue.store(context, action, reward, np.full(STATE_DIM, i+1), False)
        self.assertEqual(len(replay.rows), 1)
        self.assertAlmostEqual(replay.rows[0][2], 1+.9*2+.9**2*3)
        self.assertAlmostEqual(replay.rows[0][5], .9**3)  # Not gamma**6.
        # No offload decision exists at this final failed-movement slot.
        queue.store(None, action, -4., np.full(STATE_DIM, 4), True)
        self.assertEqual(len(replay.rows), 3)
        self.assertAlmostEqual(replay.rows[1][2], 2+.9*3-.9**2*4)
        self.assertAlmostEqual(replay.rows[2][2], 3-.9*4)
        self.assertTrue(replay.rows[1][4]); self.assertTrue(replay.rows[2][4])
        self.assertEqual(len(queue.pending), 0)


class ConditionalPolicyTest(unittest.TestCase):
    def context(self):
        env = SequentialUAVEnv(use_cvx_alloc=True, seed=42); env.reset(42)
        return env, service_observation(env, env.prepare_service(np.zeros(3*K)))

    def test_offload_mask_entropy_and_permutation(self):
        _, obs = self.context()
        tensor = torch.as_tensor(obs, device=DEVICE)[None]
        actor = StageActor(True).to(DEVICE).eval(); critic = StageTwinCritic(True).to(DEVICE).eval()
        permutation = torch.randperm(M, device=DEVICE)
        global_features, users = split_context(tensor)
        permuted = torch.cat([global_features, users[:, permutation].reshape(1, -1)], 1)
        noise = torch.randn((1, M), device=DEVICE)
        with torch.no_grad():
            action, logp, target = actor.sample(tensor, noise=noise)
            pa, plogp, ptarget = actor.sample(permuted, noise=noise[:, permutation])
            torch.testing.assert_close(pa, action[:, permutation], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(plogp, logp, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(ptarget, target)
            for x, y in zip(critic(tensor, action), critic(permuted, pa)):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
            mask = eligible_users(tensor)
            self.assertTrue(torch.all(action[~mask] == -1))
            self.assertEqual(target.item(), -mask.sum().item())
            empty = tensor.clone(); eg, eu = split_context(empty)
            eu[:, :, 3] = 0
            ea, ep, et = actor.sample(empty)
            self.assertTrue(torch.all(ea == -1)); self.assertEqual(ep.item(), 0); self.assertEqual(et.item(), 0)
            # Destination information is actually available to the policy.
            changed = tensor.clone(); _, cu = split_context(changed)
            cu[:, :, 4:4+K] = cu[:, :, 4:4+K].flip(-1)
            self.assertGreater((actor(tensor)[0]-actor(changed)[0]).abs().max().item(), 0)

    def test_engineering_updates_checkpoint_and_frozen_evaluation(self):
        agent = ConditionalDARSAgent(capacity=32, batch_size=4)
        cfg = EnvironmentConfig(horizon=6, hotspot_mode='diverse')
        env = SequentialUAVEnv(use_cvx_alloc=True, experiment_config=cfg, seed=42)
        normalizer = StateNormalization(cfg.battery_capacity)
        state = env.reset(42); agent.configure_environment(env)
        for step in range(6):
            normalized = normalizer.normalize(state)
            # Covers the random warmup path, which bypasses select_action entirely.
            action = np.zeros(ACTION_DIM) if step == 0 else agent.select_action(normalized, noise_std=1.)
            state, reward, done, info = env.step(action)
            agent.store(normalized, action, reward, normalizer.normalize(state), done)
            if done: break
        self.assertTrue(agent.buffer.ready(4)); self.assertTrue(agent.offload_buffer.ready(4))
        self.assertTrue(np.all(agent.buffer.discounts[:agent.buffer.size] == 1.))
        self.assertTrue(np.all(agent.buffer.rewards[:agent.buffer.size] == 0.))
        before_move = [p.detach().clone() for p in agent.move_actor.parameters()]
        before_offload = [p.detach().clone() for p in agent.offload_actor.parameters()]
        # Learning uses stage critics, never a differentiable/hard association callback.
        env.prepare_service = lambda *_: (_ for _ in ()).throw(AssertionError('No environment call in learn'))
        for _ in range(2):
            report = agent.learn()
            self.assertTrue(np.isfinite(report['critic_loss']))
            self.assertTrue(all(v is not None for v in report['stages'].values()))
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before_move, agent.move_actor.parameters())))
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before_offload, agent.offload_actor.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for m in (agent.move_actor, agent.offload_actor, agent.move_critic, agent.offload_critic) for p in m.parameters()))
        temp_root = ROOT/'tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(temp_root.resolve()))
            checkpoint = Path(temp)/'conditional.pth'; agent.save(checkpoint)
            restored = ConditionalDARSAgent(capacity=8, batch_size=4); restored.load(checkpoint)
            self.assertEqual(model_digest(agent), model_digest(restored))
            old = Path(temp)/'old.pth'; torch.save({'architecture':'dars_userwise_v1'}, old)
            with self.assertRaises(ValueError): restored.load(old)
        digest = model_digest(agent); replay_count = agent.buffer.size
        rows = evaluate_policy(agent, cfg, {'use_cvx_alloc':True}, [931000], keep_trace=False)
        self.assertEqual(len(rows), 1); self.assertEqual(model_digest(agent), digest)
        self.assertEqual(agent.buffer.size, replay_count)
        self.assertEqual(rows[0]['energy_violation_steps'], 0)


if __name__ == '__main__': unittest.main()
