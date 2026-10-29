import unittest
from dataclasses import replace
from unittest.mock import patch
import time
import numpy as np
import torch
import config
from experiment_config import EnvironmentConfig
from UAV_env import UAVEnv
from state_normalization import StateNormalization
from evaluation import EpisodeMetrics, evaluate_policy, model_digest, summarize
from train_unified import FixedOffloadAgent


class EvaluationProtocolTests(unittest.TestCase):
    def test_initial_energy_does_not_change_capacity_or_normalization(self):
        cfg = EnvironmentConfig(initial_energy=10000.)
        e = UAVEnv(experiment_config=cfg)
        state = e.reset(seed=1)
        self.assertEqual(e.battery_capacity, 40000.)
        np.testing.assert_array_equal(StateNormalization(e.battery_capacity).normalize(state)[:config.K], .25)

    def test_horizon_preserves_hotspot_prefix_time_and_exogenous_draws(self):
        for mode in ('fixed', 'diverse', 'unseen_order', 'unseen_timing'):
            short = UAVEnv(experiment_config=EnvironmentConfig(hotspot_mode=mode), seed=8)
            long = UAVEnv(experiment_config=EnvironmentConfig(hotspot_mode=mode, horizon=500), seed=8)
            np.testing.assert_array_equal(short.reset(seed=8), long.reset(seed=8))
            np.testing.assert_array_equal(short.hotspot_schedule, long.hotspot_schedule[:51])
            for t in (0, 16, 17, 33, 34, 49, 50):
                short.current_step = long.current_step = t
                np.testing.assert_array_equal(short._get_obs(), long._get_obs())
            long.current_step = 400
            self.assertEqual(long._get_obs()[-1], 1.)

    def test_holdout_order_and_timing_are_disjoint(self):
        forward = {(0, 1), (1, 2), (2, 0)}
        backward = {(0, 2), (2, 1), (1, 0)}
        for mode in ('diverse', 'unseen_order', 'unseen_timing'):
            for seed in range(5):
                cfg = EnvironmentConfig(hotspot_mode=mode, horizon=500)
                env = UAVEnv(experiment_config=cfg)
                env.reset(seed=seed)
                phases = [segment['phase'] for segment in env.hotspot_segments]
                self.assertTrue(set(zip(phases, phases[1:])).issubset(backward if mode == 'unseen_order' else forward))
                for seg in env.hotspot_segments:
                    d = seg['duration']
                    self.assertTrue((4 <= d <= 8 or 24 <= d <= 32) if mode == 'unseen_timing' else 12 <= d <= 20)

    def test_reset_and_next_observation_use_correct_phase(self):
        env = UAVEnv(experiment_config=EnvironmentConfig(horizon=10, hotspot_reference_slots=3))
        env.reset(seed=7)
        env.current_step = 2
        with patch.object(env, '_generate_tasks', wraps=env._generate_tasks) as generate:
            env.reset(seed=7)
            self.assertEqual(env.current_step, 0)
            self.assertEqual(env._get_current_phase(), 0)
            self.assertEqual(generate.call_count, 1)
        phases = []
        original = env._generate_tasks
        def record_phase():
            phases.append(env._get_current_phase())
            original()
        with patch.object(env, '_generate_tasks', side_effect=record_phase):
            _, _, _, info = env.step(FixedOffloadAgent(0).select_action(None))
        self.assertEqual(info['hotspot_phase'], 0)
        self.assertEqual(phases, [1])

    def test_rng_separation_and_seed_replay(self):
        a = UAVEnv(experiment_config=EnvironmentConfig(hotspot_mode='diverse'))
        b = UAVEnv(experiment_config=EnvironmentConfig(hotspot_mode='unseen_timing'))
        a.reset(seed=23)
        b.reset(seed=23)
        np.testing.assert_array_equal(a.loc_ue, b.loc_ue)
        np.testing.assert_array_equal(a.loc_uav, b.loc_uav)
        # Agent/PER global draws must not change future exogenous environment draws.
        action = FixedOffloadAgent(0).select_action(None)
        reference = a.step(action)[0]
        a.reset(seed=23)
        np.random.random(1000)
        np.testing.assert_array_equal(reference, a.step(action)[0])

    def test_wct_subset_power_and_fixed_state_dimension(self):
        env = UAVEnv(experiment_config=EnvironmentConfig(wct_indices=(2,), wct_power=80.))
        state = env.reset(seed=1)
        self.assertEqual(len(state), config.STATE_DIM)
        self.assertEqual(env.wct_positions.shape, (1, 2))
        locs = np.repeat(env.wct_positions, config.K, axis=0)
        self.assertLessEqual(env._compute_charging(locs).sum(), 80. + 1e-9)

    def test_phase_minimum_overflow_and_energy_conservation(self):
        env = UAVEnv(experiment_config=EnvironmentConfig(battery_capacity=1000., initial_energy=1000.))
        env.reset(seed=1)
        env.loc_uav[:] = env.wct_positions[:config.K]
        env.active_mask[:] = False
        metrics = EpisodeMetrics(env)
        _, reward, _, info = env.step(FixedOffloadAgent(0).select_action(None))
        self.assertTrue(info['action_executed'])
        self.assertTrue(np.all(info['energy_charge_overflow'] > 0))
        self.assertTrue(np.all(info['battery_after_flight'] < info['battery_after_service']))
        np.testing.assert_allclose(info['battery_after_service'],
            info['battery_before'] - info['energy_out'] + info['energy_charge_accepted'], atol=1e-9)
        np.testing.assert_allclose(info['energy_in'], info['energy_charge_accepted'] + info['energy_charge_overflow'])
        metrics.record(info, reward, 0.)
        row = metrics.finish()
        self.assertAlmostEqual(row['min_battery_j'], min(info['battery_after_flight']))
        self.assertAlmostEqual(row['energy_out_j'], row['energy_fly_j'] + row['energy_hover_j'] + row['energy_comp_j'])
        self.assertAlmostEqual(row['slrr_fixed_horizon'], info['rho_mean'] / env.slot_num)

    def test_rejection_not_counted_as_execution_or_energy(self):
        cfg = EnvironmentConfig(initial_energy=1.)
        env = UAVEnv(experiment_config=cfg)
        env.reset(seed=1)
        metrics = EpisodeMetrics(env)
        _, reward, done, info = env.step(FixedOffloadAgent(0).select_action(None))
        metrics.record(info, reward, 0.)
        row = metrics.finish()
        self.assertTrue(done)
        self.assertFalse(row['survived'])
        self.assertEqual(row['executed_steps'], 0)
        self.assertEqual(row['min_battery_j'], 1.)
        self.assertEqual(row['energy_out_j'], 0.)
        self.assertIsNone(row['mean_energy_per_executed_slot_j'])
        self.assertEqual(row['slrr_fixed_horizon'], 0.)
        self.assertEqual(row['rejected_steps'], 1)
        self.assertEqual(summarize([row])['survival_probability'], 0.)

    def test_decision_timer_excludes_task_generation(self):
        env = UAVEnv()
        env.reset(seed=8)
        original = env._update_ue
        def slow_simulation():
            time.sleep(.03)
            original()
        with patch.object(env, '_update_ue', side_effect=slow_simulation):
            start = time.perf_counter()
            _, _, _, info = env.step(FixedOffloadAgent(0).select_action(None))
            elapsed = time.perf_counter() - start
        self.assertGreater(elapsed - info['decision_seconds'], .025)

    def test_frozen_evaluation_matches_repeated_scenes_and_no_learning(self):
        from DARS import PERSACAgent
        torch.set_num_threads(1)
        agent = PERSACAgent()
        cfg = EnvironmentConfig(horizon=2, hotspot_mode='diverse')
        before = model_digest(agent)
        with patch.object(agent, 'learn', side_effect=AssertionError('learn during evaluation')), \
             patch.object(agent, 'store', side_effect=AssertionError('replay during evaluation')):
            rows = evaluate_policy(agent, cfg, {'use_cvx_alloc': True}, [200001, 200001])
        self.assertEqual(before, model_digest(agent))
        self.assertEqual(rows[0]['slrr_fixed_horizon'], rows[1]['slrr_fixed_horizon'])
        np.testing.assert_array_equal(rows[0]['final_battery_j'], rows[1]['final_battery_j'])
        self.assertTrue(rows[0]['survived'])
        self.assertTrue(agent.actor.training)

    def test_invalid_configuration_is_rejected(self):
        for args in (dict(initial_energy=50000.), dict(wct_indices=(0, 0)), dict(horizon=0),
                     dict(horizon=1.2), dict(wct_power=float('nan')),
                     dict(test_orders=((0, 1, 2),)), dict(test_dwell_ranges=((15, 30),))):
            with self.assertRaises(ValueError):
                EnvironmentConfig(**args)


class HeuristicInterfaceTests(unittest.TestCase):
    def test_heuristic_receives_physical_units_and_actual_chargers(self):
        from greedy_agent import GreedyAgent
        from priority_greedy_agent import PriorityGreedyAgent
        from evaluation import configure_policy_environment, policy_input
        cfg = EnvironmentConfig(initial_energy=20000., wct_indices=(1,))
        env = UAVEnv(experiment_config=cfg)
        raw = env.reset(seed=9)
        norm = StateNormalization(env.battery_capacity)
        for agent in (GreedyAgent(), PriorityGreedyAgent()):
            configure_policy_environment(agent, env)
            state = policy_input(agent, norm, raw)
            np.testing.assert_array_equal(state, raw)
            np.testing.assert_array_equal(agent.wct_positions, config.WCT_POSITIONS[[1]])
            self.assertEqual(agent.battery_capacity, 40000.)
            action = agent.select_action(state)
            self.assertTrue(np.isfinite(action).all())
            self.assertEqual(action.shape, (config.ACTION_DIM,))


if __name__ == '__main__':
    unittest.main()
