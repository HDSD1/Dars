"""Run with: python -m unittest test_energy_shield -v"""
import unittest
import numpy as np
import UAV_env as model


class EnergyShieldTests(unittest.TestCase):
    def setUp(self):
        self.env = model.UAVEnv(use_cvx_alloc=True)
        self.env.reset()
        self.action = np.zeros(model.ACTION_DIM)

    def test_preview_has_no_side_effects(self):
        env = self.env
        state = env._get_obs().copy()
        rng = np.random.get_state()
        env._step_candidate(self.action, evaluate_only=True)
        np.testing.assert_array_equal(state, env._get_obs())
        self.assertEqual(len(env.reward_log), 0)
        after = np.random.get_state()
        np.testing.assert_array_equal(rng[1], after[1])
        self.assertEqual(rng[2:], after[2:])

    def test_feasible_action_unchanged(self):
        env = self.env
        env._fly_power = lambda v: 1.0
        env._hover_power = lambda: 1.0
        env.active_mask[:] = False
        expected = env._step_candidate(self.action, evaluate_only=True)
        _, _, _, info = env.step(self.action)
        self.assertEqual(info['shield_status'], 'accepted')
        self.assertEqual(info['shield_checks'], 1)
        np.testing.assert_array_equal(info['executed_action'], self.action)
        np.testing.assert_allclose(env.e_battery_uav, expected['battery_end'])

    def test_motion_repair_and_no_future_flight_credit(self):
        env = self.env
        env.active_mask[:] = False
        env.e_battery_uav[:] = 20.0 * model.T_FLY
        env._fly_power = lambda v: 10.0 + 100.0 * v
        env._hover_power = lambda: 1.0
        env._compute_charging = lambda p: np.full(model.K, 1000.0)
        check = env._step_candidate(self.action, evaluate_only=True)
        self.assertEqual(check['phase'], 'flight')
        before = env._get_obs().copy()
        selected = None
        for candidate in env._safety_candidates(self.action):
            if env._step_candidate(candidate, evaluate_only=True)['feasible']:
                selected = candidate
                break
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(before, env._get_obs())
        _, _, done, info = env.step(self.action)
        self.assertFalse(done)
        self.assertEqual(info['shield_status'], 'corrected')
        self.assertTrue(np.all(info['battery_postflight'] >= 0))
        self.assertTrue(np.all(env.e_battery_uav >= 0))

    def test_service_repair(self):
        env = self.env
        env._fly_power = lambda v: 1.0
        env._hover_power = lambda: 1.0
        env._compute_charging = lambda p: np.zeros(model.K)
        env._distance_association = lambda p: np.zeros(model.M, dtype=int)
        env._compute_allocation = lambda *args: np.full(model.M, 1e9)
        env._compute_bandwidth_allocation = lambda *args: np.full(model.M, 1e6)
        env.active_mask[:] = True
        env.task_list[:] = 1e6
        env.e_battery_uav[:] = model.T_FLY + model.T_COM + 0.01
        self.assertEqual(env._step_candidate(self.action, True)['phase'], 'service')
        _, _, done, info = env.step(self.action)
        self.assertFalse(done)
        self.assertEqual(info['shield_status'], 'corrected')
        self.assertTrue(np.all(env.e_battery_uav >= 0))

    def test_edge_deadline_repair(self):
        env = self.env
        env.active_mask[:] = False
        env.active_mask[0] = True
        env.task_list[:] = 0.0
        env.task_list[0] = 1e6
        env._distance_association = lambda p: np.zeros(model.M, dtype=int)
        # Force a long edge execution time even though energy is sufficient.
        env._compute_allocation = lambda *args: np.full(model.M, 1e6)
        env._compute_bandwidth_allocation = lambda *args: np.full(model.M, 1e9)
        check = env._step_candidate(self.action, evaluate_only=True)
        self.assertEqual(check['phase'], 'edge_deadline')
        _, _, done, info = env.step(self.action)
        self.assertFalse(done)
        self.assertEqual(info['shield_status'], 'corrected')
        self.assertEqual(info['shield_original_failure'], 'edge_deadline')
        self.assertTrue(np.all(info['edge_delays'] <= model.T_COM))

    def test_reflection_preserves_flight_energy(self):
        env = self.env
        env.active_mask[:] = False
        speed = model.V_MAX / 2
        distance = speed * model.T_FLY
        env._fly_power = lambda v: 10.0 + v
        env._hover_power = lambda: 0.0
        env._compute_charging = lambda p: np.zeros(model.K)
        action = np.zeros(model.ACTION_DIM)
        action[0:3*model.K:3] = 1.0  # Fly towards the right boundary.
        expected_energy = (10.0 + speed) * model.T_FLY
        for start_x in (model.GROUND_LENGTH / 2,
                        model.GROUND_LENGTH - distance / 4):
            env.reset()
            env.active_mask[:] = False
            env.loc_uav[:, 0] = start_x
            env.loc_uav[:, 1] = model.GROUND_WIDTH / 2
            expected_x, _ = env._bounce(start_x + distance, model.GROUND_WIDTH / 2)
            preview = env._step_candidate(action, evaluate_only=True)
            np.testing.assert_allclose(
                preview['battery_postflight'], model.E_BATTERY_INIT - expected_energy)
            _, _, _, info = env.step(action)
            self.assertEqual(info['shield_status'], 'accepted')
            np.testing.assert_allclose(env.loc_uav[:, 0], expected_x)
            np.testing.assert_allclose(info['energy_out'], expected_energy)

    def test_no_candidate_no_execution(self):
        env = self.env
        env.e_battery_uav[:] = 0.0
        env._fly_power = lambda v: 10.0
        env._compute_charging = lambda p: np.full(model.K, 1e9)
        positions = env.loc_uav.copy()
        _, reward, done, info = env.step(self.action)
        self.assertTrue(done)
        self.assertLess(reward, 0)
        self.assertEqual(info['shield_status'], 'no_feasible_candidate_found')
        self.assertIsNone(info['executed_action'])
        self.assertEqual(info['shield_checks'], 80)
        np.testing.assert_array_equal(positions, env.loc_uav)
        self.assertTrue(np.all(env.e_battery_uav == 0))
        self.assertTrue(np.all(info['energy_in'] == 0))
        self.assertTrue(np.all(info['energy_out'] == 0))


if __name__ == '__main__':
    unittest.main()
