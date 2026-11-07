"""Run with: python -m unittest discover -s tests -p test_association_benchmarks.py -v"""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))
from association_benchmarks import AssociationSnapshot, deferred_acceptance
from UAV_env import UAVEnv, OFFLOAD_EPS
from config import K, M, S_CYCLES, F_UE, T_FLY, T_COM


class AssociationBenchmarksTest(unittest.TestCase):
    def snapshot(self, seed=9):
        env = UAVEnv(use_cvx_alloc=True, seed=seed)
        env.reset(seed=seed)
        # Ensure both UAVs cover every test user; maintain original 30-user dimension.
        env.loc_uav[:] = [250, 250]
        env.loc_ue[:] = [250, 250]
        env.active_mask[:] = 0
        env.active_mask[:4] = 1
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[3*K:] = np.random.default_rng(seed).uniform(-0.8, 0.8, M)
        action[2:3*K:3] = -1
        previous = copy.deepcopy(env)
        _, _, _, info = env.step(action)
        self.assertTrue(info['action_executed'])
        return AssociationSnapshot(previous, env.loc_uav, action, info), info

    def test_real_environment_projection_reconstruction(self):
        env = UAVEnv(use_cvx_alloc=True, disable_wct=True, seed=17)
        env.reset(seed=17)
        env.loc_uav[:] = [250, 250]
        env.loc_ue[:] = [250, 250]
        env.active_mask[:] = 0
        env.active_mask[:4] = 1
        env.e_battery_uav[:] = env._hover_power() * (T_FLY + T_COM) + 0.01
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[2:3*K:3] = -1
        previous = copy.deepcopy(env)
        _, _, _, info = env.step(action)
        self.assertTrue(info['action_executed'])
        self.assertTrue(info['offload_projected'])
        snap = AssociationSnapshot(previous, env.loc_uav, action, info)
        result = snap.run('greedy')
        for ours, actual in [('offload','offload_ratios'), ('cpu','cpu_allocation'),
                             ('bandwidth','bandwidth_allocation'), ('delay','delays'),
                             ('computation_energy_j','energy_comp')]:
            np.testing.assert_allclose(result[ours], info[actual], rtol=1e-10, atol=1e-9)

    def test_request_rejection_and_capacity(self):
        scores = np.array([[3., 2.], [4., 1.], [-1., -2.]])
        a, count = deferred_acceptance(scores, np.ones_like(scores, bool), 1)
        np.testing.assert_array_equal(a, [1, 0, -1])
        self.assertEqual(count, 3)

    def test_actual_environment_reconstruction(self):
        for seed in range(5):
            snap, info = self.snapshot(seed)
            r = snap.run('greedy')
            for ours, actual in [('association','association'), ('offload','offload_ratios'),
                                 ('cpu','cpu_allocation'), ('bandwidth','bandwidth_allocation'),
                                 ('delay','delays'), ('computation_energy_j','energy_comp')]:
                np.testing.assert_allclose(r[ours], info[actual], rtol=1e-12, atol=1e-9)

    def test_raw_proposal_survives_new_association(self):
        snap, _ = self.snapshot()
        a = np.full(M, -1, int)
        first = snap.evaluate(a)
        a[0] = 1
        second = snap.evaluate(a)
        self.assertEqual(first['offload'][0], 0)
        self.assertEqual(second['offload'][0], snap.proposal[0])
        self.assertGreater(second['cpu'][0], 0)

    def test_zero_budget_means_local_execution(self):
        snap, _ = self.snapshot()
        snap.budget[:] = 0
        snap.available = snap.hover.copy()
        r = snap.run('greedy')
        self.assertTrue(r['feasible'])
        self.assertTrue(r['projected'])
        np.testing.assert_array_equal(r['offload'], np.zeros(M))
        np.testing.assert_array_equal(r['computation_energy_j'], np.zeros(K))
        self.assertAlmostEqual(r['J'], np.sum(snap.active))

    def test_small_budget_is_respected(self):
        snap, _ = self.snapshot()
        snap.budget[:] = 1e-4
        snap.available = snap.hover + snap.budget
        for method in ('greedy', 'matching', 'swap'):
            r = snap.run(method, max_evaluations=500)
            self.assertTrue(r['feasible'])
            self.assertLessEqual(r['energy_excess_j'], 1e-9)
            self.assertTrue(np.all(r['offload'] <= snap.proposal))
            self.assertTrue(np.all(r['cpu'][r['offload'] <= OFFLOAD_EPS] == 0))

    def test_swap_monotonicity_and_no_improving_neighbor(self):
        snap, _ = self.snapshot()
        r = snap.run('swap', max_evaluations=2000, max_rounds=100)
        self.assertLessEqual(r['J'], r['initial_J']+1e-12)
        self.assertEqual(r['search_termination'], 'no_improving_neighbor')
        for a in snap.neighbors(r['association']):
            self.assertGreaterEqual(snap.evaluate(a)['J'], r['J']-1e-8)

    def test_budget_flag_is_not_convergence(self):
        snap, _ = self.snapshot()
        r = snap.run('swap', max_evaluations=1)
        self.assertTrue(r['budget_exhausted'])
        self.assertEqual(r['search_termination'], 'evaluation_limit')

    def test_no_coverage_and_no_active(self):
        snap, _ = self.snapshot()
        snap.env.loc_ue[:] = [1e6, 1e6]
        snap.eligible[:] = False
        for method in ('greedy', 'matching', 'swap'):
            r = snap.run(method)
            self.assertEqual(r['associated_users'], 0)
            self.assertAlmostEqual(r['slrr'], 0)
        snap.active[:] = False
        snap.env.active_mask[:] = False
        r = snap.run('swap')
        self.assertEqual(r['J'], 0)
        self.assertEqual(r['slrr'], 0)

    def test_repeated_runs_leave_snapshot_unchanged(self):
        snap, _ = self.snapshot()
        state = [snap.env.loc_ue.copy(), snap.env.task_list.copy(), snap.proposal.copy()]
        for method in ('swap', 'matching', 'greedy'):
            a, b = snap.run(method), snap.run(method)
            np.testing.assert_array_equal(a['association'], b['association'])
            self.assertEqual(a['J'], b['J'])
        for old, current in zip(state, [snap.env.loc_ue, snap.env.task_list, snap.proposal]):
            np.testing.assert_array_equal(old, current)

    def test_invalid_association_rejected(self):
        snap, _ = self.snapshot()
        a = np.full(M, -1, int)
        a[-1] = 0
        with self.assertRaises(ValueError):
            snap.evaluate(a)


if __name__ == '__main__':
    unittest.main()
