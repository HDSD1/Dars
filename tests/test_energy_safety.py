"""Focused regression checks for energy admission and terminal replay."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import config as cfg
import UAV_env as env_module
from UAV_env import UAVEnv, ENERGY_TOL
from DARS import NStepBuffer, PERSACAgent, SACActor


def proposal(offload=1.0):
    a = np.zeros(cfg.ACTION_DIM)
    a[:3*cfg.K:3] = 1.0
    a[2:3*cfg.K:3] = -1.0
    a[3*cfg.K:] = 2.0*offload - 1.0
    return a


def environment(cvx=True):
    e = UAVEnv(use_cvx_alloc=cvx, disable_wct=True)
    e.reset()
    e.loc_uav[:] = [100.0, 100.0]
    e.loc_ue[:] = [100.0, 100.0]
    e.active_mask[:] = True
    e.task_list[:] = cfg.TASK_MAX
    return e


class EnergySafetyTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(71)

    def assert_rejected_without_transition(self, e, a):
        state = e._get_obs().copy()
        step = e.current_step
        rng = np.random.get_state()
        nxt, reward, done, info = e.step(a)
        np.testing.assert_array_equal(nxt, state)
        self.assertEqual(e.current_step, step)
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertFalse(info['action_executed'])
        self.assertIsNone(info['executed_action'])
        for key in ('energy_in', 'energy_out', 'energy_fly', 'energy_service'):
            np.testing.assert_array_equal(info[key], np.zeros(cfg.K))
        after = np.random.get_state()
        np.testing.assert_array_equal(rng[1], after[1])
        self.assertEqual(rng[2:], after[2:])
        return info

    def test_feasible_step_commits_correct_battery_and_preserves_proposal(self):
        e = environment()
        a = proposal(0.5)
        original = a.copy()
        battery = e.e_battery_uav.copy()
        _, _, done, info = e.step(a)
        self.assertFalse(done)
        self.assertTrue(info['action_executed'])
        np.testing.assert_array_equal(a, original)
        np.testing.assert_array_equal(info['raw_action'], original)
        np.testing.assert_allclose(e.e_battery_uav,
                                   battery + info['energy_in'] - info['energy_out'])
        self.assertTrue(np.all(info['flight_ok'] & info['service_ok']))
        self.assertEqual(e.current_step, 1)

    def test_unassociated_inactive_and_zero_offload_users_have_no_resources(self):
        for cvx in (False, True):
            with self.subTest(cvx=cvx):
                e = environment(cvx)
                e.loc_ue[-1] = [499.0, 499.0]
                e.active_mask[-2] = False
                a = proposal()
                a[3*cfg.K] = -1.0
                _, _, _, info = e.step(a)
                self.assertEqual(info['offload_ratios'][0], 0.0)
                self.assertEqual(info['offload_ratios'][-1], 0.0)
                self.assertEqual(info['offload_ratios'][-2], 0.0)
                zero = info['offload_ratios'] == 0.0
                np.testing.assert_array_equal(info['cpu_allocation'][zero], 0.0)
                np.testing.assert_array_equal(info['bandwidth_allocation'][zero], 0.0)
                for k in range(cfg.K):
                    users = info['association'] == k
                    self.assertLessEqual(info['cpu_allocation'][users].sum(), cfg.F_UAV+1e-5)
                    self.assertLessEqual(info['bandwidth_allocation'][users].sum(), cfg.BANDWIDTH+1e-8)

    def test_budget_scaling_preserves_conditional_kkt_and_enforces_energy(self):
        e = environment()
        a = proposal()
        association = e._distance_association(e.loc_uav)
        x = (association >= 0).astype(float)
        counts = np.bincount(association[association >= 0], minlength=cfg.K)
        frequencies = e._compute_allocation(association, counts, x)
        energy = e._compute_comp_energy(association, x, frequencies)
        baseline = e._fly_power(0.0)*cfg.T_FLY + e._hover_power()*cfg.T_COM
        e.e_battery_uav[:] = baseline + 0.25*energy
        _, _, _, info = e.step(a)
        self.assertTrue(info['action_executed'])
        self.assertTrue(info['offload_projected'])
        active_uavs = energy > 0.0
        np.testing.assert_allclose(info['offload_scale_ideal'][active_uavs], 0.25,
                                   rtol=1e-9)
        np.testing.assert_allclose(info['cpu_allocation'], frequencies, rtol=1e-10)
        np.testing.assert_allclose(info['energy_comp'],
                                   info['offload_scale']*energy, rtol=1e-8)
        self.assertTrue(np.all(info['energy_comp'] <= info['computation_budget']+ENERGY_TOL))
        self.assertFalse(np.array_equal(info['executed_action'], info['raw_action']))

    def test_zero_computation_budget_disables_offloading(self):
        e = environment()
        e.e_battery_uav[:] = e._fly_power(0.0)*cfg.T_FLY + e._hover_power()*cfg.T_COM
        _, _, _, info = e.step(proposal())
        self.assertTrue(info['action_executed'])
        np.testing.assert_array_equal(info['offload_ratios'], 0.0)
        np.testing.assert_array_equal(info['cpu_allocation'], 0.0)
        np.testing.assert_array_equal(info['bandwidth_allocation'], 0.0)

    def test_movement_search_restores_hover_feasibility(self):
        e = environment()
        minimum_flight = min(e._fly_power(v)*cfg.T_FLY
                             for v in np.linspace(0, cfg.V_MAX, 513))
        e.e_battery_uav[:] = minimum_flight + e._hover_power()*cfg.T_COM + 2.0
        a = proposal(0.0)
        a[2:3*cfg.K:3] = 1.0
        _, _, _, info = e.step(a)
        self.assertTrue(info['action_executed'])
        self.assertTrue(info['movement_projected'])
        self.assertTrue(np.all(info['flight_ok'] & info['service_ok']))

    def test_flight_speed_correction_with_endpoint_charging(self):
        e = environment()
        e.disable_wct = False
        e.loc_uav[:] = e.wct_positions[np.arange(cfg.K) % len(e.wct_positions)]
        minimum_flight = min(e._fly_power(v)*cfg.T_FLY
                             for v in np.linspace(0, cfg.V_MAX, 513))
        e.e_battery_uav[:] = minimum_flight + 2.0
        a = proposal(0.0)
        a[2:3*cfg.K:3] = 1.0
        _, _, _, info = e.step(a)
        self.assertTrue(info['action_executed'])
        self.assertTrue(info['movement_projected'])
        self.assertTrue(np.all(info['flight_ok'] & info['service_ok']))
        self.assertTrue(np.all(info['energy_in'] > 0.0))

    def test_future_charging_cannot_finance_flight(self):
        e = environment()
        e.disable_wct = False
        e.loc_uav[:] = e.wct_positions[0]
        self.assertTrue(np.all(e._compute_charging(e.loc_uav) > 0))
        e.e_battery_uav[:] = 1.0
        self.assert_rejected_without_transition(e, proposal())

    def test_hover_infeasibility_does_not_commit_flight(self):
        e = environment()
        e.e_battery_uav[:] = e._hover_power()*cfg.T_COM*0.5
        self.assert_rejected_without_transition(e, proposal())

    def test_final_check_failure_is_also_atomic(self):
        e = environment()
        with patch.object(e, '_compute_comp_energy', return_value=np.full(cfg.K, np.inf)):
            info = self.assert_rejected_without_transition(e, proposal())
        self.assertEqual(info['failure_phase'], 'final_energy_check')
        self.assertTrue(info['movement_feasible'])

    def test_shared_wct_cap_is_applied_to_joint_candidate(self):
        e = environment()
        e.disable_wct = False
        e.loc_uav[:] = e.wct_positions[0]
        with patch.object(env_module, 'N_WCT', 1), patch.object(env_module, 'P_MAX_WCT', 1.0):
            _, _, charge, _, _, _ = e._movement_energy_state(np.zeros(cfg.K), np.zeros(cfg.K))
        np.testing.assert_allclose(charge, np.full(cfg.K, 1.0/cfg.K))

    def test_nonfinite_action_is_rejected_before_mutation(self):
        e = environment()
        state = e._get_obs().copy()
        a = proposal()
        a[0] = np.nan
        with self.assertRaises(ValueError):
            e.step(a)
        np.testing.assert_array_equal(e._get_obs(), state)


class ReplayTests(unittest.TestCase):
    def test_terminal_prefixes_are_emitted_once_and_truncated(self):
        for length in (1, 2, 3, 5):
            with self.subTest(length=length):
                b = NStepBuffer(n_step=3, gamma=0.5)
                emitted = []
                for t in range(length):
                    item = b.store((np.array([t]), np.array([t+10]), t+1,
                                    np.array([t+1]), t == length-1))
                    if item is not None:
                        emitted.append(item)
                emitted.extend(b.flush())
                self.assertEqual(len(emitted), length)
                self.assertEqual(len(b.buffer), 0)
                for t, (state, action, reward, nxt, done) in enumerate(emitted):
                    end = min(t+3, length)
                    self.assertEqual(state[0], t)
                    self.assertEqual(action[0], t+10)
                    self.assertAlmostEqual(reward, sum(0.5**j*(t+j+1) for j in range(end-t)))
                    self.assertEqual(nxt[0], end)
                    self.assertEqual(done, end == length)

    def test_agent_replay_retains_original_proposal_after_correction(self):
        e = environment()
        baseline = e._fly_power(0.0)*cfg.T_FLY + e._hover_power()*cfg.T_COM
        e.e_battery_uav[:] = baseline
        state = e._get_obs().copy()
        a = proposal()
        nxt, reward, _, info = e.step(a)
        agent = object.__new__(PERSACAgent)
        agent.n_step_buffer = NStepBuffer()
        stored = []
        class Collector:
            def store(self, *transition):
                stored.append(transition)
        agent.buffer = Collector()
        agent.store(state, a, reward, nxt, True)
        self.assertEqual(len(stored), 1)
        np.testing.assert_array_equal(stored[0][1], a)
        self.assertFalse(np.array_equal(stored[0][1], info['executed_action']))
        a[:] = -1
        np.testing.assert_array_equal(stored[0][1], info['raw_action'])
        self.assertTrue(stored[0][-1])

    def test_sac_update_runs_with_safety_layer_transitions(self):
        import DARS
        from state_normalization import StateNormalization
        torch.set_num_threads(1)
        with patch.object(DARS, 'MEMORY_CAPACITY', 32), patch.object(DARS, 'BATCH_SIZE', 4):
            agent = PERSACAgent()
            norm = StateNormalization()
            e = environment()
            state = e._get_obs()
            for _ in range(8):
                s_norm = norm.normalize(state)
                a = agent.select_action(s_norm, noise_std=1.0)
                nxt, reward, done, _ = e.step(a)
                agent.store(s_norm, a, reward, norm.normalize(nxt), done)
                state = nxt
            before = [p.detach().clone() for p in agent.actor.parameters()]
            first = agent.learn()
            second = agent.learn()
        self.assertTrue(all(np.isfinite(v) for v in first.values()))
        self.assertTrue(all(np.isfinite(v) for v in second.values()))
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, agent.actor.parameters())))

    def test_actor_proposal_log_probability_and_gradient_are_finite(self):
        actor = SACActor(cfg.STATE_DIM, cfg.ACTION_DIM)
        actions, log_prob, _ = actor.sample(torch.zeros(2, cfg.STATE_DIM))
        self.assertEqual(tuple(actions.shape), (2, cfg.ACTION_DIM))
        self.assertEqual(tuple(log_prob.shape), (2, 1))
        self.assertTrue(torch.isfinite(log_prob).all())
        (log_prob.mean() + actions.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in actor.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
