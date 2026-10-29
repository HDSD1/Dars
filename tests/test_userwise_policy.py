"""User/action correspondence and Algorithm 1 training regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Normal

import DARS
from config import K, M, STATE_DIM, ACTION_DIM
from DARS import SACActor, TwinCritic, PERSACAgent, decompose_state
from UAV_env import UAVEnv
from state_normalization import StateNormalization


def permute_state(s, p):
    result = s.clone()
    result[:, 3*K:3*K+2*M] = s[:, 3*K:3*K+2*M].reshape(-1, M, 2)[:, p].reshape(-1, 2*M)
    result[:, 3*K+2*M:3*K+3*M] = s[:, 3*K+2*M:3*K+3*M][:, p]
    result[:, 3*K+3*M:3*K+4*M] = s[:, 3*K+3*M:3*K+4*M][:, p]
    return result


def permute_action(a, p):
    return torch.cat([a[:, :3*K], a[:, 3*K:][:, p]], dim=1)


class UserwisePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3202)
        np.random.seed(3202)
        env = UAVEnv(use_cvx_alloc=True)
        norm = StateNormalization()
        self.s = torch.tensor(np.stack([norm.normalize(env.reset()) for _ in range(8)]),
                              dtype=torch.float32)
        self.actor = SACActor(STATE_DIM, ACTION_DIM).eval()
        self.critic = TwinCritic(STATE_DIM, ACTION_DIM).eval()
        self.p = torch.randperm(M)
        self.sp = permute_state(self.s, self.p)

    def test_actor_tokens_parameters_and_actions_follow_users(self):
        with torch.no_grad():
            tokens, context = self.actor._encode(self.s)
            tp, cp = self.actor._encode(self.sp)
            torch.testing.assert_close(tp, tokens[:, self.p], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(cp, context, atol=2e-6, rtol=1e-5)
            for original, permuted in zip(self.actor(self.s), self.actor(self.sp)):
                torch.testing.assert_close(permuted, permute_action(original, self.p),
                                           atol=1e-6, rtol=1e-5)
            a = self.actor.deterministic(self.s)
            ap = self.actor.deterministic(self.sp)
            torch.testing.assert_close(ap, permute_action(a, self.p), atol=1e-6, rtol=1e-5)
            # Reject the trivial "all users get the same action" solution.
            self.assertGreater(a[:, 3*K:].std(dim=1).max().item(), 1e-5)
            # Identical user tuples must produce identical offloading parameters.
            same = self.s.clone()
            loc = same[:, 3*K:3*K+2*M].reshape(-1, M, 2)
            loc[:, 1] = loc[:, 0]
            for start in (3*K+2*M, 3*K+3*M):
                same[:, start+1] = same[:, start]
            for params in self.actor(same):
                torch.testing.assert_close(params[:, 3*K], params[:, 3*K+1])

    def test_joint_log_probability_and_paired_noise(self):
        noise = torch.randn(8, ACTION_DIM) * 0.5
        a, logp, mean = self.actor.sample(self.s, noise=noise)
        ap, logpp, _ = self.actor.sample(self.sp, noise=permute_action(noise, self.p))
        torch.testing.assert_close(ap, permute_action(a, self.p), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(logpp, logp, atol=1e-5, rtol=1e-5)
        _, logstd = self.actor(self.s)
        latent = mean + logstd.exp() * noise
        per_dim = Normal(mean, logstd.exp()).log_prob(latent) - torch.log(1-a.square()+1e-6)
        expected = per_dim[:, :3*K].sum(1, keepdim=True) + per_dim[:, 3*K:].sum(1, keepdim=True)
        torch.testing.assert_close(logp, expected)
        (logp.mean() + a.square().mean()).backward()
        for module in (self.actor.user_attn, self.actor.offload_body,
                       self.actor.offload_mean_head, self.actor.offload_log_std_head):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_critic_preserves_pairs_and_detects_mismatched_pairs(self):
        a = torch.rand(8, ACTION_DIM) * 2 - 1
        ap = permute_action(a, self.p)
        q = self.critic(self.s, a)
        qp = self.critic(self.sp, ap)
        wrong = self.critic(self.sp, a)
        for old, aligned, mismatched in zip(q, qp, wrong):
            torch.testing.assert_close(aligned, old, atol=1e-6, rtol=1e-5)
            self.assertGreater((old-mismatched).abs().max().item(), 1e-7)
        actor_ids = {id(p) for p in self.actor.user_attn.parameters()}
        critic_ids = {id(p) for p in self.critic.user_attn.parameters()}
        self.assertFalse(actor_ids & critic_ids)
        self.assertFalse({id(p) for p in self.critic.pair_encoder1.parameters()} &
                         {id(p) for p in self.critic.pair_encoder2.parameters()})

    def make_agent(self):
        with patch.object(DARS, 'MEMORY_CAPACITY', 32), patch.object(DARS, 'DEVICE', torch.device('cpu')):
            return PERSACAgent()

    def test_terminal_targets_and_delayed_updates_match_algorithm(self):
        agent = self.make_agent()
        for episode in range(4):
            for t in range(2):
                state = self.s[2*episode+t].numpy()
                agent.store(state, np.zeros(ACTION_DIM), float(t+episode+1), state, t == 1)
        self.assertEqual(agent.buffer.size, 8)
        self.assertEqual(len(agent.n_step_buffer.buffer), 0)
        self.assertTrue(np.all(agent.buffer.dones[:8] == 1))
        batch = agent.buffer.sample(4)
        actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        alpha_before = agent.log_alpha.detach().clone()
        target_before = [p.detach().clone() for p in agent.critic_target.parameters()]
        priority_before = agent.buffer.tree.tree.copy()
        huber = DARS.F.smooth_l1_loss
        with patch.object(DARS, 'BATCH_SIZE', 4), patch.object(agent.buffer, 'sample', return_value=batch):
            with patch.object(DARS.F, 'smooth_l1_loss', wraps=huber) as spy:
                first = agent.learn()
                for call in spy.call_args_list:
                    torch.testing.assert_close(call.args[1], batch[2], rtol=0, atol=0)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
            torch.testing.assert_close(agent.log_alpha, alpha_before, rtol=0, atol=0)
            for before, online, target in zip(target_before, agent.critic.parameters(), agent.critic_target.parameters()):
                torch.testing.assert_close(target, (1-DARS.TAU_SOFT)*before+DARS.TAU_SOFT*online)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(target_before, agent.critic_target.parameters())))
            self.assertFalse(np.array_equal(priority_before, agent.buffer.tree.tree))
            second = agent.learn()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertFalse(torch.equal(agent.log_alpha, alpha_before))
        self.assertTrue(all(np.isfinite(v) for r in (first, second) for v in r.values()))
        with torch.no_grad():
            torch.testing.assert_close(agent.actor.deterministic(self.sp),
                                       permute_action(agent.actor.deterministic(self.s), self.p),
                                       atol=1e-6, rtol=1e-5)

    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        agent = self.make_agent()
        restored = self.make_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            agent.save(path)
            restored.load(path)
            torch.testing.assert_close(agent.actor.deterministic(self.s), restored.actor.deterministic(self.s))
            old = torch.load(path, weights_only=True)
            old.pop('architecture')
            old['actor'].pop('offload_mean_head.weight')
            torch.save(old, path)
            with self.assertRaisesRegex(ValueError, 'Legacy pooled-only'):
                restored.load(path)


if __name__ == '__main__':
    unittest.main()
