"""Association-conditioned DARS with two SAC decision stages.

Movement: Q_m(s,u) <- V_o(context), discount 1, no physical reward yet.
Offload: Q_o(c,x) <- r + gamma*V_m(s'), one physical slot later.
The existing DARS three-physical-slot return approximation is retained for
offload replay; it is not three internal decisions, and does not claim an
exact off-policy entropy-corrected multistep Bellman target. Movement replay
uses the one internal transition. Both stages retain dual-priority PER,
importance-weighted Huber losses, delayed actors, temperature adaptation and
Polyak targets. Each stage has its own twin critic and temperature.

Hard association/safety remain environment transitions. No derivative through
sorting, association switches or safety projection is assumed. Reward and
resource allocation are inherited unchanged from UAVEnv.
"""
from collections import deque
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from DARS import (UserAttention, decompose_state, PERBufferSAC, DEVICE,
                  ATTENTION_EMBED_DIM, UAV_FEAT_DIM, ACTOR_DELAY, N_STEP,
                  LOG_SIG_MIN, LOG_SIG_MAX)
from config import (K, M, STATE_DIM, ACTION_DIM, MEMORY_CAPACITY, BATCH_SIZE,
                    GAMMA, TAU_SOFT, LR_ACTOR, LR_CRITIC, GRAD_CLIP,
                    HIDDEN_DIM_1, HIDDEN_DIM_2, GROUND_LENGTH, GROUND_WIDTH,
                    F_UAV, BANDWIDTH, N_MAX, TASK_MAX, P_TX, SIGMA2, T_COM)
from sequential_environment import SequentialUAVEnv
from state_normalization import StateNormalization

ARCHITECTURE = 'dars_association_conditioned_two_stage_v1'
OFF_GLOBAL_DIM = UAV_FEAT_DIM + 5*K
# Original four user features; K+1 association one-hot; relative XY;
# spectral efficiency, associated count/load, CPU and bandwidth capacities.
OFF_USER_DIM = 4 + K+1 + 7
CONTEXT_DIM = OFF_GLOBAL_DIM + M*OFF_USER_DIM


def service_observation(env, context):
    """Encode prepared facts only; never read post-service users or energy."""
    if not context.feasible:
        return np.zeros(CONTEXT_DIM, dtype=np.float32)
    state = StateNormalization(env.battery_capacity).normalize(context.state)
    old_global = np.r_[state[:3*K], state[3*K+4*M:]]
    positions, flight, _, harvest = context.projection[2:6]
    after_flight = context.state[:K] - flight
    budget = after_flight + harvest - env._hover_power()*T_COM
    spatial_scale = np.array([GROUND_LENGTH, GROUND_WIDTH])
    global_features = np.r_[old_global, (positions/spatial_scale).ravel(),
                            after_flight/env.battery_capacity,
                            harvest/env.battery_capacity, budget/env.battery_capacity]
    locations = context.state[3*K:3*K+2*M].reshape(M, 2)
    tasks = context.state[3*K+2*M:3*K+3*M]
    active = context.state[3*K+3*M:3*K+4*M]
    users = np.zeros((M, OFF_USER_DIM), dtype=np.float64)
    users[:, :2] = locations/spatial_scale
    users[:, 2] = tasks/TASK_MAX
    users[:, 3] = active
    for i, k in enumerate(context.association):
        users[i, 4+(int(k) if k >= 0 else K)] = 1
        if k < 0:
            continue
        start = 4+K+1
        users[i, start:start+2] = (positions[k]-locations[i])/spatial_scale
        gain = env._channel_gain(positions[k], locations[i])
        users[i, start+2] = math.log2(1+max(P_TX*gain/SIGMA2, 1e-10))/20
        members = (context.association == k) & (active > 0)
        users[i, start+3:start+7] = [members.sum()/max(N_MAX, 1),
                                   tasks[members].sum()/(max(N_MAX, 1)*TASK_MAX),
                                   F_UAV/3e9, BANDWIDTH/1e6]
    obs = np.r_[global_features, users.ravel()].astype(np.float32)
    assert obs.shape == (CONTEXT_DIM,) and np.all(np.isfinite(obs))
    return obs


def split_context(observation):
    return (observation[:, :OFF_GLOBAL_DIM],
            observation[:, OFF_GLOBAL_DIM:].reshape(-1, M, OFF_USER_DIM))


def eligible_users(observation):
    _, users = split_context(observation)
    return (users[:, :, 3] > .5) & (users[:, :, 4+K] < .5)


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM_1),
                         nn.LayerNorm(HIDDEN_DIM_1), nn.ReLU(),
                         nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
                         nn.LayerNorm(HIDDEN_DIM_2), nn.ReLU(),
                         nn.Linear(HIDDEN_DIM_2, output_dim))


def initialize(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)


class StageActor(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, offload):
        super().__init__(); self.offload = offload
        self.attention = UserAttention(feat_dim=OFF_USER_DIM if offload else 4)
        global_dim = OFF_GLOBAL_DIM if offload else UAV_FEAT_DIM
        self.body = mlp(global_dim + ATTENTION_EMBED_DIM*(2 if offload else 1),
                        2 if offload else 6*K)
        initialize(self)
        nn.init.uniform_(self.body[-1].weight, -3e-3, 3e-3)

    def forward(self, state):
        global_features, users = split_context(state) if self.offload else decompose_state(state)
        tokens = self.attention.encode_users(users)
        global_context = torch.cat([global_features, tokens.mean(dim=1)], dim=-1)
        if self.offload:
            x = torch.cat([tokens, global_context[:, None, :].expand(-1, M, -1)], dim=-1)
            output = self.body(x)
            return output[:, :, 0], output[:, :, 1].clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        mean, log_std = self.body(global_context).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_SIG_MIN, LOG_SIG_MAX)

    def sample(self, state, deterministic=False, noise=None):
        mean, log_std = self(state)
        normal = Normal(mean, log_std.exp())
        if deterministic:
            latent = mean
        elif noise is None:
            latent = normal.rsample()
        else:
            if noise.shape != mean.shape: raise ValueError('Noise shape mismatch')
            latent = mean + log_std.exp()*noise
        action = latent.tanh()
        terms = normal.log_prob(latent) - torch.log(1-action.square()+1e-6)
        mask = eligible_users(state) if self.offload else torch.ones_like(action, dtype=torch.bool)
        action = torch.where(mask, action, -torch.ones_like(action))
        log_prob = torch.where(mask, terms, torch.zeros_like(terms)).sum(-1, keepdim=True)
        target_entropy = -mask.sum(-1, keepdim=True).to(action.dtype)
        return action, log_prob, target_entropy


class StageTwinCritic(nn.Module):
    def __init__(self, offload):
        super().__init__(); self.offload = offload
        self.attention = UserAttention(feat_dim=OFF_USER_DIM if offload else 4)
        if offload:
            self.pairs = nn.ModuleList([nn.Sequential(
                nn.Linear(ATTENTION_EMBED_DIM+1, ATTENTION_EMBED_DIM), nn.ReLU(),
                nn.Linear(ATTENTION_EMBED_DIM, ATTENTION_EMBED_DIM), nn.ReLU()) for _ in range(2)])
        dim = OFF_GLOBAL_DIM+ATTENTION_EMBED_DIM if offload else UAV_FEAT_DIM+ATTENTION_EMBED_DIM+3*K
        self.heads = nn.ModuleList([mlp(dim, 1) for _ in range(2)])
        initialize(self)

    def forward(self, state, action):
        global_features, users = split_context(state) if self.offload else decompose_state(state)
        tokens = self.attention.encode_users(users)
        if self.offload:
            # Pair every user's proposal with that same user's association features before pooling.
            paired = torch.cat([tokens, action.unsqueeze(-1)], dim=-1)
            return tuple(head(torch.cat([global_features, encoder(paired).mean(1)], -1))
                         for head, encoder in zip(self.heads, self.pairs))
        x = torch.cat([global_features, tokens.mean(1), action], -1)
        return tuple(head(x) for head in self.heads)


class StageReplay(PERBufferSAC):
    def __init__(self, capacity, state_dim, action_dim, next_dim):
        super().__init__(capacity, state_dim, action_dim)
        self.next_states = np.zeros((capacity, next_dim), np.float32)
        self.discounts = np.zeros((capacity, 1), np.float32)
        self.priority_rewards = np.zeros(capacity, np.float32)

    def store_stage(self, state, action, reward, next_state, done, discount, priority_reward=None):
        i = self.tree.data_pointer
        physical_reward = reward if priority_reward is None else priority_reward
        super().store(state, action, physical_reward, next_state, done)
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.priority_rewards[i] = physical_reward

    def sample_stage(self, batch):
        sample = super().sample(batch)
        indices = np.asarray(sample[-1])-self.capacity+1
        discounts = torch.as_tensor(self.discounts[indices], device=DEVICE)
        return sample, discounts, self.priority_rewards[indices].copy()


class PhysicalReturns:
    """Queue every physical slot, including a failed movement with no offload decision."""
    def __init__(self, replay, n_step=N_STEP, gamma=GAMMA):
        self.replay = replay; self.n_step = n_step; self.gamma = gamma; self.pending = deque()

    def store(self, context, offload, reward, next_state, done):
        self.pending.append((None if context is None else context.copy(), offload.copy(),
                             float(reward), next_state.copy(), bool(done)))
        if len(self.pending) >= self.n_step: self._emit()
        if done:
            while self.pending: self._emit()

    def _emit(self):
        start = self.pending[0]
        reward = 0.; last = None; length = 0
        for length, row in enumerate(list(self.pending)[:self.n_step], start=1):
            reward += self.gamma**(length-1)*row[2]; last = row
            if row[4]: break
        if start[0] is not None:
            self.replay.store_stage(start[0], start[1], reward, last[3], last[4], self.gamma**length)
        self.pending.popleft()


class ConditionalDARSAgent:
    environment_class = SequentialUAVEnv
    supports_batch_size_override = True
    architecture = ARCHITECTURE
    decision_structure = 'Two SAC stages; within-slot discount1; physical-slot discount GAMMA'

    def __init__(self, capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE):
        self.s_dim = STATE_DIM; self.a_dim = ACTION_DIM; self.batch_size = batch_size
        self.learn_step = 0; self.last_preparation_seconds = 0.; self.env = None
        self.move_actor = StageActor(False).to(DEVICE); self.offload_actor = StageActor(True).to(DEVICE)
        self.move_critic = StageTwinCritic(False).to(DEVICE); self.offload_critic = StageTwinCritic(True).to(DEVICE)
        self.move_target = copy.deepcopy(self.move_critic); self.offload_target = copy.deepcopy(self.offload_critic)
        self.move_log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.offload_log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.optimizers = {}
        for name in ('move', 'offload'):
            self.optimizers[name] = (
                torch.optim.Adam(getattr(self, name+'_actor').parameters(), lr=LR_ACTOR),
                torch.optim.Adam(getattr(self, name+'_critic').parameters(), lr=LR_CRITIC),
                torch.optim.Adam([getattr(self, name+'_log_alpha')], lr=LR_ACTOR))
        self.buffer = StageReplay(capacity, STATE_DIM, 3*K, CONTEXT_DIM)
        self.offload_buffer = StageReplay(capacity, CONTEXT_DIM, M, STATE_DIM)
        self.n_step_buffer = PhysicalReturns(self.offload_buffer)
        self._last_stored_context = None

    def configure_environment(self, env):
        if not isinstance(env, SequentialUAVEnv):
            raise TypeError('Conditional DARS requires SequentialUAVEnv')
        self.env = env; self.last_preparation_seconds = 0.

    def select_action(self, state, noise_std=0., **kwargs):
        if self.env is None: raise RuntimeError('Configure the environment before selecting an action')
        expected = StateNormalization(self.env.battery_capacity).normalize(self.env._get_obs())
        if not np.allclose(state, expected, atol=1e-6, rtol=1e-6):
            raise ValueError('Policy state does not match the configured current environment')
        tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=DEVICE)[None]
        with torch.no_grad():
            move, _, _ = self.move_actor.sample(tensor, deterministic=noise_std == 0)
            move = move[0].cpu().numpy()
            context = self.env.prepare_service(move)
            self.last_preparation_seconds = context.preparation_seconds
            if context.feasible:
                obs = torch.as_tensor(service_observation(self.env, context), device=DEVICE)[None]
                offload, _, _ = self.offload_actor.sample(obs, deterministic=noise_std == 0)
                offload = offload[0].cpu().numpy()
            else:
                offload = np.full(M, -1., dtype=np.float32)
        return np.r_[move, offload]

    def store(self, state, action, reward, next_state, done):
        context = self.env.last_service_context
        if context is None or context is self._last_stored_context:
            raise ValueError('Store exactly once after committing a physical slot')
        expected = StateNormalization(self.env.battery_capacity).normalize(context.state)
        if not np.allclose(state, expected, atol=1e-6, rtol=1e-6):
            raise ValueError('Replay state differs from the prepared state')
        if not np.array_equal(np.asarray(action[:3*K], dtype=np.float64), context.move_proposal):
            raise ValueError('Replay must contain the sampled movement proposal')
        obs = service_observation(self.env, context)
        self.buffer.store_stage(state, action[:3*K], 0. if context.feasible else reward,
                                obs, not context.feasible, 1., priority_reward=reward)
        offload = np.asarray(action[3*K:], dtype=np.float32).copy()
        mask = (context.association >= 0) & (context.state[3*K+3*M:3*K+4*M] > 0)
        offload[~mask] = -1.  # Deterministic dimensions have no policy density.
        self.n_step_buffer.store(obs if context.feasible else None, offload, reward,
                                 np.asarray(next_state, dtype=np.float32), done)
        self._last_stored_context = context

    def _update_stage(self, name):
        replay = self.buffer if name == 'move' else self.offload_buffer
        if not replay.ready(self.batch_size): return None
        other = 'offload' if name == 'move' else 'move'
        actor = getattr(self, name+'_actor'); critic = getattr(self, name+'_critic')
        target = getattr(self, name+'_target'); log_alpha = getattr(self, name+'_log_alpha')
        actor_opt, critic_opt, alpha_opt = self.optimizers[name]
        sample, discount, priority_rewards = replay.sample_stage(self.batch_size)
        states, actions, rewards, next_states, dones, weights, indices = sample
        with torch.no_grad():
            next_actions, next_logp, _ = getattr(self, other+'_actor').sample(next_states)
            q1, q2 = getattr(self, other+'_target')(next_states, next_actions)
            next_value = torch.minimum(q1, q2)-getattr(self, other+'_log_alpha').exp()*next_logp
            target_value = rewards+discount*(1-dones)*next_value
        q1, q2 = critic(states, actions)
        loss = (weights*(F.smooth_l1_loss(q1, target_value, reduction='none')+
                         F.smooth_l1_loss(q2, target_value, reduction='none'))).mean()
        critic_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP); critic_opt.step()
        errors = ((q1-target_value).abs()+(q2-target_value).abs()).detach().cpu().numpy().ravel()/2
        replay.update_priorities(indices, errors, priority_rewards)
        actor_loss = None
        if self.learn_step % ACTOR_DELAY == 0:
            for parameter in critic.parameters(): parameter.requires_grad_(False)
            try:
                proposed, logp, target_entropy = actor.sample(states)
                a, b = critic(states, proposed)
                policy_loss = (weights.detach()*(log_alpha.exp().detach()*logp-torch.minimum(a, b))).mean()
                actor_opt.zero_grad(); policy_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP); actor_opt.step()
            finally:
                for parameter in critic.parameters(): parameter.requires_grad_(True)
            boost = 1+min(2., replay.surprise_ema/40.)
            alpha_loss = -(log_alpha*(logp+target_entropy*boost).detach()).mean()
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
            actor_loss = policy_loss.item()
        with torch.no_grad():
            for t, source in zip(target.parameters(), critic.parameters()):
                t.mul_(1-TAU_SOFT).add_(source, alpha=TAU_SOFT)
        return dict(critic_loss=loss.item(), actor_loss=actor_loss,
                    avg_q=(q1.mean().item()+q2.mean().item())/2,
                    avg_td_error=float(errors.mean()), alpha=log_alpha.exp().item())

    def learn(self):
        if not self.buffer.ready(self.batch_size): return {}
        self.learn_step += 1
        results = {name:self._update_stage(name) for name in ('move', 'offload')}
        valid = [x for x in results.values() if x is not None]
        return dict(critic_loss=float(np.mean([x['critic_loss'] for x in valid])),
                    avg_q=float(np.mean([x['avg_q'] for x in valid])),
                    avg_td_error=float(np.mean([x['avg_td_error'] for x in valid])), stages=results)

    def save(self, path):
        payload = dict(architecture=ARCHITECTURE, K=K, M=M, context_dim=CONTEXT_DIM,
                       physical_n_step=self.n_step_buffer.n_step,
                       checkpoint_scope='Weights only; not optimizer/replay/RNG continuation')
        for name in ('move', 'offload'):
            for kind in ('actor', 'critic', 'target'):
                payload[name+'_'+kind] = getattr(self, name+'_'+kind).state_dict()
            payload[name+'_log_alpha'] = getattr(self, name+'_log_alpha').detach().clone()
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(Path(path), map_location=DEVICE, weights_only=True)
        if (payload.get('architecture') != ARCHITECTURE or payload.get('K') != K
                or payload.get('M') != M or payload.get('context_dim') != CONTEXT_DIM):
            raise ValueError('Checkpoint requires the exact two-stage architecture and dimensions; old DARS weights cannot be relabeled')
        for name in ('move', 'offload'):
            for kind in ('actor', 'critic', 'target'):
                getattr(self, name+'_'+kind).load_state_dict(payload[name+'_'+kind])
            with torch.no_grad(): getattr(self, name+'_log_alpha').copy_(payload[name+'_log_alpha'])
