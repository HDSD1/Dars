"""Two decisions per physical slot, using the unchanged UAVEnv physics.

Preparation does not advance time, RNG, batteries, users or metrics. The service
commit reuses the exact prepared projection and association inside UAVEnv.step.
Only one pending context can be committed, once, and only against its state.
"""
from dataclasses import dataclass
import hashlib
import math
import time
import numpy as np
from UAV_env import UAVEnv
from config import K, M, ACTION_DIM, V_MAX


def _owned(value):
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
    return value


@dataclass(frozen=True)
class ServiceContext:
    owner: int
    fingerprint: str
    move_proposal: np.ndarray
    state: np.ndarray
    projection: tuple
    association: np.ndarray
    preparation_seconds: float

    @property
    def feasible(self):
        return bool(self.projection[-1])


class SequentialUAVEnv(UAVEnv):
    def reset(self, seed=None):
        self._pending_service = None
        self._committing_service = None
        self.last_service_context = None
        return super().reset(seed)

    def _state_fingerprint(self):
        h = hashlib.sha256()
        h.update(str(self.current_step).encode('ascii'))
        # Include all realized decision-state variables; preparation consumes no RNG.
        for array in (self.e_battery_uav, self.loc_uav, self.loc_ue,
                      self.task_list, self.active_mask, self.hotspot_schedule):
            h.update(array.tobytes())
        return h.hexdigest()

    def prepare_service(self, move_proposal):
        began = time.perf_counter()
        move = np.asarray(move_proposal, dtype=np.float64)
        if move.shape != (3*K,) or not np.all(np.isfinite(move)):
            raise ValueError('Movement proposal must have 3K finite entries')
        clipped = np.clip(move, -1, 1)
        headings = np.array([math.atan2(clipped[3*k+1], clipped[3*k]) for k in range(K)])
        speeds = (clipped[2:3*K:3]+1)*0.5*V_MAX
        projected = super()._project_movement_for_energy(headings, speeds)
        association = (super()._distance_association(projected[2]) if projected[-1]
                       else np.full(M, -1, dtype=int))
        context = ServiceContext(id(self), self._state_fingerprint(), _owned(move),
                                 _owned(self._get_obs()), tuple(_owned(x) for x in projected),
                                 _owned(association), time.perf_counter()-began)
        self._pending_service = context
        return context

    def finish_service(self, context, offload_proposal):
        offload = np.asarray(offload_proposal, dtype=np.float64)
        if offload.shape != (M,) or not np.all(np.isfinite(offload)):
            raise ValueError('Offloading proposal must have M finite entries')
        if (context is not self._pending_service or context.owner != id(self)
                or context.fingerprint != self._state_fingerprint()):
            raise ValueError('Service context is stale, already committed, or belongs to another environment')
        self._pending_service = None
        self._committing_service = context
        try:
            outcome = super().step(np.r_[context.move_proposal, offload])
        finally:
            self._committing_service = None
        self.last_service_context = context
        outcome[3]['decision_seconds'] += context.preparation_seconds
        outcome[3]['preparation_seconds'] = context.preparation_seconds
        outcome[3]['decision_structure'] = 'movement_then_association_conditioned_offload_v1'
        return outcome

    def step(self, action):
        proposal = np.asarray(action, dtype=np.float64)
        if proposal.shape != (ACTION_DIM,) or not np.all(np.isfinite(proposal)):
            raise ValueError('Expected a finite action vector of shape (action_dim,)')
        context = getattr(self, '_pending_service', None)
        if (context is None or not np.array_equal(context.move_proposal, proposal[:3*K])
                or context.fingerprint != self._state_fingerprint()):
            context = self.prepare_service(proposal[:3*K])
        return self.finish_service(context, proposal[3*K:])

    def _project_movement_for_energy(self, headings, speeds):
        context = getattr(self, '_committing_service', None)
        if context is None:
            return super()._project_movement_for_energy(headings, speeds)
        return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in context.projection)

    def _distance_association(self, uav_locs):
        context = getattr(self, '_committing_service', None)
        if context is None:
            return super()._distance_association(uav_locs)
        return context.association.copy()
