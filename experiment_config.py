"""Serializable environment settings; no mutation of global config.py required."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from config import E_BATTERY_CAPACITY, E_BATTERY_INIT, SLOT_NUM, N_WCT, P_MAX_WCT


@dataclass(frozen=True)
class EnvironmentConfig:
    battery_capacity: float = E_BATTERY_CAPACITY
    initial_energy: float = E_BATTERY_INIT
    horizon: int = SLOT_NUM
    wct_indices: tuple = tuple(range(N_WCT))
    wct_power: float = P_MAX_WCT
    hotspot_mode: str = 'fixed'
    hotspot_reference_slots: int = SLOT_NUM
    time_reference_slots: int = SLOT_NUM
    # Cyclic orders in training; the reverse orders are reserved for testing.
    train_orders: tuple = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    test_orders: tuple = ((0, 2, 1), (2, 1, 0), (1, 0, 2))
    train_dwell: tuple = (12, 20)
    test_dwell_ranges: tuple = ((4, 8), (24, 32))

    def __post_init__(self):
        for key in ('wct_indices', 'train_dwell'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for key in ('train_orders', 'test_orders', 'test_dwell_ranges'):
            object.__setattr__(self, key, tuple(tuple(v) for v in getattr(self, key)))
        if not all(math.isfinite(x) for x in (self.battery_capacity, self.initial_energy, self.wct_power)):
            raise ValueError('Energy and power settings must be finite')
        if not 0 <= self.initial_energy <= self.battery_capacity or self.battery_capacity <= 0:
            raise ValueError('Require 0 <= initial_energy <= battery_capacity and positive capacity')
        for x in (self.horizon, self.hotspot_reference_slots, self.time_reference_slots):
            if not isinstance(x, int) or isinstance(x, bool) or x <= 0:
                raise ValueError('Horizons/reference periods must be positive integers')
        if self.wct_power < 0 or not self.wct_indices or len(set(self.wct_indices)) != len(self.wct_indices):
            raise ValueError('Use distinct WCT indices and nonnegative power; use disable_wct for no charging')
        if any(not isinstance(i, int) or not 0 <= i < N_WCT for i in self.wct_indices):
            raise ValueError('Invalid WCT index')
        if self.hotspot_mode not in ('fixed', 'diverse', 'unseen_order', 'unseen_timing'):
            raise ValueError('Unknown hotspot mode')
        if not self.train_orders or not self.test_orders or set(self.train_orders) & set(self.test_orders):
            raise ValueError('Training and held-out order sets must be nonempty and disjoint')
        if any(sorted(order) != [0, 1, 2] for order in self.train_orders + self.test_orders):
            raise ValueError('Each order must permute the three original hotspot pairs')
        for bounds in (self.train_dwell,) + self.test_dwell_ranges:
            if len(bounds) != 2 or any(not isinstance(x, int) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError('Invalid dwell range')
        if not self.test_dwell_ranges or any(max(a, self.train_dwell[0]) <= min(b, self.train_dwell[1])
                                            for a, b in self.test_dwell_ranges):
            raise ValueError('Held-out timing ranges must not overlap training dwell range')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def write(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n', encoding='utf-8')
