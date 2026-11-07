"""Meaningful checks for the independent frozen evaluation adapter."""
import json
from pathlib import Path
import unittest
from unittest import mock
import tempfile
import numpy as np
import torch
import hotspot_generalization_v2 as h


class ProtocolTests(unittest.TestCase):
    def test_windows_transient_log_lock_does_not_abort_or_change_data(self):
        original = Path.replace
        attempts = []
        def briefly_locked(path, destination):
            attempts.append(path)
            if len(attempts) < 3:
                raise PermissionError("simulated Windows reader holding the progress file")
            return original(path, destination)
        with tempfile.TemporaryDirectory(dir=h.ROOT / "results", prefix="hotspot_v2_io_test_") as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((h.ROOT / "results").resolve()))
            path = Path(folder) / "progress.json"
            with mock.patch.object(Path, "replace", briefly_locked):
                h.dump(path, {"completed_episodes": 55})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"completed_episodes": 55})
            self.assertEqual(len(attempts), 3)

    def test_schedule_support(self):
        starts = [0, 0, 0]
        for index in range(201):
            seed = 700000 + index
            fwd, fm = h.make_schedule("matched_forward", seed, index)
            rev, rm = h.make_schedule("matched_reverse", seed, index)
            self.assertEqual(len(fwd), 51)
            self.assertEqual(fwd[0], rev[0])
            starts[fwd[0]] += 1
            np.testing.assert_array_equal(np.bincount(fwd[:50]), np.bincount(rev[:50]))
            for metadata, direction in ((fm, 1), (rm, 2)):
                seg = metadata["segments"]
                self.assertEqual(sum(x["duration"] for x in seg), 50)
                self.assertTrue(all(12 <= x["duration"] <= 20 for x in seg))
                self.assertTrue(all((b["phase"]-a["phase"]) % 3 == direction for a, b in zip(seg, seg[1:])))
            for case, low, high in (("nonperiodic", 12, 20), ("fast_switch", 4, 8), ("slow_switch", 24, 32)):
                schedule, metadata = h.make_schedule(case, seed, index)
                self.assertTrue(all(low <= x["duration"] <= high for x in metadata["segments"]))
                if case == "nonperiodic":
                    observed = schedule[:50]
                    changes = observed[1:][observed[1:] != observed[:-1]] - observed[:-1][observed[1:] != observed[:-1]]
                    self.assertEqual(set(changes % 3), {1, 2})
        self.assertEqual(starts, [67, 67, 67])
        self.assertFalse(set(h.PAIRS) & set(h.NOVEL_PAIRS))

    def test_actual_load_and_initial_conditions_match(self):
        for index, seed in enumerate((699990, 699991)):
            forward, _ = h.make_bank_entry("matched_forward", seed, index)
            reverse, _ = h.make_bank_entry("matched_reverse", seed, index)
            np.testing.assert_array_equal(forward["ue"], reverse["ue"])
            np.testing.assert_array_equal(forward["initial_uav"], reverse["initial_uav"])
            for phase in range(3):
                a = forward["schedule"][:50] == phase
                b = reverse["schedule"][:50] == phase
                np.testing.assert_array_equal(forward["tasks"][:50][a], reverse["tasks"][:50][b])
                np.testing.assert_array_equal(forward["active"][:50][a], reverse["active"][:50][b])

    def test_replay_preserves_original_physics_and_id(self):
        seed = 699990
        bank, _ = h.make_bank_entry("natural_id", seed, 0)
        ordinary = h.UAVEnv(experiment_config=h.base_config(), seed=seed, use_cvx_alloc=True)
        replay = h.ReplayEnv(bank, experiment_config=h.base_config(), seed=seed, use_cvx_alloc=True)
        np.testing.assert_array_equal(ordinary.reset(seed=seed), replay.reset(seed=seed))
        rng = np.random.default_rng(93)
        for t in range(50):
            action = rng.uniform(-1, 1, ordinary.action_dim)
            a, ra, da, ia = ordinary.step(action)
            b, rb, db, ib = replay.step(action)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ra, rb)
            self.assertEqual(da, db)
            for field in ("offload_ratios", "delays", "energy_comp", "energy_out", "rho"):
                np.testing.assert_array_equal(ia[field], ib[field])
            if da:
                break

    def test_frozen_digest_covers_list_actors(self):
        agent = type("Dummy", (), {})()
        agent.actors = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        before = h.model_digest(agent)
        with h.frozen_policy(agent):
            self.assertTrue(all(not a.training for a in agent.actors))
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(all(a.training for a in agent.actors))
        with torch.no_grad():
            agent.actors[0].weight.add_(1)
        self.assertNotEqual(before, h.model_digest(agent))

    def test_failure_is_included_in_switch_service(self):
        bank, _ = h.make_bank_entry("matched_reverse", 699990, 0)
        events = h.switch_events(bank, [])
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e["eligible_four_slots"] for e in events))
        self.assertTrue(all(e["offloaded_bits"] == 0 and e["distance_before_m"] is None for e in events))
        self.assertTrue(all(not e["reached"] and not e["complete_four_slots"] for e in events))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
