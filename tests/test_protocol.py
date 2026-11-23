import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from roboinf_benchmark.contracts import CAMERAS, JOINT, ActionChunk, Observation, PolicyDescription
from roboinf_benchmark.runner import run_case
from roboinf_benchmark.scoring import summarize
from roboinf_benchmark.suite import load_suite


class ProtocolTests(unittest.TestCase):
    def test_policy_boundary_replanning_and_endpoint(self):
        class Env:
            step_limit, action_steps, physics_steps, closed = 23, 0, 0, False

            def observe(self):
                return Observation({k: np.zeros((2, 3, 3), dtype=np.uint8) for k in CAMERAS},
                                   np.zeros(14), None, self.action_steps, self.physics_steps * .004)

            def execute(self, action, spec):
                self.action_steps += 1
                self.physics_steps += 4

            def settle(self, steps):
                self.physics_steps += steps

            def close(self):
                self.closed = True

        class Policy:
            def describe(self):
                return PolicyDescription("fixture", JOINT.name)

            def reset(self, ctx):
                self.context = vars(ctx)
                self.starts = []

            def predict(self, obs):
                self.starts.append(obs.action_step)
                return ActionChunk(np.full((32, 14), len(self.starts)), JOINT.name)

        env, policy = Env(), Policy()
        case = {"case_id": "example", "task": "open_laptop", "scene_id": 7, "instruction": "Open the laptop."}
        result = run_case(case, {"scene_seed": 91}, "test", lambda *_: env, policy)
        self.assertEqual(policy.context, {"instruction": "Open the laptop.", "seed": 0})
        self.assertEqual(policy.starts, [0, 20])
        self.assertEqual([r[0] for r in result["actions"]], [1] * 20 + [2] * 3)
        self.assertEqual(env.physics_steps, 23 * 4 + 250)
        self.assertTrue(env.closed)
        self.assertNotIn("scene_seed", result)

    def test_privileged_fields_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name) / "evaluation"
            directory.mkdir()
            case = dict(case_id="x", task="place_fan", scene_id=0, instruction="Place the fan.")
            scene = dict(task="place_fan", scene_id=0, scene_seed=20)
            for key, row in (("cases", case), ("scenes", scene)):
                (directory / f"val-{key}.jsonl").write_text(json.dumps(row) + "\n")
            _, cases = load_suite(name, "val")
            self.assertEqual(list(cases), ["x"])
            (directory / "val-cases.jsonl").write_text(json.dumps({**case, "task_success": True}))
            with self.assertRaises(ValueError):
                load_suite(name, "val")

    def test_task_macro_average_and_completeness(self):
        cases = {c: {"task": "place_fan" if c == "a" else "open_laptop"} for c in "abc"}
        rows = [{"case_id": c, "domain": "val", "task_success": c != "a", "instruction_success": c == "b"}
                for c in cases]
        result = summarize(cases, "val", rows)
        self.assertEqual((result["macro_TR"], result["macro_JR"]), (.5, .25))
        for invalid in (rows[:1], rows + [rows[0]], [{**row, "task_success": None} for row in rows]):
            with self.assertRaises(ValueError):
                summarize(cases, "val", invalid)


if __name__ == "__main__":
    unittest.main()
