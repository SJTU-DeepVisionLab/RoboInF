import unittest

from roboinf_benchmark.checker import score_episode
from roboinf_benchmark.checker.scoring import group_checks
from roboinf_benchmark.checker.spec import PROTOCOL, TASK_CHECKS, task_name
from roboinf_benchmark.suite import BUDGETS


class CheckerTaskTests(unittest.TestCase):
    def test_each_released_task_has_one_checker(self):
        self.assertEqual({task_name(key) for key in BUDGETS}, set(TASK_CHECKS))
        self.assertEqual(len(TASK_CHECKS), 8)
        for name, (tr, jr) in TASK_CHECKS.items():
            with self.subTest(task=name):
                self.assertTrue(tr and jr)
                checks = dict.fromkeys(tr + jr, True)
                self.assertTrue(group_checks(name, checks)["control_success"])
                for key in tr + jr:
                    result = group_checks(name, {**checks, key: False})
                    self.assertEqual(result["task_success"], key not in tr)
                    self.assertFalse(result["control_success"])

    def test_sorting_names_score_the_same_record(self):
        name = "sort_blocks_in_order"
        tr, _ = TASK_CHECKS[name]
        evidence = {
            "protocol": PROTOCOL,
            "dt": 0.01,
            "metric_spec_hash": "fixture",
            "control_spec": {"stack_order": ["red", "blue", "green"]},
            "native_final": {"checks": dict.fromkeys(tr, True)},
            "samples": [{"physics_step": 0, "time_sec": 0}],
            "placement_events": [{"colors": [c]} for c in ("red", "blue", "green")],
        }
        expected = score_episode({**evidence, "task": name})
        self.assertTrue(expected["control_success"])
        for key in ("stack_blocks", "sort_blocks_three_in_order"):
            self.assertEqual(score_episode({**evidence, "task": key}), expected)
