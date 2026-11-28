import unittest
from unittest.mock import Mock

import numpy as np

from examples.groot.serve import RoboInFEndpoint


class EndpointTest(unittest.TestCase):
    def test_state_packing_and_absolute_targets(self):
        policy = Mock()
        actions = {side + key: np.full((1, 40, n), i + offset)
                   for i, side in enumerate(("left", "right"))
                   for key, n, offset in (("_arm", 6, 0), ("_gripper", 1, 2))}
        policy.get_action.return_value = actions, {}
        joints = np.arange(14).reshape(1, 1, 14)
        video = {key: object() for key in ("front", "left_wrist", "right_wrist")}
        obs = {"video": video, "language": {"annotation.human.task_description": [["open"]]},
               "state": {"left_arm": joints[..., :7], "right_arm": joints[..., 7:]}}
        packed, _ = RoboInFEndpoint(policy, 0).get_action(obs)
        transformed = policy.get_action.call_args.args[0]
        self.assertIs(transformed["video"]["head"], video["front"])
        self.assertEqual(set(transformed["state"]), set(actions))
        np.testing.assert_array_equal(transformed["state"]["right_gripper"], joints[..., 13:])
        np.testing.assert_array_equal(packed["left_arm"][0, 0], [0] * 6 + [2])
        np.testing.assert_array_equal(packed["right_arm"][0, 0], [1] * 6 + [3])
