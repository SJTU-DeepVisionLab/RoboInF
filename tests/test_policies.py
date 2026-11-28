import unittest
from unittest.mock import Mock

import numpy as np

from roboinf_benchmark.contracts import EEF, EpisodeContext, Observation
from roboinf_benchmark.policies import POLICY_NAMES
from roboinf_benchmark.policies.adapters import Gr00t, LingBotVA, OpenWAM, Pi05


def observation(step=0):
    images = {name: np.full((4, 5, 3), step + i, np.uint8)
              for i, name in enumerate(("head", "left_wrist", "right_wrist"))}
    pose = np.zeros(16, np.float32)
    pose[[0, 8]] = [1, 2]
    pose[[3, 11, 6, 14]] = 2**-0.5
    return Observation(images, np.arange(14, dtype=np.float32), pose, step, step / 30)


class PolicyTest(unittest.TestCase):
    def test_pi05_camera_layout_and_gr00t_arm_batches(self):
        self.assertEqual(len(set(POLICY_NAMES)), 4)
        ctx, obs = EpisodeContext("press the blue button", 1), observation()
        pi_client = Mock()
        pi_client.infer.return_value = {"actions": np.zeros((50, 14))}
        pi = Pi05(pi_client)
        pi.reset(ctx)
        self.assertEqual(pi.predict(obs).actions.shape, (50, 14))
        request = pi_client.infer.call_args.args[0]
        self.assertEqual(set(request), {"prompt", "state", "images"})
        self.assertEqual(request["images"]["cam_high"].shape, (3, 4, 5))
        np.testing.assert_array_equal(request["images"]["cam_right_wrist"], 2)

        gr_client = Mock()
        gr_client.reset.return_value = {"policy_seed": ctx.seed}
        gr_client.get_action.return_value = ({"left_arm": np.ones((1, 2, 7)),
                                              "right_arm": np.zeros((1, 2, 7))}, {})
        gr = Gr00t(gr_client)
        gr.reset(ctx)
        gr_client.reset.assert_called_once_with(options={"policy_seed": ctx.seed})
        actions = gr.predict(obs).actions
        np.testing.assert_array_equal(actions[:, :7], 1)
        np.testing.assert_array_equal(actions[:, 7:], 0)
        packet = gr_client.get_action.call_args.args[0]
        self.assertEqual(packet["video"]["front"].shape, (1, 1, 4, 5, 3))
        self.assertEqual(packet["language"]["annotation.human.task_description"], [[ctx.instruction]])
        np.testing.assert_array_equal(packet["state"]["right_arm"][0, 0], np.arange(7, 14))
        gr_client.reset.return_value = {"policy_seed": 0}
        with self.assertRaisesRegex(ValueError, "episode seed"):
            gr.reset(ctx)

    def test_openwam_quaternion_order_is_explicit(self):
        obs = observation()
        conv = Mock()
        client = Mock()
        client.step.return_value = np.zeros(20)
        native = np.zeros(16, dtype=np.float32)
        native[[6, 12]] = 1
        conv.eef20d_to_ee16d.return_value = native
        model = OpenWAM(client, EEF, conv)
        model.reset(EpisodeContext("open the laptop", 2))
        action = model.predict(obs).actions[0]
        self.assertEqual(action[3], 1)
        self.assertEqual(action[13], 1)
        pose = conv.robotwin_endpose_to_eef20d.call_args.args[0]
        np.testing.assert_allclose(pose[3:], obs.eef_state[[4, 5, 6, 3]])
        legacy = OpenWAM(client, EEF, conv, "legacy_wxyz_values_as_xyzw")
        legacy.reset(EpisodeContext("open the laptop", 3))
        np.testing.assert_array_equal(legacy.predict(obs).actions[0], native)
        np.testing.assert_array_equal(conv.robotwin_endpose_to_eef20d.call_args.args[0], obs.eef_state[:7])

    def test_lingbot_cache_contains_only_executed_frames(self):
        native = np.zeros((16, 2, 4), np.float32)
        native[[6, 14]] = 1
        native[0, 0] = 99
        native[0, 1] = 0.25
        calls = []

        def infer(msg):
            calls.append(msg)
            return {"action": native}

        client = Mock(infer=infer)
        policy = LingBotVA(client)
        policy.reset(EpisodeContext("orient the fan", 3))
        for step in range(4):
            action = policy.predict(observation(step)).actions[0]
            self.assertAlmostEqual(action[0], 1.25)
            np.testing.assert_allclose(action[3:7], observation().eef_state[3:7])
        self.assertEqual(len(calls), 2)
        policy.predict(observation(4))
        self.assertEqual(len(calls), 4)
        self.assertEqual([v["observation.images.cam_high"][0, 0, 0] for v in calls[2]["obs"]], [1, 2, 3, 4])
        self.assertIs(calls[2]["state"], native)
        with self.assertRaisesRegex(ValueError, "executed step"):
            policy.predict(observation(8))
        policy.reset(EpisodeContext("another instruction", 4))
        self.assertAlmostEqual(policy.predict(observation()).actions[0, 0], 1.25)
