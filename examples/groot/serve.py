"""Serve a locally provisioned RoboInF GR00T checkpoint for one active episode."""

import argparse

import numpy as np


class RoboInFEndpoint:
    """Translate the released client wire format into the training modalities."""

    def __init__(self, policy, seed):
        self.policy = policy
        self.seed = seed

    def reset(self, options=None):
        import torch

        seed = (options or {}).get("policy_seed", self.seed)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Expected a uint32 policy seed")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return {"policy_seed": seed}

    def get_action(self, observation, options=None):
        video = observation["video"]
        state = {}
        for side in ("left", "right"):
            group = np.asarray(observation["state"][side + "_arm"])
            if group.shape != (1, 1, 7):
                raise ValueError(f"Expected one {side} arm observation (six joints and gripper)")
            state[side + "_arm"] = group[..., :6]
            state[side + "_gripper"] = group[..., 6:]
        actions, info = self.policy.get_action({
            "video": {"head": video["front"], "left_wrist": video["left_wrist"],
                      "right_wrist": video["right_wrist"]},
            "state": state, "language": observation["language"],
        })
        result = {}
        for side in ("left", "right"):
            joints, grip = actions[side + "_arm"], actions[side + "_gripper"]
            if joints.shape != (1, 40, 6) or grip.shape != (1, 40, 1):
                raise ValueError("Expected 40 decoded absolute targets")
            result[side + "_arm"] = np.concatenate((joints, grip), axis=-1)
        return result, info


def main():
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, args.checkpoint, device="cuda:0", strict=True)
    PolicyServer.start_server(RoboInFEndpoint(policy, args.seed), port=args.port, host="127.0.0.1")


if __name__ == "__main__":
    main()
