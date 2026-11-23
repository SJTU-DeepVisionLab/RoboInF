from collections import deque

import numpy as np

from ..contracts import ActionChunk, EEF, JOINT, PolicyDescription


class _Adapter:
    def __init__(self, name, spec, client):
        self.client = client
        self.description = PolicyDescription(name, spec.name)
        self.instruction = None

    def describe(self):
        return self.description

    def reset(self, context):
        if not isinstance(context.instruction, str) or not context.instruction.strip():
            raise ValueError("An instruction is required")
        self.instruction = context.instruction

    def predict(self, obs):
        if self.instruction is None:
            raise ValueError("Call reset before predict")
        chunk = ActionChunk(np.asarray(self._predict(obs)), self.description.action_spec)
        return ActionChunk(chunk.checked(self.description.action_spec), chunk.action_spec)

    def close(self):
        self.instruction = None
        close = getattr(self.client, "close", None)
        if close is not None:
            close()


class Pi05(_Adapter):
    def __init__(self, client):
        super().__init__("pi_05", JOINT, client)

    def _predict(self, obs):
        names = {"cam_high": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
        images = {key: np.ascontiguousarray(obs.images[view].transpose(2, 0, 1))
                  for key, view in names.items()}
        result = self.client.infer({"images": images, "state": obs.joint_state, "prompt": self.instruction})
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 14):
            raise ValueError("Expected 50 by 14 denormalized absolute ALOHA joint commands")
        return actions.astype(np.float32)


class Gr00t(_Adapter):
    def __init__(self, client):
        super().__init__("gr00t", JOINT, client)

    def reset(self, context):
        super().reset(context)
        self.client.reset()

    def _predict(self, obs):
        views = {"front": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
        observation = {
            "video": {key: obs.images[view][None, None, ...] for key, view in views.items()},
            "state": {"left_arm": obs.joint_state[None, None, :7],
                      "right_arm": obs.joint_state[None, None, 7:]},
            "language": {"annotation.human.task_description": [[self.instruction]]},
        }
        actions, _ = self.client.get_action(observation)
        left, right = (np.asarray(actions[key]) for key in ("left_arm", "right_arm"))
        if left.shape != right.shape or left.ndim != 3 or left.shape[0] != 1 or left.shape[2] != 7:
            raise ValueError("Expected GR00T left_arm and right_arm with shape (1, horizon, 7)")
        return np.concatenate([left[0], right[0]], axis=1)


class OpenWAM(_Adapter):
    def __init__(self, client, spec, conversion, layout="physical"):
        super().__init__("openwam", spec, client)
        self.eef = spec == EEF
        self.conversion = conversion
        self.physical = layout == "physical"

    def reset(self, context):
        super().reset(context)
        self.client.reset(self.instruction)

    def _predict(self, obs):
        state = obs.joint_state
        if self.eef:
            e = obs.eef_state
            if e is None:
                raise ValueError("OpenWAM EEF requires measured end-effector poses")
            order = [0, 1, 2, 4, 5, 6, 3] if self.physical else list(range(7))
            state = self.conversion.robotwin_endpose_to_eef20d(e[:7][order], e[8:15][order], e[7], e[15])
        action = self.client.step({
            "cams": {"head": obs.images["head"], "left": obs.images["left_wrist"],
                     "right": obs.images["right_wrist"]},
            "lang": self.instruction, "state": state,
        })
        if self.eef:
            if np.shape(action) != (20,):
                raise ValueError("OpenWAM EEF requires a 20-dimensional native response")
            action = self.conversion.eef20d_to_ee16d(action)
            if self.physical:
                action = np.asarray(action)[[0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 14, 11, 12, 13, 15]]
        return np.asarray(action)[None, :]


class LingBotVA(_Adapter):
    def __init__(self, client):
        super().__init__("lingbot-va", EEF, client)

    def reset(self, context):
        super().reset(context)
        self.client.infer({"reset": True, "prompt": self.instruction})
        self.anchor = None
        self.frames = []
        self.pending = deque()
        self.issued = 0

    def _predict(self, obs):
        names = {"cam_high": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
        view = {"observation.images." + k: obs.images[v] for k, v in names.items()}
        view.update({"observation.state": obs.joint_state, "task": self.instruction})
        first = self.anchor is None
        if first:
            if obs.eef_state is None:
                raise ValueError("LingBot-VA requires the measured initial EEF pose")
            self.anchor = obs.eef_state.copy()
            self.first_view = view
            self.next_step = obs.action_step
        if obs.action_step != self.next_step:
            raise ValueError("LingBot-VA must receive each executed step in order")
        if not first and self.issued % self.stride == 0:
            self.frames.append(view)
        if not self.pending:
            if not first:
                self.client.infer({"obs": self.frames, "state": self.native, "compute_kv_cache": True,
                                   "imagine": False, "save_visualization": False})
            result = self.client.infer({"obs": self.first_view, "prompt": self.instruction,
                                        "save_visualization": False})
            self.native = np.asarray(result["action"])
            if (self.native.ndim != 3 or self.native.shape[0] != 16 or self.native.shape[1] <= int(first)
                    or self.native.shape[2] < 4 or self.native.shape[2] % 4
                    or self.native.dtype.kind not in "fiu" or not np.isfinite(self.native).all()):
                raise ValueError("Expected LingBot-VA Robotwin 16-channel [D, frames, actions] output")
            self.stride = self.native.shape[2] // 4
            steps = self.native[:, int(first):, :].transpose(1, 2, 0).reshape(-1, 16)
            actions = steps.astype(np.float64, copy=True)
            for start in (0, 8):
                actions[:, start:start+3] += self.anchor[start:start+3]
                q = slice(start+3, start+7)
                # The upstream Robotwin checkpoint encodes relative poses from the stored
                # quaternion components with scipy's xyzw convention, without reordering.
                delta = steps[:, q]
                norm = np.linalg.norm(delta, axis=1, keepdims=True)
                if (norm < 1e-8).any():
                    raise ValueError("LingBot-VA returned a zero quaternion")
                vector, scalar = (delta / norm)[:, :3], (delta / norm)[:, 3:4]
                xyz, w = self.anchor[q][:3], self.anchor[q][3]
                composed = np.column_stack((w * vector + scalar * xyz + np.cross(xyz, vector),
                                            w * scalar[:, 0] - vector @ xyz))
                actions[:, q] = composed / np.linalg.norm(composed, axis=1, keepdims=True)
            self.pending.extend(actions)
            self.frames = []
            self.issued = 0
        self.issued += 1
        self.next_step += 1
        return self.pending.popleft()[None, :]
