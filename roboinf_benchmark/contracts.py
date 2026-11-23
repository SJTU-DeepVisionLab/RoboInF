from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    name: str
    dimension: int


JOINT = ActionSpec("joint14_absolute", 14)
EEF = ActionSpec("eef16_absolute_wxyz", 16)
ACTION_SPECS = {spec.name: spec for spec in (JOINT, EEF)}
CAMERAS = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class PolicyDescription:
    method_id: str
    action_spec: str
    input_mode: str = "task_instruction"


@dataclass(frozen=True)
class EpisodeContext:
    instruction: str
    seed: int


@dataclass(frozen=True)
class Observation:
    images: Mapping[str, np.ndarray]
    joint_state: np.ndarray
    eef_state: np.ndarray | None
    action_step: int
    time_sec: float

    def checked_copy(self):
        if set(self.images) != set(CAMERAS):
            raise ValueError("Expected head, left_wrist and right_wrist cameras")
        images = {}
        for name in CAMERAS:
            frame = np.asarray(self.images[name])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape) < 1:
                raise ValueError(f"{name}: expected HWC uint8 RGB")
            images[name] = frame.copy()
        joint = np.asarray(self.joint_state, dtype=np.float32)
        eef = None if self.eef_state is None else np.asarray(self.eef_state, dtype=np.float32)
        if joint.shape != (14,) or not np.isfinite(joint).all():
            raise ValueError("Expected 14 finite joint coordinates")
        if eef is not None:
            if eef.shape != (16,) or not np.isfinite(eef).all():
                raise ValueError("Expected 16 finite EEF coordinates")
            _unit_quaternions(eef)
        if type(self.action_step) is not int or self.action_step < 0 or not isfinite(self.time_sec) or self.time_sec < 0:
            raise ValueError("Invalid observation clock")
        return Observation(images, joint.copy(), None if eef is None else eef.copy(), self.action_step, self.time_sec)


def _unit_quaternions(actions):
    for start in (3, 11):
        if not np.allclose(np.linalg.norm(actions[..., start:start+4], axis=-1), 1.0, rtol=0.0, atol=1e-3):
            raise ValueError("Expected unit wxyz EEF quaternions")


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    action_spec: str

    def checked(self, expected: str):
        array = np.asarray(self.actions)
        if self.action_spec != expected or expected not in ACTION_SPECS:
            raise ValueError("Action specification mismatch")
        if (array.ndim != 2 or array.shape[1] != ACTION_SPECS[expected].dimension
                or not 1 <= len(array) <= 2048 or array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
            raise ValueError("Expected a nonempty finite [horizon, action dimension] array")
        if expected == EEF.name:
            _unit_quaternions(array)
        return array.copy()
