"""Replay one public joint-action prediction and emit its Kit judge receipt."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from ..contracts import JOINT, ActionChunk
from ..robotwin import environment_factory
from ..suite import BUDGETS, load_suite
from . import judgment_record
from .spec import PROTOCOL, TASKS, task_name


class ContactRecorder:
    def __init__(self, episode, target, arm):
        self.episode = episode
        self.target, self.arm = target, arm
        env = episode.env
        self.links = {}
        for arm in ("left", "right"):
            names = set(getattr(env.robot, arm + "_fix_gripper_name"))
            names.update(
                joint.child_link.get_name()
                for joint, _, _ in getattr(env.robot, arm + "_gripper")
            )
            self.links.update(dict.fromkeys(names, arm))
        self.samples = []
        self.capture(0)

    def capture(self, step):
        arms = set()
        for contact in self.episode.env.scene.get_contacts():
            names = [body.entity.name for body in contact.bodies]
            if self.target in names and contact.points:
                arms.update(self.links[name] for name in names if name in self.links)
        self.samples.append(
            {
                "physics_step": step,
                "time_sec": step * self.episode.dt,
                "target_contacts": [{"arm": arm} for arm in sorted(arms)],
            }
        )

    def evidence(self, checks):
        return {"native_final": {"checks": checks}, "samples": self.samples,
                "control_spec": {"active_arm": self.arm}, "dt": self.episode.dt}


class BottleRecorder(ContactRecorder):
    def __init__(self, episode, instruction):
        match = re.fullmatch(r"Move (bottle_(?:left|right)) to (goal_(?:left|right)) using the (left|right) arm\.", instruction)
        if not match:
            raise ValueError("Unknown Place Bottle instruction")
        target, self.goal, arm = match.groups()
        super().__init__(episode, target, arm)
        self.initial_z = {name: self.center(name)[2] for name in episode.env.objects}

    def center(self, name):
        return np.asarray(self.episode.env.objects[name].get_functional_point(0, "pose").p, dtype=np.float64)

    def finish(self):
        env = self.episode.env
        centers = {name: self.center(name) for name in env.objects}
        upright = {
            name: bool(center[2] >= self.initial_z[name] - 0.02)
            for name, center in centers.items()
        }
        passive = "bottle_left" if self.target == "bottle_right" else "bottle_right"
        offset = centers[self.target][:2] - env.goals[self.goal].get_pose().p[:2]
        checks = {
            "target_center_inside_goal": bool(np.all(np.abs(offset) <= [0.045, 0.04])),
            "target_bottle_upright": upright[self.target],
            "passive_bottle_upright": upright[passive],
            "active_gripper_open": bool(
                getattr(env.robot, f"is_{self.arm}_gripper_open")()
            ),
        }
        return self.evidence(checks)


class FanRecorder(ContactRecorder):
    DIRECTIONS = ("front", "front right", "right", "back right", "back", "back left", "left", "front left")

    def __init__(self, episode, instruction):
        match = re.fullmatch(r"Use the (left|right) arm to place the fan facing ([a-z ]+) on the pad\.", instruction)
        if not match or match[2] not in self.DIRECTIONS:
            raise ValueError("Unknown Orient Fan instruction")
        super().__init__(episode, episode.env.fan.get_name(), match[1])
        self.heading = 45 * self.DIRECTIONS.index(match[2])
        self.pad = episode.env.pad.get_pose().p.copy()

    def finish(self):
        from transforms3d.quaternions import quat2mat

        env = self.episode.env
        pose = env.fan.get_pose()
        rotation = quat2mat(np.asarray(pose.q, dtype=np.float64))
        x, y = rotation[:2, 2]
        heading = np.rad2deg(np.arctan2(x, y))
        error = abs((heading - self.heading + 180) % 360 - 180)
        return self.evidence({
            "position": bool(np.all(np.abs(pose.p - self.pad) < 0.04)),
            "orientation": bool(np.hypot(x, y) > 1e-8 and error <= 10),
            "upright": bool(np.rad2deg(np.arccos(np.clip(rotation[2, 1], -1, 1))) <= 10),
            "left_gripper_open": bool(env.robot.is_left_gripper_open()),
            "right_gripper_open": bool(env.robot.is_right_gripper_open()),
        })


RECORDERS = {"place_bottle": BottleRecorder, "place_fan": FanRecorder}


def replay(raw, data, robotwin):
    row = json.loads(raw)
    domain = row["domain"]
    scenes, cases = load_suite(data, domain, verify_release=True)
    case = cases[row["case_id"]]
    if case["task"] != row["task"] or row["action_spec"] != JOINT.name:
        raise ValueError("Task or action specification mismatch")
    if case["task"] not in RECORDERS:
        raise NotImplementedError(f"No prediction recorder for {case['task']}")
    actions = ActionChunk(np.asarray(row["actions"]), row["action_spec"]).checked(
        JOINT.name
    )
    factory = environment_factory(robotwin)
    with contextlib.closing(
        factory(scenes[case["task"], case["scene_id"]], case, domain)
    ) as episode:
        limit = min(BUDGETS[case["task"]], episode.step_limit)
        if len(actions) != limit or row["action_count"] != limit:
            raise ValueError("Prediction does not reach the native endpoint")
        recorder = RECORDERS[case["task"]](episode, case["instruction"])
        episode.clock.on_step = recorder.capture
        for index, action in enumerate(actions):
            if index % 20 == 0:
                episode.observe()
            previous = episode.physics_steps
            episode.execute(action, JOINT.name)
            if episode.action_steps != index + 1 or episode.physics_steps <= previous:
                raise RuntimeError("Replay clock did not advance")
        previous = episode.physics_steps
        episode.settle(250)
        if (
            episode.physics_steps != previous + 250
            or len(recorder.samples) != episode.physics_steps + 1
        ):
            raise RuntimeError("Incomplete physics evidence")
        evidence = recorder.finish()
    result = judgment_record(
        {
            **evidence,
            "task": case["task"],
            "case_id": case["case_id"],
            "domain": domain,
            "protocol": PROTOCOL,
            "metric_spec_hash": hashlib.sha256(
                Path(TASKS[task_name(case["task"])].__file__).read_bytes()
            ).hexdigest(),
        }
    )
    return {**result, "prediction_sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--robotwin", required=True, type=Path)
    parser.add_argument("prediction", type=Path)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        result = replay(args.prediction.read_bytes(), args.data, args.robotwin)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
