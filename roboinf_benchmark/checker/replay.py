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
from .spec import PROTOCOL


class BottleRecorder:
    def __init__(self, episode, instruction):
        match = re.fullmatch(
            r"Move (bottle_(?:left|right)) to (goal_(?:left|right)) using the (left|right) arm\.",
            instruction,
        )
        if not match:
            raise ValueError("Unknown Place Bottle instruction")
        self.target, self.goal, self.arm = match.groups()
        self.episode = episode
        env = episode.env
        self.initial_z = {name: self.center(name)[2] for name in env.objects}
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

    def center(self, name):
        return np.asarray(
            self.episode.env.objects[name].get_functional_point(0, "pose").p,
            dtype=np.float64,
        )

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
        return {
            "native_final": {"checks": checks},
            "samples": self.samples,
            "control_spec": {"active_arm": self.arm},
            "dt": self.episode.dt,
        }


def replay(raw, data, robotwin):
    row = json.loads(raw)
    domain = row["domain"]
    scenes, cases = load_suite(data, domain, verify_release=True)
    case = cases[row["case_id"]]
    if case["task"] != row["task"] or row["action_spec"] != JOINT.name:
        raise ValueError("Task or action specification mismatch")
    if case["task"] != "place_bottle":
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
        recorder = BottleRecorder(episode, case["instruction"])
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
                Path(__file__)
                .with_name("tasks")
                .joinpath("place_bottle.py")
                .read_bytes()
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
