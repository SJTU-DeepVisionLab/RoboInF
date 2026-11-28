"""Physics samples shared by Select Switch and Press Buttons in Order."""

from dataclasses import asdict
import re

import numpy as np

from .tasks.press_buttons_in_order import validate_control_spec as button_spec
from .tasks.select_switch import POSITIONS, RowConfig


def controls(instruction):
    match = re.fullmatch(r"Turn on the (left|middle|right) switch using the (left|right) arm\.", instruction)
    if match:
        return {"target_switch": match[1], "active_arm": match[2]}
    match = re.fullmatch(r"Using the (left|right) arm, press the (\w+), (\w+), and (\w+) buttons in that order\.", instruction)
    if match:
        return button_spec({"arm": match[1], "press_order": match.groups()[1:]})
    raise ValueError("Unknown switch row instruction")


def descendants(entity, names):
    joints = entity.get_joints()
    for _ in joints:
        for joint in joints:
            if joint.parent_link and joint.parent_link.get_name() in names:
                names.add(joint.child_link.get_name())
    return names


class SwitchRecorder:
    def __init__(self, episode, instruction):
        self.episode = episode
        self.spec = controls(instruction)
        self.arm = self.spec.get("arm") or self.spec["active_arm"]
        self.config = RowConfig(**vars(episode.env.row_config))
        robot = episode.env.robot
        self.switches = {p: episode.env.objects[f"switch_{p}"] for p in POSITIONS}
        self.switch_links = {link.entity.per_scene_id: p for p, obj in self.switches.items()
                             for link in obj.actor.get_links()}
        self.robot_links = {}
        self.hand = []
        for arm in ("left", "right"):
            entity = getattr(robot, arm + "_entity")
            joints = list(getattr(robot, arm + "_arm_joints"))
            fingers = [j for j, _, _ in getattr(robot, arm + "_gripper") if j is not None]
            hand = {j.child_link.get_name() for j in fingers}
            hand.update(getattr(robot, arm + "_fix_gripper_name"))
            hand.add(getattr(robot, arm + "_ee").child_link.get_name())
            hand = descendants(entity, hand)
            limb = descendants(entity, {j.child_link.get_name() for j in joints})
            for link in entity.get_links():
                name = link.get_name()
                if name not in limb | hand:
                    continue
                region = "hand" if name in hand else "arm"
                self.robot_links[link.entity.per_scene_id] = {"arm": arm, "region": region}
                if arm == self.arm and region == "hand":
                    self.hand.append(link)
            if arm != self.arm:
                active = list(entity.get_active_joints())
                self.inactive = entity
                self.indices = [active.index(j) for j in joints + fingers]
                self.initial = entity.get_qpos()[self.indices]
        self.samples = []
        self.capture(0)

    def capture(self, step):
        env = self.episode.env
        tcp = getattr(env.robot, f"get_{self.arm}_tcp_pose")()[:3]
        hand = np.array([tcp, *(link.get_pose().p for link in self.hand)])
        points, clearance, progress = {}, {}, []
        for p, obj in self.switches.items():
            points[p] = obj.get_contact_point(self.config.contact_point_id, "pose").p
            clearance[p] = float(np.linalg.norm(hand - points[p], axis=1).min())
            lo, hi = obj.get_qlimits()[0]
            if not hi > lo:
                raise ValueError("Invalid switch joint limits")
            fraction = float((float(obj.get_qpos()[0]) - float(lo)) / (float(hi) - float(lo)))
            progress.append(fraction if self.config.on_at_upper_limit else 1 - fraction)
        events, collisions = [], []
        for contact in env.scene.get_contacts():
            ids = [body.entity.per_scene_id for body in contact.bodies]
            robots = [self.robot_links.get(i) for i in ids]
            if not any(robots):
                continue
            for pt in contact.points:
                if not (pt.separation <= 0 or np.linalg.norm(pt.impulse) > 1e-8):
                    continue
                if all(robots) and robots[0]["arm"] != robots[1]["arm"]:
                    collisions.append({"actual": True, "robot_links": robots})
                for i, robot in enumerate(robots):
                    switch = self.switch_links.get(ids[1-i])
                    if robot and switch:
                        events.append({"switch": switch, "arm": robot["arm"], "robot_region": robot["region"],
                                       "distance_m": float(np.linalg.norm(pt.position - points[switch]))})
        drift = np.abs(self.inactive.get_qpos()[self.indices] - self.initial)
        sample = {"sim_step": step, "progress": progress, "contacts": events, "raw_contacts": collisions,
                  "inactive_arm_drift": float(drift.max()), "inactive_arm_joint_drift": drift.tolist(),
                  "switch_clearance_m": clearance}
        if "target_switch" in self.spec:
            sample["target_clearance_m"] = clearance[self.spec["target_switch"]]
        self.samples.append({"physics_step": step, "time_sec": step * self.episode.dt, "native_sample": sample})

    def finish(self):
        return {"samples": self.samples, "dt": self.episode.dt, "control_spec": self.spec,
                "task_config": asdict(self.config)}
