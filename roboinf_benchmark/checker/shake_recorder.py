"""Measure held-bottle motion in the post-grasp window and passive tail."""

from dataclasses import asdict
import re

import numpy as np

from .physical import PhysicsRecorder
from .switch_recorder import descendants
from .tasks.shake_bottle import ShakeConfig


class ShakeRecorder(PhysicsRecorder):
    def __init__(self, episode, instruction):
        super().__init__(episode)
        match = re.fullmatch(
            r"Use the (left|right) arm to hold the bottle (horizontal|vertical)ly and shake it (left and right three times|with three small wrist rotations around the horizontal grasp axis)\.",
            instruction,
        )
        if not match:
            raise ValueError("Unknown Shake Bottle instruction")
        self.arm, self.pose = match[1], match[2]
        self.lateral = match[3].startswith("left")
        self.config = ShakeConfig()
        self.inactive = "right" if self.arm == "left" else "left"
        for arm in ("left", "right"):
            entity = getattr(self.robot, arm + "_entity")
            hand = {name for name, side in self.grippers.items() if side == arm}
            hand.add(getattr(self.robot, arm + "_ee").child_link.get_name())
            for name in descendants(entity, hand):
                self.grippers.setdefault(name, arm)
        self.other = getattr(self.robot, self.inactive + "_entity")
        joints = list(getattr(self.robot, self.inactive + "_arm_joints"))
        joints.extend(
            j
            for j, _, _ in getattr(self.robot, self.inactive + "_gripper")
            if j is not None
        )
        active = list(self.other.get_active_joints())
        self.indices = [active.index(j) for j in joints if j in active]
        self.initial = self.other.get_qpos()[self.indices]
        self.origin = None
        self.capture(0)

    def capture(self, step):
        from transforms3d.quaternions import qinverse, qmult, quat2mat

        pose = self.env.bottle.get_pose()
        q = np.asarray(pose.q, dtype=float)
        p = np.asarray(pose.p, dtype=float)
        tcp = np.asarray(
            getattr(self.robot, f"get_{self.arm}_tcp_pose")()[:3], dtype=float
        )
        tilt = np.rad2deg(np.arccos(np.clip(abs(quat2mat(q)[2, 1]), 0, 1)))
        error = tilt if self.pose == "vertical" else abs(90 - tilt)
        height = p[2] - self.env.table_z
        arms = set()
        for contact in self.env.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            if "bottle" in names and contact.points:
                arms.update(self.grippers[n] for n in names & self.grippers.keys())
        holding = {a for a in arms if getattr(self.robot, f"is_{a}_gripper_close")()}
        moving = self.origin is not None
        axis = None
        scalar = angle = center_drift = tcp_drift = 0.0
        if moving:
            p0, q0, tcp0, axis = self.origin
            relative = qmult(q, qinverse(q0))
            if relative[0] < 0:
                relative = -relative
            angle = np.rad2deg(
                2
                * np.arccos(
                    np.clip(
                        abs(q @ q0) / (np.linalg.norm(q) * np.linalg.norm(q0)), 0, 1
                    )
                )
            )
            scalar = (
                (p - p0) @ axis
                if self.lateral
                else np.rad2deg(2 * np.arccos(np.clip(relative[0], -1, 1)))
                * (1 if relative[1:] @ axis >= 0 else -1)
            )
            center_drift = np.linalg.norm(p - p0)
            tcp_drift = np.linalg.norm(tcp - tcp0)
        drift = np.abs(self.other.get_qpos()[self.indices] - self.initial)
        native = {
            "sim_step": step,
            "phase": "settle" if self.passive else "shake" if moving else "initialized",
            "pose_check_enabled": moving,
            "trace_check_enabled": moving and not self.passive,
            "pose_error_deg": float(error),
            "trace_scalar": float(scalar),
            "trace_orientation_error_deg": float(angle),
            "trace_axis_world": axis.tolist() if moving else None,
            "tcp_position_drift_m": float(tcp_drift),
            "bottle_center_drift_m": float(center_drift),
            "active_arm_contact": self.arm in arms,
            "inactive_arm_contact": self.inactive in arms,
            "active_gripper_closed": bool(
                getattr(self.robot, f"is_{self.arm}_gripper_close")()
            ),
            "bottle_height_above_table_m": float(height),
            "contacts": [{"arm": a} for a in sorted(arms)],
            "inactive_arm_drift": float(drift.max()),
            "inactive_arm_joint_drift": drift.tolist(),
        }
        self.sample(step, native_sample=native, holding_arms=sorted(holding))
        pose_limit = (
            max(self.config.pose_tolerance_deg, 25.0)
            if self.lateral
            else self.config.pose_tolerance_deg
        )
        if (
            step
            and not self.passive
            and not moving
            and self.arm in holding
            and height >= self.config.min_lift_clearance_m
            and error <= pose_limit
        ):
            axis = np.array(self.config.translation_axis_world, dtype=float)
            if not self.lateral:
                wrist = getattr(self.robot, self.arm + "_arm_joints")[-1]
                frame = (
                    wrist.get_global_pose().q
                    if hasattr(wrist, "get_global_pose")
                    else getattr(self.robot, f"get_{self.arm}_ee_pose")()[3:]
                )
                axis = quat2mat(frame)[:, 0]
            self.origin = (p, q, tcp, axis / np.linalg.norm(axis))

    def finish(self):
        return self.evidence(
            task_config=asdict(self.config),
            control_spec={
                "target_object": "bottle",
                "active_arm": self.arm,
                "pose": self.pose,
                "trace": "lateral_translation" if self.lateral else "wrist_rotation",
            },
        )
