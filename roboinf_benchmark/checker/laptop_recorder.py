"""Measure hinge angle, stability and the active gripper's lid contact patch."""

import re

import numpy as np

from .physical import PhysicsRecorder


class LaptopRecorder(PhysicsRecorder):
    OFFSETS = {"left": -0.045, "center": 0.0, "right": 0.045}

    def __init__(self, episode, instruction):
        from transforms3d.quaternions import quat2mat

        super().__init__(episode)
        match = re.fullmatch(
            r"Open the laptop to (45|90|135) degrees by contacting the (left|center|right) region of the lid\.",
            instruction,
        )
        if not match:
            raise ValueError("Unknown Open Laptop instruction")
        self.angle, self.region = int(match[1]), match[2]
        self.laptop = self.env.laptop
        self.arm = "left" if quat2mat(self.laptop.get_pose().q)[0, 0] > 0 else "right"
        self.lid = self.laptop.config["contact_points"][1]["base"]
        self.links = {link.get_name() for link in self.laptop.actor.get_links()} | {
            "laptop"
        }
        (joint,) = self.laptop.actor.get_active_joints()
        # SAPIEN revolute joints rotate about local +X. Give the hinge a
        # consistent sign across scene yaw without perturbing the joint.
        axis = np.asarray(
            joint.get_global_pose().to_transformation_matrix()[:3, 0], dtype=float
        )
        self.axis = axis if axis[0] >= 0 else -axis
        self.arms = set()
        self.best_error, self.best_region = float("inf"), None
        self.capture(0)

    def capture(self, step):
        patch = {}
        for contact in self.env.scene.get_contacts():
            names = [body.entity.name for body in contact.bodies]
            for side, name in enumerate(names):
                other = names[1 - side]
                if (
                    name not in self.links
                    or other not in self.grippers
                    or not contact.points
                ):
                    continue
                arm = self.grippers[other]
                self.arms.add(arm)
                if name == self.lid and arm == self.arm:
                    for point in contact.points:
                        pos = np.asarray(point.position, dtype=float)
                        patch.setdefault((other, *np.round(pos, 5)), pos)
        if patch:
            center = self.laptop.get_contact_point(1, "matrix")[:3, 3]
            delta = np.mean(list(patch.values()), axis=0) - center
            error = np.linalg.norm(delta - self.OFFSETS[self.region] * self.axis)
            if error < self.best_error:
                offset = delta @ self.axis
                self.best_error = error
                self.best_region = min(
                    self.OFFSETS, key=lambda r: abs(offset - self.OFFSETS[r])
                )
        self.sample(step)

    def finish(self):
        q = float(self.laptop.get_qpos()[0])
        lo, hi = self.laptop.get_qlimits()[0]
        checks = {
            "target_angle_reached": bool(abs(np.rad2deg(q - lo) - self.angle) <= 8),
            "lid_stable": bool(abs(self.laptop.get_qvel()[0]) <= 0.25),
            "joint_within_limits": bool(lo - 1e-6 <= q <= hi + 1e-6),
            "active_arm_contact": self.arm in self.arms,
            "inactive_arm_no_contact": not (self.arms - {self.arm}),
            "requested_contact_region": bool(
                self.best_region == self.region and self.best_error <= 0.035
            ),
        }
        return self.evidence(
            assigned_arm=self.arm,
            control_spec={
                "target_angle_deg": self.angle,
                "contact_region": self.region,
            },
            native_final={"checks": checks},
        )
