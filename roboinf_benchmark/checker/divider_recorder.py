"""Observe the first physical panel crossing, collisions and terminal placement."""

import re

import numpy as np

from .physical import PhysicsRecorder


class DividerRecorder(PhysicsRecorder):
    def __init__(self, episode, instruction):
        super().__init__(episode)
        match = re.fullmatch(
            r"Move the beverage bottle to the target zone using the (left|right) arm, passing (?:directly over|around the (left|right) side of) the divider\.",
            instruction,
        )
        if not match:
            raise ValueError("Unknown Transport via Divider instruction")
        self.arm, self.route = match[1], match[2] or "direct_over"
        self.robot_links = {
            link.get_name()
            for entity in (self.robot.left_entity, self.robot.right_entity)
            for link in entity.get_links()
        }
        self.initial_z = float(self.env.payload.get_pose().p[2])
        self.upright_z = (
            float(self.env.payload.get_functional_point(0, "pose").p[2]) - 0.02
        )
        self.initial_divider = self.env.divider.get_pose().p.copy()
        self.observed_route = self.previous = None
        self.arms = set()
        self.collision = self.transport_started = False
        self.capture(0)

    def capture(self, step):
        env = self.env
        p = np.asarray(env.payload.get_pose().p, dtype=float)
        for contact in env.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            if "divider" in names and (names & self.robot_links or "payload" in names):
                self.collision = True
            if "payload" in names and contact.points:
                self.arms.update(self.grippers[n] for n in names & self.grippers.keys())
        if (
            not self.passive
            and not self.transport_started
            and step
            and self.arm in self.arms
            and p[2] - self.initial_z >= 0.8 * 0.10
        ):
            self.transport_started = True
            self.collision = False
        if self.previous is not None:
            delta = p - self.previous
            self.last_motion = np.linalg.norm(delta)
            if self.observed_route is None and abs(delta[1]) > 1e-12:
                fraction = (env.divider_y - self.previous[1]) / delta[1]
                if 0 <= fraction <= 1:
                    x, _, z = self.previous + fraction * delta
                    x -= env.scene_x
                    boundary = float(env.divider_half_size[0]) + 0.025 + 0.01
                    over_height = (
                        env.table_z + 2 * env.divider_half_size[2] + 0.15 - 0.005
                    )
                    self.observed_route = (
                        "left"
                        if x <= -boundary
                        else "right"
                        if x >= boundary
                        else "direct_over"
                        if abs(x) <= 0.05 and z >= over_height
                        else "invalid"
                    )
        self.previous = p
        self.sample(step)

    def finish(self):
        env = self.env
        center = env.payload.get_functional_point(0, "pose").p
        checks = {
            "payload_at_target": bool(
                np.all(
                    np.abs(center[:2] - [env.scene_x, env.target_y])
                    <= env.target_half_size[:2]
                )
            ),
            "payload_upright": bool(center[2] >= self.upright_z),
            "payload_stable": bool(self.last_motion <= 0.005),
            "both_grippers_open": bool(
                self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open()
            ),
            "requested_arm_contact": self.arm in self.arms,
            "inactive_arm_no_contact": not (self.arms - {self.arm}),
            "requested_route_followed": self.observed_route == self.route,
            "divider_unchanged": bool(
                np.linalg.norm(env.divider.get_pose().p - self.initial_divider) <= 1e-5
            ),
            "divider_collision_free": not self.collision,
        }
        return self.evidence(
            control_spec={"active_arm": self.arm, "route": self.route},
            native_final={"checks": checks},
        )
