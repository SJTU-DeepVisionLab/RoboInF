"""Measure placement order and terminal stability for Sort Blocks in Order."""

from collections import deque
import re

import numpy as np


class SortRecorder:
    def __init__(self, episode, instruction):
        self.episode = episode
        self.env = episode.env
        match = re.fullmatch(r"Sort the blocks in this order: (\w+), then (\w+), then (\w+)\.", instruction)
        if not match or set(match.groups()) != set(self.env.objects):
            raise ValueError("Unknown sort order")
        self.order = match.groups()
        self.robot_links = {link.name for entity in (self.env.robot.left_entity, self.env.robot.right_entity)
                            for link in entity.get_links()}
        self.resting_z = np.mean([b.get_pose().p[2] for b in self.env.objects.values()])
        self.samples, self.placements = [], []
        self.placed = set()
        self.stability = deque(maxlen=10)
        self.capture(0)

    def capture(self, step):
        matrices = np.array([b.get_pose().to_transformation_matrix() for b in self.env.objects.values()])
        self.in_bins = set()
        self.on_table = set()
        for color, matrix in zip(self.env.objects, matrices):
            p = matrix[:3, 3]
            if np.all(np.abs(p[:2] - self.env.sort_target_xy[color]) <= 0.015):
                self.in_bins.add(color)
            if abs(p[2] - self.resting_z) <= 0.012:
                self.on_table.add(color)
        region = self.in_bins & self.on_table
        touched = set()
        for contact in self.env.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            if contact.points and names & self.robot_links:
                touched.update(names & self.env.objects.keys())
        self.placed.intersection_update(region)
        entered = region - self.placed - touched
        if entered and step:
            self.placements.append({"physics_step": step, "colors": sorted(entered)})
        self.placed.update(entered)
        if step % 10 == 0:
            self.stability.append(matrices)
        self.samples.append({"physics_step": step, "time_sec": step * self.episode.dt})

    def finish(self):
        matrices = np.asarray(self.stability)
        delta = matrices - matrices[-1]
        robot = self.env.robot
        checks = {
            "blocks_in_matching_bins": self.in_bins == self.env.objects.keys(),
            "blocks_on_table": self.on_table == self.env.objects.keys(),
            "both_grippers_open": bool(robot.is_left_gripper_open() and robot.is_right_gripper_open()),
            "sort_stable": bool(len(matrices) == 10 and np.max(np.linalg.norm(delta[:, :, :3, 3], axis=-1)) <= .003
                                and np.max(np.abs(delta[:, :, :3, :3])) <= .03),
        }
        return {"samples": self.samples, "dt": self.episode.dt, "native_final": {"checks": checks},
                "control_spec": {"stack_order": self.order}, "placement_events": self.placements}
