"""Physical scene construction adapted from the MIT-licensed RoboTwin tasks."""

from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import sapien
import transforms3d as t3d
from envs.utils import (
    Actor,
    create_actor,
    create_box,
    create_sapien_urdf_obj,
    create_visual_box,
    rand_create_actor,
    rand_create_sapien_urdf_obj,
    rand_pose,
)
from envs._GLOBAL_CONFIGS import ROOT_PATH
from .switch_geometry import inspect_asset, rotation

POSITIONS = ("left", "middle", "right")
COLORS = ("red", "green", "blue")


def place_bottle(self):
    self.bottle_left = rand_create_actor(
        self,
        xlim=list(self.bottle_left_x_range),
        ylim=list(self.bottle_y_range),
        modelname="001_bottle",
        rotate_rand=True,
        rotate_lim=[0, 1, 0],
        qpos=[0.66, 0.66, -0.25, -0.25],
        convex=True,
        model_id=13,
    )
    self.bottle_right = rand_create_actor(
        self,
        xlim=list(self.bottle_right_x_range),
        ylim=list(self.bottle_y_range),
        modelname="001_bottle",
        rotate_rand=True,
        rotate_lim=[0, 1, 0],
        qpos=[0.65, 0.65, 0.27, 0.27],
        convex=True,
        model_id=16,
    )
    if self.bottle_left is None or self.bottle_right is None:
        raise FileNotFoundError("Missing RoboTwin bottle assets")
    self.bottle_left.set_name("bottle_left")
    self.bottle_right.set_name("bottle_right")
    self.bottle_left.set_mass(0.05)
    self.bottle_right.set_mass(0.05)
    self.objects = {"bottle_left": self.bottle_left, "bottle_right": self.bottle_right}
    goal_z = 0.744 + self.table_z_bias
    self.goal_left = create_visual_box(
        scene=self.scene,
        pose=sapien.Pose([-0.06, -0.105, goal_z]),
        half_size=[0.045, 0.04, 0.002],
        color=[0.15, 0.65, 0.95],
        name="goal_left",
    )
    self.goal_right = create_visual_box(
        scene=self.scene,
        pose=sapien.Pose([0.06, -0.105, goal_z]),
        half_size=[0.045, 0.04, 0.002],
        color=[0.95, 0.55, 0.15],
        name="goal_right",
    )
    self.goals = {"goal_left": self.goal_left, "goal_right": self.goal_right}
    self.add_prohibit_area(self.bottle_left, padding=0.08)
    self.add_prohibit_area(self.bottle_right, padding=0.08)
    self.prohibited_area.append([-0.14, -0.16, 0.14, -0.05])


def place_fan(self):
    rand_pos = rand_pose(
        xlim=[-0.1, 0.1],
        ylim=[-0.15, -0.05],
        qpos=[0.0, 0.0, 0.707, 0.707],
        rotate_rand=True,
        rotate_lim=[0, 2 * np.pi, 0],
    )
    id_list = [4, 5]
    self.fan_id = np.random.choice(id_list)
    self.fan = create_actor(
        scene=self,
        pose=rand_pos,
        modelname="099_fan",
        convex=True,
        model_id=self.fan_id,
    )
    self.fan.set_mass(0.01)
    xlim = [0.15, 0.25] if self.fan.get_pose().p[0] > 0 else [-0.25, -0.15]
    rand_pos = rand_pose(xlim=xlim, ylim=[-0.15, -0.05])
    colors = {
        "Red": (1, 0, 0),
        "Green": (0, 1, 0),
        "Blue": (0, 0, 1),
        "Yellow": (1, 1, 0),
        "Cyan": (0, 1, 1),
        "Magenta": (1, 0, 1),
        "Black": (0, 0, 0),
        "Gray": (0.5, 0.5, 0.5),
        "Orange": (1, 0.5, 0),
        "Purple": (0.5, 0, 0.5),
        "Brown": (0.65, 0.4, 0.16),
        "Pink": (1, 0.75, 0.8),
        "Lime": (0.5, 1, 0),
        "Olive": (0.5, 0.5, 0),
        "Teal": (0, 0.5, 0.5),
        "Maroon": (0.5, 0, 0),
        "Navy": (0, 0, 0.5),
        "Coral": (1, 0.5, 0.31),
        "Turquoise": (0.25, 0.88, 0.82),
        "Indigo": (0.29, 0, 0.51),
        "Beige": (0.96, 0.91, 0.81),
        "Tan": (0.82, 0.71, 0.55),
        "Silver": (0.75, 0.75, 0.75),
    }
    color_items = list(colors.items())
    idx = np.random.choice(len(color_items))
    (self.color_name, self.color_value) = color_items[idx]
    self.pad = create_box(
        scene=self.scene,
        pose=rand_pos,
        half_size=(0.05, 0.05, 0.001),
        color=self.color_value,
        name="box",
    )
    self.pad.set_mass(1)
    self.objects = {"fan": self.fan}
    self.goals = {"pad": self.pad}
    self.add_prohibit_area(self.fan, padding=0.07)
    self.prohibited_area.append(
        [
            rand_pos.p[0] - 0.15,
            rand_pos.p[1] - 0.15,
            rand_pos.p[0] + 0.15,
            rand_pos.p[1] + 0.15,
        ]
    )


def shake_bottle(self):
    config = self.shake_config
    layout = scene_layout(self.scene_seed, config)
    position = np.asarray(layout["position"], dtype=np.float64)
    quaternion = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], float(layout["yaw"]))
    self.bottle_id = int(config.model_id)
    self.bottle = create_actor(
        scene=self,
        pose=sapien.Pose(position, quaternion),
        modelname="001_bottle",
        convex=True,
        model_id=self.bottle_id,
    )
    if self.bottle is None:
        raise RuntimeError("001_bottle asset could not be created")
    self.bottle.set_name("bottle")
    self.bottle.set_mass(0.01)
    self.add_prohibit_area(self.bottle, padding=0.05)


def scene_layout(seed: int, config) -> dict[str, object]:
    """Return a deterministic conservative common-reach scene layout."""
    rng = np.random.default_rng(int(seed))
    position = np.asarray(
        [
            config.center_x + rng.uniform(-config.xy_jitter, config.xy_jitter),
            config.center_y + rng.uniform(-config.y_jitter, config.y_jitter),
            config.table_z,
        ],
        dtype=np.float64,
    )
    yaw = float(
        config.initial_yaw
        + rng.uniform(-config.initial_yaw_jitter, config.initial_yaw_jitter)
    )
    return {"position": position.tolist(), "yaw": yaw, "seed": int(seed)}


def row_layout(seed, config):
    """Local RNG: branch choices and planner RNG consumption cannot affect layout."""
    rng = np.random.default_rng(seed)
    center = np.array([config.center_x, config.center_y]) + rng.uniform(
        -config.xy_jitter, config.xy_jitter, 2
    )
    center[1] = max(float(center[1]), config.min_center_y)
    yaw = float(rng.uniform(-config.yaw_jitter, config.yaw_jitter))
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    return {
        "center": center.tolist(),
        "yaw": yaw,
        "positions": {
            position: [*center + (i - 1) * config.spacing * axis, config.root_z]
            for (i, position) in enumerate(POSITIONS)
        },
    }


def _physical_disk(scene, pose, *, radius, half_thickness, color, name):
    """Create a static cylindrical route divider with render and collision."""
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)
    material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        sapien.render.RenderShapeCylinder(
            radius=radius, half_length=half_thickness, material=material
        )
    )
    collision_component = sapien.physx.PhysxRigidStaticComponent()
    collision_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_thickness,
            material=scene.default_physical_material,
        )
    )
    entity.add_component(render_component)
    entity.add_component(collision_component)
    scene.add_entity(entity)
    return Actor(
        entity,
        {
            "bbox_half_extents": [half_thickness, radius, radius],
            "extents": [2 * half_thickness, 2 * radius, 2 * radius],
            "scale": [1.0, 1.0, 1.0],
            "functional_matrix": [],
            "contact_points_pose": [],
            "target_pose": [],
        },
    )


def place_via_divider(self):
    self.payload = rand_create_actor(
        self,
        xlim=[self.scene_x, self.scene_x],
        ylim=[self.payload_y, self.payload_y],
        modelname=self.payload_model_name,
        rotate_rand=False,
        rotate_lim=[0, 0, 0],
        qpos=[0.66, 0.66, -0.25, -0.25],
        convex=True,
        model_id=self.payload_model_id,
    )
    if self.payload is None:
        raise FileNotFoundError("Missing RoboTwin bottle asset 001_bottle/model 13")
    self.payload.set_name("payload")
    self.payload.set_mass(0.05)
    self.divider = _physical_disk(
        scene=self.scene,
        pose=sapien.Pose(
            [
                self.scene_x,
                self.divider_y,
                self.table_z + float(self.divider_half_size[2]),
            ],
            [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0],
        ),
        radius=float(self.divider_half_size[0]),
        half_thickness=float(self.divider_half_size[2]),
        color=(0.25, 0.35, 0.85),
        name="divider",
    )
    self.target_zone = create_visual_box(
        scene=self.scene,
        pose=sapien.Pose(
            [
                self.scene_x,
                self.target_y,
                self.table_z + float(self.target_half_size[2]),
            ],
            [1, 0, 0, 0],
        ),
        half_size=self.target_half_size.tolist(),
        color=(0.15, 0.75, 0.3),
        name="target_zone",
    )
    self.objects = {"payload": self.payload, "divider": self.divider}
    self.goals = {"target_zone": self.target_zone}
    self.prohibited_area.extend(
        [
            [
                self.scene_x - 0.08,
                self.payload_y - 0.08,
                self.scene_x + 0.08,
                self.payload_y + 0.08,
            ],
            [
                self.scene_x - self.route_offset - 0.07,
                self.divider_y - 0.08,
                self.scene_x + self.route_offset + 0.07,
                self.divider_y + 0.08,
            ],
            [
                self.scene_x - 0.08,
                self.target_y - 0.08,
                self.scene_x + 0.08,
                self.target_y + 0.08,
            ],
        ]
    )


def open_laptop(self):
    self.laptop: object = rand_create_sapien_urdf_obj(
        scene=self,
        modelname=self.model_name,
        modelid=self.model_id,
        xlim=[-0.05, 0.05],
        ylim=[-0.1, 0.05],
        rotate_rand=True,
        rotate_lim=[0, 0, np.pi / 3],
        qpos=[0.7, 0, 0, 0.7],
        fix_root_link=True,
    )
    if self.laptop is None:
        raise FileNotFoundError(
            f"Missing {self.model_name} model index {self.model_id}"
        )
    self.laptop.set_name("laptop")
    limits = np.asarray(self.laptop.get_qlimits()[0], dtype=np.float64)
    self.closed_qpos = float(limits[0])
    self.open_qpos_limit = float(limits[1])
    self.joint_range_rad = self.open_qpos_limit - self.closed_qpos
    if self.joint_range_rad <= 0:
        raise ValueError(f"Invalid laptop joint limits: {limits.tolist()}")
    initial_qpos = self.closed_qpos + self.joint_range_rad * self.initial_open_fraction
    self.laptop.set_qpos([initial_qpos])
    self.laptop.set_qvel([0.0])
    self.laptop.set_mass(0.01)
    self.laptop.set_properties(1, 0)
    self.objects = {"laptop": self.laptop}
    self.goals = {}
    self.add_prohibit_area(self.laptop, padding=0.1)


def _blocks(self):
    block_half_size = 0.025
    block_pose_lst = []
    for i in range(3):
        block_pose = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.08, 0.05],
            zlim=[0.741 + block_half_size],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        def check_block_pose(block_pose):
            for j in range(len(block_pose_lst)):
                if np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01:
                    return False
            return True

        while (
            abs(block_pose.p[0]) < 0.05
            or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
            or (not check_block_pose(block_pose))
        ):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )
        block_pose_lst.append(deepcopy(block_pose))

    def create_block(block_pose, color):
        return create_box(
            scene=self,
            pose=block_pose,
            half_size=(block_half_size, block_half_size, block_half_size),
            color=color,
            name="box",
        )

    self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
    self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
    self.block3 = create_block(block_pose_lst[2], (0, 0, 1))
    self.add_prohibit_area(self.block1, padding=0.05)
    self.add_prohibit_area(self.block2, padding=0.05)
    self.add_prohibit_area(self.block3, padding=0.05)
    target_pose = [-0.04, -0.13, 0.04, -0.05]
    self.prohibited_area.append(target_pose)


def stack_blocks(self):
    _blocks(self)
    self.objects = dict(zip(COLORS, (self.block1, self.block2, self.block3)))
    self.goals = {}
    base_slots = ((-0.12, 0.03), (0.0, 0.03), (0.12, 0.03))
    safe_slots = tuple(
        (
            (x + random.uniform(-0.02, 0.02), y + random.uniform(-0.01, 0.01))
            for (x, y) in base_slots
        )
    )
    for index, (name, block) in enumerate(self.objects.items()):
        pose = block.get_pose()
        (x, y) = safe_slots[index]
        yaw = random.uniform(-np.deg2rad(10.0), np.deg2rad(10.0))
        block.actor.set_pose(
            sapien.Pose(
                [x, y, float(pose.p[2])], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            )
        )
    self.prohibited_area = []
    for block in self.objects.values():
        self.add_prohibit_area(block, padding=0.05)
    self.prohibited_area.append([-0.04, -0.13, 0.04, -0.05])
    for name, block in self.objects.items():
        block.actor.set_name(name)
        block.config["bbox_half_extents"] = [0.025, 0.025, 0.025]
    self.sort_bins = {}
    table_z = 0.74 + float(getattr(self, "table_z_bias", 0.0))
    bin_z = table_z + 0.005
    for color, center in self.sort_target_xy.items():
        (cx, cy) = map(float, center)
        bin_color = {
            "red": (1, 0.15, 0.15),
            "green": (0.15, 1, 0.15),
            "blue": (0.15, 0.3, 1),
        }[color]
        parts = []
        parts.append(
            create_box(
                self,
                sapien.Pose([cx, cy, bin_z]),
                half_size=(0.04, 0.04, 0.005),
                color=bin_color,
                is_static=True,
                name=f"{color}_bin_floor",
            )
        )
        for dx, dy, hs in (
            (-0.045, 0, (0.005, 0.045, 0.025)),
            (0.045, 0, (0.005, 0.045, 0.025)),
            (0, -0.045, (0.045, 0.005, 0.025)),
            (0, 0.045, (0.045, 0.005, 0.025)),
        ):
            parts.append(
                create_box(
                    self,
                    sapien.Pose([cx + dx, cy + dy, table_z + 0.03]),
                    half_size=hs,
                    color=bin_color,
                    is_static=True,
                    name=f"{color}_bin_wall",
                )
            )
        self.sort_bins[color] = parts


def switches(self):
    config = self.row_config
    self.asset_info = inspect_asset(
        Path(ROOT_PATH), config.model_id, config.contact_point_id
    )
    self.layout = row_layout(self.scene_seed, config)
    yaw = self.layout["yaw"]
    root_rotation = rotation([0, 0, 1], np.pi / 2)
    world_rotation = rotation([0, 0, 1], yaw) @ root_rotation
    root_quat = t3d.quaternions.mat2quat(world_rotation)
    (self.objects, self.goals) = ({}, {})
    self.switch_link_names = {}
    bounds = np.array(self.asset_info["root_bounds"])
    corners = np.array(
        [[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]]
    )
    row_corners = corners @ root_rotation.T
    (row_min, row_max) = (row_corners.min(0), row_corners.max(0))
    if config.spacing < row_max[0] - row_min[0] + 0.005:
        raise ValueError(
            "Switch spacing is smaller than asset width plus 5 mm clearance"
        )
    if config.root_z + row_min[2] < 0.742:
        raise ValueError("Switch geometry intersects the table; increase root_z")
    for position in POSITIONS:
        name = f"switch_{position}"
        actor = create_sapien_urdf_obj(
            self,
            pose=sapien.Pose(self.layout["positions"][position], root_quat),
            modelname="056_switch",
            modelid=config.model_id,
            fix_root_link=True,
        )
        actor.set_name(name)
        for original_name, link in actor.link_dict.items():
            unique_name = f"{name}/{original_name}"
            link.set_name(unique_name)
            link.entity.set_name(unique_name)
            self.switch_link_names[unique_name] = position
        actor.config = dict(actor.config)
        actor.config["center"] = (bounds.mean(0) / self.asset_info["scale"]).tolist()
        actor.config["extents"] = (
            (bounds[1] - bounds[0]) / self.asset_info["scale"]
        ).tolist()
        limits = actor.get_qlimits()[0]
        actor.set_qpos([limits[0] if config.on_at_upper_limit else limits[1]])
        actor.set_qvel([0.0])
        self.objects[name] = actor
    plate_min = np.array(
        [-config.spacing + row_min[0] - 0.006, row_max[1] + 0.0005, 0.74]
    )
    plate_max = np.array(
        [
            config.spacing + row_max[0] + 0.006,
            row_max[1] + 0.0125,
            config.root_z + row_max[2] + 0.006,
        ]
    )
    center = (plate_min + plate_max) / 2
    center[:2] = rotation([0, 0, 1], yaw)[:2, :2] @ center[:2] + self.layout["center"]
    self.mount = create_box(
        self,
        pose=sapien.Pose(center, t3d.quaternions.mat2quat(rotation([0, 0, 1], yaw))),
        half_size=(plate_max - plate_min) / 2,
        color=[0.28, 0.3, 0.33],
        is_static=True,
        name="switch_row_mount",
    )
    self.prohibited_area.append([-0.4, -0.2, 0.4, 0.25])
