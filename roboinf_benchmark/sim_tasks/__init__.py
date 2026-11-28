import random
from types import SimpleNamespace

import numpy as np
from envs._base_task import Base_Task

from ..suite import BUDGETS
from . import layouts

NATIVE_TASKS = {
    "place_bottle": "phase0_pick_place", "shake_bottle": "phase1_shake_bottle",
    "place_fan": "place_fan_orientation", "turn_switch": "turn_switch_row",
    "place_via_divider": "place_via_divider", "open_laptop": "open_laptop_controlled",
    "stack_blocks": "sort_blocks_in_order", "press_buttons": "press_buttons_order",
}


class Scene(Base_Task):
    appearance = None

    def setup_scene(self, **kwargs):
        import sapien

        if self.appearance is not None:
            kwargs.update({key: self.appearance[key] for key in ("ambient_light", "point_lights")})
        result = super().setup_scene(**kwargs)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        return result

    def check_success(self):
        # Held-out policies run to their prescribed horizon.
        return False

    def _step_scene(self):
        self.scene.step()

    @property
    def table_z(self):
        return 0.74 + self.table_z_bias


def make_task(task, scene, case, args, configure):
    if task not in BUDGETS or (case["task"], case["scene_id"]) != (
        scene["task"],
        scene["scene_id"],
    ):
        raise ValueError("Case and scene must refer to the same public task and scene")
    seed = scene["scene_seed"]
    random.seed(seed)
    env = Scene()
    env.scene_seed = seed
    if task == "place_bottle":
        env.bottle_left_x_range, env.bottle_right_x_range = (-0.16, -0.08), (0.08, 0.16)
        env.bottle_y_range = (0.08, 0.16)
    elif task == "shake_bottle":
        env.shake_config = SimpleNamespace(
            model_id=0,
            center_x=0.0,
            center_y=-0.10,
            xy_jitter=0.020,
            y_jitter=0.010,
            table_z=0.785,
            initial_yaw=0.0,
            initial_yaw_jitter=np.pi / 4,
        )
    elif task == "place_via_divider":
        env.payload_model_id, env.payload_model_name = 13, "001_bottle"
        env.target_half_size, env.divider_half_size = (
            np.array([0.045, 0.04, 0.002]),
            np.array([0.03, 0.03, 0.04]),
        )
        rng = random.Random(f"place_via_divider:layout_v2:{seed}")
        env.payload_y, env.divider_y, env.target_y = (
            rng.uniform(0.07, 0.10),
            rng.uniform(-0.01, 0.01),
            rng.uniform(-0.14, -0.11),
        )
        env.scene_x = random.Random(f"place_via_divider:x_v3:{seed}").uniform(-0.1, 0.1)
        env.route_offset = 0.20
    elif task == "open_laptop":
        env.model_name, env.model_id, env.initial_open_fraction = "015_laptop", 0, 0.2
    elif task == "stack_blocks":
        env.sort_target_xy = {
            "red": (-0.12, -0.13),
            "green": (0, -0.20),
            "blue": (0.12, -0.13),
        }
    elif task in {"turn_switch", "press_buttons"}:
        env.row_config = SimpleNamespace(
            model_id=1,
            contact_point_id=0,
            center_x=0.0,
            center_y=0.08,
            xy_jitter=0.02,
            min_center_y=0.06,
            yaw_jitter=0.08,
            root_z=0.86,
            spacing=0.12,
            on_at_upper_limit=True,
        )
    build = (
        layouts.switches
        if task in {"turn_switch", "press_buttons"}
        else getattr(layouts, task)
    )
    env.load_actors = build.__get__(env)
    configure(env)
    try:
        env._init_task_env_(
            **{**args, "task_name": NATIVE_TASKS[task], "seed": seed, "now_ep_num": 0}
        )
        return env
    except BaseException:
        if hasattr(env, "scene"):
            env.close_env(clear_cache=True)
        raise
