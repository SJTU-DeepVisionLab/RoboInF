from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import os
import sys

import numpy as np
import yaml

from .contracts import CAMERAS, EEF, JOINT, Observation
from .suite import validate_appearance


def _task_args(root, task):
    def read(path):
        with path.open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    config = root / "env_cfg" / "task_config"
    args = read(config / "demo_clean.yml")
    embodiment = "aloha-agilex"
    path = (root / read(config / "_embodiment_config.yml")[embodiment]["file_path"]).resolve()
    robot = read(path / "config.yml")
    dual = bool(robot.get("dual_arm", False))
    args.update(embodiment=[embodiment] if dual else [embodiment, embodiment, 0.6],
                embodiment_name=embodiment if dual else f"{embodiment}+{embodiment}",
                left_robot_file=str(path), right_robot_file=str(path), left_embodiment_config=robot,
                right_embodiment_config=robot, dual_arm_embodied=dual, task_config="demo_clean",
                collect_data=False, save_data=False, eval_video_log=False, render_freq=0, eval_mode=True)
    if not dual:
        args["embodiment_dis"] = 0.6
    args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
    args["data_type"].update(rgb=True, qpos=True, endpose=True)
    args["domain_randomization"].update(random_background=False, cluttered_table=False,
        random_head_camera_dis=0, random_table_height=0.03 if task == "place_bottle" else 0.0,
        random_light=False, crazy_random_light_rate=0)
    return args


def _appearance(env, value, root):
    validate_appearance(value)
    for key in ("wall_texture", "table_texture"):
        if not (root / "assets/background_texture" / (value[key] + ".png")).is_file():
            raise FileNotFoundError(f"Missing RoboTwin texture: {value[key]}")
    env.appearance = value


class _SceneClock:
    def __init__(self, scene):
        self.scene = scene
        self.count = 0

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self):
        result = self.scene.step()
        self.count += 1
        return result


class RoboTwinEpisode:
    def __init__(self, root, scene, case, domain):
        self.stack = ExitStack()
        try:
            previous_cwd = Path.cwd()
            os.chdir(root)
            self.stack.callback(os.chdir, previous_cwd)
            sys.path.insert(0, str(root))
            self.stack.callback(sys.path.remove, str(root))
            loaded = sys.modules.get("envs")
            if loaded is not None and (root / "envs").resolve() not in map(Path.resolve, map(Path, loaded.__path__)):
                raise RuntimeError("A different RoboTwin runtime is already loaded in this Python process")
            from .sim_tasks import make_task

            def configure(task):
                if domain == "test_random":
                    _appearance(task, scene["test_random_appearance"], root)

            env = make_task(case["task"], scene, case, _task_args(root, case["task"]), configure)
            self.env = env
            self.stack.callback(env.close_env, clear_cache=True)
            env.check_success = lambda: False
            env.eval_success = False
            self.clock = _SceneClock(env.scene)
            self.dt = float(env.scene.get_timestep())
            if not np.isfinite(self.dt) or self.dt <= 0:
                raise ValueError("Simulator returned an invalid physics timestep")
            env.scene = self.clock
            self.stack.callback(setattr, env, "scene", self.clock.scene)
        except BaseException:
            self.stack.close()
            raise

    @property
    def action_steps(self):
        return int(self.env.take_action_cnt)

    @property
    def physics_steps(self):
        return self.clock.count

    @property
    def step_limit(self):
        return self.env.step_lim

    def observe(self):
        raw = self.env.get_obs()
        views = dict(zip(CAMERAS, ("head_camera", "left_camera", "right_camera")))
        images = {name: raw["observation"][source]["rgb"] for name, source in views.items()}
        eef = []
        for arm in ("left", "right"):
            pose = np.asarray(self.env.get_arm_pose(arm))
            if pose.shape != (7,):
                raise ValueError("Expected native xyz+wxyz EEF pose")
            eef.extend(pose)
            eef.append(float(getattr(self.env.robot, f"get_{arm}_gripper_val")()))
        return Observation(images, raw["joint_action"]["vector"], np.asarray(eef), self.action_steps,
                           self.physics_steps * self.dt)

    def execute(self, action, spec):
        mode = {JOINT.name: "qpos", EEF.name: "ee"}[spec]
        self.env.take_action(action, action_type=mode)

    def settle(self, steps):
        step = getattr(self.env, "_step_scene", self.clock.step)
        for _ in range(steps):
            step()

    def close(self):
        self.stack.close()


def environment_factory(root):
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        raise RuntimeError("RoboTwin's native planner requires NumPy 1.x. Use a clean simulator environment.")
    root = Path(root).resolve(strict=True)
    for line in Path(__file__).with_name("robotwin.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = root / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name}: expected the bundled RoboInF runtime; see third_party/robotwin/README.md")
    return lambda scene, case, domain: RoboTwinEpisode(root, scene, case, domain)
