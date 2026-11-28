"""Shared state for observers of a single SAPIEN episode."""


class PhysicsRecorder:
    def __init__(self, episode):
        self.episode = episode
        self.env = episode.env
        self.robot = self.env.robot
        self.samples = []
        self.passive = False
        self.grippers = {}
        for arm in ("left", "right"):
            names = set(getattr(self.robot, arm + "_fix_gripper_name"))
            for joint, _, _ in getattr(self.robot, arm + "_gripper"):
                if joint is not None:
                    names.add(joint.child_link.get_name())
            self.grippers.update(dict.fromkeys(names, arm))

    def sample(self, step, **measurements):
        self.samples.append(
            {"physics_step": step, "time_sec": step * self.episode.dt, **measurements}
        )

    def evidence(self, **measurements):
        return {"samples": self.samples, "dt": self.episode.dt, **measurements}
