"""Select Switch: task and instruction success criteria."""

from dataclasses import asdict, dataclass

import numpy as np

TR = ("initially_all_off", "target_on", "released_and_stable")

JR = (
    "correct_arm_contact",
    "inactive_arm_stationary",
    "inter_arm_collision_free",
    "others_untouched",
)

POSITIONS = ("left", "middle", "right")

ARMS = ("left", "right")


@dataclass(frozen=True)
class RowConfig:
    model_id: int = 1
    spacing: float = 0.12
    center_x: float = 0.0
    center_y: float = 0.08
    root_z: float = 0.86
    min_center_y: float = 0.06
    min_root_z: float = 0.84
    min_initial_arm_clearance_m: float = 0.08
    xy_jitter: float = 0.02
    yaw_jitter: float = 0.08
    contact_point_id: int = 0
    on_at_upper_limit: bool = True
    pre_press_distance: float = 0.06
    press_distance: float = -0.005
    press_hold_seconds: float = 0.12
    settle_seconds: float = 0.4
    on_threshold: float = 0.95
    off_tolerance: float = 0.10
    inactive_arm_tolerance: float = 0.03
    contact_tolerance_m: float = 0.025

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Row configuration must contain finite values")
        if self.model_id < 0 or self.contact_point_id < 0:
            raise ValueError("Asset and contact IDs must be nonnegative")
        if not isinstance(self.on_at_upper_limit, bool):
            raise TypeError("on_at_upper_limit must be a boolean")
        if not 0 <= self.off_tolerance < self.on_threshold <= 1:
            raise ValueError("Require 0 <= off_tolerance < on_threshold <= 1")
        if (
            self.spacing <= 0
            or self.xy_jitter < 0
            or not 0 <= self.yaw_jitter < np.pi / 4
        ):
            raise ValueError("Invalid spacing or randomization range")
        if self.root_z < self.min_root_z or self.min_center_y < 0:
            raise ValueError("Switch row is too low or too close to the robot")
        if self.min_initial_arm_clearance_m <= 0:
            raise ValueError("Initial arm clearance must be positive")
        if (
            min(
                self.settle_seconds,
                self.pre_press_distance,
                self.inactive_arm_tolerance,
                self.contact_tolerance_m,
            )
            <= 0
        ):
            raise ValueError(
                "Distances, tolerances and settling duration must be positive"
            )
        if self.press_hold_seconds < 0:
            raise ValueError("Press hold duration must be nonnegative")


def validate_control_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"target_switch", "active_arm"}:
        raise ValueError("Expected exactly target_switch and active_arm")
    if spec["target_switch"] not in POSITIONS or spec["active_arm"] not in ARMS:
        raise ValueError("Invalid switch position or active arm")
    return dict(spec)


def check_rollout(samples, spec, config, dt):
    """Use all physics steps, so a transient wrong-switch toggle remains a failure."""
    spec = validate_control_spec(spec)
    target = POSITIONS.index(spec["target_switch"])
    others = [i for i in range(3) if i != target]
    checks = {
        "initially_all_off": False,
        "target_on": False,
        "others_untouched": False,
        "correct_arm_contact": False,
        "inactive_arm_stationary": False,
        "inter_arm_collision_free": False,
        "released_and_stable": False,
    }
    if not samples:
        return {"success": False, "checks": checks}
    progress = np.asarray([sample["progress"] for sample in samples], dtype=float)
    drift = np.asarray(
        [sample["inactive_arm_drift"] for sample in samples], dtype=float
    )
    if (
        progress.shape != (len(samples), 3)
        or not np.isfinite(progress).all()
        or not np.isfinite(drift).all()
    ):
        return {"success": False, "checks": checks}
    contacts = [event for sample in samples for event in sample.get("contacts", [])]
    inter_arm_collisions = [
        contact
        for sample in samples
        for contact in sample.get("raw_contacts", [])
        if contact.get("actual")
        and len({robot.get("arm") for robot in contact.get("robot_links", []) if robot})
        > 1
    ]
    intended = [
        event
        for event in contacts
        if event["switch"] == spec["target_switch"]
        and event["arm"] == spec["active_arm"]
        and event.get("robot_region", "hand") == "hand"
        and event["distance_m"] <= config.contact_tolerance_m
    ]
    wrong_arm = any(event["arm"] != spec["active_arm"] for event in contacts)
    stable_steps = max(1, int(np.ceil(config.settle_seconds / dt)))
    tail = samples[-stable_steps:]
    checks.update(
        initially_all_off=bool(np.max(np.abs(progress[0])) <= config.off_tolerance),
        target_on=bool(progress[-1, target] >= config.on_threshold),
        others_untouched=bool(
            np.max(np.abs(progress[:, others])) <= config.off_tolerance
        ),
        correct_arm_contact=bool(intended and not wrong_arm),
        inactive_arm_stationary=bool(np.max(drift) <= config.inactive_arm_tolerance),
        inter_arm_collision_free=not inter_arm_collisions,
        released_and_stable=bool(
            len(samples) >= stable_steps
            and np.all(progress[-stable_steps:, target] >= config.on_threshold)
            and all(
                not sample.get("contacts")
                and sample.get("target_clearance_m", 0) >= config.pre_press_distance / 2
                for sample in tail
            )
        ),
    )
    return {
        "success": all(checks.values()),
        "checks": checks,
        "final_progress": progress[-1].tolist(),
        "max_other_progress": float(np.max(np.abs(progress[:, others]))),
        "max_inactive_arm_drift": float(np.max(drift)),
        "actual_contact_arms": sorted({event["arm"] for event in contacts}),
        "valid_target_contact_count": len(intended),
        "inter_arm_collision_count": len(inter_arm_collisions),
    }
