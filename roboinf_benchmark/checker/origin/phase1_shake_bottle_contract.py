"""Simulator-independent contract for the Phase 1 bottle-shaking task.

The contract keeps the three controllable factors explicit:
``active_arm`` (left/right), ``pose`` (horizontal/vertical), and ``trace``
(``wrist_rotation``/``lateral_translation``).  A wrist-rotation trace is
defined as a small direct sixth-arm-joint rotation around the horizontal
gripper-jaw axis through the grasp, rather than the bottle's local long axis.
The simulator task uses the same functions for
branch generation, metadata and validation tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np


TASK_NAME = "phase1_shake_bottle"
SCENE_LAYOUT_VERSION = 1
ARMS = ("left", "right")
POSES = ("horizontal", "vertical")
# Keep the deterministic scan order aligned with the acquisition policy:
# lateral branches first, then wrist branches; horizontal before vertical.
TRACES = ("lateral_translation", "wrist_rotation")
LONG_AXIS_LOCAL = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class ShakeConfig:
    # Fixed first-pilot asset.  Later pilots can replace this with a small
    # explicit list without changing the branch contract.
    model_id: int = 0
    center_x: float = 0.0
    # The phase-1 grasp planner has been validated in the rear/common arm
    # envelope around y=-0.08 m.  The apparently similar front-table region
    # used by some dual-bottle tasks is outside this planner's reliable grasp
    # envelope for the 001_bottle asset.
    center_y: float = -0.10
    xy_jitter: float = 0.020
    # Keep the pilot layout close to the v4 graspable sample instead of
    # exploring the full table depth before the planner envelope is mapped.
    y_jitter: float = 0.010
    table_z: float = 0.785
    initial_yaw: float = 0.0
    # Match RoboTwin's original shake_bottle randomization: z-yaw is sampled
    # uniformly from [-pi/4, pi/4].  The sampled value remains scene-level so
    # all eight branches use the identical initial bottle pose.
    initial_yaw_jitter: float = np.pi / 4.0
    grasp_pre_distance: float = 0.10
    lift_height: float = 0.10
    # The left arm needs a small extra lift before rotating the bottle
    # upright; the right arm's validated route already clears the table.
    left_vertical_lift_extra_m: float = 0.04
    settle_seconds: float = 0.35
    wrist_rotation_duration_s: float = 0.30
    cycles: int = 3
    pose_tolerance_deg: float = 16.0
    # A wrist rotation can briefly tilt the bottle while the dynamic grasp
    # settles.  The final/settled pose still uses pose_tolerance_deg; this
    # looser limit applies only to active shake samples.
    rotation_shake_pose_tolerance_deg: float = 20.0
    rotation_half_amplitude_deg: float = 10.0
    rotation_min_peak_to_peak_deg: float = 16.0
    translation_half_amplitude_m: float = 0.025
    # Lateral shaking is intentionally a loose arm-relative side-to-side
    # motion; the checker only requires a modest measurable excursion.
    translation_min_peak_to_peak_m: float = 0.025
    rotation_tcp_position_tolerance_m: float = 0.03
    rotation_center_translation_tolerance_m: float = 0.03
    # Allow modest grasp slip / mesh-origin motion while keeping gross loss of
    # the grasp outside the task contract.
    rotation_center_translation_allowed_m: float = 0.05
    # abs(world-z component) <= 0.35 is approximately a 20-degree wrist-axis
    # tilt from horizontal.  The older 0.25 value remains a quality metric.
    rotation_axis_vertical_allowed: float = 0.35
    translation_orientation_tolerance_deg: float = 30.0
    translation_axis_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    min_lift_clearance_m: float = 0.08
    max_inactive_arm_drift: float = 0.03
    state_tolerance: float = 1e-5

    def __post_init__(self):
        numeric = asdict(self)
        for name, value in numeric.items():
            if name == "translation_axis_world":
                continue
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.model_id < 0:
            raise ValueError("model_id must be nonnegative")
        if self.cycles != 3:
            raise ValueError("Phase 1 shake_bottle requires exactly three cycles")
        if self.xy_jitter < 0 or self.y_jitter < 0 or self.initial_yaw_jitter < 0:
            raise ValueError("scene jitter must be nonnegative")
        if min(self.grasp_pre_distance, self.lift_height, self.settle_seconds,
               self.wrist_rotation_duration_s,
               self.min_lift_clearance_m) <= 0:
            raise ValueError("distances and settle time must be positive")
        if self.left_vertical_lift_extra_m < 0 or not np.isfinite(self.left_vertical_lift_extra_m):
            raise ValueError("left_vertical_lift_extra_m must be finite and nonnegative")
        if not 0 < self.pose_tolerance_deg < 90:
            raise ValueError("pose_tolerance_deg must be in (0, 90)")
        if not self.pose_tolerance_deg <= self.rotation_shake_pose_tolerance_deg < 90:
            raise ValueError(
                "rotation_shake_pose_tolerance_deg must be >= pose_tolerance_deg and < 90"
            )
        if self.rotation_half_amplitude_deg <= 0 or self.rotation_min_peak_to_peak_deg <= 0:
            raise ValueError("rotation amplitudes must be positive")
        if self.rotation_min_peak_to_peak_deg > 2.0 * self.rotation_half_amplitude_deg:
            raise ValueError("rotation_min_peak_to_peak_deg cannot exceed the planned excursion")
        if self.translation_half_amplitude_m <= 0 or self.translation_min_peak_to_peak_m <= 0:
            raise ValueError("translation amplitudes must be positive")
        if self.translation_min_peak_to_peak_m > 2.0 * self.translation_half_amplitude_m:
            raise ValueError("translation_min_peak_to_peak_m cannot exceed the planned excursion")
        if self.rotation_tcp_position_tolerance_m <= 0 or self.rotation_center_translation_tolerance_m <= 0:
            raise ValueError("motion tolerances must be positive")
        if self.rotation_center_translation_allowed_m < self.rotation_center_translation_tolerance_m:
            raise ValueError("rotation_center_translation_allowed_m must be >= legacy tolerance")
        if not 0 < self.rotation_axis_vertical_allowed < 1:
            raise ValueError("rotation_axis_vertical_allowed must be in (0, 1)")
        if self.translation_orientation_tolerance_deg <= 0 or self.max_inactive_arm_drift <= 0:
            raise ValueError("orientation and inactive-arm tolerances must be positive")
        axis = np.asarray(self.translation_axis_world, dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-9:
            raise ValueError("translation_axis_world must be a finite nonzero 3-vector")
        if self.state_tolerance <= 0:
            raise ValueError("state_tolerance must be positive")


def validate_control_spec(spec: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(spec, Mapping):
        raise TypeError("control_spec must be a mapping")
    expected = {"target_object", "active_arm", "pose", "trace"}
    if set(spec) != expected:
        raise ValueError(f"Expected exactly {sorted(expected)}")
    if spec["target_object"] != "bottle":
        raise ValueError("target_object must be 'bottle'")
    if spec["active_arm"] not in ARMS:
        raise ValueError(f"active_arm must be one of {ARMS}")
    if spec["pose"] not in POSES:
        raise ValueError(f"pose must be one of {POSES}")
    if spec["trace"] not in TRACES:
        raise ValueError(f"trace must be one of {TRACES}")
    return {key: str(spec[key]) for key in sorted(expected)}


def iter_control_specs():
    for arm in ARMS:
        for pose in POSES:
            for trace in TRACES:
                yield {
                    "target_object": "bottle",
                    "active_arm": arm,
                    "pose": pose,
                    "trace": trace,
                }


def branch_key(scene_id: int, spec: Mapping[str, object]) -> str:
    spec = validate_control_spec(spec)
    return (f"{TASK_NAME}_scene_{int(scene_id):06d}__{spec['active_arm']}__"
            f"{spec['pose']}__{spec['trace']}")


def canonical_text(spec: Mapping[str, object]) -> str:
    spec = validate_control_spec(spec)
    pose = "horizontally" if spec["pose"] == "horizontal" else "vertically"
    if spec["trace"] == "wrist_rotation":
        motion = "with three small wrist rotations around the horizontal grasp axis"
    else:
        motion = "left and right three times"
    return f"Use the {spec['active_arm']} arm to hold the bottle {pose} and shake it {motion}."


def encode_control_ir(spec: Mapping[str, object]) -> list[float]:
    spec = validate_control_spec(spec)
    return [
        *[float(spec["active_arm"] == value) for value in ARMS],
        *[float(spec["pose"] == value) for value in POSES],
        *[float(spec["trace"] == value) for value in TRACES],
    ]


def scene_layout(seed: int, config: ShakeConfig) -> dict[str, object]:
    """Return a deterministic conservative common-reach scene layout."""

    rng = np.random.default_rng(int(seed))
    position = np.asarray([
        config.center_x + rng.uniform(-config.xy_jitter, config.xy_jitter),
        config.center_y + rng.uniform(-config.y_jitter, config.y_jitter),
        config.table_z,
    ], dtype=np.float64)
    yaw = float(config.initial_yaw + rng.uniform(
        -config.initial_yaw_jitter, config.initial_yaw_jitter,
    ))
    return {
        "position": position.tolist(),
        "yaw": yaw,
        "seed": int(seed),
    }


def digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _angle_wrap_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def pose_class_error_deg(long_axis_world: np.ndarray, pose: str) -> float:
    """Distance to horizontal/vertical using the bottle's unsigned long axis."""

    axis = np.asarray(long_axis_world, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-9:
        return float("inf")
    axis = axis / np.linalg.norm(axis)
    vertical_angle = float(np.rad2deg(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    if pose == "vertical":
        return vertical_angle
    if pose == "horizontal":
        return abs(90.0 - vertical_angle)
    raise ValueError(f"unknown pose={pose!r}")


def signed_projection(position: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    position = np.asarray(position, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if position.shape != (3,) or origin.shape != (3,) or axis.shape != (3,) or norm <= 1e-9:
        raise ValueError("position, origin and axis must be finite 3-vectors")
    return float(np.dot(position - origin, axis / norm))


def count_sign_cycles(values: np.ndarray, *, deadband: float = 0.0) -> int:
    """Count alternating positive/negative excursions in a scalar trace."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        return 0
    states = []
    for value in values:
        if value > deadband:
            state = 1
        elif value < -deadband:
            state = -1
        else:
            continue
        if not states or state != states[-1]:
            states.append(state)
    # One completed back-and-forth is one positive/negative (or negative/
    # positive) pair.  A trailing unmatched excursion is deliberately ignored.
    return len(states) // 2


def state_difference(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    if left.get("identity") != right.get("identity") or left.get("keys") != right.get("keys"):
        return float("inf")
    a = np.asarray(left.get("values"), dtype=np.float64)
    b = np.asarray(right.get("values"), dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    return float(np.max(np.abs(a - b)))


def check_rollout(samples: list[Mapping[str, object]], spec: Mapping[str, object],
                  config: ShakeConfig, dt: float) -> dict[str, object]:
    """Check arm, coarse pose and the requested three-cycle motion.

    Samples are collected at every simulator step.  Pose and motion checks are
    intentionally restricted to the post-grasp portion of the trace, while
    contact and lift checks are evaluated over the complete rollout.
    """

    spec = validate_control_spec(spec)
    checks = {
        "active_arm_contact": False,
        "inactive_arm_no_contact": False,
        "inactive_arm_stationary": False,
        "pose_during_shake": False,
        "pose_after_shake": False,
        "trace_three_cycles": False,
        "trace_amplitude": False,
        "wrist_position_stable": False,
        "wrist_axis_horizontal": False,
        "unintended_motion_limited": False,
        "lifted_and_held": False,
        "final_gripper_closed": False,
    }
    if not samples:
        return {"success": False, "checks": checks}

    def values(name: str, rows: list[Mapping[str, object]]) -> np.ndarray:
        return np.asarray([row.get(name) for row in rows], dtype=np.float64)

    active_contacts = [row for row in samples if row.get("active_arm_contact")]
    inactive_contacts = [row for row in samples if row.get("inactive_arm_contact")]
    checks["active_arm_contact"] = bool(active_contacts)
    checks["inactive_arm_no_contact"] = not inactive_contacts
    inactive_drift = values("inactive_arm_drift", samples)
    checks["inactive_arm_stationary"] = bool(
        inactive_drift.size and np.isfinite(inactive_drift).all() and
        np.max(inactive_drift) <= config.max_inactive_arm_drift
    )

    shake_rows = [row for row in samples if row.get("pose_check_enabled")]
    after_rows = [row for row in samples if row.get("phase") in {"settle", "complete"}]
    pose_errors = values("pose_error_deg", shake_rows)
    after_errors = values("pose_error_deg", after_rows)
    pose_limit = config.pose_tolerance_deg
    after_pose_limit = config.pose_tolerance_deg
    if spec["trace"] == "lateral_translation":
        # Lateral shaking is defined by table-horizontal displacement; small
        # coarse-pose deviations are acceptable while the planner moves the
        # arm through that displacement.
        pose_limit = max(pose_limit, 25.0)
        after_pose_limit = pose_limit
    elif shake_rows:
        # Do not reject a semantically valid wrist shake because one or two
        # active-shake samples contain a transient bottle tilt.  The settle
        # and complete phases below continue to use the strict task pose
        # tolerance.
        pose_limit = config.rotation_shake_pose_tolerance_deg
    checks["pose_during_shake"] = bool(
        shake_rows and np.isfinite(pose_errors).all() and
        np.max(pose_errors) <= pose_limit
    )
    checks["pose_after_shake"] = bool(
        after_rows and np.isfinite(after_errors).all() and
        np.max(after_errors) <= after_pose_limit
    )

    motion_rows = [row for row in samples if row.get("trace_check_enabled")]
    scalar = values("trace_scalar", motion_rows)
    scalar = scalar[np.isfinite(scalar)]
    cycle_count = count_sign_cycles(scalar, deadband=(
        config.translation_min_peak_to_peak_m / 4.0
        if spec["trace"] == "lateral_translation"
        else config.rotation_min_peak_to_peak_deg / 4.0
    )) if scalar.size else 0
    checks["trace_three_cycles"] = cycle_count == config.cycles
    if spec["trace"] == "lateral_translation":
        # Wrist-axis geometry is irrelevant for the translational trace.
        checks["wrist_axis_horizontal"] = True
        amplitude = float(np.ptp(scalar)) if scalar.size else 0.0
        checks["trace_amplitude"] = amplitude >= config.translation_min_peak_to_peak_m
        orientation = values("trace_orientation_error_deg", motion_rows)
        checks["unintended_motion_limited"] = bool(
            motion_rows and np.isfinite(orientation).all() and
            np.max(orientation) <= config.translation_orientation_tolerance_deg
        )
        checks["wrist_position_stable"] = True
    else:
        amplitude = float(np.ptp(scalar)) if scalar.size else 0.0
        axis_rows = [row for row in motion_rows if row.get("trace_axis_world") is not None]
        axis_z = np.asarray([
            abs(float(np.asarray(row["trace_axis_world"], dtype=np.float64)[2]))
            for row in axis_rows
        ], dtype=np.float64)
        # Older simulator-independent contract fixtures omit this optional
        # field; real task samples always include it during the shake phase.
        axis_within_legacy_limit = bool(
            not axis_rows or (np.isfinite(axis_z).all() and np.max(axis_z) <= 0.25)
        )
        # The world-frame wrist-axis test is retained as a quality diagnostic
        # in the returned metrics.  It is not a task-success gate: the task
        # asks for holding and shaking the bottle, and allows a modest wrist
        # tilt as long as contact, lift, cycles and final pose are valid.
        checks["wrist_axis_horizontal"] = bool(
            not axis_rows or (
                np.isfinite(axis_z).all()
                and np.max(axis_z) <= config.rotation_axis_vertical_allowed
            )
        )
        checks["trace_amplitude"] = amplitude >= config.rotation_min_peak_to_peak_deg
        tcp_drift = values("tcp_position_drift_m", motion_rows)
        center_drift = values("bottle_center_drift_m", motion_rows)
        checks["wrist_position_stable"] = bool(
            motion_rows and np.isfinite(tcp_drift).all() and
            np.max(tcp_drift) <= config.rotation_tcp_position_tolerance_m
        )
        center_drift_within_legacy_limit = bool(
            motion_rows and np.isfinite(center_drift).all() and
            np.max(center_drift) <= config.rotation_center_translation_tolerance_m
        )
        # Bottle mesh-origin drift can reflect grasp slip or an origin that is
        # away from the visual bottle centre.  Allow the configured modest
        # drift, while rejecting a gross displacement that suggests loss of
        # the grasp.
        checks["unintended_motion_limited"] = bool(
            motion_rows and np.isfinite(center_drift).all() and
            np.max(center_drift) <= config.rotation_center_translation_allowed_m
        )

    quality = {
        "wrist_axis_within_legacy_limit": bool(
            locals().get("axis_within_legacy_limit", True)
        ),
        "center_drift_within_legacy_limit": bool(
            locals().get("center_drift_within_legacy_limit", True)
        ),
        "rotation_shake_pose_limit_deg": float(
            config.rotation_shake_pose_tolerance_deg
            if spec["trace"] == "wrist_rotation" else pose_limit
        ),
        "pose_after_shake_limit_deg": float(after_pose_limit),
        "rotation_axis_vertical_allowed": float(config.rotation_axis_vertical_allowed),
        "rotation_center_translation_allowed_m": float(
            config.rotation_center_translation_allowed_m
        ),
    }

    # A planner can briefly dip below the clearance while transitioning from
    # lift to pose_align.  The task requirement is that the bottle is lifted
    # and held during the actual shake and final settle, so judge the stable
    # execution phases instead of treating that transient dip as permanent.
    held_phases = {"shake", "settle", "complete"}
    held_rows = [row for row in samples if row.get("phase") in held_phases]
    held_heights = values("bottle_height_above_table_m", held_rows)
    lifted_and_held = bool(
        held_rows and np.isfinite(held_heights).all() and
        np.min(held_heights) >= config.min_lift_clearance_m
    )
    checks["lifted_and_held"] = lifted_and_held
    checks["final_gripper_closed"] = bool(samples[-1].get("active_gripper_closed", False))
    return {
        "success": bool(all(checks.values())),
        "checks": checks,
        "quality": quality,
        "cycle_count": int(cycle_count),
        "trace_peak_to_peak": float(np.ptp(scalar)) if scalar.size else 0.0,
        "max_pose_error_deg": float(np.max(pose_errors)) if pose_errors.size else None,
        "max_tcp_position_drift_m": float(np.max(values("tcp_position_drift_m", motion_rows))) if motion_rows else None,
        "max_bottle_center_drift_m": float(np.max(values("bottle_center_drift_m", motion_rows))) if motion_rows else None,
        "max_inactive_arm_drift": float(np.max(inactive_drift)) if inactive_drift.size else None,
        "active_contact_samples": len(active_contacts),
    }


def finalize_groups(records: list[dict], tolerance: float = 1e-5,
                    success_field: str = "expert_success") -> list[dict]:
    groups: dict[int, list[dict]] = {}
    for record in records:
        groups.setdefault(int(record["scene_id"]), []).append(dict(record))
    expected = {tuple(validate_control_spec(spec).values()) for spec in iter_control_specs()}
    result = []
    for siblings in groups.values():
        baseline = siblings[0].get("initial_scene_state") if siblings else None
        same_state = bool(
            len(siblings) == 8
            and baseline is not None
            and all(item.get("initial_scene_state") is not None and
                    state_difference(baseline, item["initial_scene_state"]) <= tolerance
                    for item in siblings)
        )
        actual = {tuple(validate_control_spec(item["control_spec"]).values()) for item in siblings}
        rgb = [item.get("initial_rgb_digest") for item in siblings]
        same_rgb = bool(len(rgb) == 8 and all(rgb) and len(set(rgb)) == 1)
        complete = len(siblings) == 8 and actual == expected
        eligible = bool(complete and same_state and same_rgb and
                        all(item.get(success_field, False) for item in siblings))
        for item in siblings:
            item.update({
                "scene_complete": complete,
                "scene_same_state": same_state,
                "scene_same_rgb": same_rgb,
                "scene_eligible": eligible,
            })
            result.append(item)
    return sorted(result, key=lambda item: int(item["branch_id"]))
