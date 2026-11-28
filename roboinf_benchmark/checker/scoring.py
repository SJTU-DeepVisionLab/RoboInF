"""Pure, repeatable scoring of recorded physics evidence; never drives a robot."""
from copy import deepcopy
import math

from .origin import phase1_shake_bottle_contract as shake
from .origin import press_buttons_order_contract as buttons
from .origin import switch_row_contract as switch
from .spec import PROTOCOL, TASK_CHECKS


class EvidenceError(ValueError):
    """Missing/corrupt observations are not policy failures."""


def validate_evidence(evidence):
    if evidence.get("protocol") != PROTOCOL:
        raise EvidenceError("Wrong checker protocol")
    rows = evidence.get("samples", [])
    dt = evidence.get("dt")
    if not rows or not isinstance(dt, (float, int)) or not math.isfinite(dt) or dt <= 0:
        raise EvidenceError("Missing physics samples or invalid timestep")
    for index, row in enumerate(rows):
        if type(row.get("physics_step")) is not int or row["physics_step"] != index:
            raise EvidenceError("Physics samples must be unique and consecutive from step zero")
        if not math.isfinite(row.get("time_sec", float("nan"))) or abs(row["time_sec"] - index * dt) > 1e-8:
            raise EvidenceError("Physics timestamp mismatch")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvidenceError("Nonfinite checker evidence")
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        if isinstance(value, (list, tuple)):
            for item in value: finite(item)
    finite(evidence)
    return rows


def group_checks(task, checks, *, control_spec=None):
    tr_names, jr_names = TASK_CHECKS[task]
    missing = [name for name in tr_names + jr_names if type(checks.get(name)) is not bool]
    if missing:
        raise EvidenceError(f"Missing boolean measurements: {missing}")
    tr = all(checks[k] for k in tr_names)
    jr = tr and all(checks[k] for k in jr_names)
    na = set()
    if task == "phase1_shake_bottle" and (control_spec or {}).get("trace") == "lateral_translation":
        na = {"wrist_position_stable", "wrist_axis_horizontal"}
    return {"task_success": tr, "control_success": jr,
            "checks": {k: checks[k] for k in tr_names + jr_names},
            "conditions": {k: {"group": "TR" if k in tr_names else "JR",
                                "status": "not_applicable" if k in na else "pass" if checks[k] else "fail"}
                           for k in tr_names + jr_names},
            "failed_tr_checks": [k for k in tr_names if not checks[k]],
            "failed_jr_checks": [k for k in jr_names if not checks[k]]}


def inactive_arm_stationary(samples):
    """Aloha: six revolute arm joints (rad), then two prismatic fingers (m)."""
    maxima = [0.0] * 8
    for sample in samples:
        drift = sample.get("inactive_arm_joint_drift")
        if not isinstance(drift, (list, tuple)) or len(drift) != 8:
            raise EvidenceError("Stationary checker requires six arm and two finger joint drifts")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x < 0 for x in drift):
            raise EvidenceError("Invalid inactive arm joint drift")
        maxima = [max(a, b) for a, b in zip(maxima, drift)]
    arm, fingers = max(maxima[:6]), max(maxima[6:])
    return arm <= 0.06 and fingers <= 0.03, {
        "max_arm_joint_drift_rad": arm, "arm_joint_tolerance_rad": 0.06,
        "max_finger_joint_drift_m": fingers, "finger_joint_tolerance_m": 0.03,
        "max_per_joint_drift": maxima,
    }


def score_episode(evidence):
    rows = validate_evidence(evidence)
    task, spec = evidence["task"], evidence["control_spec"]
    native = deepcopy(evidence.get("native_final", {}))
    if task in ("phase1_shake_bottle", "turn_switch_row", "turn_the_switches_in_order"):
        try:
            samples = [r["native_sample"] for r in rows]
            common = {'sim_step', 'inactive_arm_drift', 'contacts'}
            required = (common | {'phase', 'pose_check_enabled', 'trace_check_enabled', 'trace_scalar',
                'trace_orientation_error_deg', 'pose_error_deg', 'trace_axis_world', 'tcp_position_drift_m',
                'bottle_center_drift_m', 'bottle_height_above_table_m', 'active_arm_contact',
                'inactive_arm_contact', 'active_gripper_closed'} if task == 'phase1_shake_bottle'
                else common | {'progress', 'raw_contacts',
                    'target_clearance_m' if task == 'turn_switch_row' else 'switch_clearance_m'})
            for index, sample in enumerate(samples):
                if required - sample.keys():
                    raise EvidenceError(f'Missing physics fields at sample {index}: {required - sample.keys()}')
                if index and sample['sim_step'] != samples[index-1]['sim_step'] + 1:
                    raise EvidenceError('Native physics clock is not consecutive')
                if (task == 'phase1_shake_bottle' and spec['trace'] == 'wrist_rotation'
                        and sample['trace_check_enabled'] and sample['trace_axis_world'] is None):
                    raise EvidenceError('Missing wrist-axis measurement during shaking')
            if task == "phase1_shake_bottle":
                config = shake.ShakeConfig(**evidence["task_config"])
                native = shake.check_rollout(samples, spec, config, evidence["dt"])
            else:
                config = switch.RowConfig(**evidence["task_config"])
                checker = switch if task == "turn_switch_row" else buttons
                native = checker.check_rollout(samples, spec, config, evidence["dt"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceError(f"Invalid archived contract inputs: {error}") from error
    checks = deepcopy(native.get("checks", {}))
    if task in ("phase1_shake_bottle", "turn_switch_row", "turn_the_switches_in_order"):
        checks["inactive_arm_stationary"], stationary = inactive_arm_stationary(samples)
        # Keep archived diagnostics intact; explicitly report the revised paper gate.
        native["stationary_override"] = {
            **stationary, "passed": checks["inactive_arm_stationary"],
            "archived_passed": native["checks"]["inactive_arm_stationary"],
        }
    if task in ("place_the_bottle", "place_fan_orientation"):
        if any("target_contacts" not in r for r in rows):
            raise EvidenceError("Missing target contact observations")
        checks["requested_arm_contact"] = any(c.get("arm") == spec["active_arm"]
                                              for r in rows for c in r["target_contacts"])
    elif task == "phase1_shake_bottle":
        last = rows[-1]
        if "holding_arms" not in last:
            raise EvidenceError("Missing final holding measurement")
        checks["bottle_held"] = bool(last["holding_arms"])
        checks["final_gripper_closed"] = bool(last["holding_arms"])
    elif task == "turn_the_switches_in_order":
        if native["checks"].get("valid_samples") is not True:
            raise EvidenceError("Archived button checker rejected sample integrity")
        checks["all_switches_on"] = all(x >= config.on_threshold for x in samples[-1]["progress"])
    elif task == "sort_blocks_three_in_order":
        if "placement_events" not in evidence:
            raise EvidenceError("Missing placement event history")
        events = evidence["placement_events"]
        observed = [e["colors"][0] for e in events if len(e["colors"]) == 1]
        checks["requested_sort_order"] = (all(len(e["colors"]) == 1 for e in events)
                                           and observed == list(spec["stack_order"]))
        native.setdefault("metrics", {})["observed_sort_order"] = observed
    elif task == "open_the_laptop":
        if evidence.get("assigned_arm") not in ("left", "right"):
            raise EvidenceError("Laptop task-assigned arm was not initialized")
    result = group_checks(task, checks, control_spec=spec)
    result.update(protocol=PROTOCOL, metric_spec_hash=evidence["metric_spec_hash"],
                  evidence_valid=True, native_diagnostics=native, physics_samples=len(rows))
    return result
