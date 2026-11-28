"""Pure, repeatable scoring of recorded physics evidence; never drives a robot."""

import math

from .measurements import EvidenceError
from .spec import PROTOCOL, TASK_CHECKS, TASKS, task_name


def validate_evidence(evidence):
    if evidence.get("protocol") != PROTOCOL:
        raise EvidenceError("Wrong checker protocol")
    rows = evidence.get("samples", [])
    dt = evidence.get("dt")
    if not rows or not isinstance(dt, (float, int)) or not math.isfinite(dt) or dt <= 0:
        raise EvidenceError("Missing physics samples or invalid timestep")
    for index, row in enumerate(rows):
        if type(row.get("physics_step")) is not int or row["physics_step"] != index:
            raise EvidenceError(
                "Physics samples must be unique and consecutive from step zero"
            )
        if (
            not math.isfinite(row.get("time_sec", float("nan")))
            or abs(row["time_sec"] - index * dt) > 1e-8
        ):
            raise EvidenceError("Physics timestamp mismatch")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvidenceError("Nonfinite checker evidence")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        if isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    finite(evidence)
    return rows


def group_checks(task, checks, *, control_spec=None):
    tr_names, jr_names = TASK_CHECKS[task]
    missing = [
        name for name in tr_names + jr_names if type(checks.get(name)) is not bool
    ]
    if missing:
        raise EvidenceError(f"Missing boolean measurements: {missing}")
    tr = all(checks[k] for k in tr_names)
    jr = tr and all(checks[k] for k in jr_names)
    na = set()
    if (
        task == "shake_bottle"
        and (control_spec or {}).get("trace") == "lateral_translation"
    ):
        na = {"wrist_position_stable", "wrist_axis_horizontal"}
    return {
        "task_success": tr,
        "control_success": jr,
        "checks": {k: checks[k] for k in tr_names + jr_names},
        "conditions": {
            k: {
                "group": "TR" if k in tr_names else "JR",
                "status": "not_applicable"
                if k in na
                else "pass"
                if checks[k]
                else "fail",
            }
            for k in tr_names + jr_names
        },
        "failed_tr_checks": [k for k in tr_names if not checks[k]],
        "failed_jr_checks": [k for k in jr_names if not checks[k]],
    }


def score_episode(evidence):
    rows = validate_evidence(evidence)
    task = task_name(evidence["task"])
    try:
        checks, native = TASKS[task].measure(evidence)
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"Invalid {task} measurements: {error}") from error
    result = group_checks(task, checks, control_spec=evidence["control_spec"])
    result.update(
        protocol=PROTOCOL,
        metric_spec_hash=evidence["metric_spec_hash"],
        evidence_valid=True,
        native_diagnostics=native,
        physics_samples=len(rows),
    )
    return result
