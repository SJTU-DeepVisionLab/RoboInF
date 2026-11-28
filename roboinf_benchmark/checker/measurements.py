"""Shared measurements from simulator evidence."""

import math
from copy import deepcopy


class EvidenceError(ValueError):
    """Missing/corrupt observations are not policy failures."""


def inactive_arm_stationary(samples):
    """Aloha: six revolute arm joints (rad), then two prismatic fingers (m)."""
    maxima = [0.0] * 8
    for sample in samples:
        drift = sample.get("inactive_arm_joint_drift")
        if not isinstance(drift, (list, tuple)) or len(drift) != 8:
            raise EvidenceError(
                "Stationary checker requires six arm and two finger joint drifts"
            )
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x < 0
            for x in drift
        ):
            raise EvidenceError("Invalid inactive arm joint drift")
        maxima = [max(a, b) for a, b in zip(maxima, drift)]
    arm, fingers = max(maxima[:6]), max(maxima[6:])
    return arm <= 0.06 and fingers <= 0.03, {
        "max_arm_joint_drift_rad": arm,
        "arm_joint_tolerance_rad": 0.06,
        "max_finger_joint_drift_m": fingers,
        "finger_joint_tolerance_m": 0.03,
        "max_per_joint_drift": maxima,
    }


def final_checks(evidence):
    native = deepcopy(evidence.get("native_final", {}))
    return dict(native.get("checks", {})), native


def contact_checks(evidence):
    checks, native = final_checks(evidence)
    rows = evidence["samples"]
    if any("target_contacts" not in row for row in rows):
        raise EvidenceError("Missing target contact observations")
    checks["requested_arm_contact"] = any(
        contact.get("arm") == evidence["control_spec"]["active_arm"]
        for row in rows
        for contact in row["target_contacts"]
    )
    return checks, native


def native_samples(evidence, required):
    samples = [row["native_sample"] for row in evidence["samples"]]
    required = required | {"sim_step", "inactive_arm_drift", "contacts"}
    for index, sample in enumerate(samples):
        missing = required - sample.keys()
        if missing:
            raise EvidenceError(
                f"Missing physics fields at sample {index}: {sorted(missing)}"
            )
        if index and sample["sim_step"] != samples[index - 1]["sim_step"] + 1:
            raise EvidenceError("Native physics clock is not consecutive")
    return samples


def arm_checks(native, samples):
    checks = dict(native["checks"])
    passed, metrics = inactive_arm_stationary(samples)
    native["stationary_override"] = {
        **metrics,
        "passed": passed,
        "archived_passed": checks["inactive_arm_stationary"],
    }
    checks["inactive_arm_stationary"] = passed
    return checks, native
