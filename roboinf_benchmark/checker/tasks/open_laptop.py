"""Open Laptop: task and instruction success criteria."""

from ..measurements import EvidenceError, final_checks

TR = ("target_angle_reached", "lid_stable", "joint_within_limits")

JR = ("requested_contact_region", "active_arm_contact", "inactive_arm_no_contact")


def measure(evidence):
    if evidence.get("assigned_arm") not in ("left", "right"):
        raise EvidenceError("Laptop task-assigned arm was not initialized")
    return final_checks(evidence)
