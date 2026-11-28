"""Transport via Divider: task and instruction success criteria."""

from ..measurements import final_checks as measure

__all__ = ["JR", "TR", "measure"]

TR = ("payload_at_target", "payload_upright", "payload_stable", "both_grippers_open")

JR = (
    "requested_arm_contact",
    "inactive_arm_no_contact",
    "requested_route_followed",
    "divider_unchanged",
    "divider_collision_free",
)
