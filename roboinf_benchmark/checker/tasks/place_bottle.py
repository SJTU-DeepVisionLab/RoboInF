"""Place Bottle: task and instruction success criteria."""

from ..measurements import contact_checks as measure

__all__ = ["JR", "TR", "measure"]

TR = ("target_center_inside_goal", "target_bottle_upright", "active_gripper_open")

JR = ("requested_arm_contact", "passive_bottle_upright")
