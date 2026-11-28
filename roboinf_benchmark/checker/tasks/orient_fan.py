"""Orient Fan: task and instruction success criteria."""

from ..measurements import contact_checks as measure

__all__ = ["JR", "TR", "measure"]

TR = ("position", "orientation", "upright", "left_gripper_open", "right_gripper_open")

JR = ("requested_arm_contact",)
