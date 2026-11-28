PROTOCOL = "roboinf.paper_table.origin_archive.v2"
TASK_CHECKS = {
    "place_the_bottle": (
        ("target_center_inside_goal", "target_bottle_upright", "active_gripper_open"),
        ("requested_arm_contact", "passive_bottle_upright")),
    "phase1_shake_bottle": (
        ("bottle_held", "final_gripper_closed", "pose_after_shake"),
        ("active_arm_contact", "inactive_arm_no_contact", "inactive_arm_stationary",
         "pose_during_shake", "trace_three_cycles", "trace_amplitude", "lifted_and_held",
         "wrist_position_stable", "wrist_axis_horizontal", "unintended_motion_limited")),
    "place_fan_orientation": (
        ("position", "orientation", "upright", "left_gripper_open", "right_gripper_open"),
        ("requested_arm_contact",)),
    "turn_switch_row": (
        ("initially_all_off", "target_on", "released_and_stable"),
        ("correct_arm_contact", "inactive_arm_stationary", "inter_arm_collision_free", "others_untouched")),
    "place_via_divider": (
        ("payload_at_target", "payload_upright", "payload_stable", "both_grippers_open"),
        ("requested_arm_contact", "inactive_arm_no_contact", "requested_route_followed",
         "divider_unchanged", "divider_collision_free")),
    "open_the_laptop": (
        ("target_angle_reached", "lid_stable", "joint_within_limits"),
        ("requested_contact_region", "active_arm_contact", "inactive_arm_no_contact")),
    "sort_blocks_three_in_order": (
        ("blocks_in_matching_bins", "blocks_on_table", "both_grippers_open", "sort_stable"),
        ("requested_sort_order",)),
    "turn_the_switches_in_order": (
        ("all_switches_on",),
        ("initially_all_off", "requested_press_order", "correct_arm_contact",
         "inactive_arm_stationary", "inter_arm_collision_free", "released_and_stable")),
}
