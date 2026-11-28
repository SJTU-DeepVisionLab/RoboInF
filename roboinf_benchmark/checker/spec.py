"""Resolve public dataset and recorder IDs at the evidence boundary."""

from .tasks import (
    open_laptop,
    orient_fan,
    place_bottle,
    press_buttons_in_order,
    select_switch,
    shake_bottle,
    sort_blocks_in_order,
    transport_via_divider,
)

# Serialized protocol ID; do not invalidate recorded evidence.
PROTOCOL = "roboinf.paper_table.origin_archive.v2"

TASK_CHECKS = {
    "place_bottle": (place_bottle.TR, place_bottle.JR),
    "shake_bottle": (shake_bottle.TR, shake_bottle.JR),
    "orient_fan": (orient_fan.TR, orient_fan.JR),
    "select_switch": (select_switch.TR, select_switch.JR),
    "transport_via_divider": (transport_via_divider.TR, transport_via_divider.JR),
    "open_laptop": (open_laptop.TR, open_laptop.JR),
    "sort_blocks_in_order": (sort_blocks_in_order.TR, sort_blocks_in_order.JR),
    "press_buttons_in_order": (press_buttons_in_order.TR, press_buttons_in_order.JR),
}

# IDs on disk may differ from the project catalogue.
_TASK_IDS = {
    "place_the_bottle": "place_bottle",
    "phase1_shake_bottle": "shake_bottle",
    "place_fan_orientation": "orient_fan",
    "place_fan": "orient_fan",
    "turn_switch_row": "select_switch",
    "turn_switch": "select_switch",
    "place_via_divider": "transport_via_divider",
    "open_the_laptop": "open_laptop",
    "sort_blocks_three_in_order": "sort_blocks_in_order",
    "stack_blocks": "sort_blocks_in_order",
    "turn_the_switches_in_order": "press_buttons_in_order",
    "press_buttons": "press_buttons_in_order",
}


def task_name(identifier):
    name = _TASK_IDS.get(identifier, identifier)
    if name not in TASK_CHECKS:
        raise ValueError(f"Unknown RoboInF task: {identifier!r}")
    return name
