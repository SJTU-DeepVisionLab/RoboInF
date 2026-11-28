"""Press Buttons in Order: task and instruction success criteria."""

import itertools

from ..measurements import EvidenceError, arm_checks, native_samples
from .select_switch import RowConfig

TR = ("all_switches_on",)

JR = (
    "initially_all_off",
    "requested_press_order",
    "correct_arm_contact",
    "inactive_arm_stationary",
    "inter_arm_collision_free",
    "released_and_stable",
)

ARMS = ("left", "right")

BUTTONS = ("left", "middle", "right")


def validate_control_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"arm", "press_order"}:
        raise ValueError("Expected only arm and press_order")
    order = spec["press_order"]
    if spec["arm"] not in ARMS:
        raise ValueError("arm must be left or right")
    if (
        not isinstance(order, (list, tuple))
        or len(order) != 3
        or any(not isinstance(item, str) for item in order)
        or set(order) != set(BUTTONS)
    ):
        raise ValueError("press_order must permute left, middle, right")
    return {"arm": spec["arm"], "press_order": list(order)}


def evaluate_press_events(spec, events):
    """Evaluate a chronological list of measured events, including extra presses.

    Malformed/ambiguous events fail closed. The simulator adapter must reject
    simultaneous activations rather than arbitrarily ordering them. This checks
    control fidelity only; final release and planning are separate task checks.
    """
    spec = validate_control_spec(spec)
    events = list(events)
    valid = all(
        isinstance(event, dict)
        and event.get("button") in BUTTONS
        and event.get("arm") in ARMS
        for event in events
    )
    order = [
        event.get("button") if isinstance(event, dict) else None for event in events
    ]
    checks = {
        "valid_press_events": valid,
        "requested_press_order": valid and order == spec["press_order"],
        "requested_arm": valid
        and bool(events)
        and all(event["arm"] == spec["arm"] for event in events),
    }
    return checks, {
        "observed_press_order": order,
        "control_success": all(checks.values()),
    }


def check_rollout(samples, spec, config, dt):
    """Detect ON transitions from physics, associating contact during each press.

    Contacts are accumulated only since the last OFF sample/ON event. Multiple
    ON transitions in one step are ambiguous and fail, regardless of array order.
    """
    import numpy as np

    spec = validate_control_spec(spec)
    checks = {
        "valid_samples": False,
        "initially_all_off": False,
        "requested_press_order": False,
        "correct_arm_contact": False,
        "inactive_arm_stationary": False,
        "inter_arm_collision_free": False,
        "released_and_stable": False,
    }
    result = {"success": False, "checks": checks, "press_events": []}
    if not samples or not np.isfinite(dt) or dt <= 0:
        return result
    progress = np.asarray([s["progress"] for s in samples], dtype=float)
    drift = np.asarray([s["inactive_arm_drift"] for s in samples], dtype=float)
    clearance = np.asarray(
        [
            [s.get("switch_clearance_m", {}).get(p, float("nan")) for p in BUTTONS]
            for s in samples
        ],
        dtype=float,
    )
    if (
        progress.shape != (len(samples), 3)
        or drift.shape != (len(samples),)
        or not all(np.isfinite(a).all() for a in (progress, drift, clearance))
    ):
        return result
    steps = [s.get("sim_step") for s in samples]
    checks["valid_samples"] = all(isinstance(x, int) for x in steps) and all(
        b == a + 1 for a, b in itertools.pairwise(steps)
    )
    checks["initially_all_off"] = bool(
        np.max(np.abs(progress[0])) <= config.off_tolerance
    )
    contacts_by_button = {p: set() for p in BUTTONS}
    previous = progress[0] >= config.on_threshold
    ambiguous = False
    wrong_contact = False
    events = []
    for i, sample in enumerate(samples):
        for j, button in enumerate(BUTTONS):
            if progress[i, j] <= config.off_tolerance:
                contacts_by_button[button].clear()
        for contact in sample.get("contacts", []):
            arm, button = contact.get("arm"), contact.get("switch")
            if arm != spec["arm"]:
                wrong_contact = True
            distance = contact.get("distance_m", float("nan"))
            if (
                button in BUTTONS
                and arm in ARMS
                and contact.get("robot_region", "hand") == "hand"
                and np.isfinite(distance)
                and 0 <= distance <= config.contact_tolerance_m
            ):
                contacts_by_button[button].add(arm)
        current = progress[i] >= config.on_threshold
        rising = np.flatnonzero(current & ~previous)
        ambiguous |= len(rising) > 1
        for j in rising:
            button = BUTTONS[j]
            arms = contacts_by_button[button]
            events.append(
                {
                    "button": button,
                    "arm": next(iter(arms)) if len(arms) == 1 else None,
                    "sim_step": steps[i],
                }
            )
            contacts_by_button[button].clear()
        previous = current
    event_checks, _ = evaluate_press_events(spec, events)
    stable_steps = max(1, int(np.ceil(config.settle_seconds / dt)))
    collisions = [
        c
        for s in samples
        for c in s.get("raw_contacts", [])
        if c.get("actual")
        and len({r.get("arm") for r in c.get("robot_links", []) if r}) > 1
    ]
    checks.update(
        requested_press_order=not ambiguous and event_checks["requested_press_order"],
        correct_arm_contact=not wrong_contact and event_checks["requested_arm"],
        inactive_arm_stationary=bool(np.max(drift) <= config.inactive_arm_tolerance),
        inter_arm_collision_free=not collisions,
        released_and_stable=bool(
            len(samples) >= stable_steps
            and np.all(progress[-stable_steps:] >= config.on_threshold)
            and np.all(clearance[-stable_steps:] >= config.pre_press_distance / 2)
            and all(not s.get("contacts") for s in samples[-stable_steps:])
        ),
    )
    result.update(
        success=all(checks.values()),
        press_events=events,
        observed_press_order=[e["button"] for e in events],
        simultaneous_activation=bool(ambiguous),
        final_progress=progress[-1].tolist(),
    )
    return result


def measure(evidence):
    samples = native_samples(
        evidence, {"progress", "raw_contacts", "switch_clearance_m"}
    )
    config = RowConfig(**evidence["task_config"])
    native = check_rollout(samples, evidence["control_spec"], config, evidence["dt"])
    if not native["checks"].get("valid_samples"):
        raise EvidenceError("Button checker rejected sample integrity")
    checks, native = arm_checks(native, samples)
    checks["all_switches_on"] = all(
        x >= config.on_threshold for x in samples[-1]["progress"]
    )
    return checks, native
