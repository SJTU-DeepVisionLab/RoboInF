"""Portable control contract; scene integration awaits the existing button task.

Events must come from measured button activations, never expert action labels.
Each event carries button and arm; sustained contact must produce one event.
"""
from itertools import permutations

ARMS = ("left", "right")
BUTTONS = ("left", "middle", "right")


def validate_control_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"arm", "press_order"}:
        raise ValueError("Expected only arm and press_order")
    order = spec["press_order"]
    if spec["arm"] not in ARMS:
        raise ValueError("arm must be left or right")
    if (not isinstance(order, (list, tuple)) or len(order) != 3
            or any(not isinstance(item, str) for item in order)
            or set(order) != set(BUTTONS)):
        raise ValueError("press_order must permute left, middle, right")
    return {"arm": spec["arm"], "press_order": list(order)}


def iter_control_specs():
    """Stable arm-major IDs: left 0–5, right 6–11."""
    for arm in ARMS:
        for order in permutations(BUTTONS):
            yield {"arm": arm, "press_order": list(order)}


def canonical_text(spec):
    spec = validate_control_spec(spec)
    first, second, third = spec["press_order"]
    return (f"Using the {spec['arm']} arm, press the {first}, {second}, "
            f"and {third} buttons in that order.")


def evaluate_press_events(spec, events):
    """Evaluate a chronological list of measured events, including extra presses.

    Malformed/ambiguous events fail closed. The simulator adapter must reject
    simultaneous activations rather than arbitrarily ordering them. This checks
    control fidelity only; final release and planning are separate task checks.
    """
    spec = validate_control_spec(spec)
    events = list(events)
    valid = all(isinstance(event, dict)
                and event.get("button") in BUTTONS
                and event.get("arm") in ARMS for event in events)
    order = [event.get("button") if isinstance(event, dict) else None
             for event in events]
    checks = {
        "valid_press_events": valid,
        "requested_press_order": valid and order == spec["press_order"],
        "requested_arm": valid and bool(events) and all(
            event["arm"] == spec["arm"] for event in events),
    }
    return checks, {"observed_press_order": order,
                    "control_success": all(checks.values())}


TASK_NAME = "press_buttons_order"


def branch_key(scene_id, spec):
    spec = validate_control_spec(spec)
    return f"{TASK_NAME}_scene_{scene_id:06d}__{spec['arm']}__{'_'.join(spec['press_order'])}"


def check_rollout(samples, spec, config, dt):
    """Detect ON transitions from physics, associating contact during each press.

    Contacts are accumulated only since the last OFF sample/ON event. Multiple
    ON transitions in one step are ambiguous and fail, regardless of array order.
    """
    import numpy as np
    spec = validate_control_spec(spec)
    checks = dict(valid_samples=False, initially_all_off=False, requested_press_order=False,
                  correct_arm_contact=False, inactive_arm_stationary=False,
                  inter_arm_collision_free=False, released_and_stable=False)
    result = {"success": False, "checks": checks, "press_events": []}
    if not samples or not np.isfinite(dt) or dt <= 0:
        return result
    progress = np.asarray([s["progress"] for s in samples], dtype=float)
    drift = np.asarray([s["inactive_arm_drift"] for s in samples], dtype=float)
    clearance = np.asarray([[s.get("switch_clearance_m", {}).get(p, float("nan"))
                             for p in BUTTONS] for s in samples], dtype=float)
    if (progress.shape != (len(samples), 3) or drift.shape != (len(samples),)
            or not all(np.isfinite(a).all() for a in (progress, drift, clearance))):
        return result
    steps = [s.get("sim_step") for s in samples]
    checks["valid_samples"] = all(isinstance(x, int) for x in steps) and all(
        b == a + 1 for a, b in zip(steps, steps[1:]))
    checks["initially_all_off"] = bool(np.max(np.abs(progress[0])) <= config.off_tolerance)
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
            if (button in BUTTONS and arm in ARMS and contact.get("robot_region", "hand") == "hand"
                    and np.isfinite(distance) and 0 <= distance <= config.contact_tolerance_m):
                contacts_by_button[button].add(arm)
        current = progress[i] >= config.on_threshold
        rising = np.flatnonzero(current & ~previous)
        ambiguous |= len(rising) > 1
        for j in rising:
            button = BUTTONS[j]
            arms = contacts_by_button[button]
            events.append({"button": button, "arm": next(iter(arms)) if len(arms) == 1 else None,
                           "sim_step": steps[i]})
            contacts_by_button[button].clear()
        previous = current
    event_checks, _ = evaluate_press_events(spec, events)
    stable_steps = max(1, int(np.ceil(config.settle_seconds / dt)))
    collisions = [c for s in samples for c in s.get("raw_contacts", []) if c.get("actual")
                  and len({r.get("arm") for r in c.get("robot_links", []) if r}) > 1]
    checks.update(
        requested_press_order=not ambiguous and event_checks["requested_press_order"],
        correct_arm_contact=not wrong_contact and event_checks["requested_arm"],
        inactive_arm_stationary=bool(np.max(drift) <= config.inactive_arm_tolerance),
        inter_arm_collision_free=not collisions,
        released_and_stable=bool(len(samples) >= stable_steps
            and np.all(progress[-stable_steps:] >= config.on_threshold)
            and np.all(clearance[-stable_steps:] >= config.pre_press_distance / 2)
            and all(not s.get("contacts") for s in samples[-stable_steps:])),
    )
    result.update(success=all(checks.values()), press_events=events,
                  observed_press_order=[e["button"] for e in events],
                  simultaneous_activation=bool(ambiguous), final_progress=progress[-1].tolist())
    return result


def finalize_groups(records, tolerance=1e-5, success_field="expert_success"):
    from .switch_row_contract import state_difference
    groups = {}
    for record in records:
        groups.setdefault(record["scene_id"], []).append(dict(record))
    expected = {(s["arm"], tuple(s["press_order"])) for s in iter_control_specs()}
    result = []
    for group in groups.values():
        actual = {(r["control_spec"]["arm"], tuple(r["control_spec"]["press_order"])) for r in group}
        baseline = group[0].get("initial_state")
        same = baseline is not None and all(r.get("initial_state") is not None and
            state_difference(baseline, r["initial_state"]) <= tolerance for r in group)
        images = [r.get("initial_rgb_digest") for r in group]
        same_rgb = bool(images[0]) and len(set(images)) == 1
        complete = len(group) == 12 and actual == expected
        for r in group:
            r.update(scene_complete=complete, scene_same_state=same, scene_same_rgb=same_rgb,
                     scene_eligible=bool(complete and same and same_rgb and all(
                         x.get(success_field, False) for x in group)))
            result.append(r)
    return sorted(result, key=lambda r: r["branch_id"])

