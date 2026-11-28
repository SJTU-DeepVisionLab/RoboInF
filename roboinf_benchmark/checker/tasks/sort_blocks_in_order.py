"""Sort Blocks in Order: task and instruction success criteria."""

from ..measurements import EvidenceError, final_checks

TR = ("blocks_in_matching_bins", "blocks_on_table", "both_grippers_open", "sort_stable")

JR = ("requested_sort_order",)


def measure(evidence):
    checks, native = final_checks(evidence)
    if "placement_events" not in evidence:
        raise EvidenceError("Missing placement event history")
    events = evidence["placement_events"]
    observed = [e["colors"][0] for e in events if len(e["colors"]) == 1]
    checks["requested_sort_order"] = all(
        len(e["colors"]) == 1 for e in events
    ) and observed == list(evidence["control_spec"]["stack_order"])
    native.setdefault("metrics", {})["observed_sort_order"] = observed
    return checks, native
