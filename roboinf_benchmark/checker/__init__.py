"""Evaluate measured RoboInF physics evidence, not policy predictions or demos."""

from ..suite import DOMAINS
from .scoring import EvidenceError, score_episode

__all__ = ["EvidenceError", "judgment_record", "score_episode"]


def judgment_record(evidence):
    """Export the scored episode in the public aggregator's input format."""
    case_id, domain = evidence.get("case_id"), evidence.get("domain")
    if not isinstance(case_id, str) or not case_id or domain not in DOMAINS:
        raise EvidenceError(
            "Judgment export requires a case_id and domain recorded with the evidence"
        )
    result = score_episode(evidence)
    return {
        "case_id": case_id,
        "domain": domain,
        "task_success": result["task_success"],
        "instruction_success": result["control_success"],
    }
