"""Evaluate measured RoboInF physics evidence, not policy predictions or demos."""

from .scoring import EvidenceError, score_episode

__all__ = ["EvidenceError", "score_episode"]
