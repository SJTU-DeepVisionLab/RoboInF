"""Score one complete physics-evidence JSON object."""

import argparse
import json
from pathlib import Path

from . import judgment_record, score_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--judgment", action="store_true", help="Emit one JSONL row for roboinf score"
    )
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = judgment_record(evidence) if args.judgment else score_episode(evidence)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
