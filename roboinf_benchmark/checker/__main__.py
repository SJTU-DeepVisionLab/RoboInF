"""Score one complete physics-evidence JSON object."""

import argparse
import json
from pathlib import Path

from . import score_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    result = score_episode(json.loads(args.evidence.read_text(encoding="utf-8")))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
