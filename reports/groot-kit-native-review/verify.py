"""Verify published prediction bytes, coverage, paired judgments and repeatability."""

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HORIZONS = {
    "place_bottle": 400,
    "turn_switch": 400,
    "place_fan": 400,
    "place_via_divider": 400,
    "press_buttons": 600,
    "open_laptop": 700,
    "shake_bottle": 700,
    "stack_blocks": 1000,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(reference, candidate):
    a, b = reference["actions"], candidate["actions"]
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
    return {
        "TR_disagrees": reference["task_success"] != candidate["task_success"],
        "JR_disagrees": reference["instruction_success"]
        != candidate["instruction_success"],
        "first_different_action": first,
    }


def verify(root, case_table=None):
    protocol = json.loads((root / "review-sample-v1.json").read_bytes())
    rounds = protocol["rounds"]
    require(len(rounds) == 5 and rounds[0] == rounds[-1], "Invalid repeat definition")
    for task in HORIZONS:
        require(
            len({r[task] for r in rounds[:4]}) == 4,
            f"Non-distinct primary cases: {task}",
        )
    if case_table is not None:
        raw = case_table.read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == protocol["input_sha256"],
            "Not the released case table",
        )
        table = [json.loads(line) for line in raw.splitlines()]
        for task in HORIZONS:
            ids = [r["case_id"] for r in table if r["task"] == task]
            selected = sorted(
                ids,
                key=lambda c: hashlib.sha256(
                    f"{protocol['seed']}:{task}:{c}".encode()
                ).digest(),
            )[:4]
            require(
                selected == [r[task] for r in rounds[:4]],
                f"Hash selection mismatch: {task}",
            )
    expected = {
        (i, d, t, p): c
        for i, r in enumerate(rounds, 1)
        for t, c in r.items()
        for p in ("kit", "native")
        for d in ("test", "test_random")
    }
    require(len(expected) == 160, "Invalid workload size")
    rows = {}
    metrics = collections.defaultdict(collections.Counter)
    for line in (root / "episodes.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        key = (r["round"], r["domain"], r["task"], r["path"])
        require(
            key not in rows and expected.get(key) == r["case_id"],
            f"Duplicate or unexpected {key}",
        )
        require(
            type(r["task_success"]) is bool
            and type(r["instruction_success"]) is bool
            and (r["task_success"] or not r["instruction_success"]),
            f"Invalid outcome {key}",
        )
        prediction = (
            root
            / "predictions"
            / f"r{key[0]}"
            / key[1]
            / key[2]
            / (key[3] + ".jsonl.gz")
        )
        data = gzip.decompress(prediction.read_bytes())
        require(
            hashlib.sha256(data).hexdigest() == r["prediction_sha256"],
            f"Receipt hash mismatch {key}",
        )
        pred = json.loads(data)
        require(
            all(
                pred[k] == r[k]
                for k in ("case_id", "task", "domain", "action_count", "policy_seed")
            )
            and pred["action_spec"] == "joint14_absolute"
            and pred["policy"] == "gr00t"
            and pred["policy_seed"] == 0,
            f"Incorrect trajectory identity {key}",
        )
        actions = pred["actions"]
        require(
            len(actions) == pred["action_count"] == HORIZONS[key[2]],
            f"Wrong horizon {key}",
        )
        require(
            all(
                len(a) == 14
                and all(type(x) in (int, float) and math.isfinite(x) for x in a)
                for a in actions
            ),
            f"Invalid action {key}",
        )
        digest = hashlib.sha256(
            json.dumps(actions, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        require(digest == r["actions_sha256"], f"Action hash mismatch {key}")
        rows[key] = {**r, "actions": actions}
        metrics[(key[1], key[2], key[3])].update(
            episodes=1,
            steps=len(actions),
            TR=int(r["task_success"]),
            JR=int(r["instruction_success"]),
        )
    require(
        rows.keys() == expected.keys(),
        f"Missing {sorted(expected.keys() - rows.keys())}",
    )
    pairs, repeats, appearance = [], [], []
    for (i, domain, task, path), row in sorted(rows.items()):
        if path == "kit":
            pairs.append(
                dict(
                    round=i,
                    domain=domain,
                    task=task,
                    case_id=row["case_id"],
                    **compare(rows[i, domain, task, "native"], row),
                )
            )
        if i == 5:
            repeats.append(
                dict(
                    domain=domain,
                    task=task,
                    case_id=row["case_id"],
                    path=path,
                    **compare(rows[1, domain, task, path], row),
                )
            )
        if i < 5 and domain == "test_random" and path == "kit":
            appearance.append(
                dict(
                    round=i,
                    task=task,
                    case_id=row["case_id"],
                    **compare(rows[i, "test", task, path], row),
                )
            )

    def counts(items):
        return {
            "comparisons": len(items),
            "TR_disagreements": sum(x["TR_disagrees"] for x in items),
            "JR_disagreements": sum(x["JR_disagrees"] for x in items),
            "unequal_action_traces": sum(
                x["first_different_action"] is not None for x in items
            ),
        }

    return {
        "episodes": len(rows),
        "paired": counts(pairs),
        "repeat": counts(repeats),
        "clear_vs_random_kit_distinct_cases": counts(appearance),
        "tasks": [
            dict(domain=d, task=t, path=p, **v)
            for (d, t, p), v in sorted(metrics.items())
        ],
        "pairs": pairs,
        "repeats": repeats,
        "clear_vs_random": appearance,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        help="Verify hash sampling against released test-cases.jsonl",
    )
    result = verify(ROOT, parser.parse_args().cases)
    print(json.dumps(result, indent=2))
    gate = ("TR_disagreements", "JR_disagreements", "unequal_action_traces")
    raise SystemExit(
        int(any(result[group][k] for group in ("paired", "repeat") for k in gate))
    )
