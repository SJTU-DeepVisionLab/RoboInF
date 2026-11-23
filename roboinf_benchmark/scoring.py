from collections import defaultdict
from statistics import mean


def summarize(cases, domain, judgments):
    """Aggregate measured task and instruction predicates for this policy's rollouts."""
    per_task, seen = defaultdict(list), set()
    for row in judgments:
        if not isinstance(row, dict) or set(row) != {"case_id", "domain", "task_success", "instruction_success"}:
            raise ValueError("Unexpected checker result fields")
        cid = row["case_id"]
        if cid in seen or cid not in cases or row["domain"] != domain:
            raise ValueError("Unexpected domain, duplicate or unknown case ID")
        tr, ir = row["task_success"], row["instruction_success"]
        if type(tr) is not bool or type(ir) is not bool:
            raise ValueError("Missing boolean checker decision; incomplete runs cannot be scored")
        seen.add(cid)
        per_task[cases[cid]["task"]].append((tr, tr and ir))
    if seen != cases.keys() or not seen:
        raise ValueError("Require one checker decision per case; run errors must be resolved")
    tasks = {task: {"cases": len(rows), "TR": mean(r[0] for r in rows), "JR": mean(r[1] for r in rows)}
             for task, rows in sorted(per_task.items())}
    return {"domain": domain, "tasks": tasks, "macro_TR": mean(t["TR"] for t in tasks.values()),
            "macro_JR": mean(t["JR"] for t in tasks.values())}
