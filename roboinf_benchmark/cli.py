import argparse
import json
import os
from pathlib import Path

from .scoring import summarize
from .suite import BUDGETS, DOMAINS, load_suite, read_jsonl


def main(argv=None):
    from .policies import POLICY_NAMES
    parser = argparse.ArgumentParser(prog="roboinf")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run", "score"):
        sub = subs.add_parser(name)
        sub.add_argument("--data", required=True, type=Path, help="Root of the downloaded RoboInF-4K dataset")
        sub.add_argument("--domain", choices=DOMAINS, required=True)
        if name == "score":
            sub.add_argument("--judgments", required=True, type=Path, help="Checker judgments for the evaluated rollouts")
        if name == "run":
            sub.add_argument("--robotwin", required=True, type=Path)
            sub.add_argument("--policy", required=True, choices=POLICY_NAMES)
            sub.add_argument("--config", required=True, type=Path, help="Policy JSON configuration")
            sub.add_argument("--output", required=True, type=Path, help="New output JSONL file")
            sub.add_argument("--policy-seed", type=int, default=0)
            sub.add_argument("--task", choices=sorted(BUDGETS))
            sub.add_argument("--limit", type=int, help="Limit number of cases, for installation checks")
    args = parser.parse_args(argv)
    scenes, cases = load_suite(args.data, args.domain, verify_release=True)
    if args.command == "inspect":
        print(json.dumps({"domain": args.domain, "scenes": len(scenes), "cases": len(cases),
                          "tasks": sorted({key[0] for key in scenes})}, indent=2))
    elif args.command == "score":
        print(json.dumps(summarize(cases, args.domain, read_jsonl(args.judgments)), indent=2))
    else:
        from .policies import load_policy
        from .robotwin import environment_factory
        from .runner import run_case
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Policy config must be a JSON object")
        todo = [case for case in cases.values() if not args.task or case["task"] == args.task][:args.limit]
        if not todo:
            raise ValueError("No matching cases")
        output = args.output.resolve()
        partial = output.with_name(output.name + ".partial")
        if output.exists():
            raise FileExistsError(output)
        env_factory = environment_factory(args.robotwin.resolve())
        policy = load_policy(args.policy, config)
        try:
            with partial.open("x", encoding="utf-8") as stream:
                for case in todo:
                    result = run_case(case, scenes[case["task"], case["scene_id"]], args.domain,
                                      env_factory, policy, args.policy_seed)
                    stream.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
            os.link(partial, output)
            partial.unlink()
        finally:
            policy.close()
