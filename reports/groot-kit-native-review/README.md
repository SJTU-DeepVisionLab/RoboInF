# GR00T: Kit vs. released RoboInF on the frozen review sample

[PR #217](https://github.com/MINT-SJTU/RoboScope-Kit/pull/217) requested formal
review for joint-action GR00T on `test` and `test_random`. This report follows
the [maintainer-approved reduced workload](https://github.com/MINT-SJTU/RoboScope-Kit/pull/217#issuecomment-7238217073).
It is not a new full-suite score or a statement of upstream acceptance.

**Result:** 160/160 real rollouts; 80 paired comparisons with 0 TR and 0 JR disagreements; 0 unequal action traces.
The independent repeat compared 32 trajectories with 0 TR and 0 JR disagreements; 0 unequal action traces.
0 non-terminal scheduling attempts;
0 Kit worker failures.
Each file hash binds a judgment to the prediction from its own runner.
Initial observations were consumed by the policy, not retained for a pixel-by-pixel comparison.

## Four distinct cases per task

| Domain | Path | Episodes | TR | JR | Executed actions |
| --- | --- | ---: | ---: | ---: | ---: |
| `test` | native | 32 | 8/32 (25.0%) | 3/32 (9.4%) | 18,400 |
| `test` | kit | 32 | 8/32 (25.0%) | 3/32 (9.4%) | 18,400 |
| `test_random` | native | 32 | 7/32 (21.9%) | 6/32 (18.8%) | 18,400 |
| `test_random` | kit | 32 | 7/32 (21.9%) | 6/32 (18.8%) | 18,400 |

Both runners produced the following task-level results. Each denominator is
four distinct cases; the repeat is excluded. All 32 clear action traces differ
from the randomized traces for the same case IDs, even where judgments agree.

| Task ID | Clear TR | Clear JR | Random TR | Random JR |
| --- | ---: | ---: | ---: | ---: |
| `open_laptop` | 0/4 (0.0%) | 0/4 (0.0%) | 1/4 (25.0%) | 1/4 (25.0%) |
| `place_bottle` | 2/4 (50.0%) | 2/4 (50.0%) | 2/4 (50.0%) | 2/4 (50.0%) |
| `place_fan` | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| `place_via_divider` | 2/4 (50.0%) | 1/4 (25.0%) | 3/4 (75.0%) | 3/4 (75.0%) |
| `press_buttons` | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| `shake_bottle` | 2/4 (50.0%) | 0/4 (0.0%) | 1/4 (25.0%) | 0/4 (0.0%) |
| `stack_blocks` | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| `turn_switch` | 2/4 (50.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |

## Repeat of round one

The fifth round used fresh model services, new simulators and separate output directories.

| Domain | Path | Episodes | TR | JR | Executed actions |
| --- | --- | ---: | ---: | ---: | ---: |
| `test` | native | 8 | 2/8 (25.0%) | 1/8 (12.5%) | 4,600 |
| `test` | kit | 8 | 2/8 (25.0%) | 1/8 (12.5%) | 4,600 |
| `test_random` | native | 8 | 2/8 (25.0%) | 2/8 (25.0%) | 4,600 |
| `test_random` | kit | 8 | 2/8 (25.0%) | 2/8 (25.0%) | 4,600 |

## Evidence

- [Revisions, environment, commands and scope](reproduce.md).
- [Frozen case IDs](review-sample-v1.json) and [per-episode terminal records](episodes.jsonl).
- [Predictions](predictions/): one gzip stream per episode containing the exact native-format prediction bytes.
- [Comparison report](comparison.json): every task/domain/path, paired result, repeat and first differing action (null for equal trajectories).
- [Execution records](execution.jsonl) and [simulator logs](simulation-logs.jsonl.gz). The latter is gzip JSONL with one `text` field per stream; local installation paths and terminal formatting are removed.

To verify all hashes and comparisons offline with Python 3.10+, from this directory:

```sh
python verify.py --cases /path/to/RoboInF-4K/evaluation/test-cases.jsonl
```

Omit `--cases` to verify against the included frozen IDs without downloading
the release table. This checks coverage, action dimensions, native horizons,
per-episode identity, prediction and action hashes, paired TR/JR, and the
repeat. It does not re-simulate the physics; use the public replay command for that.

The historical 861-case clear comparison with the **internal** native runner
is [separate evidence](../groot-clear-861/). These measurements do not validate
other checkpoints, the EEF action profile or the pi_05, OpenWAM and LingBot-VA clients.
