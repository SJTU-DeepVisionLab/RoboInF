# Physics-evidence checker

The source in `roboinf_benchmark/checker` evaluates recorded physics evidence
for the eight RoboInF tasks. The predicates are the same in `test` and
`test_random`; initial appearance and trajectories need not be the same.
`task_success` is TR. `control_success` is JR and requires TR.
The task contract modules retain the terms in `licenses/RoboTwin-MIT.txt`.

Each [task module](../roboinf_benchmark/checker/tasks) lists its TR criteria
and the additional criteria for JR. Module names follow the project catalogue.
Use the dataset ID when selecting cases from a released manifest:

| Project task | Dataset ID |
| --- | --- |
| Place Bottle | `place_bottle` |
| Shake Bottle | `shake_bottle` |
| Orient Fan | `place_fan` |
| Select Switch | `turn_switch` |
| Transport via Divider | `place_via_divider` |
| Open Laptop | `open_laptop` |
| Sort Blocks in Order | `stack_blocks` |
| Press Buttons in Order | `press_buttons` |

`python -m roboinf_benchmark.checker evidence.json` scores the native physics
evidence object for one rollout. This is **not** a command to score raw action
predictions: those must first be replayed from the correct case and domain
with a physics recorder. A prediction file, final image, or historical label
is not an acceptable substitute for the trajectory evidence. Publishing this
layer alone does not certify end-to-end scoring from the public runner.

For evidence containing the rollout's own `case_id` and `domain`, run
`python -m roboinf_benchmark.checker --judgment evidence.json` to emit one
JSONL judgment for `roboinf score`. Concatenate one row per case in the domain.
Without `--judgment`, the command still returns detailed diagnostics.

### Prediction replay (partial)

For a one-case `joint14_absolute` Place Bottle prediction, the replay command
loads the released case and scene, measures contacts on each physics step, and
prints a judgment bound to the SHA-256 of the prediction file:

```sh
python -m roboinf_benchmark.checker.replay --data ./RoboInF-4K \
  --robotwin ./RoboTwin prediction.jsonl
```

Unsupported tasks and incomplete trajectories raise errors, not failed policy
judgments. This path has been checked against a historical clear receipt; it
is not yet a validated replay evaluator for the full benchmark.
