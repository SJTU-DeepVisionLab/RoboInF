# Physics-evidence checker

The source in `roboinf_benchmark/checker` evaluates recorded physics evidence
for the eight RoboInF tasks. The predicates are the same in `test` and
`test_random`; initial appearance and trajectories need not be the same.
`task_success` is TR. `control_success` is JR and requires TR.
The task contract modules retain the terms in `licenses/RoboTwin-MIT.txt`.

`python -m roboinf_benchmark.checker evidence.json` scores the native physics
evidence object for one rollout. This is **not** a command to score raw action
predictions: those must first be replayed from the correct case and domain
with a physics recorder. A prediction file, final image, or historical label
is not an acceptable substitute for the trajectory evidence. Publishing this
layer alone does not certify end-to-end scoring from the public runner.
