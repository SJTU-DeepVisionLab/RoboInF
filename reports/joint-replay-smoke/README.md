# Joint-action replay checks

`historical-clear.jsonl` contains one receipt per supported task. The public
checker replayed the historical Kit prediction in SAPIEN to the native action
horizon plus 250 passive physics steps. The case, domain, TR, JR and prediction
hash matched the corresponding historical judgment. These examples cover both
positive and negative outcomes; they are not an estimate of policy performance.

`scene-checks.json` records the exit status of Kit `sim_case.py --self-test`
at `145b31a` on round one's eight frozen cases in each domain. The check
constructs the scene, renders and validates the initial observation, then closes
the simulator. It does not load a policy, execute actions or score a trajectory.
Neither file counts toward the paired 160-rollout comparison.
