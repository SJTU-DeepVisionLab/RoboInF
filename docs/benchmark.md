# Evaluation protocol

RoboInF measures whether a policy completes the physical task (TR) and whether the same rollout also satisfies every instruction constraint (JR). A task's rate is the fraction of successful cases. The benchmark average gives equal weight to each of the eight tasks, separately for validation, clear test, and appearance-randomized test.

## Inputs

Download the evaluation directory from [RoboInF-4K](https://huggingface.co/datasets/JGuo666/RoboInF-4K). The normal-language instruction is a policy input. The environment alone reads scene identifiers, scene seeds and appearance settings. The policy receives a fixed policy RNG seed, independently of the scene seed.

| Domain | Cases | Base scenes | Scene table | Case table |
| --- | ---: | ---: | --- | --- |
| val | 861 | 115 | `evaluation/val-scenes.jsonl` | `evaluation/val-cases.jsonl` |
| test | 861 | 115 | `evaluation/test-scenes.jsonl` | `evaluation/test-cases.jsonl` |
| test_random | 861 | same 115 | same as test, with the specified wall/table textures and lighting | same as test |

Each case has a unique `case_id`, `task`, `scene_id`, and `instruction`. The two test domains use identical case IDs; include the domain when comparing rollouts. Scene names and seeds do not identify a unique instruction. Test expert trajectories and judgments are withheld.

```sh
roboinf inspect --data ./RoboInF-4K --domain test_random
```

## Running a policy

Use Python 3.10 and the [RoboTwin installation and asset instructions](https://github.com/RoboTwin-Platform/RoboTwin). The simulator was exercised with [this RoboTwin revision](https://github.com/RoboTwin-Platform/RoboTwin/tree/266f3aadf505a4f7fe9af0faa41a20f5f47cd123), MPLib 0.2.1 and NumPy 1.26.4. Use a clean environment; on Linux, `PYTHONNOUSERSITE=1` prevents a user-installed NumPy 2 from shadowing the version expected by MPLib's native extension. Set up the simulator and CuRobo planner in the environment that runs the benchmark. The adapter expects the RoboTwin 2 source layout with `envs/_base_task.py`, `env_cfg/task_config/demo_clean.yml`, the Aloha-AgileX embodiment and its expanded planner configuration. Obtain the assets separately under the terms of their providers. Set up the policy being evaluated according to [the policy interfaces](policies.md).

```sh
roboinf run --robotwin ./RoboTwin --data ./RoboInF-4K --domain val \
  --policy pi_05 --config pi05.json --limit 1 --output trial.jsonl
```

Remove `--limit` to evaluate all cases in the domain. Supply a new output file for each policy and domain. The runner writes completed episodes to a temporary `.partial` file until the run finishes; an error leaves the partial output for diagnosis and does not assign a failure judgment. The benchmark does not install weights or a model service.

The policy observes three current RGB images (head, left wrist, right wrist), 14 joint/gripper coordinates, 16 absolute end-effector/gripper coordinates when available, the action index and simulation time. The front camera included with the training data is not a policy input. At step zero the policy resets with the case instruction. The two supported output spaces are:

- `joint14_absolute`: left six joint angles in radians and left gripper, then the same for the right arm.
- `eef16_absolute_wxyz`: position in metres, unit quaternion in wxyz order and gripper for each arm, left first, in the native RoboTwin arm-pose frame.

The runner executes at most 20 actions from each predicted chunk, or fewer if the chunk or remaining budget is shorter. It discards the unexecuted suffix, obtains a fresh observation, and never stops on early task success. The action budget is the smaller of the task budget and the initialized simulator limit. The fixed endpoint includes 250 passive physics steps after the last executed action.

| Task | Action budget |
| --- | ---: |
| place_bottle, place_fan, turn_switch, place_via_divider | 400 |
| press_buttons | 600 |
| open_laptop, shake_bottle | 700 |
| stack_blocks | 1,200 |

## Predictions and scores

A predictions JSONL file contains one row per completed case. Each row records the case ID, task, domain, policy, policy seed, action specification, action count and the ordered list of **executed** actions. A one-case installation check is not a full benchmark result.

Official judgments require replay or equivalent simulator evidence for the policy's own rollout. Expert demonstration labels must never be reused to score a new policy. The [physics checker](physics-checker.md) scores measured evidence. The public aggregation command consumes judgments produced by an evaluator; it does not infer physical success from action arrays.

A checker export has exactly these fields per case (illustrative values):

```json
{"case_id":"example","domain":"val","task_success":true,"instruction_success":false}
```

```sh
roboinf score --data ./RoboInF-4K --domain val --judgments val-judgments.jsonl
```

TR uses `task_success`; JR requires both booleans. Dropouts and infrastructure errors must be resolved before producing a score. The command rejects unknown, duplicate and missing cases rather than changing the denominator. Comparisons with paper results additionally require matching simulator dependencies, assets, dynamics, rendering, and policy settings; merely matching the scene seed does not verify the full initial state.
