# Reproduce the paired workload

The comparison is about joint-action GR00T in **simulated** release `test`
and `test_random`, not throughput, val, EEF, or the other three policy clients.

| Component | Revision |
| --- | --- |
| [RoboScope Kit PR head](https://github.com/MINT-SJTU/RoboScope-Kit/commit/bbd0d3f8cfe407dab40a28ef562e3ed72f0077f0) | `bbd0d3f8cfe407dab40a28ef562e3ed72f0077f0` |
| [Kit simulator wrapper and joint replay judge](https://github.com/SJTU-DeepVisionLab/RoboInF/tree/ca694f8) | `ca694f81088a5b7871a1b7d6cc9ca075d3af87cd` |
| [Released native CLI](https://github.com/SJTU-DeepVisionLab/RoboInF/tree/af9d8d5) | `af9d8d583885df48b300ae97f1a45bd07d318b6a` |
| [Native RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/266f3aa) | `266f3aadf505a4f7fe9af0faa41a20f5f47cd123` |
| [Test tables](https://huggingface.co/datasets/JGuo666/RoboInF-4K/tree/1893df2369cd9316f4ef56928252ae309d90471d) | `1893df2369cd9316f4ef56928252ae309d90471d` |
| [Checkpoint](https://huggingface.co/JGuo666/GR00T-N1.7-RoboInF-C2C-30K/tree/8ef7a7426ebdf97c4b1e732a67a9e26e0677fbcd) | `8ef7a7426ebdf97c4b1e732a67a9e26e0677fbcd` |
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/708617f) | `708617f5716c63fc0a327391492ea2b91977ae90` |

Use the [Kit runtime instructions](https://github.com/MINT-SJTU/RoboScope-Kit/blob/bbd0d3f/benchmarks/roboinf/README.md)
and [model card](https://huggingface.co/JGuo666/GR00T-N1.7-RoboInF-C2C-30K/blob/8ef7a7426ebdf97c4b1e732a67a9e26e0677fbcd/README.md)
to provision the dependencies and separately licensed assets. The measured run
used an already provisioned Python 3.10 SAPIEN environment; it is not a
successful clean-build test. The Kit uses the bundled RoboTwin source; the
released native CLI uses the original RoboTwin revision. Both see the same
asset files. In particular, do not apply the release's random texture override
twice.

The host used six NVIDIA A100-SXM4-80GB GPUs and driver 570.133.20.
After one pilot pair from the frozen sample, up to 12 independent episodes
ran at a time (two endpoints per GPU). Each Kit shard used one replica,
one worker and no result cache. Simulator: Python 3.10.20, SAPIEN 3.0.0b1,
MPLib 0.2.1, PyTorch 2.4.1, NumPy 1.26.4 and TOPPRA 0.6.10.
Model server: Python 3.12.13, PyTorch 2.8.0+cu128 and Transformers 4.57.3.

Both paths used the same server implementation with one active episode per
endpoint, seed 0, and the same checkpoint tensors. For offline inference
`config.json:model_name` and `processor_config.json:processor_kwargs.model_name`
pointed to a local copy of the upstream backbone; the public config uses its
Hub ID. Both local shard hashes were checked against the release:

```text
8f37b5b115bdbe5ecaac9f5caf79a0db8e9c01778608de91f93ef5c50c96cbf9  model-00001-of-00002.safetensors
617fe1d70a9c306d27fb978d275f1c369716aa40066dc1945a69f857ce1d3b6f  model-00002-of-00002.safetensors
```

## One case on both paths

Use absolute paths: `KIT` is the Kit worktree, `RT` is its installed simulator
directory, `NATIVE` and `NATIVE_TWIN` are the two pinned native worktrees,
`CKPT` is the policy checkpoint directory, and `OUT` is a new output directory.
`MODEL_PY` is the Isaac-GR00T model environment. `KIT_PY` can be the same
interpreter with the Kit installed. In a separate terminal, start a server;
use a distinct port for each concurrent case:

```bash
"$MODEL_PY" "$RT/RoboInF/examples/groot/serve.py" --checkpoint "$CKPT" --port 5555 --seed 0
```

Set `ISAAC_GROOT_SOURCE` to the pinned model source. For example, generate
both client configs with `jq` before invoking either runner:

```sh
mkdir -p "$OUT"
printf '%s\n' '{"host":"127.0.0.1","port":5555}' > "$OUT/client.json"
jq -n --arg cwd "$OUT" --arg py "$MODEL_PY" \
  --arg pythonpath "$KIT/packages/roboscope-kit/src:$RT/RoboInF:$ISAAC_GROOT_SOURCE" \
  --arg entry "$KIT/examples/roboinf/policy.py" \
  '{schema_version:2,cwd:$cwd,command:["/usr/bin/env",("PYTHONPATH="+$pythonpath),
    $py,$entry,"gr00t","--client-config",($cwd+"/client.json")]}' > "$OUT/policy.json"
```

The selector is the same file on both sides. It filters the full verified
release table, not a synthetic dataset. For example:

```bash
ROUND=1 DOMAIN=test_random TASK=place_bottle
SEL="$KIT/benchmarks/roboinf/run_review.py"
export ROBOSCOPE_ROBOINF_METHOD=gr00t ROBOSCOPE_ROBOINF_JUDGE="$RT/replay-judge.sh"
PYTHONPATH="$KIT/packages/roboscope-kit/src" "$KIT_PY" "$SEL" kit --round "$ROUND" --task "$TASK" -- \
  --benchmark roboinf --suite "$DOMAIN" --checkpoint-source "$CKPT" \
  --benchmark-manifest "$RT/benchmark-manifest.json" --policy-config "$OUT/policy.json" \
  --output-dir "$OUT/kit" --eval-mode strict --concurrency 1 --replicas 1 --run-cache off

PYTHONPATH="$NATIVE:$GROOT_CLIENT_PATH" "$RT/sim-env/bin/python" "$SEL" native --round "$ROUND" --task "$TASK" -- \
  --data "$RT/data" --robotwin "$NATIVE_TWIN" --domain "$DOMAIN" \
  --policy gr00t --config "$OUT/client.json" --policy-seed 0 --output "$OUT/native.jsonl"
"$RT/replay-judge.sh" "$OUT/native.jsonl" > "$OUT/native-judgment.json"
```

The native command uses the **released** `roboinf run` implementation and its
original GR00T adapter. The native interpreter needs the GR00T `PolicyClient`
and its wire dependencies, not the GPU model. For the recorded run the Python
3.10 client path contained unchanged upstream modules `gr00t/policy/{policy,server_client}.py`,
`gr00t/data/{types,embodiment_tags,utils}.py`, and `gr00t/configs/data/embodiment_configs.py`,
without the model-importing package initializers, plus `pyzmq==26.4.0`,
`msgpack` and `msgpack_numpy`. The published server runs in its own Python 3.12 model
environment. The released client's reset has no seed argument; the serving
endpoint's default is explicitly 0. The Kit client sends seed 0 on reset.

Run for the eight task IDs and both domains in rounds 1 through 4; start
fresh servers and simulators for round 5 (same cases as round 1). Use a new
output directory for every path and round. Do not use historical judgments,
demonstration labels, or one path's actions as the other path's predictions.
The native endpoint is the minimum of the task and simulator limits followed
by exactly 250 passive physics steps; `stack_blocks` executes 1,000 rather
than 1,200 policy actions in this runtime. Both paths use the same public
physics replay judge. An individual one-case shard is not a full benchmark.
