# RoboInF

[Project and results](https://sjtu-deepvisionlab.github.io/RoboInF/) · [RoboInF-4K dataset](https://huggingface.co/datasets/JGuo666/RoboInF-4K) · [Evaluation protocol](docs/benchmark.md) · [Policy setup](docs/policies.md)

RoboInF evaluates instruction fidelity in simulated bimanual manipulation. Several instructions share an initial scene: a policy might place the same object using a different arm, choose a different destination, or perform a sequence in a different order. Task success (TR) measures completion; joint success (JR) also requires satisfying every instruction constraint.

## Inspect the evaluation inputs

Use Python 3.10 or later. Inspecting the suite does not require a GPU, a policy checkpoint or simulator assets.

```bash
git clone --branch main https://github.com/SJTU-DeepVisionLab/RoboInF.git
cd RoboInF
python -m pip install .
python -m pip install huggingface_hub
hf download JGuo666/RoboInF-4K --repo-type dataset --include 'evaluation/*' --local-dir ./RoboInF-4K
roboinf inspect --data ./RoboInF-4K --domain test
```

Expect eight tasks, 115 base scenes and 861 instruction cases. The `test` and `test_random` domains reuse the same cases with different appearances. The validation cases use a separate set of 115 scenes. No held-out expert recordings or case judgments are needed to run a policy.

## Run a policy

Install [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/266f3aadf505a4f7fe9af0faa41a20f5f47cd123), its simulator dependencies and assets. Start a compatible inference server following the [policy guide](docs/policies.md). The four supported client interfaces are `pi_05`, `openwam`, `gr00t`, and `lingbot-va`. For a local OpenPI server on port 8000, put `{"host":"localhost","port":8000}` in `pi05.json` and run:

```bash
roboinf run --data ./RoboInF-4K --robotwin ./RoboTwin --domain val \
  --policy pi_05 --config pi05.json --limit 1 --output trial.jsonl
```

The output records the actions executed in that rollout. See the [evaluation protocol](docs/benchmark.md) for camera and action conventions, step budgets, and how to aggregate judgments. Official held-out scoring requires an evaluator with the withheld reference information; this repository does not host a scoring service or distribute checkpoints.

## Citation and license

Jiawei Guo, Yu Huang, Dijiang Dong, Yue Liu, Peiran Ren, and Wei Shen. *RoboInF: A Benchmark for Instruction Fidelity in Robotic Manipulation* (2026). See [CITATION.cff](CITATION.cff).

The RoboInF authors license their contributions under [CC BY 4.0](LICENSE). Third-party components retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
