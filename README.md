<div align="center">
<div style="text-align: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://sjtu-deepvisionlab.github.io/RoboInF/assets/img/logo_dark.svg">
        <img src="https://sjtu-deepvisionlab.github.io/RoboInF/assets/img/logo.svg" alt="RoboInF Logo" style="height: 52px;" height="52">
    </picture>
    <h2>A Benchmark for Instruction Fidelity in Robotic Manipulation</h2>
</div>

<div>
    <a href="https://guocalix.github.io/" target='_blank'>Jiawei Guo<sup>1,*,§</sup></a>&emsp;
    <a href="https://github.com/yellowfish0331" target='_blank'>Yu Huang<sup>1,*</sup></a>&emsp;
    <a href="https://github.com/Timekeeper-J" target='_blank'>Dijiang Dong<sup>1,*</sup></a>&emsp;
    <a href="https://github.com/miracle-techlink" target='_blank'>Yue Liu<sup>2</sup></a>&emsp;
    <a href="https://scholar.google.com/citations?user=x5dEuxsAAAAJ&hl=en" target='_blank'>Peiran Ren<sup>2,†</sup></a>&emsp;
    <a href="https://shenwei1231.github.io/" target='_blank'>Wei Shen<sup>1,†</sup></a>
</div>
<div>
    <sup>1</sup>Shanghai Jiao Tong University&emsp;
    <sup>2</sup>Tongji University
</div>
<div>
    <sup>*</sup>Co-first authors&emsp;<sup>§</sup>Project leader&emsp;<sup>†</sup>Corresponding authors
</div>

<div>
    <h4 align="center">
        <a href="https://sjtu-deepvisionlab.github.io/RoboInF/" target='_blank'>
        <img src="https://img.shields.io/badge/🤖-Project%20Page-029C8D">
        </a>
        <a href="https://arxiv.org/abs/2611.30555" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2611.30555-b31b1b.svg">
        </a>
        <a href="https://www.alphaxiv.org/abs/2611.roboinf" target='_blank'>
        <img src="https://img.shields.io/badge/alphaXiv-2611.roboinf-b31b1b.svg">
        </a>
        <a href="https://huggingface.co/datasets/JGuo666/RoboInF-4K" target='_blank'>
        <img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20RoboInF--4K-blue">
        </a>
        <a href="https://discord.gg/h8jPpwNvf" target='_blank'>
        <img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white">
        </a>
        <a href="./LICENSE" target='_blank'>
        <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg">
        </a>
        <img src="https://visitor-badge.laobi.icu/badge?page_id=SJTU-DeepVisionLab.RoboInF&left_text=Visitors&left_color=%23112D44&right_color=%23029C8D">
    </h4>
</div>

<strong>RoboInF evaluates instruction fidelity in simulated bimanual manipulation: from the same initial scene, different instructions ask for a different arm, destination, route, order or manipulation detail. Task success (TR) measures completion; joint success (JR) also requires satisfying every instruction constraint.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="https://sjtu-deepvisionlab.github.io/RoboInF/assets/img/final_teaser.png">
</div>

For task videos, per-task checkers and full results, go check out our <a href="https://sjtu-deepvisionlab.github.io/RoboInF/" target="_blank">project page</a>.

---
</div>


## ⭐ Update
- [2026.11] RoboInF adapters for `test` and `test_random` are available in [RoboScope Kit PR #217](https://github.com/MINT-SJTU/RoboScope-Kit/pull/217). The [historical clear comparison](reports/groot-clear-861) matched on all 861 cases; formal Kit acceptance is pending.
- [2026.11] **🔥 RoboInF v1.0.0 is released!** The evaluation runner and four policy interfaces (`pi_05`, `openwam`, `gr00t`, `lingbot-va`) are now open source.
- [2026.11] The **RoboInF-4K** dataset is available on [Hugging Face](https://huggingface.co/datasets/JGuo666/RoboInF-4K) 🤗, and the [project page](https://sjtu-deepvisionlab.github.io/RoboInF/) is online.

### ✅ TODO
- [x] Release the evaluation runner and policy interfaces
- [x] Release the RoboInF-4K dataset
- [x] Release the project page


## 🎃 Overview
![data_collection_engine](https://sjtu-deepvisionlab.github.io/RoboInF/assets/img/final_data_collection_engine.png)

RoboInF contains eight base tasks built on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). Each task has six to twelve instruction branches that start from the same initial scene and robot state, so a change in behavior can only come from the instruction. Task-specific checkers verify the terminal goal (TR) and use recorded trajectories, contacts and event sequences to check every instruction constraint (JR).


## 📷 RoboInF-4K Dataset
![dataset_overview](https://sjtu-deepvisionlab.github.io/RoboInF/assets/img/final_dataset_overview.png)

RoboInF-4K pairs 4,242 expert demonstrations with simulator-grounded annotations of motion and contact. The evaluation suite has eight tasks, 115 base scenes and 861 instruction cases; the `test` and `test_random` domains reuse the same cases with clear and randomized appearance.


## ⚙️ Installation
Use Python 3.10 or later. Inspecting the suite does not require a GPU, a policy checkpoint or simulator assets.

```bash
git clone --branch main https://github.com/SJTU-DeepVisionLab/RoboInF.git
cd RoboInF
python -m pip install .
python -m pip install huggingface_hub
hf download JGuo666/RoboInF-4K --repo-type dataset --include 'evaluation/*' --local-dir ./RoboInF-4K
```


## ⚡ Evaluation
### Inspect the evaluation inputs
```bash
roboinf inspect --data ./RoboInF-4K --domain test
```

Expect eight tasks, 115 base scenes and 861 instruction cases. The validation cases use a separate set of 115 scenes. No held-out expert recordings or case judgments are needed to run a policy.

### Run a policy
Prepare the [bundled RoboTwin runtime](third_party/robotwin/README.md), its simulator dependencies and licensed assets. Start a compatible inference server following the [policy guide](docs/policies.md). The four supported client interfaces are `pi_05`, `openwam`, `gr00t`, and `lingbot-va`. For a local OpenPI server on port 8000, put `{"host":"localhost","port":8000}` in `pi05.json` and run:

```bash
roboinf run --data ./RoboInF-4K --robotwin ./RoboTwin --domain val \
  --policy pi_05 --config pi05.json --limit 1 --output trial.jsonl
```

The output records the actions executed in that rollout. See the [evaluation protocol](docs/benchmark.md) for camera and action conventions, step budgets, and how to aggregate judgments.

### Score judgments
```bash
roboinf score --data ./RoboInF-4K --domain val --judgments val-judgments.jsonl
```

TR uses `task_success`; JR requires both `task_success` and `instruction_success`.
The [physics checker](docs/physics-checker.md) evaluates measured trajectory evidence.
`roboinf score` aggregates judgments; neither command assigns a score directly to an action file.
See the [GR00T model card](https://huggingface.co/JGuo666/GR00T-N1.7-RoboInF-C2C-30K)
for checkpoint release status.


## 📝 License
The RoboInF authors license their contributions under [CC BY 4.0](LICENSE). Third-party components retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).


## 📑 Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@article{guo2026roboinf,
  title   = {{RoboInF}: A Benchmark for Instruction Fidelity in Robotic Manipulation},
  author  = {Guo, Jiawei and Huang, Yu and Dong, Dijiang and Liu, Yue and Ren, Peiran and Shen, Wei},
  journal = {arXiv preprint arXiv:2611.30555},
  year    = {2026}
}
```
See also [CITATION.cff](CITATION.cff).


## 🤝 Acknowledgement
RoboInF is built on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). We thank the authors for their open-source simulator and assets.


## 📧 Contact
If you have any questions, please open an issue or join our [Discord](https://discord.gg/h8jPpwNvf).
