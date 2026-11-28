# Historical GR00T clear comparison

These are policy rollout judgments, not demonstration labels. The policy was
[GR00T N1.7 RoboInF C2C 30K](https://huggingface.co/JGuo666/GR00T-N1.7-RoboInF-C2C-30K/tree/8ef7a7426ebdf97c4b1e732a67a9e26e0677fbcd).

| File | Source |
| --- | --- |
| `internal-native.jsonl` | Historical internal evaluation, **not** the released `roboinf run` |
| `kit.jsonl` | Kit clear rollout at `1dfce0b`; `prediction_sha256` hashes the exact one-case prediction file used to produce the receipt |

Join by `case_id` in the released `evaluation/test-cases.jsonl`. Both files
contain 861 unique `test` cases. TR passed for 175 and JR for 54. Neither
predicate disagreed on any paired case. This comparison does not cover the
current Kit implementation, `test_random`, or independent repeatability.
