# Policy interfaces

Run the checkpoint in its upstream inference server and install the matching
client in the environment running RoboInF. Each adapter receives only the
instruction, current camera images and robot state. The server is responsible
for checkpoint-specific preprocessing, normalization and inference. No weights
or training code are part of this repository.

| Policy ID | Upstream client | Robot action |
| --- | --- | --- |
| `pi_05` | [OpenPI](https://github.com/Physical-Intelligence/openpi), `openpi_client.websocket_client_policy.WebsocketClientPolicy` | Absolute joint positions, 14 values per step |
| `openwam` | [OpenWAM RoboTwin ModelClient](https://github.com/OpenWAM-Official/OpenWAM/blob/1dcaeea76501f2431286b511284a671778645996/benchmarks/robotwin/openwam2robotwin_interface.py) | End-effector pose and gripper for each arm, or 14 joint values |
| `gr00t` | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), `gr00t.policy.server_client.PolicyClient` | Absolute joint targets from two seven-value arm groups |
| `lingbot-va` | [LingBot-VA RoboTwin client](https://github.com/Robbyant/lingbot-va/tree/main/evaluation/robotwin), `evaluation.robotwin.websocket_client_policy.WebsocketClientPolicy` | End-effector poses from the 16-channel RoboTwin checkpoint |

The config is a JSON object. Supply the address at which you started the
upstream server, for example `{"host":"localhost","port":8000}` for OpenPI.
The default ports, in table order, are 8000, 8848, 5555 and 10001. The OpenWAM
and LingBot clients import code under `benchmarks/` and `evaluation/`
respectively; put the corresponding **upstream repository root** on the client
Python path. Importing the RoboInF benchmark itself does not require any of
these clients. Use one active rollout per server. Configure the sampling seed
in the model service if supported; these native remote calls do not promise a
per-episode RNG reset.

- **π0.5**: Serve a checkpoint with OpenPI's ALOHA input transform. The adapter
  sends CHW RGB as `cam_high`, `cam_left_wrist` and `cam_right_wrist`, the
  14-value joint state and the original prompt. The server must return
  `actions` shaped `[50,14]` after its output transform. The runner executes
  the permitted prefix of the chunk. Deployments returning padded or delta
  actions need the matching OpenPI output transform.
- **OpenWAM**: Match `action_mode` to the checkpoint: `"ee"` (default) or
  `"qpos"`. The upstream client sends one action per request and manages its
  own action queue. For EEF models, both proprioception and returned actions use
  upstream rotation conversion. The default `"eef_layout":"physical"` converts
  simulator `wxyz` quaternions into the client's `xyzw` order and back.
  Only for checkpoints trained by treating the original `wxyz` component
  values as `xyzw`, choose `"eef_layout":"legacy_wxyz_values_as_xyzw"`.
- **GR00T**: Use the N1.7-style native policy API with a compatible bimanual
  modality configuration: camera keys `front`, `left_wrist`, `right_wrist`
  (the first is the benchmark's head camera), state/action groups
  `left_arm` and `right_arm` (each six joints plus gripper), and language key
  `annotation.human.task_description`. Images are batched as
  `[1,1,H,W,3]`, state as `[1,1,7]`. The server must return decoded absolute
  targets as `[1,horizon,7]` per group. A different embodiment requires a
  compatible server-side modality configuration.
- **LingBot-VA**: Use the upstream
  [RoboTwin checkpoint](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin).
  The adapter requests native action blocks, skips the initial conditioning
  frame, composes relative poses with the measured episode-start pose, and
  supplies one absolute action per call. To match this checkpoint's
  relative-pose encoding, it composes the stored four components exactly as
  the upstream RoboTwin client does (SciPy `xyzw` multiplication on the
  stored values). When a block has finished, quarter-frame RGB
  observations and that block update the model's KV cache. This interface
  expects the 16-channel RoboTwin output, not a 30-channel joint checkpoint.

The adapter tests use synthetic observations and deterministic replies to
check camera layout, channel grouping, quaternion ordering and cache timing.
They do not run the four checkpoints or reproduce leaderboard scores.
