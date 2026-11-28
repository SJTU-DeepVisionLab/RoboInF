# RoboTwin evaluation runtime

This directory contains the simulator modules and four task configuration files
used by RoboInF, from [RoboTwin commit `266f3aa`](https://github.com/RoboTwin-Platform/RoboTwin/tree/266f3aadf505a4f7fe9af0faa41a20f5f47cd123).
The sources retain their [MIT license](LICENSE). The only local source change
is in `Base_Task.create_table_and_wall`: use the released textures when the
episode supplies an appearance. It does not change geometry or sample an RNG.

From a RoboInF source checkout, provision the Python simulator dependencies and
the separately licensed RoboTwin assets, then run:

```sh
cp -R third_party/robotwin ./RoboTwin
ln -s /absolute/path/to/provisioned/assets ./RoboTwin/assets
```

Pass `--robotwin ./RoboTwin` to `roboinf run`. The assets directory must include
the expanded Aloha planner configuration. No upstream Git checkout is required.
