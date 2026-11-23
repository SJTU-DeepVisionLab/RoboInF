from contextlib import closing

from .contracts import ACTION_SPECS, ActionChunk, EpisodeContext
from .suite import BUDGETS


def run_case(case, scene, domain, env_factory, policy, seed=0):
    desc = policy.describe()
    if desc.action_spec not in ACTION_SPECS or desc.input_mode != "task_instruction":
        raise ValueError("Expected a policy with a supported action space and task instructions")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid policy seed")
    with closing(env_factory(scene, case, domain)) as env:
        if type(env.step_limit) is not int or env.step_limit < 1 or env.action_steps != 0 or env.physics_steps != 0:
            raise ValueError("Simulator must start at action/physics step zero with a positive step limit")
        limit = min(BUDGETS[case["task"]], env.step_limit)
        policy.reset(EpisodeContext(case["instruction"], seed))
        actions = []
        last_time = -1.0
        while len(actions) < limit:
            obs = env.observe().checked_copy()
            if obs.action_step != len(actions) or obs.time_sec < last_time:
                raise ValueError("Unexpected observation clock")
            last_time = obs.time_sec
            chunk = policy.predict(obs)
            if not isinstance(chunk, ActionChunk):
                raise TypeError("Policy must return ActionChunk")
            planned = chunk.checked(desc.action_spec)
            for action in planned[:min(20, limit - len(actions))]:
                before = env.physics_steps
                env.execute(action, desc.action_spec)
                if env.action_steps != len(actions) + 1 or env.physics_steps <= before:
                    raise RuntimeError("Simulator failed to advance")
                actions.append(action.tolist())
        before = env.physics_steps
        env.settle(250)
        if env.physics_steps != before + 250 or env.action_steps != limit:
            raise RuntimeError("Simulator did not complete exactly 250 passive physics steps")
        return {"case_id": case["case_id"], "task": case["task"], "domain": domain,
                "policy": desc.method_id, "policy_seed": seed, "action_spec": desc.action_spec,
                "action_count": limit, "actions": actions}
