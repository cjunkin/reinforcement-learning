
# MetaRacer
**A reinforcement learning agent trained in MetaDrive Racing Environment**

To be updated...

## Environment overview

The MetaDrive racing environment contains a long two-lane track with no intersection, no traffic vehicles and with a high wall that blocks vehicles driving out of the road.

The environment is natively a multi-agent RL environment. The input and output data is all in the form of Python dicts, whose keys are the agents' name such as `agent0`, `agent1`, ..., and values are the corresponding information.

In `train_ppo_in_multiagent_env.py` and `eval_single_agent_env.py`, a wrapper of the environment that sets `config["num_agents"] = 1` and wraps and unwraps the data passing in and out from the environment. Therefore, the environment behaves like a single-agent RL environment and we can reuse the single-agent RL algorithm such as PPO to train agent in this environment.

In `train_ppo_in_multiagent_env.py`, a wrapper that still makes the environment behaves like a single-agent env, but this time we can load a trained agent serving as the `agent1` and let the learning agent to control `agent0`.