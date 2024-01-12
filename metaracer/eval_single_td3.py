"""
Training script for PPO in the single-agent environment.

This file is almost identical to assignment 3 train_ppo.py. Differences:

1. rename `num_envs` argument to `num_processes` to avoid confusion.
2. remove `env_id` argument to avoid issue in assignment 3 and initialize the environment explicitly here.
3. We move make_envs function in this file so that you can easily change the setting of the environment.

-----
2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import argparse
import os
from collections import defaultdict

import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from agents import load_policies
from core.envs import make_envs
from agents.chrispark_td3_agent.td3_trainer import ReplayBuffer, TD3Trainer
from core.utils import pretty_print, Timer, step_envs
from vis import evaluate_in_batch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent-name",
    default="example_agent",
    type=str,
    help="The name of the agent to be evaluated, aka the subfolder name in 'agents/'. Default: example_agent"
)
parser.add_argument(
    "--log-dir",
    default="data/",
    type=str,
    help="The directory where you want to store the data. "
         "Default: ./data/"
)
parser.add_argument(
    "--num-processes",
    default=10,
    type=int,
    help="The number of parallel environments for evaluation. Default: 10"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="The random seed. Default: 0"
)
parser.add_argument(
    "--num-episodes",
    default=10,
    type=int,
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Whether to launch both the top-down renderer and the 3D renderer. Default: False."
)
args = parser.parse_args()


class SingleAgentRacingEnv(MultiAgentRacingEnv):
    """
    MetaDrive provides a MultiAgentRacingEnv class, where all the input/output data is dict. This wrapper class let the
    environment "behaves like a single-agent RL environment" by unwrapping the output dicts from the environment and
    wrapping the action to be a dict for feeding to the environment.
    """

    AGENT_NAME = "agent0"

    def __init__(self, config):
        assert config["num_agents"] == 1
        super(SingleAgentRacingEnv, self).__init__(config)

    @property
    def action_space(self) -> gym.Space:
        return super(SingleAgentRacingEnv, self).action_space[self.AGENT_NAME]

    @property
    def observation_space(self) -> gym.Space:
        return super(SingleAgentRacingEnv, self).observation_space[self.AGENT_NAME]

    def reset(self, *args, **kwargs):
        obs, info = super(SingleAgentRacingEnv, self).reset(*args, **kwargs)
        return obs[self.AGENT_NAME], info[self.AGENT_NAME]

    def step(self, action):
        o, r, tm, tc, i = super(SingleAgentRacingEnv, self).step({self.AGENT_NAME: action})
        return o[self.AGENT_NAME], r[self.AGENT_NAME], tm[self.AGENT_NAME], tc[self.AGENT_NAME], i[self.AGENT_NAME]

    def reward_function(self, vehicle_id):
        """Only the longitudinal movement is in the reward."""
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        longitudinal_last, _ = current_lane.local_coordinates(vehicle.last_position)
        longitudinal_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        self.movement_between_steps[vehicle_id].append(abs(longitudinal_now - longitudinal_last))
        reward = longitudinal_now - longitudinal_last
        step_info["progress"] = (longitudinal_now - longitudinal_last)
        step_info["speed_km_h"] = vehicle.speed_km_h
        step_info["step_reward"] = reward
        step_info["crash_sidewalk"] = False
        if vehicle.crash_sidewalk:
            step_info["crash_sidewalk"] = True
        return reward, step_info

if __name__ == '__main__':
    args = parser.parse_args()

    log_dir = args.log_dir
    num_episodes = args.num_episodes
    num_envs = args.num_processes

    # Create environments
    def single_env_factory():
        return SingleAgentRacingEnv(dict(
            num_agents=1,
        ))

    envs = make_envs(
        single_env_factory=single_env_factory,
        num_envs=num_envs,
        asynchronous=True,
    )

    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    max_action = float(envs.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
    trainer = TD3Trainer(**kwargs)
    trainer.load(os.path.join("agents/" + args.agent_name))


    def _policy(obs):
        return trainer.select_action_in_batch(obs)

    render = args.render
    if render:
        assert num_envs == 1

    eval_reward, eval_info = evaluate_in_batch(
        policy=_policy,
        envs=envs,
        num_episodes=num_episodes
    )

    df = pd.DataFrame({"rewards": eval_info["rewards"], "successes": eval_info["successes"]})
    path = "{}/eval_results.csv".format(log_dir)
    df.to_csv(path)

    print("The average return after running {} agent for {} episodes: {}.\n" \
          "Result is saved at: {}".format(
        "TD3", num_episodes, eval_reward, path
    ))
