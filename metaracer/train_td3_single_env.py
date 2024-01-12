import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from core.envs import make_envs
from core.utils import verify_log_dir, pretty_print, Timer
from agents.chrispark_td3_agent.td3_trainer import ReplayBuffer, TD3Trainer, TD3Config

parser = argparse.ArgumentParser()

# You can adjust the environment config here.
# You can check out the environment below to see how the config affects the reward function.
parser.add_argument(
    "--log-dir",
    default="data/td3_agent_single_overhaul",
    type=str,
    help="The path of directory that you want to store the data to. "
            "Default: ./data/"
)
parser.add_argument(
    "--num-processes",
    default=1,
    type=int,
    help="The number of parallel RL environments. Default: 1 (for TD3)"
)
parser.add_argument(
    "--max-steps", 
    default=1e6, 
    type=int,
    help="Max time steps to run environment"
) 
parser.add_argument(
    "--start-steps", 
    default=1e4, 
    type=int
)  # Time steps initial random policy is used
parser.add_argument(
    "--seed", 
    default=0
)
parser.add_argument(
    "--save_freq", 
    default=2e3, 
    type=int
)  # How often (time steps) we save model
parser.add_argument(
    "--log_freq", 
    default=1e3, 
    type=int
)  # How often (time steps) we print stats of model
parser.add_argument(
    "--pretrained-model-log-dir",
    default="",
    type=str,
    help="The folder that hosts the pretrained model. Example: agents/youragentname"
)
parser.add_argument(
    "--pretrained-model-suffix",
    default="",
    type=str,
    help="The suffix of the checkpoint (if you are using PPO during pretraining). Example: iter275"
)

# ===================
# Hyperparameters
# ===================
parser.add_argument(
    "--lr", 
    default=5e-6, 
    type=float
)
parser.add_argument(
    "--explore_noise", 
    default=0.1, 
    type=float
)  # Std of Gaussian exploration noise
parser.add_argument(
    "--batch_size", 
    default=256, 
    type=int
)  # Batch size for both actor and critic
parser.add_argument(
    "--discount", 
    default=0.99, 
    type=float
)  # Discount factor
parser.add_argument(
    "--tau", 
    default=0.005, 
    type=float
)  # Target network update rate
parser.add_argument(
    "--policy_noise", 
    default=0.2, 
    type=float
)  # Noise added to target policy during critic update
parser.add_argument(
    "--noise_clip", 
    default=0.5, 
    type=float
)  # Range to clip target policy noise
parser.add_argument(
    "--policy_freq", 
    default=2, 
    type=int
)  # Frequency of delayed policy updates
parser.add_argument(
    "--progress_multiplier", 
    default=1, 
    type=float
)  # How much we weight progress for agent
parser.add_argument(
    "--idle_multiplier", 
    default=1, 
    type=float
)  

parser.add_argument(
    "--load_model", 
    action="store_true"
)
args = parser.parse_args()


# You can adjust the environment config here.
# You can check out the environment below to see how the config affects the reward function.
ENVIRONMENT_CONFIG = dict(
    num_agents=1,  # Don't change

    # Reward function
    crash_sidewalk_penalty = 50,
    success_reward = 100,
    speed_reward = 0,
    driving_reward = args.progress_multiplier,  # Multiplier for distance covered

    # DEFAULT VALUES
    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    # success_reward=10.0,
    # out_of_road_penalty=5.0,
    # crash_vehicle_penalty=5.0,
    # crash_object_penalty=5.0,
    # driving_reward=1.0,
    # speed_reward=0.1,
    # use_lateral_reward=False,

    # # ===== Cost Scheme =====
    # crash_vehicle_cost=1.0,
    # crash_object_cost=1.0,
    # out_of_road_cost=1.0,
)

EXTRA_ENV_CONFIG = dict(
    driving_cutoff = 100,  # driving_reward cutoff
    idle_multiplier = args.idle_multiplier
)


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
        """
        Reward function copied from metadrive.envs.marl_envs.mark_racing_env
        You can freely adjust the config or add terms.
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        longitudinal_last, _ = current_lane.local_coordinates(vehicle.last_position)
        longitudinal_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        self.movement_between_steps[vehicle_id].append(abs(longitudinal_now - longitudinal_last))

        reward = 0.0
        progress = (longitudinal_now - longitudinal_last)
        reward += self.config["driving_reward"] * progress
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        step_info["progress"] = progress
        step_info["speed_km_h"] = vehicle.speed_km_h

        # [ADDED]
        # progress_reward = min(step_info["progress"] * self.config.progress_multiplier, 100)
        # reward += progress_reward  # Add progress reward to the existing reward

        step_info["step_reward"] = reward
        step_info["crash_sidewalk"] = False
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
            step_info["crash_sidewalk"] = True
        elif self._is_idle(vehicle_id):
            reward = -self.config["idle_penalty"] * EXTRA_ENV_CONFIG["idle_multiplier"]

        return reward, step_info

if __name__ == "__main__":
    # Clean log directory
    algo = "td3"
    log_dir = verify_log_dir(args.log_dir, algo)

    def single_env_factory():
        return SingleAgentRacingEnv(ENVIRONMENT_CONFIG)
    
    trainer_path = log_dir
    progress_path = os.path.join(log_dir, "progress.csv")

    if not os.path.exists(trainer_path):
        os.makedirs(trainer_path)

    # RECORD HYPER-PARAMETERS
    hyperparams_path = os.path.join(log_dir, "hyperparams.csv")
    hyperparams = pd.DataFrame([{
        "lr": args.lr,
        "explore_noise": args.explore_noise,
        "batch_size": args.batch_size,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq, 
        "crash_penalty": ENVIRONMENT_CONFIG["crash_sidewalk_penalty"],
        "success_reward": ENVIRONMENT_CONFIG["success_reward"],
        "speed_reward": ENVIRONMENT_CONFIG["speed_reward"],
        "progress_mulitplier": args.progress_multiplier
    }])
    hyperparams.to_csv(hyperparams_path)  

    # Create vectorized environments
    num_processes = args.num_processes

    environments = make_envs(
        single_env_factory=single_env_factory,
        num_envs=num_processes,
        asynchronous=False,
    )
    env = environments.envs[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["lr"] = args.lr
    config = TD3Config(**kwargs)
    policy = TD3Trainer(config)

    discrete = False
    max_size = 1e-6
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset()
    done = False

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Setup some stats helpers
    log_count = 0
    reward_recorder = deque(maxlen=100)
    success_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    loss_stats = {"target_q": np.nan, "actor_loss": np.nan, "critic_loss": np.nan}

    for t in tqdm.trange(args.max_steps, desc="Training Step"):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            # [TWEAKED] How TD3 generates exploratory actions.
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.explore_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        done_bool = float(done)  # if episode_timesteps < env._max_episode_steps else 0

        if args.load_model:
            # Modify this to load proper models!
            policy.load(f"{args.pretrained_model_log_dir}")

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_steps:
            loss_stats = policy.train(replay_buffer, args.batch_size)

        if done:
            reward_recorder.append(episode_reward)
            if "arrive_dest" in info:
                success_recorder.append(info.get("arrive_dest", 0))

            # Reset environment
            state, _ = env.reset()
            done = False

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # ===== Log information =====
            if t - log_count * args.log_freq > args.log_freq:
                log_count = int(t // args.log_freq)
                stats = dict(
                    log_dir=log_dir,
                    frame_per_second=int(t / total_timer.now),
                    episode_reward=np.mean(reward_recorder),
                    total_steps=t,
                    total_episodes=episode_num,
                    total_time=total_timer.now,
                    **loss_stats
                )

                if success_recorder:
                    stats["success_rate"] = np.mean(success_recorder)

                progress.append(stats)
                pretty_print({
                    "===== TD3 Training Step {} =====".format(t): stats
                })

        if (t + 1) % args.save_freq == 0:
            policy.save(trainer_path)
            pd.DataFrame(progress).to_csv(progress_path)
            print("Trainer is saved at <{}>. Progress is saved at <{}>.".format(
                trainer_path, progress_path
            ))