import pathlib

# Use dot here to denote importing the file in the folder hosting this file.
from .ppo_trainer import PPOTrainer, PPOConfig
from .td3.td3_trainer import TD3Trainer, TD3Config
from .td7.td7_trainer import TD7Trainer, TD7Config

########################################
# CHANGE TO "ppo"/"td3"/"td7" if BUGGY #
########################################
MODE = "ppo" 
FOLDER_ROOT = pathlib.Path(__file__).parent  # The path to the folder hosting this file.


class Policy:
    """
    This class is the interface where the evaluation scripts communicates with your trained agent.

    You can initialize your model and load weights in the __init__ function. At each environment interactions,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim=161), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.

    If you use any external package, please import it here and EXPLICITLY describe how to setup package in the REPORT.
    """

    # FILLED YOUR PREFERRED NAME & UID HERE!
    CREATOR_NAME = "Chris Park"  # Your preferred name here in a string
    CREATOR_UID = "806183297"  # Your UID here in a string

    def __init__(self):
        if MODE == "td3":
            config = TD3Config(lr=5e-5)
            #config = TD3Config(**kwargs)
            # if kwargs.get("state_dim"):
            #     config.state_dim = kwargs.get("state_dim")
            # if kwargs.get("action_dim"):
            #     config.action_dim = kwargs.get("action_dim")
            # if kwargs.get("max_action"):
            #     config.max_action = kwargs.get("max_action")

            self.agent = TD3Trainer(config)
            self.agent.load(FOLDER_ROOT + "/td3")
        elif MODE == "td7":
            config = TD7Config()  # Can change parameters
            self.agent = TD7Trainer(hp=config)
            self.agent.load(FOLDER_ROOT + "/td7")
        else:
            config = PPOConfig()
            self.agent = PPOTrainer(config=config)
            self.agent.load_w(log_dir=FOLDER_ROOT, suffix="iter150")

    def reset(self, done_batch=None):
        """
        Optionally reset the latent state of your agent, if any.

        Args:
            done_batch: an array with shape (batch_size,) in vectorized environment or a boolean in single environment.
            True represents the latent state of this episode should be reset.
            If it's None, you should reset the latent state for all episodes.

        Returns:
            None
        """
        pass

    def __call__(self, obs):
        if MODE == "td3":
            action = self.agent.select_action_in_batch(obs)
            return action
        elif MODE == "td7":
            action = self.agent.select_action(obs)
            return action
        else:
            value, action, action_log_prob = self.agent.compute_action(obs)
            action = action.detach().cpu().numpy()
            return action
