from typing import List
import gymnasium as gym
from omegaconf import DictConfig


class Demo(list):
    def __init__(self, transition_tuples: List[tuple]):
        super().__init__(transition_tuples)


class EnvFactory:
    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        pass

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        pass

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        """Collect demonstrations or fetch stored demonstrations.

        Args:
            cfg (DictConfig): Config
            num_demos (int): Number of demonstrations to fetch or collect
        """
        raise NotImplementedError("This env does not support demo loading.")

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        """Post-process demonstrations after collecting or storing them.
        This is required for a case when such post-processing needs some
        information from environments, which were often not available when
        we call `collect_or_fetch_demos`

        Args:
            cfg (DictConfig): Config
        """
        raise NotImplementedError("This env does not support demo loading.")

    def load_demos_into_replay(self, cfg: DictConfig, buffer):
        """Load the collected or fetched demos into the replay buffer.

        Args:
            cfg (DictConfig): Config
            buffer (_type_): Replay buffer to save the demonstrations.
        """
        raise NotImplementedError("This env does not support demo loading.")


class DemoEnv(gym.Env):
    def __init__(self, demos: List[Demo], action_space, observation_space):
        """Init.

        Args:
            demos: A list of demos
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.is_demo_env = True
        self._active_demo = []
        self._loaded_demos = demos

    def modify_actions(self):
        pass

    def render(self):
        raise NotImplementedError("Not supported for demo env.")

    def step(self, action):
        return self._active_demo.pop(0)

    def reset(self, seed=None, options=None):
        self._active_demo = self._loaded_demos.pop(0)
        return self._active_demo.pop(0)
