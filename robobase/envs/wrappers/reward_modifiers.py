"""Shape Rewards."""
import gymnasium as gym


class ShapeRewards(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Shape Rewards."""

    def __init__(self, env: gym.Env, reward_shaping_fn: callable):
        """General function to shape the rewards.

        Args:
            env: The environment to apply the wrapper
            reward_shaping_fn: The reward shaping function.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, reward_shaping_fn=reward_shaping_fn
        )
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.fn = reward_shaping_fn

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        observations, reward, *rest = self.env.step(action)
        return observations, self.fn(reward), *rest


class ScaleReward(ShapeRewards):
    """Scale Rewars."""

    def __init__(self, env: gym.Env, scale: float):
        """Scale the rewards.

        Args:
            env: The environment to apply the wrapper
            scale: The scale value
        """
        super().__init__(env, lambda r: r * scale)


class ClipReward(ShapeRewards):
    """Clip Rewards."""

    def __init__(self, env: gym.Env, lower_bound: float, upper_bound: float):
        """Clip the rewards.

        Args:
            env: The environment to apply the wrapper
            lower_bound: The lower bound
            upper_bound: The upper bound
        """
        super().__init__(env, lambda r: max(min(r, upper_bound), lower_bound))
