from pathlib import Path

import imageio
import numpy as np
import gymnasium as gym


def _render_single_env_if_vector(env: gym.vector.VectorEnv):
    if getattr(env, "is_vector_env", False):
        if getattr(env, "parent_pipes", False):
            # Async env
            old_parent_pipes = env.parent_pipes
            env.parent_pipes = old_parent_pipes[:1]
            img = env.call("render")[0]
            env.parent_pipes = old_parent_pipes
        elif getattr(env, "envs", False):
            # Sync env
            old_envs = env.envs
            env.envs = old_envs[:1]
            img = env.call("render")[0]
            env.envs = old_envs
        else:
            raise ValueError("Unrecognized vector env.")
    else:
        img = env.render()
    return img


class VideoRecorder:
    def __init__(self, save_dir: Path, render_size=256, fps=20):
        self.save_dir = save_dir
        if save_dir is not None:
            self.save_dir.mkdir(exist_ok=True)
        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = _render_single_env_if_vector(env)
            if frame is not None:
                self.frames.append(frame)

    def save(self, file_name):
        if self.enabled and len(self.frames) > 0:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), np.array(self.frames), fps=self.fps)
