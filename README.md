# RoboBase: Robot Learning Baselines

RoboBase, robot learning baselines, covering:
- Reinforcement Learning
- Demo-driven Reinforcement Learning (aka Offline-RL)
- Imitation Learning

Top Features of RoboBase:
- Well-tuned algorithms with a focus on methods that take both low-dimensional proprioceptive robot data **and** (multiple) high-dimensional vision-sensor data.
  This is in contrast to other common frameworks (StableBaselines, CleanRL, etc) that often prioritise **only** low-dimensional **or only** high-dimensional inputs.
- "Single-file" implementation of algorithms.
- First-class support for vectorised training environments.
- Wrappers around common environments, e.g. DMC and RLBench.

## Table of Contents

1. [Install](#install)
2. [Implemented Algorithms](#implemented-algorithms)
3. [Framework Overview ](#framework-overview)
4. [Usage](#usage)

## Install

System installs:

```commandline
sudo apt-get install ffmpeg  # Usually pre-installed on most systems
```

```commandline
pip install .
```

### DeepMind Control

```commandline
pip install ".[dmc]"
```

### RLBench

```commandline
sudo apt-get install python3.10-dev   # if using python3.10
./extra_install_scripts/install_coppeliasim.sh  # If you dont have CoppeliaSim already installed
pip install ".[rlbench]"
```

<details>
<summary>RLBench Issues?</summary>
<br>

Note: If you got an error about not finding libGL.so.1, then you need to install the following:
```commandline
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
sudo apt-get install libgl1-mesa-dev libxrender1 libxkbcommon-x11-0
```
If you still get an error, then set the following environment variable to see if the error is more informative:
```commandline
export QT_DEBUG_PLUGINS=1
```
</details>

### BiGym

```commandline
pip install ".[bigym]"
```


## Implemented Algorithms

:white_check_mark: = High confidence that it is implemented correctly and thoroughly evaluated.

:warning: = Lower confidence that it is implemented correctly and/or thoroughly evaluated.

### (Demo-driven) RL

| Method                                        | Paper                                                                                                                                 | 1-line Summary                                                                                                                                | Differences to paper?             | Stable             |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------|
| [drqv2](robobase/cfgs/method/drqv2.yaml)         | [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/abs/2107.09645)               | Uses augmentation (4-pixel shifting) and layer-norm bottleneck to aid learning from pixels.                                                   | None.                             | :white_check_mark: |
| [alix](robobase/cfgs/method/alix.yaml)           | [Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2207.00986)                                    | Rather then augmentation (as in DrQV2), uses a Adaptive Local SIgnal MiXing (LIX) layer that explicitly enforces smooth featuremap gradients. | None.                             | :white_check_mark: |
| [sac_lix](robobase/cfgs/method/sac_lix.yaml)     | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) | Maximum entropy RL algorithm that has adaptive exploration.                                                                                   | Uses ALIX as the base algorithm.  | :white_check_mark: |
| [drm](robobase/cfgs/method/drm.yaml)             | [DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization](https://arxiv.org/abs/2310.19668)                   | Uses dormant ratio as a metric to measure inactivity in the RL agent's network to allow effective exploration.                                | None.                             | :warning:          |
| [dreamerv3](robobase/cfgs/method/dreamerv3.yaml) | [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)                                                            | Learns world models with CNN/MLP encoder and decoder.                                                                                 | None.                             | :white_check_mark: |
| [mwm](robobase/cfgs/method/mwm.yaml)             | [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)                                                            | World model (similar to DreamerV2) that uses Masked Autoencoders (MAE) for visual feature learning.                                           | None.                             | :white_check_mark: |
| [iql_drqv2](robobase/cfgs/method/iql_drqv2.yaml) | [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)                                           | Does not evaluate "unseen" actions to limit Q-value overestimation.                                                                           | Uses DrQv2 as the base algorithm. | :white_check_mark: |
| [CQN](robobase/cfgs/method/cqn.yaml)             | [Coarse-to-fine Q-Network](https://younggyo.me/cqn/)  | Value-based agent (without a separate actor) for continuous control that zooms into discrete action space multiple times.     | None.                             | :white_check_mark: |

### Imitation Learning

| Method                                        | Paper                                                                                                   | 1-line Summary                              | Differences to paper?             | Stable    |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------|-----------------------------------|-----------|
| [diffusion](robobase/cfgs/method/diffusion.yaml) | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)   | Brings diffusion to robotics.               | None.                             | :warning: |
| [act](robobase/cfgs/method/act.yaml)             | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)  | Transformer and action-sequence prediction. | None.                             | :white_check_mark: |

### Algorithmic Features

| Feature (argument name)                                      | Description                                                                                                                                                                           | Methods supported |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Action sequence (action_sequence)                            | Same as action chunking in ACT, it allows a model to predict a sequence of actions per inference time                                                                                 | All methods       |
| Frame stacking (frame_stack)                                 | Stacking current frame with previous ones to provide recent input history to the model                                                                                                | All methods       |
| Action standardization (use_standardization)                 | Based on demonstration data, perform z-score normalization on actions. Note that default option clips actions beyond $3\sigma$                                                        | All methods       |
| Action min/max normalization (use_min_max_normalization)     | Based on demonstration data, perform min/max normalization on actions.                                                        | All methods       |
| Distributional critic (method.distributional_critic)         | A distributional version of the critic model based on [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), known to improve learning stability | All RL methods    |
| Critic ensembling (method.num_critics)                       | Using multiple critics to mitigate value overestimation                                                                                                                               | All RL methods    |
| Intrinsic exploration algorithms (intrinsic_reward_module)   | Advanced exploration through the use of intrinsic rewards                                                                                                                             | All RL methods    |

Below is an example of launching a method using all of the above features:

```commandline
python3 train.py method=sac_lix env=dmc/cartpole_swingup action_sequence=3 \
frame_stack=3 use_standardization=true method.num_critics=4 intrinsic_reward_module=rnd \
method.distributional_critic=true method.critic_model.output_shape=\[251, 1\]
```

## Framework Overview :memo:

### Method

All implemented methods should extend `Method`:

```python
class Method:
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: torch.device,
        num_train_envs: int,
        replay_alpha: float,
        replay_beta: float,
    ):
        ...

    @property
    def random_explore_action(self) -> torch.Tensor:
        # Produces a random action for exploration
        ...

    @abstractmethod
    def act(
        self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool
    ) -> BatchedActionSequence:
        # Called when an action is needed in the environment. Outputs tensor: (B, T, A)
        ...

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> Metrics:
        # Called when gradient updates should be performed
        ...

    @abstractmethod
    def reset(self, step: int, agents_to_reset: list[int]):
        # Called on each environment.
        ...
```

### Replay Buffer / Updates

Within the `update` method, we can access batch data from the replay buffer via:
```python
batch = next(replay_iter)
```
Batch will be a dictionary mapping strings to `torch.Tensor`. All observation data will have the following shape: `(B, T, ...)`, where `B` is batch size, and `T` is an observation history (aka frame stack).

### RoboBaseModules

Networks should be passed into the `Method` class so that they can be parameterised through Hydra.
Most of the methods in RoboBase assume 3 networks (`RoboBaseModule`) to be passed in:

If you are frame stacking on channel, i.e. `frame_stack_on_channel=true`, then:
```
(B, V, T, C, W, H)
 ⌄
(B, V, T * C, W, H)
 ⌄
|EnoderModule|
 ⌄
(B, V, Z)
 ⌄
|FusionModule|
 ⌄
(B, Z',)
 ⌄
|FullyConnectedModule|
 ⌄
(B, T', A)
```

If you are using an rnn to roll in the frame stack, i.e. `frame_stack_on_channel=false`, then:, then:
```
(B, V, T, C, W, H)
 ⌄
(B * T, V, C, W, H)
 ⌄
|EnoderModule|
 ⌄
(B * T, V, Z)
 ⌄
|FusionModule|
 ⌄
(B * T, Z')
 ⌄
(B, T, Z')
 ⌄
|FullyConnectedModule|
 ⌄
(B, T', A)
```
where `V` is the number of cameras/views, and `T'` is the action output sequence.
Note that `FullyConnectedModule` can have either a 1-dim `(Z,)` input or a 2-dim `(T, Z)` input.

To stop training, execute `ctrl-c` in the terminal. This will cleanly terminate the training process.

## Usage :chart_with_upwards_trend:

There are 4 common ways to use RoboBase:
1. Running existing algorithms/networks on existing environments.
2. Running existing algorithms/networks on custom environments.
3. Running novel/experimental algorithms/networks on existing environments.
4. Running novel/experimental algorithms/networks on custom environments.

Option **2**, **3**, and **4** requires you importing RoboBase into your project, while option **1** you can install and use directly in the terminal with no new code.
See below for details on each of these options.

### Running existing algorithms on existing environments

From the root of the project, you can launch experiments from any of the supported environments.
Here are some examples:

#### DeepMind Control Suite (DMC) Examples

Launch the `sac_lix` method on the `cartpole_swingup` task, with `episode_length` 1000.
```commandline
python3 train.py method=sac_lix env=dmc/cartpole_swingup env.episode_length=1000
```

Let's launch this as a pixel-based experiment, using a prioritised replay buffer, and with some tensorboard logging:
```commandline
python3 train.py method=sac_lix env=dmc/cartpole_swingup env.episode_length=1000 \
pixels=true replay.prioritization=true tb.use=true \
tb.log_dir=/tmp/robobase_tb_logs tb.name="my_experiment"
```

You can now track that experiment in tensorboard by running:
```commandline
tensprboard --logdir=/tmp/robobase_tb_logs --port
```

and then in your browser, navigate to: [http://localhost:6006/](http://localhost:6006/)

For a full list of launch configs, [see here](robobase/cfgs/robobase_config.yaml).

#### RLBench Examples

Launch the `drqv2` method on the `reach_target` task, with `episode_length` 100, and 10 demos with pixels.

```commandline
python3 train.py method=drqv2 env=rlbench/reach_target env.episode_length=100 demos=10 pixels=true
```

Let's reduce the number of channels in the CNN of our vision encoder, and the number of nodes in our critic MPL:

```commandline
python3 train.py method=drqv2 env=rlbench/reach_target env.episode_length=100 demos=10 \
pixels=true method.encoder_model.channels=16 method.critic_model.mlp_nodes=\[128,128\]
```

#### Launch Configs

You can create your own handy config file in `robobase.cfgs.launch` and use them to launch your experiments.
Here are some examples:
```commandline
python3 train.py launch=drqv2 env=rlbench/reach_target env.episode_length=100

python3 train.py launch=drqv2_pixel_dmc env=dmc/cartpole_balance

python3 train.py launch=mwm env=dmc/walker_walk

python3 train.py launch=mwm_rlbench env=rlbench/open_drawer

python3 train.py method=act pixels=true env=rlbench/reach_target is_imitation_learning=true

python3 train.py method=act launch=act_pixel_bigym env=bigym/dishwasher_close wandb.name=act_bigym_dishwasher_close batch_size=256 demos=-1

python3 train.py launch=dp_pixel_bigym env=bigym/dishwasher_close
```

### Running existing algorithms/networks on custom environments.

In a new project/repo, you will need to create a minimum of 3 files:
1. A Hydra config for your environment, e.g. `myenv.yaml`
2. An environment and a corresponding `Factory` to build it, e.g. `myenv.py`
3. A launch file that hooks everything together, e.g. `train.py`

**myenv.yaml**
```yaml
# @package _global_
env:
  env_name: my_env_name
  physics_dt: 0.004  # The time passed per simulation step
  # Others ways to configure your environment
```

**myenv.py**
```python
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    AppendDemoInfo,
    ConcatDim,
)


class MyEnv(gym.Env):
  pass


class MyEnvFactory(EnvFactory):

    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = ConcatDim(env, 1, 0, "low_dim_state")
        env = TimeLimit(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = AppendDemoInfo(env)
        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(MyEnv(), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(MyEnv(), cfg)
```

**train.py**
```python
import hydra
from robobase.workspace import Workspace
from myenv import MyEnvFactory

@hydra.main(
    config_path="cfgs", config_name="my_cfg", version_base=None
)
def main(cfg):
    workspace = Workspace(cfg, env_factory=MyEnvFactory())
    workspace.train()


if __name__ == "__main__":
    main()
```


### Running novel/experimental algorithms/networks on existing environments

In a new project/repo, you will need to create a minimum of 2 files:
1. A Hydra config for your method, e.g. `mymethod.yaml`
2. A method class e.g. `mymethod.py`

**method/mymethod.yaml**
```yaml
# @package _global_
method:
  _target_: mymethod.MyMethod
  my_special_parameter: 1
  # Others ways to configure your environment
```

**mymethod.py**
```python
import torch
from robobase.method.core import Method, BatchedActionSequence, Metrics
from typing import Iterator
from robobase.replay_buffer.replay_buffer import ReplayBuffer

class MyMethod(Method):
  def __init__(self, my_special_parameter: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_special_parameter = my_special_parameter

  def reset(self, step: int, agents_to_reset: list[int]):
    pass

  def update(self, replay_iter: Iterator[dict[str, torch.Tensor]], step: int,
             replay_buffer: ReplayBuffer = None) -> Metrics:
    pass

  def act(self, observations: dict[str, torch.Tensor], step: int,
          eval_mode: bool) -> BatchedActionSequence:
    pass
```

You can then launch that algorithm on an environment, e.g.
```commandline
python3 train.py --config-dir=. method=mymethod env=dmc/cartpole_swingup env.episode_length=1000
```
where `config-dir` adds a config directory to the Hydra config search path.

### Running novel/experimental algorithms/networks on custom environments

A combination of the two configurations described above.

## Optimisations

### Logging

In your method, only log when logging is True; this will be slight more efficient, especially if you log a lot.

```python
from robobase.method.core import OffPolicyMethod

class MyMethod(OffPolicyMethod):
  def update(self, *args):
    metrics = {}
    if self.logging:
      metrics["loss"] = 0
    return metrics

```
