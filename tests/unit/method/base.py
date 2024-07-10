"""Base test"""
import multiprocessing
import tempfile

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from robobase.envs import dmc, bigym

from robobase.workspace import Workspace

import os
import torch

dmc.UNIT_TEST = True
bigym.UNIT_TEST = True


def train_and_shutdown(cfg, tempdir):
    w = Workspace(cfg, work_dir=tempdir)
    w.train()
    w.shutdown()


def _train_process_helper(cfg, tempdir):
    # Initialize Workspace inside the subprocess
    workspace = Workspace(cfg, work_dir=tempdir)

    # Store the initial state_dict
    prev_state_dict = {k: v.clone() for k, v in workspace.agent.state_dict().items()}

    # Perform training
    workspace.train()

    # Get the updated state_dict and save it to temp directory
    state_dict = {k: v.clone() for k, v in workspace.agent.state_dict().items()}
    with open(f"{tempdir}/state_dict.pt", "wb") as f:
        torch.save(state_dict, f)

    is_param_different = []
    for k in state_dict.keys():
        is_param_different.append(not torch.allclose(state_dict[k], prev_state_dict[k]))
    assert sum(is_param_different) > 0
    workspace.save_snapshot()
    workspace.shutdown()


def _load_snapshot_process_helper(cfg, tempdir):
    # Initialize Workspace inside the subprocess
    new_workspace = Workspace(cfg, work_dir=tempdir)

    # Load state_dict from previous process
    with open(f"{tempdir}/state_dict.pt", "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    # Check the snapshot path
    snapshot_path = os.path.join(tempdir, "snapshots", "latest_snapshot.pt")
    assert os.path.exists(snapshot_path)

    # Check whether initial parameters are different from saved parameters
    new_state_dict = new_workspace.agent.state_dict()
    is_param_different = []
    for k in new_state_dict.keys():
        is_param_different.append(not torch.allclose(state_dict[k], new_state_dict[k]))
    assert sum(is_param_different) > 0

    # Load snapshot
    new_workspace.load_snapshot()
    new_state_dict = new_workspace.agent.state_dict()

    # Check whether the parameters are the same after loading snapshot
    assert len(state_dict) == len(new_state_dict)
    for k in new_state_dict.keys():
        assert torch.allclose(state_dict[k], new_state_dict[k])

    new_workspace.shutdown()


class Base:
    def test_save_load_snapshot(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + [
                "pixels=true",
                "env=dmc/acrobot_swingup",
                "save_snapshot=true",
                "snapshot_every_n=1",
            ]
            + cfg_params,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(
                target=_train_process_helper, args=(cfg, tempdir)
            )
            p.start()
            p.join()
            assert not p.exitcode

            p = multiprocessing.Process(
                target=_load_snapshot_process_helper, args=(cfg, tempdir)
            )
            p.start()
            p.join()
            assert not p.exitcode
