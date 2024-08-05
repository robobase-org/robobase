import csv
import datetime
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
import wandb
from termcolor import colored

COMMON_PRETRAIN_FORMAT = [
    ("iteration", "Iter", "int"),
    ("total_time", "T", "time"),
    ("buffer_size", "BS", "int"),
    ("agent_batched_updates_per_second", "Batched Update FPS", "float"),
]

COMMON_TRAIN_FORMAT = [
    ("iteration", "Iter", "int"),
    ("env_steps", "S", "int"),
    ("env_episodes", "E", "int"),
    ("buffer_size", "BS", "int"),
    ("buffer_sample_time", "BST", "float"),
    ("env_steps_per_second", "Env FPS", "float"),
    ("agent_batched_updates_per_second", "Batched Update FPS", "float"),
    ("total_time", "T", "time"),
]

COMMON_EVAL_FORMAT = [
    ("iteration", "Iter", "int"),
    ("env_steps", "S", "int"),
    ("env_episodes", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating, save_csv: bool):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._save_csv = save_csv
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("pretrain_eval"):
                key = key[len("pretrain_eval") + 1 :]
            elif key.startswith("pretrain"):
                key = key[len("pretrain") + 1 :]
            elif key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["episode"]) >= data["episode"]:
                    break
                rows.append(row)
        with self._csv_file_name.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(data.keys()), restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open("a")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        if prefix == "train":
            color = "yellow"
        elif prefix == "pretrain":
            color = "red"
        else:
            color = "green"
        prefix = colored(prefix, color)
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        if self._save_csv:
            self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, cfg):
        self._log_dir = log_dir
        self._pretrain_mg = MetersGroup(
            log_dir / "pretrain.csv", COMMON_PRETRAIN_FORMAT, cfg.save_csv
        )
        self._pretrain_eval_mg = MetersGroup(
            log_dir / "pretrain_eval.csv", COMMON_EVAL_FORMAT, cfg.save_csv
        )
        self._train_mg = MetersGroup(
            log_dir / "train.csv", COMMON_TRAIN_FORMAT, cfg.save_csv
        )
        self._eval_mg = MetersGroup(
            log_dir / "eval.csv", COMMON_EVAL_FORMAT, cfg.save_csv
        )
        self._use_wandb = cfg.wandb.use
        self._use_tb = cfg.tb.use
        if self._use_wandb and self._use_tb:
            raise ValueError(
                "You have request to log with both TensorBoard and W&B. "
                "We will assume this is a mistake."
            )
        self._wandb_logs = {}
        if self._use_wandb:
            import wandb

            cfg_dict = OmegaConf.to_container(cfg, resolve=False)

            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                config=cfg_dict,
            )
        elif self._use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ImportError("Please run `pip install tensorboard`") from e
            from datetime import datetime

            logdir = (
                datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                if cfg.tb.name is None
                else cfg.tb.name
            )
            self._sw = SummaryWriter(str(Path(cfg.tb.log_dir) / logdir))

    def _try_log(self, key, value, step, is_video=False):
        if self._use_wandb:
            if is_video:
                self._wandb_logs[key] = wandb.Video(
                    np.array([value["video"]]).transpose(0, 1, 4, 2, 3),
                    step,
                    fps=value["fps"],
                )

            elif np.isscalar(value) or value.size == 1:
                self._wandb_logs[key] = value
            else:
                v = value if value.ndim == 3 else value[0]  # assume image
                channel_first = v.shape[0] == 3
                if channel_first:
                    v = np.moveaxis(v, 0, -1)
                self._wandb_logs[key] = wandb.Image(v)
        elif self._use_tb:
            if is_video:
                if value["video"].ndim == 5:
                    v = value["video"]
                elif value["video"].ndim == 4:
                    v = np.expand_dims(value["video"], 0)
                else:
                    raise ValueError("Expected video to have ndim = 4 or 5.")
                v = v.transpose(0, 1, 4, 2, 3)
                self._sw.add_video(key, v, step, fps=value["fps"])
            elif np.isscalar(value) or value.size == 1:
                v = value.item() if value is np.ndarray else value
                self._sw.add_scalar(key, v, step)
            else:
                v = value if value.ndim == 3 else value[0]  # assume image
                self._sw.add_image(key, v, step)

    def _log(self, key, value, step):
        if torch.is_tensor(value):
            # If used has logged tensor, convert to numpy
            value = value.detach().cpu().numpy()
        # If plot is in the key, it is not a video.
        is_plot = (
            any(["plot" in str(key) for key in value.keys()])
            if isinstance(value, dict)
            else False
        )
        is_video = (
            any(["video" in str(key) for key in value.keys()]) and not is_plot
            if isinstance(value, dict)
            else False
        )
        if not (
            np.isscalar(value) or isinstance(value, np.ndarray) or is_plot or is_video
        ):
            # Unknown value so don't log
            return
        self._try_log(key, value, step, is_video)
        if np.isscalar(value):
            if key.startswith("train"):
                mg = self._train_mg
            elif key.startswith("pretrain_eval"):
                mg = self._pretrain_eval_mg
            elif key.startswith("pretrain"):
                mg = self._pretrain_mg
            else:
                mg = self._eval_mg
            mg.log(key, value)

    def _dump(self, step, prefix=None):
        if prefix is None or prefix == "eval":
            self._eval_mg.dump(step, "eval")
        if prefix is None or prefix == "train":
            self._train_mg.dump(step, "train")
        if prefix is None or prefix == "pretrain":
            self._pretrain_mg.dump(step, "pretrain")
        if prefix is None or prefix == "pretrain_eval":
            self._pretrain_eval_mg.dump(step, "pretrain_eval")
        if self._use_wandb and len(self._wandb_logs):
            wandb.log(self._wandb_logs, step=step)
            self._wandb_logs = {}

    def log_metrics(self, metrics, step, prefix):
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 1:
                for i, v in enumerate(value):
                    self._log(f"{prefix}/{key}{i}", v, step)
            else:
                self._log(f"{prefix}/{key}", value, step)
        self._dump(step, prefix)
