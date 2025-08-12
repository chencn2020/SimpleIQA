# -*- coding: utf-8 -*-
"""
toolkit.py
-----------
A compact collection of utilities commonly used in PyTorch-based training and evaluation.

This module provides:
- Distributed helpers: `gather_together`
- Argument logging: `printArgs`
- Configuration I/O: `load_yaml`, `recursive_update`
- Reproducibility setup: `setup_seed` (enables CUDNN deterministic mode)
- Dataset splitting: `get_data` (80/20 split with a given seed)
- Learning-rate scheduling: `adjust_learning_rate` (cosine decay)
- Checkpointing: `save_checkpoint`
- Metrics: `cal_srocc_plcc` (SROCC/PLCC/KRCC/MAE)
- Training meters: `AverageMeter`, `ProgressMeter`

Notes:
- Only formatting and documentation were changed; public interfaces and behavior remain the same.
- Dependencies: torch, scipy, numpy, pyyaml.
"""

import json
import math
import os
import random
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel  # kept for parity with original imports
import torch.optim        # kept for parity with original imports
import torch.utils.data   # kept for parity with original imports
import torch.utils.data.distributed  # kept for parity with original imports
import torch.distributed as dist
from scipy import stats
import yaml


@torch.no_grad()
def gather_together(data: Any) -> List[Any]:
    """
    Collect an object from every process in a distributed run using
    `dist.all_gather_object`.

    Args:
        data: Any picklable Python object from the current process.

    Returns:
        List[Any]: A list of gathered objects in rank order, with length equal
            to the world size.
    """
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data: List[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


import os
import argparse

def printArgs(args: argparse.Namespace, savePath: str) -> None:
    """
    打印并记录参数。支持顶层键为普通值或 Namespace；
    若为 Namespace，则展开一层键值。
    """
    os.makedirs(savePath, exist_ok=True)
    log_path = os.path.join(savePath, "args_info.log")

    with open(log_path, "w", encoding="utf-8") as f:
        print("--------------args----------------")
        f.write("--------------args----------------\n")

        # 遍历顶层 args（通常是 Namespace）
        for name in vars(args):
            val = getattr(args, name)

            # 顶层一行
            print(f"########## {name:<20} {val if not hasattr(val, '__dict__') else ''}".rstrip())
            f.write(f"{name:<20} {val if not hasattr(val, '__dict__') else ''}\n")

            # 若该值可展开（例如 argparse.Namespace），则打印其子项
            if hasattr(val, "__dict__"):
                for sub in vars(val):
                    subval = getattr(val, sub)
                    print(f"{sub + ':':<20} {subval}")
                    f.write(f"{sub + ':':<20} {subval}\n")

        print("----------------------------------")
        f.write("----------------------------------\n")


def load_yaml(path: str) -> Any:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed Python object (e.g., dict or list).
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recursive_update(namespace: Any, updates: Dict[str, Any]) -> Any:
    """
    Recursively update attributes of an object; if they do not exist, add them.

    Args:
        namespace: An object with settable attributes (e.g., argparse.Namespace or dict).
        updates: A possibly nested dictionary of updates.

    Returns:
        The updated `namespace`.
    """
    for k, v in updates.items():
        if isinstance(v, dict):
            # 如果当前属性是 dict 或 Namespace，也递归更新
            if hasattr(namespace, k):
                sub_attr = getattr(namespace, k)
                recursive_update(sub_attr, v)
            else:
                setattr(namespace, k, v)  # YAML 有，但 args 没有，直接添加
        else:
            setattr(namespace, k, v)  # 不管是否存在，都直接赋值
    return namespace



def setup_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility and enable CUDNN deterministic mode.

    Warning:
        Enabling deterministic mode may slow training and can cause unexpected
        behavior when resuming from checkpoints.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def get_data(
    dataset: str,
    split_seed: int,
    data_path: str = "./SimpleIQA/utils/dataset/dataset_info.json",
) -> Tuple[str, List[int], List[int]]:
    """
    Load dataset metadata from a JSON file and produce an 80/20 train/test split.

    The JSON file is expected to map dataset names to a tuple of
    (root_path, num_images), for example:
        {
            "LIVE": ["path/to/live", 982],
            "DatasetX": ["path/to/x", 1234]
        }

    If `dataset` contains a suffix separated by an underscore (e.g., "LIVE_split1"),
    only the part before the underscore is used to look up the entry.

    Args:
        dataset: Dataset name (optionally with a suffix after an underscore).
        split_seed: Random seed for the split.
        data_path: Path to the dataset info JSON file.

    Returns:
        Tuple[str, List[int], List[int]]:
            - path: Root directory of the dataset.
            - train_index: Indices for the training split.
            - test_index: Indices for the test split.
    """
    with open(data_path, "r") as data_info:
        data_info = json.load(data_info)

    path, img_total = data_info[dataset.split("_")[0]]
    img_idx: List[int] = list(range(img_total))

    random.seed(split_seed)
    random.shuffle(img_idx)

    cut = int(round(0.8 * len(img_idx)))
    train_index = img_idx[:cut]
    test_index = img_idx[cut:]

    print("Split_seed", split_seed)
    print("train_index", train_index[:10], len(train_index))
    print("test_index", test_index[:10], len(test_index))

    return path, train_index, test_index


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, args: Any) -> None:
    """
    Apply cosine decay to the learning rate (no restarts).

    The schedule is:
        lr(epoch) = lr0 * 0.5 * (1 + cos(pi * epoch / epochs))

    Args:
        optimizer: A PyTorch optimizer.
        epoch: Current epoch index (starting from 0).
        args: An object that must define `lr` and `epochs`.
    """
    lr = args.optimizer.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.train.epochs))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(state: Any, is_best: bool, filename: str = "checkpoint.pth.tar") -> None:
    """
    Save training state (e.g., model weights and optimizer state) to disk.

    Note:
        The `is_best` flag is currently unused; it is retained for API
        compatibility and potential future use.

    Args:
        state: Any object serializable by `torch.save` (typically a dict).
        is_best: Whether this is the best checkpoint so far.
        filename: Target file name.
    """
    torch.save(state, filename)
    # To keep an additional copy for the best model, uncomment and import shutil:
    # if is_best:
    #     shutil.copyfile(filename, "model_best.pth.tar")


def cal_srocc_plcc(
    pred_score: Sequence[float],
    gt_score: Sequence[float],
) -> Tuple[float, float, float, float]:
    """
    Compute common regression/quality-assessment metrics:
    SROCC (Spearman's rank correlation), PLCC (Pearson linear correlation),
    KRCC (Kendall's tau), and MAE (mean absolute error).

    Args:
        pred_score: Predicted scores.
        gt_score: Ground-truth scores (same length as `pred_score`).

    Returns:
        Tuple[float, float, float, float]: (srocc, plcc, krcc, mae)
    """
    srocc, _ = stats.spearmanr(pred_score, gt_score)
    plcc, _ = stats.pearsonr(pred_score, gt_score)
    krcc, _ = stats.kendalltau(pred_score, gt_score)
    mae = float(np.mean(np.abs(np.array(pred_score) - np.array(gt_score))))
    return srocc, plcc, krcc, mae


class AverageMeter:
    """
    Track running statistics for a scalar quantity such as loss or accuracy.

    Maintains current value, running sum, count, and average.
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add a new observation.

        Args:
            val: The observed value.
            n:   The number of occurrences to add.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    Pretty-print training progress for a given batch index and a set of meters.
    """

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """Print formatted progress and all meter values for the current batch."""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """Create a format string like '[ 12/500]' based on the total batch count."""
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
