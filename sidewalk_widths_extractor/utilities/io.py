import io
import os
import shutil
from datetime import datetime
from functools import reduce
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.optim import Optimizer

from sidewalk_widths_extractor.typing import _DEVICE, _PATH


def load_checkpoint(source_path: _PATH, module: Union[nn.Module, Optimizer]) -> Union[None, int]:
    """Load a nn.Module checkpoint.

    Args:
        source_path (_PATH): the path to the checkpoint file.
        module (nn.Module): nn.Module to-be-loaded from.

    Returns:
        Union[None, int]: returns the epoch number if there is one, otherwise return None.
    """
    d = torch.load(source_path)
    module.load_state_dict(d["state_dict"])
    return d["epoch"] if "epoch" in d else None


def save_checkpoint(
    target_path: _PATH,
    module: Union[nn.Module, Optimizer],
    epoch: Optional[int] = None,
    filename: str = "checkpoint",
    format_type: str = ".pth.tar",
) -> None:
    """Save a nn.Module checkpoint.

    Args:
        target_path (_PATH): the path to the target checkpoint file.
        module (nn.Module): nn.Module to-be-saved.
        epoch (Optional[int], optional): The epoch number. Defaults to None.
        filename (str, optional): checkpoint file's name. Defaults to "checkpoint".
        format_type (str, optional): file format. Defaults to ".pth.tar".
    """
    mkdir(target_path)
    cp = {"state_dict": module.state_dict()}
    if epoch:
        cp["epoch"] = epoch
    torch.save(
        cp,
        os.path.join(target_path, filename + format_type),
    )


def mkdir(target_path: _PATH) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        target_path (_PATH): The path to the directory
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def archive(source_path: _PATH, target_path: _PATH, archive_format: str = "zip") -> None:
    """
    Archive the given folder.

    Args:
        source_path (_PATH): The path to the source folder.
        target_path (_PATH): The path to the target folder.
        archive_format (str): The archive format
    """
    base = os.path.basename(target_path)
    name = base.split(".")[0]
    archive_from = os.path.dirname(source_path)
    archive_to = os.path.basename(source_path.strip(os.sep))
    shutil.make_archive(name, archive_format, archive_from, archive_to)
    shutil.move(f"{name}.{archive_format}", target_path)


def get_device() -> _DEVICE:
    """
    Get the available device.

    Returns:
        _DEVICE: device identifier
    """
    return (
        torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else "cpu"
    )


def create_path_with_timestamp(
    root: Optional[_PATH], prefix: Optional[str] = None, postfix: Optional[str] = None
) -> str:
    """Create a path with current timestamp.

    Args:
        root (Optional[_PATH]): root path.
        prefix (Optional[str], optional): string to be added before the timestamp. Defaults to None.
        postfix (Optional[str], optional): string to be added after the timestamp. Defaults to None.

    Returns:
        str: path string
    """
    timestamp = datetime.now().strftime(r"%d-%m-%Y %H-%M-%S")
    out, name = [], None
    if root:
        out.append(root)
    if prefix:
        name = prefix
    if name:
        name += " " + timestamp
    else:
        name = timestamp
    if postfix:
        name += " " + postfix
    out.append(name)
    return os.path.join(*out)


def save_writer_scalars(log_path: _PATH, filename: str = "scalars.csv") -> None:
    event = EventAccumulator(path=log_path)
    event.Reload()
    tags = event.Tags()["scalars"]

    if len(tags) == 0:
        return

    dfs = []
    for tag in tags:
        dfs.append(
            pd.DataFrame(event.Scalars(tag))
            .rename(columns={"value": tag})
            .drop(["wall_time"], axis=1)
        )
    out = reduce(lambda left, right: pd.merge(left, right, on=["step"], how="outer"), dfs)
    out.to_csv(os.path.join(log_path, filename), index=False)


def save_writer_figures(log_path: _PATH, folder_name: str = "figures") -> None:
    event = EventAccumulator(path=log_path, size_guidance={"images": 0})
    event.Reload()
    tags = event.Tags()["images"]

    if len(tags) == 0:
        return

    figure_folder = os.path.join(log_path, folder_name)
    mkdir(figure_folder)

    for tag in tags:
        for item in event.Images(tag):
            step = item.step
            encoded = item.encoded_image_string
            img = Image.open(io.BytesIO(encoded))
            path = os.path.join(figure_folder, str(step))
            mkdir(path)
            img.save(os.path.join(path, f"{tag.replace('/','_')}.png"))
