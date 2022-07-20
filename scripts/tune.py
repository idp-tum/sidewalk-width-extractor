import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any

import albumentations as A
import optuna
from torch.utils.data import DataLoader

from sidewalk_widths_extractor import Trainer, seed_all
from sidewalk_widths_extractor.dataset import SatelliteDataset
from sidewalk_widths_extractor.modules.seg import SegModule
from sidewalk_widths_extractor.utilities import get_device


def get_suggest(trial, s_name: str, s_type: str, s_values: Any) -> Any:
    assert isinstance(s_values, (list, tuple))
    if s_type == "loguniform":
        return trial.suggest_loguniform(s_name, s_values[0], s_values[1])
    elif s_type == "categorical":
        return trial.suggest_categorical(s_name, s_values)
    elif s_type == "int":
        return trial.suggest_int(s_name, s_values[0], s_values[1])
    elif s_type == "float":
        return trial.suggest_float(s_name, s_values[0], s_values[1])


def objective(trial, override_log_dir, config):
    seed_all(config["general"]["random_seed"])

    if config["general"]["force_cpu_usage"]:
        device = "cpu"
    else:
        device = get_device()

    params = {
        k: get_suggest(trial, k, v["type"], v["value"])
        for k, v in config["hyperparameters"].items()
    }

    transform = A.Compose(
        [
            A.HorizontalFlip(p=params["horizontal_flip_probability"]),
            A.VerticalFlip(p=params["vertical_flip_probability"]),
            A.RandomBrightnessContrast(
                p=params["random_brightness_contrast_probability"],
            ),
        ]
    )

    train_dataset, val_dataset = SatelliteDataset.from_split(
        config["general"]["source_images_folder"],
        config["general"]["source_masks_folder"],
        config["general"]["train_val_split_ratio"],
        train_transform=transform,
        random_seed=config["general"]["random_seed"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=config["training"]["data"]["shuffle"],
        drop_last=config["training"]["data"]["drop_last"],
        pin_memory=config["training"]["data"]["pin_memory"],
        num_workers=config["training"]["data"]["num_workers"],
        persistent_workers=config["training"]["data"]["persistent_workers"],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["data"]["batch_size"],
        shuffle=config["validation"]["data"]["shuffle"],
        drop_last=config["validation"]["data"]["drop_last"],
        pin_memory=config["validation"]["data"]["pin_memory"],
        num_workers=config["validation"]["data"]["num_workers"],
        persistent_workers=config["validation"]["data"]["persistent_workers"],
    )

    if params["encoder_weights"] == "none":
        params["encoder_weights"] = None

    module = SegModule(
        params["network"],
        {
            "encoder_name": params["network_encoder"],
            "encoder_weights": params["encoder_weights"],
            "in_channels": 3,
            "classes": 2,
        },
        "adam",
        {
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "wce",
        {"weight": [params["criterion_weights_0"], params["criterion_weights_0"]]},
        device,
        save_network_checkpoint=False,
        save_optimizer_checkpoint=False,
    )

    trainer = Trainer(
        log_folder=config["logging"]["target_log_folder"],
        log_comment=f"{config['logging']['run_id']} {trial.number}",
        override_log_dir=override_log_dir,
        progress_bar=config["logging"]["enable_progress_bar"],
        benchmark=config["general"]["pytorch_benchmark_enabled"],
        deterministic=config["general"]["pytorch_deterministic_enabled"],
        transfer_results_to_cpu=config["general"]["transfer_step_results_to_cpu"],
    )

    trainer.fit(
        module=module,
        dataloader=train_dataloader,
        validate_dataloader=val_dataloader,
        max_epochs=config["training"]["num_epochs"],
        checkpoint_path=config["general"]["source_checkpoint_path"],
        save_scalars=True,
    )

    results = trainer.validate(dataloader=val_dataloader)

    tp = sum(results["tp"])
    fp = sum(results["fp"])
    fn = sum(results["fn"])
    # tn = results["tn"].compute("sum")

    dice = 2 * tp / (2 * tp + fn + fp)

    return dice


def tune(config):

    if config["logging"]["include_date"]:
        override_log_dir = None
    else:
        override_log_dir = os.path.join(
            config["logging"]["target_log_folder"], config["logging"]["run_id"]
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(
        lambda trial: objective(trial, override_log_dir, config),
        n_trials=config["general"]["n_trials"],
    )

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    df = study.trials_dataframe()

    if override_log_dir:
        df.to_csv(os.path.join(override_log_dir, "results.csv"))
    else:
        df.to_csv(os.path.join(config["logging"]["target_log_folder"], "results.csv"))


def get_args() -> Namespace:
    """Parse given arguments

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, help="path to the config file", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert os.path.exists(args.config), f"specified config file is not found in {args.config}"

    config = None
    with open(args.config) as file:
        config = json.load(file)

    tune(config)
