import copy
import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from typing import Any

import albumentations as A
import optuna
import torch
from torch.utils.data import DataLoader

from sidewalk_widths_extractor import Trainer, seed_all
from sidewalk_widths_extractor.dataset import SatelliteDataset
from sidewalk_widths_extractor.modules.seg import SegModule
from sidewalk_widths_extractor.utilities import (
    get_device,
    save_writer_figures,
    save_writer_scalars,
)
from sidewalk_widths_extractor.utilities.io import mkdir


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


def objective(trial, config):
    torch.cuda.empty_cache()
    seed_all(config["general"]["random_seed"])

    if config["general"]["force_cpu_usage"]:
        device = "cpu"
        config["training"]["data"]["pin_memory"] = False
        config["validation"]["data"]["pin_memory"] = False
    else:
        device = get_device()

    params = {
        k: get_suggest(trial, k, v["type"], v["value"])
        for k, v in config["hyperparameters"].items()
    }
    print(f"Trial {trial.number} - Params: {params}")

    transform = A.Compose(
        [
            A.RandomCrop(
                width=config["general"]["target_image_width"],
                height=config["general"]["target_image_height"],
            ),
            A.HorizontalFlip(p=params["horizontal_flip_probability"]),
            A.VerticalFlip(p=params["vertical_flip_probability"]),
            A.RandomBrightnessContrast(
                p=params["random_brightness_contrast_probability"],
            ),
        ]
    )

    val_transform = A.Compose(
        [
            A.CenterCrop(
                width=config["general"]["target_image_width"],
                height=config["general"]["target_image_height"],
            ),
        ]
    )

    train_dataset, val_dataset = SatelliteDataset.from_split(
        config["general"]["source_images_folder"],
        config["general"]["source_masks_folder"],
        config["general"]["train_val_split_ratio"],
        train_transform=transform,
        val_transform=val_transform,
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

    final_params = copy.deepcopy(params)

    if params["encoder_weights"] == "none":
        final_params["encoder_weights"] = None
    if params["encoder_depth"] == 5:
        final_params["decoder_channels"] = (256, 128, 64, 32, 16)
    elif params["encoder_depth"] == 4:
        final_params["decoder_channels"] = (128, 64, 32, 16)
    elif params["encoder_depth"] == 3:
        final_params["decoder_channels"] = (64, 32, 16)
    else:
        raise Exception("encoder_depth must be between 3 and 5")

    module = SegModule(
        network_id=final_params["network"],
        network_params={
            "encoder_name": final_params["network_encoder"],
            "encoder_weights": final_params["encoder_weights"],
            "encoder_depth": final_params["encoder_depth"],
            "decoder_channels": final_params["decoder_channels"],
            "in_channels": 3,
            "classes": 2,
        },
        optimizer_id="adam",
        optimizer_params={
            "lr": final_params["lr"],
            "weight_decay": final_params["weight_decay"],
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        criterion_id="wce",
        criterion_params={
            "weight": [final_params["criterion_weights_0"], final_params["criterion_weights_1"]]
        },
        scheduler_id=config["training"]["scheduler"],
        scheduler_params=config["training"]["scheduler_params"],
        device=device,
        save_network_checkpoint=config["logging"]["save_network_checkpoint"],
        save_optimizer_checkpoint=config["logging"]["save_optimizer_checkpoint"],
    )

    trainer = Trainer(
        log_folder=config["logging"]["target_log_folder"],
        log_comment=f"{config['logging']['run_id']} {trial.number}",
        progress_bar=config["logging"]["enable_progress_bar"],
        benchmark=config["general"]["pytorch_benchmark_enabled"],
        deterministic=config["general"]["pytorch_deterministic_enabled"],
        transfer_results_to_cpu=config["general"]["transfer_step_results_to_cpu"],
    )

    dice = 0.0
    current_epoch_idx = 1
    pruned = False
    for epoch_idx in range(1, config["training"]["num_epochs"] + 1):
        current_epoch_idx = epoch_idx
        results = trainer.tune(
            module=module,
            dataloader=train_dataloader,
            validate_dataloader=val_dataloader,
            epoch_idx=epoch_idx,
        )

        tp = sum(results["tp"])
        fp = sum(results["fp"])
        fn = sum(results["fn"])
        dice = (2 * tp / (2 * tp + fn + fp)).item()

        trial.report(dice, epoch_idx)

        if trial.should_prune():
            pruned = True
            break

    if config["logging"]["save_settings_file"]:
        trainer.save_settings(
            {
                "no_train_samples": len(train_dataloader.dataset),
                "train_batch_size": train_dataloader.batch_size,
                "no_validation_samples": len(val_dataloader.dataset),
                "validation_batch_size": val_dataloader.batch_size,
            }
        )

    if config["logging"]["save_scalar_metrics"]:
        save_writer_scalars(trainer.log_dir)

    if config["logging"]["save_figure_images"]:
        save_writer_figures(trainer.log_dir)

    with open(os.path.join(config["logging"]["target_log_folder"], "history.json"), "r+") as file:
        history = json.load(file)
        if "distributions" not in history:
            history["distributions"] = [
                (n, optuna.distributions.distribution_to_json(x))
                for n, x in trial.distributions.items()
            ]
        if "trials" not in history:
            history["trials"] = []

        history["trials"].append(
            {
                "value": dice,
                "params": trial.params,
            }
        )
        file.seek(0)
        json.dump(history, file, indent=2, sort_keys=False)

    if pruned:
        raise optuna.exceptions.TrialPruned()

    module.save(current_epoch_idx)

    return dice


def tune(config):
    study = optuna.create_study(
        study_name=config["logging"]["run_id"],
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        # pruner=optuna.pruners.MedianPruner(
        #     n_startup_trials=5, n_warmup_steps=25, interval_steps=10
        # ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=25, max_resource=config["training"]["num_epochs"], reduction_factor=3
        ),
    )

    mkdir(config["logging"]["target_log_folder"])

    if (
        "resume" in config
        and isinstance(config["resume"], str)
        and os.path.exists(config["resume"])
    ):
        print(f"Resuming with given history from {config['resume']}")
        with open(config["resume"]) as file:
            history = json.load(file)
            study.add_trials(
                [
                    optuna.trial.create_trial(
                        params=trial["params"],
                        distributions={
                            x[0]: optuna.distributions.json_to_distribution(x[1])
                            for x in history["distributions"]
                        },
                        value=trial["value"],
                    )
                    for trial in history["trials"]
                ]
            )
        shutil.copyfile(
            config["resume"], os.path.join(config["logging"]["target_log_folder"], "history.json")
        )
    else:
        with open(
            os.path.join(config["logging"]["target_log_folder"], "history.json"), "w"
        ) as file:
            json.dump({}, file, indent=2, sort_keys=False)

    if "enqueue" in config and isinstance(config["enqueue"], dict):
        study.enqueue_trial(config["enqueue"])

    with open(os.path.join(config["logging"]["target_log_folder"], "config.json"), "w") as file:
        json.dump(config, file, indent=2, sort_keys=False)

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=config["general"]["n_trials"],
    )

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    df = study.trials_dataframe()

    df.to_csv(os.path.join(config["logging"]["target_log_folder"], "results.csv"))

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(config["logging"]["target_log_folder"], "param_importances.png"))
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(
        os.path.join(config["logging"]["target_log_folder"], "optimization_history.png")
    )
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(config["logging"]["target_log_folder"], "parallel_coordinate.png"))


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
