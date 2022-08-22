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
from sidewalk_widths_extractor.utilities import (
    get_device,
    save_writer_figures,
    save_writer_scalars,
)


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
        config["training"]["data"]["pin_memory"] = False
        config["validation"]["data"]["pin_memory"] = False
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

    if params["encoder_depth"] == 5:
        params["decoder_channels"] = (256, 128, 64, 32, 16)
    elif params["encoder_depth"] == 4:
        params["decoder_channels"] = (128, 64, 32, 16)
    elif params["encoder_depth"] == 3:
        params["decoder_channels"] = (64, 32, 16)
    else:
        raise Exception("encoder_depth must be between 3 and 5")

    module = SegModule(
        params["network"],
        {
            "encoder_name": params["network_encoder"],
            "encoder_weights": params["encoder_weights"],
            "encoder_depth": params["encoder_depth"],
            "decoder_channels": params["decoder_channels"],
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
        save_network_checkpoint=config["logging"]["save_network_checkpoint"],
        save_optimizer_checkpoint=config["logging"]["save_optimizer_checkpoint"],
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

    dice = 0.0

    for epoch_idx in range(1, config["training"]["num_epochs"] + 1):
        results = trainer.tune(
            module=module,
            dataloader=train_dataloader,
            validate_dataloader=val_dataloader,
            epoch_idx=epoch_idx,
        )

        tp = sum(results["tp"])
        fp = sum(results["fp"])
        fn = sum(results["fn"])
        # tn = results["tn"].compute("sum")

        dice = 2 * tp / (2 * tp + fn + fp)

        trial.report(dice, epoch_idx)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if config["logging"]["save_settings_file"]:
        trainer.save_settings()
    if config["logging"]["save_scalar_metrics"]:
        save_writer_scalars(trainer.log_dir)
    if config["logging"]["save_figure_images"]:
        save_writer_figures(trainer.log_dir)

    return dice


def tune(config):

    if config["logging"]["include_date"]:
        override_log_dir = None
    else:
        override_log_dir = os.path.join(
            config["logging"]["target_log_folder"], config["logging"]["run_id"]
        )

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

    if "enqueue_trial" in config:
        print("enqueued a trial")
        study.enqueue_trial(config["enqueue_trial"])

    study.optimize(
        lambda trial: objective(trial, override_log_dir, config),
        n_trials=config["general"]["n_trials"],
    )

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    with open(os.path.join(config["logging"]["target_log_folder"], "config.json"), "w") as file:
        json.dump(config, file, indent=2, sort_keys=False)

    df = study.trials_dataframe()

    if override_log_dir:
        with open(os.path.join(override_log_dir, "config.json"), "w") as file:
            json.dump(config, file, indent=2, sort_keys=False)
        df.to_csv(os.path.join(override_log_dir, "results.csv"))
    else:
        with open(os.path.join(config["logging"]["target_log_folder"], "config.json"), "w") as file:
            json.dump(config, file, indent=2, sort_keys=False)
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
