import json
import os
from argparse import ArgumentParser, Namespace

import albumentations as A
from torch.utils.data import DataLoader

from sidewalk_widths_extractor import Trainer, seed_all
from sidewalk_widths_extractor.dataset import SatelliteDataset
from sidewalk_widths_extractor.modules.seg import SegModule
from sidewalk_widths_extractor.utilities import get_device


def train(config) -> None:
    seed_all(config["general"]["random_seed"])

    if config["general"]["force_cpu_usage"]:
        device = "cpu"
    else:
        device = get_device()

    if config["training"]["data"]["transform"]:
        train_transform = A.Compose(
            [
                A.HorizontalFlip(
                    p=config["training"]["data"]["transform"]["horizontal_flip_probability"]
                ),
                A.VerticalFlip(
                    p=config["training"]["data"]["transform"]["vertical_flip_probability"]
                ),
                A.RandomBrightnessContrast(
                    p=config["training"]["data"]["transform"][
                        "random_brightness_contrast_probability"
                    ],
                    brightness_limit=config["training"]["data"]["transform"]["brightness_limit"],
                    contrast_limit=config["training"]["data"]["transform"]["contrast_limit"],
                ),
            ]
        )
    else:
        train_transform = None

    if config["general"]["enable_validation"]:
        train_dataset, val_dataset = SatelliteDataset.from_split(
            config["general"]["source_images_folder"],
            config["general"]["source_masks_folder"],
            config["general"]["train_val_split_ratio"],
            train_transform=train_transform,
            random_seed=config["general"]["random_seed"],
        )
    else:
        train_dataset = SatelliteDataset(
            config["general"]["source_images_folder"],
            config["general"]["source_masks_folder"],
            train_transform,
        )
        val_dataset = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["data"]["batch_size"],
        shuffle=config["training"]["data"]["shuffle"],
        drop_last=config["training"]["data"]["drop_last"],
        pin_memory=config["training"]["data"]["pin_memory"],
        num_workers=config["training"]["data"]["num_workers"],
        persistent_workers=config["training"]["data"]["persistent_workers"],
    )

    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["validation"]["data"]["batch_size"],
            shuffle=config["validation"]["data"]["shuffle"],
            drop_last=config["validation"]["data"]["drop_last"],
            pin_memory=config["validation"]["data"]["pin_memory"],
            num_workers=config["validation"]["data"]["num_workers"],
            persistent_workers=config["validation"]["data"]["persistent_workers"],
        )

    if config["parameters"]["network"]["encoder_weights"] == "none":
        config["parameters"]["network"]["encoder_weights"] = None

    if config["parameters"]["network"]["encoder_depth"] == 5:
        config["parameters"]["network"]["decoder_channels"] = (256, 128, 64, 32, 16)
    elif config["parameters"]["network"]["encoder_depth"] == 4:
        config["parameters"]["network"]["decoder_channels"] = (128, 64, 32, 16)
    elif config["parameters"]["network"]["encoder_depth"] == 3:
        config["parameters"]["network"]["decoder_channels"] = (64, 32, 16)
    else:
        raise Exception("encoder_depth must be between 3 and 5")

    module = SegModule(
        config["parameters"]["network_id"],
        config["parameters"]["network"],
        config["parameters"]["optimizer_id"],
        config["parameters"]["optimizer"],
        config["parameters"]["criterion_id"],
        config["parameters"]["criterion"],
        device,
        config["logging"]["save_network_checkpoint"],
        config["logging"]["save_optimizer_checkpoint"],
    )

    if config["logging"]["include_date"]:
        override_log_dir = None
    else:
        override_log_dir = os.path.join(
            config["logging"]["target_log_folder"], config["logging"]["run_id"]
        )

    trainer = Trainer(
        log_folder=config["logging"]["target_log_folder"],
        log_comment=config["logging"]["run_id"],
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
        max_steps=config["training"]["max_steps"],
        checkpoint_path=config["general"]["source_checkpoint_path"],
        save_every_n_epoch=config["logging"]["save_checkpoint_every_n_epoch"],
        save_settings=config["logging"]["save_settings_file"],
        save_scalars=config["logging"]["save_scalar_metrics"],
        save_figures=config["logging"]["save_figure_images"],
    )

    with open(os.path.join(trainer.log_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2, sort_keys=False)


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

    train(config)
