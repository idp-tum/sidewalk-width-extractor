import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob

import cv2
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torchvision.utils import draw_segmentation_masks

from sidewalk_widths_extractor.modules.seg import SegModule
from sidewalk_widths_extractor.utilities import get_device
from sidewalk_widths_extractor.utilities.io import mkdir


def get_args() -> Namespace:
    """Parse given arguments

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--settings_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--source_folder_path",
        type=str,
        required=False,
        default="demo//data//images",
    )
    parser.add_argument(
        "-t",
        "--target_folder_path",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "-f",
        "--force_cpu",
        type=bool,
        required=False,
        default=False,
    )

    return parser.parse_args()


def extract_predictions(
    model_path: str,
    settings_path: str,
    source_folder_path: str,
    target_folder_path: str,
    force_cpu: bool,
    label_train_val: bool = True,
    train_val_split_ration: float = 0.8,
    random_seed: int = 42,
):
    assert os.path.exists(model_path)
    assert os.path.exists(settings_path)
    assert os.path.exists(source_folder_path)

    settings = None
    with open(settings_path) as file:
        settings = json.load(file)

    assert isinstance(settings["module"]["network"], dict)
    assert isinstance(settings["module"]["optimizer"], dict)
    assert isinstance(settings["module"]["criterion"], dict)

    device = "cpu" if force_cpu else get_device()

    module = SegModule(
        settings["module"]["network"]["id"],
        settings["module"]["network"]["params"],
        settings["module"]["optimizer"]["id"],
        settings["module"]["optimizer"]["params"],
        settings["module"]["criterion"]["id"],
        settings["module"]["criterion"]["params"],
        None,
        None,
        device,
        False,
        False,
    )

    module.load({"network": model_path})

    try:
        if module.curr_epoch_idx != 1:
            epoch = min(module.curr_epoch_idx, settings["run"]["ending_epoch_idx"])
        else:
            epoch = settings["run"]["ending_epoch_idx"]
    except Exception as e:
        print("Epoch finding error: ", e)
        epoch = "Unknown"

    if not target_folder_path:
        target_folder_path = f"Epoch {epoch} {settings['run']['run_id']}"

    mkdir(target_folder_path)

    image_paths = [(None, x) for x in glob(os.path.join(source_folder_path, "*"))]

    tensor_transform = T.ToTensor()
    image_transform = T.ToPILImage()

    if label_train_val:
        all_indices = range(len(image_paths))
        train_indices, val_indices = train_test_split(
            all_indices, test_size=1 - train_val_split_ration, random_state=random_seed
        )
        for i in train_indices:
            image_paths[i] = ("train", image_paths[i][1])
        for i in val_indices:
            image_paths[i] = ("val", image_paths[i][1])

    for label, path in image_paths:
        if label == "train":
            image_folder_path = os.path.join(
                target_folder_path, "train", os.path.splitext(os.path.basename(path))[0]
            )
        elif label == "val":
            image_folder_path = os.path.join(
                target_folder_path, "val", os.path.splitext(os.path.basename(path))[0]
            )
        else:
            image_folder_path = os.path.join(
                target_folder_path, os.path.splitext(os.path.basename(path))[0]
            )

        mkdir(image_folder_path)

        full_image = cv2.imread(path)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        h, w, _ = full_image.shape

        images = [
            full_image[: h // 2, : w // 2],
            full_image[: h // 2, w // 2 :],
            full_image[h // 2 :, : w // 2],
            full_image[h // 2 :, w // 2 :],
        ]

        for i, x in enumerate(images):
            image = tensor_transform(x)
            pred = module.infer(image).detach().cpu()

            image = (image * 255).type(torch.uint8)
            pred = pred.type(torch.bool)

            seg = draw_segmentation_masks(image, pred, alpha=0.3, colors="blue")

            image_transform(seg).save(os.path.join(image_folder_path, f"seg_{i}.png"))
            image_transform((pred * 255).type(torch.uint8)).save(
                os.path.join(image_folder_path, f"mask_{i}.png")
            )

    return target_folder_path, len(image_paths)


if __name__ == "__main__":
    args = get_args()
    target_folder_path, total_predicted = extract_predictions(
        model_path=args.model_path,
        settings_path=args.settings_path,
        source_folder_path=args.source_folder_path,
        target_folder_path=args.target_folder_path,
        force_cpu=args.force_cpu,
    )
    print(f"Extracted {total_predicted} predictions to {target_folder_path}.")
