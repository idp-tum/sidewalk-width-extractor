import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from typing import Optional

import cv2
import numpy as np


def get_args() -> Namespace:
    """Parse given arguments

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--source_images", type=str, required=False, default="demo//raw-data//images"
    )
    parser.add_argument("--target_images", type=str, required=False, default="demo//data//images")
    parser.add_argument(
        "--source_json",
        type=str,
        required=False,
        default="demo//raw-data//sidewalk-widths-extractor.json",
    )
    parser.add_argument("--target_masks", type=str, required=False, default="demo//data//masks")

    parser.add_argument("--mask_width", type=int, required=False, default=256)
    parser.add_argument("--mask_height", type=int, required=False, default=256)

    return parser.parse_args()


def extract_masks(
    source_images_folder: str,
    souce_masks_json: str,
    target_images_folder: str,
    target_masks_folder: str,
    mask_width: int = 256,
    mask_height: int = 256,
) -> int:
    assert os.path.exists(source_images_folder)
    assert os.path.exists(souce_masks_json)

    if not os.path.exists(target_images_folder):
        os.makedirs(target_images_folder)

    if not os.path.exists(target_masks_folder):
        os.makedirs(target_masks_folder)

    count = 0  # count of total images saved
    source_masks_data = None
    mapping: dict = {}

    with open(souce_masks_json) as f:
        source_masks_data = json.load(f)

    def get_polygon_coords(data: dict, count: int) -> Optional[list]:
        x_points = data["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data["regions"][count]["shape_attributes"]["all_points_y"]
        points = []
        for i, x in enumerate(x_points):
            points.append([x, y_points[i]])
        return points

    for _, data in source_masks_data.items():
        filename_wf = data["filename"]
        filename = os.path.splitext(os.path.basename(filename_wf))[0]
        sub_count = 0

        if len(data["regions"]) >= 1:
            for _ in range(len(data["regions"])):
                bbs = get_polygon_coords(data, sub_count)
                if filename in mapping:
                    mapping[filename].append(bbs)
                else:
                    mapping[filename] = [bbs]
                sub_count += 1

    print("Images with masks: ", len(mapping))

    for file_name in os.listdir(source_images_folder):
        bfn = os.path.splitext(os.path.basename(file_name))[0]
        if bfn not in mapping:
            continue
        source_img = os.path.join(source_images_folder, file_name)
        target_img = os.path.join(target_images_folder, file_name)
        if not os.path.exists(target_img):
            shutil.copyfile(source_img, target_img)

    for filename, polygons in mapping.items():
        mask = np.zeros((mask_width, mask_height))
        arrs = []
        for i in range(0, len(polygons)):
            arrs.append(np.array(polygons[i]))

        count += 1
        cv2.fillPoly(mask, arrs, color=(255))
        cv2.imwrite(os.path.join(target_masks_folder, filename + ".png"), mask)

    return count


if __name__ == "__main__":
    args = get_args()
    count = extract_masks(
        source_images_folder=args.source_images,
        souce_masks_json=args.source_json,
        target_images_folder=args.target_images,
        target_masks_folder=args.target_masks,
        mask_width=args.mask_width,
        mask_height=args.mask_height,
    )
    print("Images saved:", count)
