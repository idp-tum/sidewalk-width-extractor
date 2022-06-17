import os
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal
from PIL import Image
from tqdm import tqdm


def get_args() -> Namespace:
    """Parse given arguments

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", type=str, help="path to the tif file", required=True)
    parser.add_argument("-o", "--output", type=str, help="path to dump image files", required=True)
    parser.add_argument(
        "-x", "--x_image_res", type=int, help="ouput images' x resolution", required=False
    )
    parser.add_argument(
        "-y", "--y_image_res", type=int, help="ouput images' y resolution", required=False
    )
    parser.add_argument(
        "--start_x",
        type=int,
        help="exctraction starting X coordinate in the main satellite image",
        required=False,
    )
    parser.add_argument(
        "--end_x",
        type=int,
        help="exctraction ending X coordinate in the main satellite image",
        required=False,
    )
    parser.add_argument(
        "--start_y",
        type=int,
        help="exctraction starting Y coordinate in the main satellite image",
        required=False,
    )
    parser.add_argument(
        "--end_y",
        type=int,
        help="exctraction ending Y coordinate in the main satellite image",
        required=False,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        help="limit the number of samples to be extracted",
        required=False,
    )

    return parser.parse_args()


def extract_images(
    input_path: str,
    output_path: str,
    x_image_res: Optional[int] = None,
    y_image_res: Optional[int] = None,
    x_start_coord: Optional[int] = None,
    x_end_coord: Optional[int] = None,
    y_start_coord: Optional[int] = None,
    y_end_coord: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    assert os.path.isfile(input_path) and input_path.lower().endswith(".tif")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ds = gdal.Open(input_path, gdal.GA_ReadOnly)

    x_blocksize = x_image_res if x_image_res else ds.GetRasterBand(1).GetBlockSize()[0]
    y_blocksize = y_image_res if y_image_res else ds.GetRasterBand(1).GetBlockSize()[1]

    assert x_blocksize > 0 and y_blocksize > 0

    total_raster_count = ds.RasterCount
    bands = [ds.GetRasterBand(i) for i in range(1, total_raster_count + 1)]

    assert len(bands) > 0

    x_s_coord = x_start_coord if x_start_coord else 0
    y_s_coord = y_start_coord if y_start_coord else 0

    x_e_coord = x_end_coord if x_end_coord else bands[0].XSize
    y_e_coord = y_end_coord if y_end_coord else bands[0].YSize

    total = 0
    total_extracted = 0
    total_pb_iter = len(range(y_s_coord, y_e_coord, y_blocksize)) * len(
        range(x_s_coord, x_e_coord, x_blocksize)
    )

    pbar = tqdm(total=total_pb_iter, desc="Extracting", mininterval=0.25, leave=False)
    for y in range(y_s_coord, y_e_coord, y_blocksize):
        if y + y_blocksize < y_e_coord:
            rows = y_blocksize
        else:
            rows = y_e_coord - y
        for x in range(x_s_coord, x_e_coord, x_blocksize):
            if x + x_blocksize < x_e_coord:
                cols = x_blocksize
            else:
                cols = x_e_coord - x

            arrays = [band.ReadAsArray(x, y, cols, rows) for band in bands]

            img = Image.fromarray(np.dstack(arrays))

            if img.size[1] == y_blocksize and img.size[0] == x_blocksize:
                if arrays[0].max() != 0 and arrays[0].min():
                    img.save(
                        os.path.join(output_path, f"{x}-{x+x_blocksize}_{y}-{y+y_blocksize}.png")
                    )
                    total_extracted += 1
            total += 1

            pbar.update(1)
            if max_samples and total_extracted >= max_samples:
                pbar.close()
                return total, total_extracted
    pbar.close()
    return total, total_extracted


if __name__ == "__main__":
    args = get_args()
    total, total_extracted = extract_images(
        input_path=args.input,
        output_path=args.output,
        x_image_res=args.x_image_res,
        y_image_res=args.y_image_res,
        x_start_coord=args.start_x,
        x_end_coord=args.end_x,
        y_start_coord=args.start_y,
        y_end_coord=args.end_y,
        max_samples=args.max_samples,
    )
    print(f"Extracted {total_extracted} out of {total}.")
