import torch
from torchvision.utils import draw_segmentation_masks, make_grid


def create_stacked_segments(
    img: torch.Tensor, predict: torch.Tensor, target: torch.Tensor, nrow: int, pad_value=1
) -> torch.Tensor:
    r"""
    TODO documentation
    """

    t_img = (img * 255).type(torch.uint8)
    t_pred = predict.type(torch.bool)
    t_target = target.type(torch.bool)

    t = torch.cat(
        [
            torch.cat(
                [
                    draw_segmentation_masks(t_img[i], t_target[i], alpha=0.3, colors="red")
                    for i in range(nrow)
                ],
                2,
            ),
            torch.cat(
                [
                    draw_segmentation_masks(t_img[i], t_pred[i], alpha=0.3, colors="blue")
                    for i in range(nrow)
                ],
                2,
            ),
        ],
        1,
    )

    grid = make_grid(t, nrow=nrow, pad_value=pad_value)
    return grid
