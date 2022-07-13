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
    out = []
    for i in range(nrow):
        out.append(draw_segmentation_masks(t_img[i], t_pred[i], alpha=0.5, colors="blue"))
    t = torch.cat(out, 2)

    grid = make_grid(t, nrow=nrow, pad_value=pad_value)
    return grid
