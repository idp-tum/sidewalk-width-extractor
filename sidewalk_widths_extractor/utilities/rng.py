import os
import random

import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    r"""
    Makes the necessary adjustments for reproducibility

    Args:
        seed (int): a random number generator seed.
            (default is 42)
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
