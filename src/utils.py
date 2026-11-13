import random

import numpy as np
import torch


def set_seed(seed):
    """Make runs repeatable: same seed gives the same shuffling and initial weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"