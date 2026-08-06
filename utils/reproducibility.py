import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for experiment reproducibility.
    
    Sets seeds for Python random, NumPy, PyTorch CPU/CUDA,
    and configures cuDNN deterministic mode.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
