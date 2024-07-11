import torch
import numpy as np
import random


# Sets the most important seeds, ensures reproducibility for benchmarks
def set_seed(seed=9001):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
