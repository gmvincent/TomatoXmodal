import random
import torch
import numpy as np

def set_seed(random_seed, rank=None, deterministic=False):
    seed = random_seed + rank if rank is not None else random_seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # A100 speedups
    torch.backends.cuda.matmul.allow_tf32 = True  # speed up matmul on A100
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    if deterministic:
        # exact reproducibility (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster (recommended default)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True