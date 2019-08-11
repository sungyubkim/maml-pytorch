import os
import random
import numpy as np
import torch

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def adjust_opt(optimizer, epoch):
    if epoch < 20: lr = 1e-3
    elif epoch == 40: lr = 3e-4
    elif epoch == 50: lr = 1e-4
    else: return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr