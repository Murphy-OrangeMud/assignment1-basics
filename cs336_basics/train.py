import torch
import torch.nn as nn
import numpy as np
import typing
from typing import Tuple
import os
import random


def get_batch(x: np.ndarray, 
              batch_size: int, 
              context_len: int, 
              device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    start_indices = random.sample(range(0, x.shape[0] - context_len), batch_size)
    batches = np.zeros((batch_size, context_len))
    targets = np.zeros((batch_size, context_len))
    
    for i in range(len(start_indices)):
        batches[i, :] = x[start_indices[i]: start_indices[i] + context_len]
        targets[i, :] = x[start_indices[i] + 1: start_indices[i] + 1 + context_len]

    return torch.Tensor(batches, device=torch.device(device)), torch.Tensor(targets, device=torch.device(device))


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj_state = {
        "t": iteration,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }
    import pickle as pkl
    with open(out, 'wb') as f:
        pkl.dump(obj_state, f)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    import pickle as pkl
    with open(src, "rb") as f:
        obj = pkl.load(f)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optim"])
    iteration = int(obj["t"])
    return iteration
