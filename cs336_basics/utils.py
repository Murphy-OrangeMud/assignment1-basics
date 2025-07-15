import math
import torch
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from typing import Optional
import einops
from cs336_basics.model import softmax


# Not passed test yet
def cross_entropy(pred: torch.Tensor, target: torch.Tensor, eps=1e-15):
    pred = softmax(pred, -1)
    pred = torch.clip(pred, eps, 1 - eps)
    target = F.one_hot(target, torch.max(target) + 1)

    return -torch.sum(target * torch.log(pred)) / target.shape[0]


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss
        

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 betas,
                 weight_decay,
                 lr=1e-3, 
                 eps=1e-6,
                 dtype=None,
                 device=None):
        defaults = {"lr": lr, 
                    "beta_1": betas[0], 
                    "beta_2": betas[1], 
                    "decay": weight_decay, 
                    "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            lr = group["lr"]
            eps = group["eps"]
            decay = group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                lr_t = state.get("lr_t", group["lr"])
                m = state.get("m", torch.zeros(p.data.shape))
                v = state.get("v", torch.zeros(p.data.shape))
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                lr_t = lr * torch.sqrt(1 - torch.pow(torch.tensor(beta_2), t)) / (1 - torch.pow(torch.tensor(beta_1), t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data *= (1 - lr * decay)

                # update states
                state["t"] = t + 1
                state["lr_t"] = lr_t
                state["m"] = m
                state["v"] = v


def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        alpha_t = t * alpha_max / T_w
    elif t <= T_c:
        alpha_t = alpha_min + (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) / 2 * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t


def gradient_clipping(params, M, eps = 1e-6):
    import numpy as np
    grad = 0.
    for param in params:
        if param.grad is None:
            continue
        grad += torch.sum(param.grad.data ** 2)

    grad = torch.sqrt(grad)
    
    if grad >= M:
        for param in params:
            if param.grad is None:
                continue
            param.grad.data *= (M / (grad + eps))


def test_sgd():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e-3)

    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
