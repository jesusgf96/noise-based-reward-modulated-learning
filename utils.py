import random
import numpy as np
import torch


def softmax():
    return lambda x: torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)


def leaky_relu(ALPHA):
    return lambda x: torch.where(x >= 0, x, ALPHA * x)


def relu():
    return lambda x: torch.where(x >= 0, x, 0)


def linear():
    return lambda x: x


def tanh():
    return lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def mse(pred, target):
    return torch.mean((pred - target)**2)


def control_randomness(seed, env):
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed) # cuda
    env.seed(seed) # gym environment