import torch
from .driver import driver


if __name__ == '__main__':
    F = torch.eye(3)
    F[0, 2] = .1
    driver(F, 1000, torch.tensor([1., 0, 0]))
