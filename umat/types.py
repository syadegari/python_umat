from typing import TypeVar
import torch

B = TypeVar("B")


class FloatTensor(torch.Tensor):
    @classmethod
    def __class_getitem__(cls, item):
        return cls
