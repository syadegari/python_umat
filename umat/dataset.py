import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .config import Config

from jaxtyping import Float
from torch import Tensor
from numpy import ndarray
from typing import Any, Dict, Iterator


def read_hdf5(path: str) -> dict:
    data = {}

    with h5py.File(path, "r") as fh:
        data_pairs = fh["data_pairs"]

        idx = 0
        for key in data_pairs:
            if isinstance(data_pairs[key], h5py.Group):
                angle = data_pairs[key]["angle"][:]
                defgrad = data_pairs[key]["defgrad"][:]
                data[idx] = {"angle": angle, "defgrad": defgrad}
                idx += 1

    return data


class UMATDataSet(Dataset):
    def __init__(self, data: dict) -> None:
        """
        Args:
            data (dict): Dictionary where each key is an index, and the value is another dict with 'angle' and 'defgrad'
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        defgrad = item["defgrad"]
        angle = item["angle"]
        return {"defgrad": defgrad, "angle": angle}


def split_dataset(dataset: UMATDataSet, cfg: Config) -> tuple:
    """
    Splits the dataset into training, validation, and test datasets.

    Args:
        dataset (Dataset): The dataset to split.
        train_prop (float): Proportion of the dataset to use for training.
        val_prop (float): Proportion of the dataset to use for validation.
        test_prop (float): Proportion of the dataset to use for testing.

    Returns:
        tuple: Containing (train_dataset, val_dataset, test_dataset)

    Example:
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.7, 0.15, 0.15)
    """
    train_prop = cfg.split_train_proportion
    val_prop = cfg.split_val_proportion
    test_prop = cfg.split_test_proportion

    total_len = len(dataset)
    train_size = int(train_prop * total_len)
    val_size = int(val_prop * total_len)
    # Use all the data
    test_size = total_len - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: UMATDataSet,
    val_dataset: UMATDataSet,
    test_dataset: UMATDataSet,
    cfg: Config,
    shuffle_train: bool = True,
):
    """
    Creates DataLoader instances for training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.
        batch_size (int, optional): The size of each batch. Defaults to 10.
        shuffle_train (bool, optional): Whether to shuffle the training dataset. Defaults to True.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset_batch_train, shuffle=shuffle_train, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset_batch_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset_batch_test, shuffle=False)

    return train_loader, val_loader, test_loader


def circular_loader(loader: DataLoader) -> Iterator:
    """
    Infinite loop over the data

    Example:
        circular_train_loader = circular_loader(train_loader)
        for data in circular_train_loader:
            defgrad_batch = data['defgrad']
            angle_batch = data['angle']
    """
    while True:
        for data in loader:
            yield data
