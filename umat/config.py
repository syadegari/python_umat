import pprint
from dataclasses import dataclass
import numpy as np


@dataclass
class LogFlags: ...


@dataclass
class Config:
    buffer_size: int = None
    min_buffer_size: int = None
    batch_size: int = None  # for training and sampling from the buffer
    buffer_n_steps: int = None  # Used for annealing beta

    N_iteration: int = None
    N_validation: int = None

    dataset_path: str = None

    n_time: int = None

    # training loss coefficients
    coeff_data: float = None
    coeff_physics: float = None
    penalty_coeff_delta_gamma: float = None
    penalty_coeff_min_slipres: float = None
    penalty_coeff_max_slipres: float = None

    lr: float = None

    use_lr_scheduler: bool = None
    lr_scheduler_T_0: int = None
    lr_scheduler_T_multi: int = None
    lr_scheduler_initial_lr: float = None
    lt_scheduler_final_lr: float = None
    lr_scheduler_eta_min: float = None

    # split dataset values
    split_train_proportion: float = None
    split_val_proportion: float = None
    split_test_proportion: float = None

    # dataset batch sizes
    dataset_batch_train: int = None
    dataset_batch_val: int = None
    dataset_batch_test: int = None

    # log directory
    log_directory: str = None

    def __str__(self) -> str:
        return pprint.pformat(vars(self))

    def __post_init__(self):
        self.coeff_data = float(self.coeff_data)
        self.coeff_physics = float(self.coeff_physics)
        self.penalty_coeff_delta_gamma = float(self.penalty_coeff_delta_gamma)
        self.penalty_coeff_min_slipres = float(self.penalty_coeff_min_slipres)
        self.penalty_coeff_max_slipres = float(self.penalty_coeff_max_slipres)

        assert np.isclose(
            self.split_train_proportion + self.split_val_proportion + self.split_test_proportion,
            1.0,
        ), "The sum of train_prop, val_prop, and test_prop must equal 1.0"
