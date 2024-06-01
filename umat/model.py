from dataclasses import dataclass
import numpy as np
from pandas import isna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

from .config import Config
from .replay_buffer import SampledValues
from .umat import (
    get_rI,
    get_rII,
    get_driving_force,
    clip_slip_resistance,
    enforce_non_negative_increment,
    get_penalty_delta_gamma,
    get_penalty_max_slip_resistance,
    get_penalty_min_slip_resistance,
)
from .constants import consts

from jaxtyping import Float, Int
from torch import Tensor
from typing import Tuple
from numpy import ndarray


@dataclass
class Xs:
    """Batched dataset for training inputs"""

    gamma: Float[Tensor, "batch 24"] = None
    slip_resistance: Float[Tensor, "batch 24"] = None
    plastic_defgrad: Float[Tensor, "batch 9"] = None
    beta: Float[Tensor, "batch"] = None
    defgrad0: Float[Tensor, "batch 9"] = None
    defgrad1: Float[Tensor, "batch 9"] = None
    theta: Float[Tensor, "batch 3"] = None
    t0: Float[Tensor, "batch"] = None
    t1: Float[Tensor, "batch"] = None


@dataclass
class Ys:
    """Batched dataset for training outputs"""

    gamma: Float[Tensor, "batch 24"] = None
    slip_resistance: Float[Tensor, "batch 24"] = None


def weighted_mse_loss(
    x: Float[Tensor, "batch ..."],
    y: Float[Tensor, "  batch ..."],
    weights: Float[Tensor, "batch ..."],
    normalize: bool = True,
) -> Tensor:
    """Normalized weighted mse loss"""
    assert x.shape[0] == y.shape[0] == weights.shape[0]
    assert x.shape == y.shape

    weights = weights.clone()
    if normalize:
        weights /= weights.sum()

    if weights.dim() == 1 and x.dim() > 1:
        weights = weights.view(-1, 1, *([1] * (x.dim() - 2)))

    return (weights * (x - y) ** 2).mean()


class Model(nn.Module):
    def __init__(self, activation_fn) -> None:
        super().__init__()

        self.f_theta = nn.Sequential(
            nn.Linear(3, 16),
            activation_fn,
            nn.Linear(16, 32),
        )
        self.f_defGrad = nn.Sequential(
            nn.Linear(18, 32),
            activation_fn,
            nn.Linear(32, 64),
            activation_fn,
            nn.Linear(64, 32),
        )
        self.f_intvars = nn.Sequential(
            nn.Linear(48, 64),
            activation_fn,
            nn.Linear(64, 128),
            activation_fn,
            nn.Linear(128, 64),
            activation_fn,
            nn.Linear(64, 32),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.flatten = nn.Linear(640, 48)

    def make_batch(self, sampled_values: SampledValues, cfg: Config) -> Tuple[Xs, Ys]:
        nbatch = cfg.batch_size

        assert len(sampled_values.samples) == nbatch

        xs_gamma = []
        xs_slip_resistance = []
        xs_defgrad0 = []
        xs_defgrad1 = []
        xs_theta = []
        xs_plastic_defgrad = []
        xs_t0 = []
        xs_t1 = []
        xs_beta = []

        ys_gamma = []
        ys_slip_resistance = []

        for experience in sampled_values.samples:
            xs_theta.append(experience.s0.angle)
            xs_defgrad0.append(experience.s0.defgrad)
            xs_defgrad1.append(experience.s1.defgrad)
            xs_gamma.append(experience.s0.intvar.gamma)
            xs_slip_resistance.append(experience.s0.intvar.slip_resistance)
            xs_plastic_defgrad.append(experience.s0.intvar.plastic_defgrad)
            xs_beta.append(experience.s0.intvar.beta)
            xs_t0.append(experience.s0.t)
            xs_t1.append(experience.s1.t)

            ys_gamma.append(experience.s1.intvar.gamma)
            ys_slip_resistance.append(experience.s1.intvar.slip_resistance)

        stack_and_torch = lambda a: torch.from_numpy(np.stack(a))
        xs = Xs(
            gamma=stack_and_torch(xs_gamma),
            slip_resistance=stack_and_torch(xs_slip_resistance),
            defgrad0=stack_and_torch(xs_defgrad0),
            defgrad1=stack_and_torch(xs_defgrad1),
            theta=stack_and_torch(xs_theta),
            plastic_defgrad=stack_and_torch(xs_plastic_defgrad),
            t0=torch.tensor(xs_t0),
            t1=torch.tensor(xs_t1),
            beta=torch.tensor(xs_beta),
        )

        ys = Ys(gamma=stack_and_torch(ys_gamma), slip_resistance=stack_and_torch(ys_slip_resistance))

        return xs, ys

    def forward(self, xs: Xs) -> Ys:

        x1 = self.f_theta(xs.theta)
        x2 = self.f_defGrad(torch.cat((xs.defgrad0, xs.defgrad1), dim=1))
        x3 = self.f_intvars(torch.cat((xs.gamma, xs.slip_resistance), dim=1))

        conv_output = self.conv_layers(torch.stack((x1, x2, x3), dim=1))

        out = self.flatten(conv_output.flatten(start_dim=1, end_dim=2))
        return Ys(gamma=out[:, :24], slip_resistance=out[:, 24:])


class LossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, ys: Ys, ys_hat: Ys, xs: Xs, weights: Float[Tensor, "batch"], cfg: Config
    ) -> Tuple[Tensor, Float[ndarray, "batch"]]:
        # clipped_slipres1_hat = torch.clip(slipres1_hat, consts.s0_F, consts.sInf_F)
        #     # ensures that dgamma >= 0.0
        # clipped_gamma1_hat = torch.where(
        #         gamma1_hat >= vals0.gamma, gamma1_hat, vals0.gamma
        # )
        dtime: Float[Tensor, "batch"] = xs.t1 - xs.t0

        clipped_slipres1_hat = clip_slip_resistance(ys_hat.slip_resistance)
        # ensures that dgamma >= 0.0
        delta_gamma = enforce_non_negative_increment(ys_hat.gamma, xs.gamma)

        g1, H_matrix, non_schmid_stress = get_driving_force(
            slip_resistance0=xs.slip_resistance,
            slip_resistance1=clipped_slipres1_hat,
            delta_gamma=delta_gamma - xs.gamma,
            beta0=xs.beta,
            Fp0=xs.plastic_defgrad.reshape(-1, 3, 3),
            theta=xs.theta,
            F1=xs.defgrad1.reshape(-1, 3, 3),
        )
        r_I = vmap(get_rI)(
            xs.slip_resistance,
            # clip s1 to be between min and max
            clipped_slipres1_hat,
            xs.gamma,
            # gamma1 should be bigger than or equal to gamma0
            delta_gamma,
            H_matrix,
        )
        r_II = vmap(get_rII)(
            g1,
            # clip s1 to be between min and max
            clipped_slipres1_hat,
            non_schmid_stress,
            xs.gamma,
            # gamma1 should be bigger than or equal to gamma0
            delta_gamma,
            torch.tensor(cfg.batch_size * [consts.GammaDot0_F], dtype=torch.float64),
            dtime,
            torch.tensor(cfg.batch_size * [consts.pExp_F], dtype=torch.float64),
        )

        penalty_delta_gamma = get_penalty_delta_gamma(ys_hat.gamma, xs.gamma)
        penalty_max_slipresistance = get_penalty_max_slip_resistance(ys_hat.slip_resistance)
        penalty_min_slipresistance = get_penalty_min_slip_resistance(ys_hat.slip_resistance)

        data_loss = weighted_mse_loss(ys_hat.gamma, ys.gamma, weights) + weighted_mse_loss(
            ys_hat.slip_resistance, ys.slip_resistance, weights
        )
        physics_loss = weighted_mse_loss(r_I, torch.zeros_like(r_I), weights) + weighted_mse_loss(
            r_II, torch.zeros_like(r_II), weights
        )

        if torch.isnan(physics_loss):
            physics_loss = torch.tensor([0.0], dtype=torch.float64)

        loss = (
            cfg.coeff_data * data_loss
            + cfg.coeff_physics * physics_loss
            + cfg.penalty_coeff_delta_gamma
            * weighted_mse_loss(penalty_delta_gamma, torch.zeros_like(penalty_delta_gamma), weights)
            + cfg.penalty_coeff_min_slipres
            * weighted_mse_loss(penalty_min_slipresistance, torch.zeros_like(penalty_min_slipresistance), weights)
            + cfg.penalty_coeff_max_slipres
            * weighted_mse_loss(penalty_max_slipresistance, torch.zeros_like(penalty_max_slipresistance), weights)
        )

        mean_abs_to_numpy = lambda t: t.abs().mean(axis=1).numpy().squeeze()

        with torch.no_grad():
            # Use `mean` for non-batch dimension. Later try with `sum` to see if it works better/worst
            # This is technically not td-error but we just use the terminology from RL.
            td_error_data = mean_abs_to_numpy(ys_hat.gamma - ys.gamma) + mean_abs_to_numpy(
                ys_hat.slip_resistance - ys.slip_resistance
            )
            if np.isnan(td_error_data).any():
                raise ValueError("Nans are detected in TD error data calculation.")

            td_error_physics = mean_abs_to_numpy(r_I) + mean_abs_to_numpy(r_II)
            if np.isnan(td_error_physics).any():
                td_error_physics = np.zeros_like(td_error_physics)

            td_error = (
                cfg.coeff_data * td_error_data
                + cfg.coeff_physics * td_error_physics
                + cfg.penalty_coeff_delta_gamma * mean_abs_to_numpy(penalty_delta_gamma)
                + cfg.penalty_coeff_max_slipres * mean_abs_to_numpy(penalty_max_slipresistance)
                + cfg.penalty_coeff_min_slipres * mean_abs_to_numpy(penalty_min_slipresistance)
            )

        return loss, td_error
