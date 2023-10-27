from dataclasses import dataclass
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap
from torch.utils.tensorboard import SummaryWriter

from .model import Model
from .umat import get_driving_force
from .constants import consts
from .dataloader import DataLoader, BatchSampler, SequenceDataset, make_batch
from .logger import Logger, log_errors, Losses


def get_rI(s0, s1, gamma0, gamma1, H_matrix):
    return (s1 - s0) - H_matrix @ (gamma1 - gamma0)


def get_rII(g1, s1, non_schmid_stress, gamma0, gamma1, gamma_dot_0, dt, pF):
    return torch.where(
        g1 > s1,
        g1
        - (s1 - non_schmid_stress) * ((gamma1 - gamma0) / (gamma_dot_0 * dt) + 1) ** pF,
        gamma1 - gamma0,
    )


def train(params):
    with open(params["sims_path"], "rb") as f:
        sims = pickle.load(f)

    writer = SummaryWriter(params["tboard_path"])

    logger = Logger(
        log_flags=params["log_flags"], log_frequencies=params["log_frequencies"]
    )
    n_steps_per_sim = sims[0]["stress"].shape[0] - 1
    n_sims = len(sims)
    n_batch = params["n_batch"]
    dtime = 1 / n_steps_per_sim

    n_total = n_sims * n_steps_per_sim
    n_epoch = params["n_epoch"]

    coeff_data = params["coeff_loss_data"]
    coeff_physics = params["coeff_loss_physics"]
    penalty_coeff_delta_gamma = params["coeff_penalty_delta_gamma"]
    penalty_coeff_min_slipres = params["coeff_penalty_min_slipres"]
    penalty_coeff_max_slipres = params["coeff_penalty_max_slipres"]
    pnorm = params["penalty_pnorm"]

    seq = torch.arange(n_total)
    dataset = SequenceDataset(seq)
    sampler = BatchSampler(dataset, n_batch)
    dataloder = DataLoader(dataset, batch_sampler=sampler)

    model = Model(nn.Tanh()).to(torch.float64)
    optimizer = optim.Adam(model.parameters(), lr=params["opt_lr"])

    idx = 0
    for epoch in range(1, n_epoch + 1):
        print(f"epoch = {epoch}")
        for batch in dataloder:
            vals0, vals1 = make_batch(
                batch.numpy(),
                sims,
                ts=np.linspace(0, 1, n_steps_per_sim + 1),
                N=n_steps_per_sim,
            )
            gamma1_hat, slipres1_hat = model.forward(
                theta=vals0.theta,
                defgrad0=vals0.F0,
                defgrad1=vals0.F1,
                gamma0=vals0.gamma,
                slip_res0=vals0.slip_res,
            )

            clipped_slipres1_hat = torch.clip(slipres1_hat, consts.s0_F, consts.sInf_F)
            # ensures that dgamma >= 0.0
            clipped_gamma1_hat = torch.where(
                gamma1_hat >= vals0.gamma, gamma1_hat, vals0.gamma
            )

            # driving force g at n+1
            g1, H_matrix, non_schmid_stress = get_driving_force(
                slip_resistance0=vals0.slip_res,
                slip_resistance1=clipped_slipres1_hat,
                delta_gamma=clipped_gamma1_hat - vals0.gamma,
                beta0=vals0.beta,
                Fp0=vals0.Fp.reshape(-1, 3, 3),
                theta=vals0.theta,
                F1=vals0.F1.reshape(-1, 3, 3),
            )

            r_I = vmap(get_rI)(
                vals0.slip_res,
                # clip s1 to be between min and max
                clipped_slipres1_hat,
                vals0.gamma,
                # gamma1 should be bigger than or equal to gamma0
                clipped_gamma1_hat,
                H_matrix,
            )
            r_II = vmap(get_rII)(
                g1,
                # clip s1 to be between min and max
                clipped_slipres1_hat,
                non_schmid_stress,
                vals0.gamma,
                # gamma1 should be bigger than or equal to gamma0
                clipped_gamma1_hat,
                torch.tensor(n_batch * [consts.GammaDot0_F], dtype=torch.float64),
                torch.tensor(n_batch * [dtime], dtype=torch.float64),
                torch.tensor(n_batch * [consts.pExp_F], dtype=torch.float64),
            )
            physics_loss = F.mse_loss(r_I, torch.zeros_like(r_I)) + F.mse_loss(
                r_II, torch.zeros_like(r_II)
            )
            data_loss = F.mse_loss(gamma1_hat, vals1.gamma) + F.mse_loss(
                slipres1_hat, vals1.slip_res
            )
            penalty_delta_gamma = torch.where(
                gamma1_hat >= vals0.gamma, 0.0, vals0.gamma - gamma1_hat
            )

            penalty_negative_gamma = torch.where(gamma1_hat > 0, 0.0, -gamma1_hat)

            penalty_max_slipresistance = torch.where(
                slipres1_hat <= consts.sInf_F, 0.0, slipres1_hat - consts.sInf_F
            )

            penalty_min_slipresistance = torch.where(
                slipres1_hat >= consts.s0_F, 0.0, consts.s0_F - slipres1_hat
            )

            log_errors(
                writer,
                model,
                optimizer,
                idx,
                Losses(
                    data=data_loss,
                    physics=physics_loss,
                    pnt_delta_gamma=penalty_delta_gamma.norm(p=1, dim=1).mean(),
                    pnt_min_slipresistance=penalty_min_slipresistance.norm(
                        p=1, dim=1
                    ).mean(),
                    pnt_max_slipresistance=penalty_max_slipresistance.norm(
                        p=1, dim=1
                    ).mean(),
                ),
                logger=logger,
            )
            if torch.isnan(physics_loss):
                physics_loss = torch.tensor([0.0], dtype=torch.float64)

            loss = (
                coeff_data * data_loss
                + coeff_physics * physics_loss
                + penalty_coeff_delta_gamma * penalty_delta_gamma.norm(p=pnorm)
                + penalty_coeff_min_slipres * penalty_min_slipresistance.norm(p=pnorm)
                + penalty_coeff_max_slipres * penalty_max_slipresistance.norm(p=pnorm)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx += 1
    writer.close()
    torch.save(
        model.state_dict(),
        f'{params["config_file_path"]}/{params["experiment_name"]}.pth',
    )
