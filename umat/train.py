from dataclasses import dataclass
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


def clone_grads(loss, model, optimizer: optim.Optimizer):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    return [p.grad.clone() for p in model.parameters()]


def norm_of_ts(ts, p):
    return sum([t.norm(p=p).item() ** 2 for t in ts]) ** 0.5


def norm_of_grad(loss, model, optimizer, p):
    return norm_of_ts(clone_grads(loss, model, optimizer), p)


@dataclass
class LogFlags:
    loss: bool = False
    loss_grad_norm: bool = False
    params_histogram: bool = False


class Logger:
    def __init__(self, log_flags: LogFlags, log_frequencies: dict):
        self.log_flags = log_flags
        self.log_frequencies = log_frequencies
        self._counters = {"loss": 0, "loss_grad_norm": 0, "params_histogram": 0}

    def should_log(self, key: str) -> bool:
        if getattr(self.log_flags, key):
            if self._counters[key] % self.log_frequencies[key] == 0:
                self._counters[key] += 1
                return True
            self._counters[key] += 1
        return False


@dataclass
class Losses:
    '''Fields with `pnt_` prefix stand for "penalty"'''

    data: torch.Tensor
    physics: torch.Tensor
    pnt_delta_gamma: torch.Tensor
    pnt_negative_gamma: torch.Tensor
    pnt_min_slipresistance: torch.Tensor
    pnt_max_slipresistance: torch.Tensor


def log_losses(writer: SummaryWriter, idx: int, losses: Losses):
    detach = lambda x: x.detach().item()

    writer.add_scalars(
        "Loss",
        {
            "data": detach(losses.data),
            "physics": detach(losses.physics),
            "penalty_delta_gamma": detach(losses.pnt_delta_gamma),
            "penalty_negative_gamma": detach(losses.pnt_negative_gamma),
            "penalty_min_slipresistance": detach(losses.pnt_min_slipresistance),
            "penalty_max_slipresistance": detach(losses.pnt_max_slipresistance),
        },
        idx,
    )


def log_gradient_norm(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
):
    writer.add_scalars(
        "GradNorm",
        {
            "data": norm_of_grad(losses.data, model, optimizer, p=2),
            "physics": norm_of_grad(losses.physics, model, optimizer, p=2),
            "delta_gamma": norm_of_grad(losses.pnt_delta_gamma, model, optimizer, p=1),
            "negative_gamma": norm_of_grad(
                losses.pnt_negative_gamma, model, optimizer, p=1
            ),
            "min_slipres": norm_of_grad(
                losses.pnt_min_slipresistance, model, optimizer, p=1
            ),
            "max_slipres": norm_of_grad(
                losses.pnt_max_slipresistance, model, optimizer, p=1
            ),
        },
        idx,
    )


def log_errors(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
):
    log_losses(writer, idx, losses)
    log_gradient_norm(writer, model, optimizer, idx, losses)
    # for name, param in model.named_parameters():
    #     writer.add_histogram(name + "/grad", param.grad.data, idx)


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

    n_steps_per_sim = sims[0]["stress"].shape[0] - 1
    n_sims = len(sims)
    n_batch = params["n_batch"]
    dtime = 1 / n_steps_per_sim

    n_total = n_sims * n_steps_per_sim
    n_epoch = params["n_epoch"]

    alpha_data = params["coeff_loss_data"]
    alpha_physics = params["coeff_loss_physics"]
    penalty_coeff_delta_gamma = params["coeff_penalty_delta_gamma"]
    penalty_coeff_negative_gamma = params["coeff_penalty_negative_gamma"]
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
            # driving force g at n+1
            g1, H_matrix, non_schmid_stress = get_driving_force(
                slip_resistance0=vals0.slip_res,
                slip_resistance1=slipres1_hat,
                delta_gamma=gamma1_hat - vals0.gamma,
                beta0=vals0.beta,
                Fp0=vals0.Fp.reshape(-1, 3, 3),
                theta=vals0.theta,
                F1=vals0.F1.reshape(-1, 3, 3),
            )
            r_I = vmap(get_rI)(
                vals0.slip_res, slipres1_hat, vals0.gamma, gamma1_hat, H_matrix
            )
            r_II = vmap(get_rII)(
                g1,
                slipres1_hat,
                non_schmid_stress,
                vals0.gamma,
                gamma1_hat,
                torch.tensor(n_batch * [consts.GammaDot0_F], dtype=torch.float64),
                torch.tensor(n_batch * [consts.pExp_F], dtype=torch.float64),
                torch.tensor(n_batch * [dtime], dtype=torch.float64),
            )
            physics_loss = F.mse_loss(r_I, torch.zeros_like(r_I)) + F.mse_loss(
                r_II, torch.zeros_like(r_II)
            )
            data_loss = F.mse_loss(gamma1_hat, vals1.gamma) + F.mse_loss(
                slipres1_hat, vals1.slip_res
            )
            penalty_delta_gamma = torch.where(
                gamma1_hat >= vals0.gamma, 0.0, gamma1_hat - vals0.gamma
            )

            penalty_negative_gamma = torch.where(gamma1_hat > 0, 0.0, -gamma1_hat)

            penalty_max_slipresistance = torch.where(
                slipres1_hat <= consts.sInf_F, 0.0, slipres1_hat - consts.sInf_F
            )

            penalty_min_slipresistance = torch.where(
                slipres1_hat >= consts.s0_F, 0.0, consts.s0_F - slipres1_hat
            )

            optimizer.zero_grad()
            log_errors(
                writer,
                model,
                optimizer,
                idx,
                Losses(
                    data=data_loss,
                    physics=physics_loss,
                    pnt_delta_gamma=penalty_delta_gamma.norm(p=1, dim=1).mean(),
                    pnt_negative_gamma=penalty_negative_gamma.norm(p=1, dim=1).mean(),
                    pnt_min_slipresistance=penalty_min_slipresistance.norm(
                        p=1, dim=1
                    ).mean(),
                    pnt_max_slipresistance=penalty_max_slipresistance.norm(
                        p=1, dim=1
                    ).mean(),
                ),
            )
            if torch.isnan(physics_loss):
                physics_loss = torch.tensor([0.0], dtype=torch.float64)

            loss = (
                alpha_data * data_loss
                + alpha_physics * physics_loss
                + penalty_coeff_delta_gamma * penalty_delta_gamma.norm(p=pnorm)
                + penalty_coeff_negative_gamma * penalty_negative_gamma.norm(p=pnorm)
                + penalty_coeff_min_slipres * penalty_min_slipresistance.norm(p=pnorm)
                + penalty_coeff_max_slipres * penalty_max_slipresistance.norm(p=pnorm)
            )

            loss.backward()
            optimizer.step()

            idx += 1
    writer.close()


if __name__ == "__main__":
    train()
