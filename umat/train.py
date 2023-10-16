import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

from .model import Model
from .umat import get_driving_force
from .constants import consts
from .dataloader import DataLoader, BatchSampler, SequenceDataset, make_batch


def log_errors(data_loss, physics_loss, penalty_gamma, penalty_slipresistance):
    precision = ".6e"
    detach = lambda x: x.detach().item()
    print(
        f"dloss={detach(data_loss):{precision}},"
        f" phloss={detach(physics_loss):{precision}},"
        f" pngamma={detach(penalty_gamma):{precision}},"
        f" pnslipres={detach(penalty_slipresistance):{precision}}"
    )


def get_rI(s0, s1, gamma0, gamma1, H_matrix):
    return (s1 - s0) - H_matrix @ (gamma1 - gamma0)


def get_rII(g1, s1, gamma0, gamma1, gamma_dot_0, dt, pF):
    return torch.where(
        g1 > s1,
        g1 - s1 * ((gamma1 - gamma0) / (gamma_dot_0 * dt) + 1) ** pF,
        gamma1 - gamma0,
    )


def main():
    with open("./sims_sample.pkl", "rb") as f:
        sims = pickle.load(f)

    n_steps_per_sim = sims[0]["stress"].shape[0] - 1
    n_sims = len(sims)
    n_batch = 4
    dtime = 1 / n_steps_per_sim

    n_total = n_sims * n_steps_per_sim

    # alpha_data = 1
    # alpha_physics = 10
    # penalty_coeff = 1e4

    alpha_data = 1.0
    alpha_physics = 0.0
    penalty_coeff = 0.0

    seq = torch.arange(n_total)
    dataset = SequenceDataset(seq)
    sampler = BatchSampler(dataset, n_batch)
    dataloder = DataLoader(dataset, batch_sampler=sampler)

    model = Model(nn.Tanh()).to(torch.float64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
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
            if torch.all(slipres1_hat < consts.sInf_F):
                g1, H_matrix = get_driving_force(
                    slip_resistance0=vals0.slip_res,
                    slip_resistance1=slipres1_hat,
                    delta_gamma=gamma1_hat - vals0.gamma,
                    beta0=vals0.beta,
                    Fp0=vals0.Fp.reshape(-1, 3, 3),
                    theta=vals0.theta,
                    F1=vals0.F1.reshape(-1, 3, 3),
                )
                r_I = vmap(get_rI)(
                    vals0.slip_res,
                    slipres1_hat,
                    vals0.gamma,
                    gamma1_hat,
                    H_matrix,
                )
                r_II = vmap(get_rII)(
                    g1,
                    slipres1_hat,
                    vals0.gamma,
                    gamma1_hat,
                    torch.tensor(
                        n_batch * [consts.GammaDot0_F], dtype=torch.float64
                    ),
                    torch.tensor(
                        n_batch * [consts.pExp_F], dtype=torch.float64
                    ),
                    torch.tensor(n_batch * [dtime], dtype=torch.float64),
                )
                physics_loss = F.mse_loss(
                    r_I, torch.zeros_like(r_I)
                ) + F.mse_loss(r_II, torch.zeros_like(r_II))
            else:
                physics_loss = torch.tensor([0.0], dtype=torch.float64)

            data_loss = F.mse_loss(gamma1_hat, vals1.gamma) + F.mse_loss(
                slipres1_hat, vals1.slip_res
            )
            penalty_gamma = torch.where(
                gamma1_hat >= vals0.gamma, 0, gamma1_hat - vals0.gamma
            )
            penalty_slipresistance = torch.where(
                slipres1_hat <= consts.sInf_F, 0, slipres1_hat - consts.sInf_F
            )

            optimizer.zero_grad()
            loss = (
                alpha_data * data_loss
                + alpha_physics * physics_loss
                + penalty_coeff
                * (penalty_gamma.norm(1) + penalty_slipresistance.norm(p=1))
            )
            loss.backward()
            optimizer.step()

            log_errors(
                data_loss,
                physics_loss,
                penalty_gamma.norm(p=2),
                penalty_slipresistance.norm(p=2),
            )


if __name__ == "__main__":
    main()
