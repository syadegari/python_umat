import numpy as np
import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader, Sampler


def getF(F_init, F_final, t):
    assert 0 <= t <= 1
    return (1 - t) * F_init + F_final * t


def get_ts(n_time):
    return np.linspace(0, 1, n_time + 1)


def pairwise(xs):
    return zip(xs[:-1], xs[1:])


@dataclass
class Input:
    """
    Except F0 anc F1, all other values correspond to time t_n
    """

    F0: torch.Tensor
    F1: torch.Tensor
    Fp: torch.Tensor
    theta: torch.Tensor
    gamma: torch.Tensor
    slip_res: torch.Tensor
    beta: torch.Tensor
    dtype: torch.dtype = field(default=torch.float64, init=True)

    def __post_init__(self):
        self.F0 = torch.tensor(self.F0).to(self.dtype)
        self.F1 = torch.tensor(self.F1).to(self.dtype)
        self.Fp = torch.tensor(self.Fp).to(self.dtype)
        self.theta = torch.tensor(self.theta).to(self.dtype)
        self.gamma = torch.tensor(self.gamma).to(self.dtype)
        self.slip_res = torch.tensor(self.slip_res).to(self.dtype)
        self.beta = torch.tensor(self.beta).to(self.dtype)


@dataclass
class Output:
    gamma: torch.Tensor
    slip_res: torch.Tensor
    dtype: torch.dtype = field(default=torch.float64, init=True)

    def __post_init__(self):
        self.gamma = torch.tensor(self.gamma).to(self.dtype)
        self.slip_res = torch.tensor(self.slip_res).to(self.dtype)


def get_indices(n, N):
    timestep0, timestep1 = n % N, n % N + 1
    sim_number = n // N
    return timestep0, timestep1, sim_number


def get_data(sims, ts, n, N):
    """
    n : drawn from dataloader
    N : datapoints per simulation (we drop the last point,
        so if we have t0, ... t10, which is 11 points, we choose N to be 10)
    ts: time vector [t0, ..., tN], where t0=0.0 and tN=1.0
    sims: dictionary of sims indexed from 0

    In the following N = 5 and we have 3 simulations.
    the simulations are stored in the following dict:
    sims = {0: sim0, 1: sim1, 2: sim2}

    We sample a number n between 0 and 15 (exclusive: 0, 1, ..., 14). For example, if we draw 3:
      - to determine the simulation sim_i = n // N
      - To determine the point: t_n = x % N.

    So for n=3, we have 3 -> 3 // 5 = 0 and 3 % 5 = 3.
    So we sample sim0 and t_n=0.6, t_n+1=0.4.
    that correspond to ts[3] and ts[4]

          sim0                sim1            sim2
      n    t              n    t         n     t
          ----                ----            ----
     (0)  0.0            (5)  0.0       (10)  0.0
     (1)  0.2            (6)  0.2       (11)  0.2
     (2)  0.4            (7)  0.4       (12)  0.4
     (3)  0.6  <- t_n    (8)  0.6       (13)  0.6
     (4)  0.8  <- t_n+1  (9)  0.8       (14)  0.8
          ----                ----            ----
          1.0                 1.0             1.0  (we do not sample the last points)

    """
    timestep0, timestep1, sim_number = get_indices(n, N)
    t0, t1 = ts[timestep0], ts[timestep1]

    theta = sims[sim_number]["theta"]

    xi0, xi1 = (
        sims[sim_number]["intvar"][timestep0, :],
        sims[sim_number]["intvar"][timestep1, :],
    )
    Fp0 = xi0[0:9]
    gamma0, gamma1 = xi0[9:33], xi1[9:33]
    slip0, slip1 = xi0[33:57], xi1[33:57]
    beta0 = xi0[57]

    defgrad = sims[sim_number]["F"]

    F0 = getF(np.eye(3), defgrad, t0)
    F1 = getF(np.eye(3), defgrad, t1)

    return F0, F1, theta, gamma0, gamma1, slip0, slip1, Fp0, beta0


def make_batch(ns: list, sims, ts, N: int):
    """
    this routine is called during the optimization loop and provides
    inputs (xs) and outputs (ys) for training as well as computation
    residuals r_I and r_II and possibly the penalties (dgamma should
    be positive and slip resistance should not be bigger than $s_{\inf}$)

    ns: list of simulation indices. The length of the list is equal to the
        batch size
    sims: all simulations results for trainingstored in a dict. key
          for the dictionary is the simulation id (int)
    ts: np.array containing the time axis
    N : number of simulation intervals in each simulation.
        len(ts) should be N + 1
    """
    F0s, F1s = [], []
    thetas = []
    gamma0s, gamma1s = [], []
    slip0s, slip1s = [], []
    Fp0s = []
    betas = []

    for n in ns:
        F0, F1, theta, gamma0, gamma1, slip0, slip1, Fp0, beta0 = get_data(
            sims, ts, n, N
        )
        F0s.append(F0.reshape(-1))
        F1s.append(F1.reshape(-1))
        thetas.append(theta)
        gamma0s.append(gamma0)
        gamma1s.append(gamma1)
        slip0s.append(slip0)
        slip1s.append(slip1)
        Fp0s.append(Fp0)
        betas.append(beta0)

    inputs = Input(
        F0=np.stack(F0s),
        F1=np.stack(F1s),
        theta=np.stack(thetas),
        gamma=np.stack(gamma0s),
        slip_res=np.stack(slip0s),
        Fp=np.stack(Fp0s),
        beta=np.stack(betas),
    )
    outputs = Output(gamma=np.stack(gamma1s), slip_res=np.stack(slip1s))
    return inputs, outputs


class SequenceDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx]


class BatchSampler(Sampler):
    def __init__(self, data_source, n_batch):
        self.data_source = data_source
        self.n_batch = n_batch

    def __iter__(self):
        total_samples = len(self.data_source)
        indices = torch.randperm(total_samples).tolist()

        # Yield batches of n_batch until all indices are used
        for i in range(0, total_samples, self.n_batch):
            yield indices[i : i + self.n_batch]

    def __len__(self):
        return (len(self.data_source) + self.n_batch - 1) // self.n_batch
