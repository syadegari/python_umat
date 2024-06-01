import torch
from collections import namedtuple
import numpy as np
from dataclasses import dataclass

# Typing
from jaxtyping import Float, Int
from torch import Tensor
from numpy import ndarray


@dataclass(frozen=True)
class IntVarIndices:
    plastic_defgrad_index = np.s_[0:9]
    gamma_index = np.s_[9:33]
    slip_res_index = np.s_[33:57]
    beta_index = np.s_[57]

    def extract_variables(self, intvar_vector: Float[ndarray, "93"]) -> "IntVarState":
        return IntVarState(
            gamma=intvar_vector[self.gamma_index].copy(),
            slip_resistance=intvar_vector[self.slip_res_index].copy(),
            plastic_defgrad=intvar_vector[self.plastic_defgrad_index].copy(),
            beta=intvar_vector[self.beta_index],  # single index doesn't need copying
        )


@dataclass
class IntVarState:
    gamma: Float[ndarray, "24"] = None
    slip_resistance: Float[ndarray, "24"] = None
    plastic_defgrad: Float[ndarray, "9"] = None
    beta: float = None


@dataclass
class State:
    total_defgrad: Float[ndarray, "9"]
    angle: Float[ndarray, "3"]
    defgrad: Float[ndarray, "9"] = None
    t: float = None
    stress: Float[ndarray, "6"] = None
    intvar: IntVarState = None


@dataclass
class Experience:
    s0: State = None
    s1: State = None


@dataclass
class SampledValues:
    samples: list[Experience] = None
    weights: Float[Tensor, "batch"] = None
    indices: Float[ndarray, "batch"] = None


class PriotorizedReplayBuffer:
    def __init__(
        self,
        buffer_size,
        batch_size,
        seed,
        n_total_steps,
        alpha: float = 0.6,
        beta_0: float = 0.4,
        eps: float = 1e-5,
    ) -> None:
        self.alpha = alpha
        self.beta_0 = beta_0
        self.eps = eps
        self.n_total_steps = n_total_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(seed)

        self.buffer: list[Experience] = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, s0: State, s1: State) -> None:
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self) < self.buffer_size:
            self.buffer.append(Experience(s0=s0, s1=s1))
        else:
            self.buffer[self.pos] = Experience(s0=s0, s1=s1)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer_size

    def get_beta(self, i_step: int) -> float:
        """
        Anneals beta linearly from its initial value, beta_0 to 1.
        Check section 3.4 of the paper on Prio Exp Replay
        """
        return (1 - self.beta_0) * i_step / self.n_total_steps + self.beta_0

    def sample(self, i_step: int) -> SampledValues:

        N = len(self)
        beta = self.get_beta(i_step)

        if N == self.buffer_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]

        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        sampled_indices = np.random.choice(
            N,
            self.batch_size,
            replace=False,
            p=probabilities,
        )
        samples = [self.buffer[idx] for idx in sampled_indices]

        weights = (N * probabilities[sampled_indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).to(torch.float32)

        return SampledValues(samples=samples, weights=weights, indices=sampled_indices)

    def update_priorities(
        self, batch_indices: Int[ndarray, "batch"], batch_priorities: Float[ndarray, "batch"]
    ) -> None:
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + self.eps
