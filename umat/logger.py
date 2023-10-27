from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


@dataclass
class LogFlags:
    """Fields here correspond (1-to-1) to values in `log_flags` in `PARAMS_SCHEMA`"""

    loss: bool = False
    grad_norm1: bool = False
    grad_norm2: bool = False
    grad_norm_max: bool = False
    params_histogram: bool = False


class Logger:
    def __init__(self, log_flags: list[str], log_frequencies: list[int]):
        self.log_flags = LogFlags()
        for log_name in log_flags:
            setattr(self.log_flags, log_name, True)
        # set the counter to zero for the log keys that are requested
        self.log_frequencies = {
            log_flag: log_freq for log_flag, log_freq in zip(log_flags, log_frequencies)
        }
        self._counters = {k: 0 for k in log_flags}

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
            "penalty_min_slipresistance": detach(losses.pnt_min_slipresistance),
            "penalty_max_slipresistance": detach(losses.pnt_max_slipresistance),
        },
        idx,
    )


def clone_grads(loss, model, optimizer: optim.Optimizer):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    return [p.grad.clone() for p in model.parameters()]


def norm_of_ts(ts, p):
    return torch.concat([t.reshape(-1) for t in ts]).norm(p)


def norm_of_grad(loss, model, optimizer, p):
    return norm_of_ts(clone_grads(loss, model, optimizer), p)


def _log_gradient_norm(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
    groupname: str,
    pnorm: int,
):
    writer.add_scalars(
        groupname,
        {
            "data": norm_of_grad(losses.data, model, optimizer, p=pnorm),
            "physics": norm_of_grad(losses.physics, model, optimizer, p=pnorm),
            "delta_gamma": norm_of_grad(
                losses.pnt_delta_gamma, model, optimizer, p=pnorm
            ),
            "min_slipres": norm_of_grad(
                losses.pnt_min_slipresistance, model, optimizer, p=pnorm
            ),
            "max_slipres": norm_of_grad(
                losses.pnt_max_slipresistance, model, optimizer, p=pnorm
            ),
        },
        idx,
    )


def log_gradient_norm2(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
):
    _log_gradient_norm(writer, model, optimizer, idx, losses, "GradNorm2", pnorm=2)


def log_gradient_norm1(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
):
    _log_gradient_norm(writer, model, optimizer, idx, losses, "GradNorm1", pnorm=1)


def log_gradient_max(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
):
    _log_gradient_norm(
        writer, model, optimizer, idx, losses, "GradMax", pnorm=torch.inf
    )


def log_params_hist(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: torch.Tensor,
    idx: int,
):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        writer.add_histogram(name + "/grad", param.grad.data, idx)


def log_errors(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: optim.Optimizer,
    idx: int,
    losses: Losses,
    logger: Logger,
):
    if logger.should_log("loss"):
        log_losses(writer, idx, losses)
    if logger.should_log("grad_norm1"):
        log_gradient_norm1(writer, model, optimizer, idx, losses)
    if logger.should_log("grad_norm2"):
        log_gradient_norm2(writer, model, optimizer, idx, losses)
    if logger.should_log("grad_norm_max"):
        log_gradient_max(writer, model, optimizer, idx, losses)
    if logger.should_log("params_histogram"):
        log_params_hist(writer, model, optimizer, losses.data, idx)
