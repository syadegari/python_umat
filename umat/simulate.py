import pandas as pd
import torch
import multiprocessing as mp
from itertools import repeat
from dataclasses import dataclass
import numpy as np
import torch.nn as nn

# Module imports
from umat.lr_scheduler import CustomCosineAnnealingWarmRestarts
from .get_results import get_results_with_state, UMATResult
from .replay_buffer import PriotorizedReplayBuffer, SampledValues
from .dataset import read_hdf5, UMATDataSet, split_dataset, create_data_loaders, circular_loader
from .model import Model, LossFunction
from .replay_buffer import IntVarIndices, State
from .config import Config
from torch.optim import Optimizer

# Typing
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler


def simulate_umat(data: dict[str, Float[Tensor, "batch_dim ..."]], n_times: int = 400) -> list[UMATResult]:
    angles = data["angle"].numpy()
    defgrads = data["defgrad"].numpy()

    # Check the batch dimension
    assert angles.shape[0] == defgrads.shape[0]
    batch_dim = angles.shape[0]
    ncpus = mp.cpu_count()

    with mp.Pool(ncpus) as pool:
        results: list[UMATResult] = pool.starmap(
            get_results_with_state, zip(defgrads, angles, repeat(n_times, batch_dim))
        )

    return results


pairwise = lambda xs: zip(xs[:-1], xs[1:])


def parse_umat_output(umat_result: UMATResult) -> list[State]:
    Ns = umat_result.ts.shape[0]
    intvar_indices = IntVarIndices()

    states = []
    for n in range(Ns):
        state = make_state(umat_result, intvar_indices, n)
        states.append(state)

    return states


def make_state(result: UMATResult, intvar_indices: IntVarIndices, step: int) -> State:
    return State(
        total_defgrad=result.F_final.reshape(1, -1).copy(),
        angle=result.theta.copy(),
        defgrad=result.F[step].copy(),
        t=result.ts[step],
        stress=result.stress[step].copy(),
        intvar=intvar_indices.extract_variables(result.intvars[step].copy()),
    )


def add_result_to_buffer(result: list[UMATResult], buffer: PriotorizedReplayBuffer) -> None:

    intvar_indices = IntVarIndices()

    for res in result:
        Ns = res.ts.shape[0]
        for n0, n1 in pairwise(np.arange(Ns)):
            s0 = make_state(res, intvar_indices, n0)
            s1 = make_state(res, intvar_indices, n1)
            buffer.add(s0, s1)


def update_buffer(circular_train_loader, buffer: PriotorizedReplayBuffer, cfg: Config, print_msg=False) -> None:
    if print_msg:
        print("Adding data to buffer.")
        print(
            f"Current buffer size: {len(buffer)}, minimum allowable buffer size: {cfg.min_buffer_size}, maximum buffer size: {cfg.buffer_size}"
        )
        print(f"Estimated number of additions to the buffer: {cfg.n_time * cfg.dataset_batch_train} experiences")
    data = next(circular_train_loader)
    result = simulate_umat(data, cfg.n_time)
    add_result_to_buffer(result, buffer)


def train_model(loss: LossFunction, optimizer: Optimizer, scheduler: _LRScheduler = None) -> None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()




def train(cfg: Config) -> None:
    dataset_hdf5 = read_hdf5(cfg.dataset_path)
    dset = UMATDataSet(dataset_hdf5)
    train, val, test = split_dataset(dset, cfg)
    train_loader, val_loader, test_loader = create_data_loaders(train, val, test, cfg)
    circular_train_loader = circular_loader(train_loader)
    circular_val_loader = circular_loader(val_loader)
    # TODO: `n_total_steps` is too small with current setup.
    #        Think how this can be enhanced. Understand how annealing works.
    buffer = PriotorizedReplayBuffer(cfg.buffer_size, cfg.batch_size, 101, n_total_steps=cfg.N_iteration)
    model = Model(nn.Tanh()).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    #
    if cfg.use_lr_scheduler:
        scheduler = CustomCosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.lr_scheduler_T_0,
            T_mult=cfg.lr_scheduler_T_multi,
            eta_min=cfg.lr_scheduler_eta_min,
            initial_lr=cfg.lr_scheduler_initial_lr,
            final_lr=cfg.lt_scheduler_final_lr,
            restarts_until_final=10,
        )
    else:
        scheduler = None
    #
    loss_fn = LossFunction()

    n_step = 0
    while True:
        while len(buffer) < cfg.min_buffer_size:
            print(len(buffer))
            update_buffer(circular_train_loader, buffer, cfg)

        for idx_iteration in range(1, cfg.N_iteration + 1):
            n_step += 1
            sampled_values = buffer.sample(idx_iteration)
            xs, ys = model.make_batch(sampled_values, cfg)
            ys_hat = model.forward(xs)
            loss, td_error, loss_items_dict = loss_fn.forward(ys, ys_hat, xs, sampled_values.weights, cfg)
            log_losses(loss, loss_items_dict, td_error, optimizer, scheduler, writer, n_step)
            if n_step % 100 == 0:
                print(loss.data)
            train_model(loss, optimizer, scheduler)
            buffer.update_priorities(sampled_values.indices, td_error)

        print("Add data to buffer")
        update_buffer(circular_train_loader, buffer, cfg, print_msg=True)


def log_losses(
    loss: Tensor,
    loss_items: dict,
    td_error: Float[ndarray, "batch"],
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    writer: SummaryWriter,
    n_step: int,
) -> None:
    writer.add_scalars("TrainingLossComponents", loss_items, global_step=n_step)
    writer.add_scalar("TrainingLoss", loss.cpu().detach().item(), global_step=n_step)

    td_statistics_1 = {
        "max": np.max(td_error),
        "min": np.min(td_error),
        "std": np.std(td_error),
        "mean": np.mean(td_error),
    }
    writer.add_scalars("TDError", td_statistics_1, n_step)

    quantiles = np.quantile(td_error, [0.25, 0.50, 0.75], axis=0)
    td_statistics_2 = {"25% quantile": quantiles[0], "50% quantile": quantiles[1], "75% quantile": quantiles[2]}
    writer.add_scalars("TDErrorQuantiles", td_statistics_2, n_step)

    if scheduler is not None:
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], global_step=n_step)


def main(cfg: Config) -> None:
    writer = SummaryWriter(cfg.log_directory, flush_secs=20)
    try:
        train(cfg, writer)
    except RuntimeError as e:
        print(f"Caught an error: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        pdb.post_mortem(exc_traceback)
    finally:
        writer.close()
