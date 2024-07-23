import pdb
import sys
import traceback
import torch
import einops
import multiprocessing as mp
from itertools import repeat
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from torch.func import vmap
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# Module imports
from umat.lr_scheduler import CustomCosineAnnealingWarmRestarts
from .replay_buffer import IntVarIndices, SampledValues, PriotorizedReplayBuffer
from .get_results import get_results_with_state, UMATResult
from .dataset import (
    read_hdf5,
    UMATDataSet,
    split_dataset,
    create_data_loaders,
    circular_loader,
)
from .generate_defgrad import fig_to_buffer
from .model import Model, LossFunction
from .replay_buffer import IntVarIndices, State
from .config import Config
from .constants import consts
from .model import Xs, Ys
from .umat import (
    clip_slip_resistance,
    enforce_positive_gamma_increment,
    plastic_def_grad,
    rotate_elastic_stiffness,
    rotate_slip_system,
    rotation_matrix,
    get_cauchy_stress,
)
from .trip_ferrite_data import ElasStif, SlipSys

# Typing
from jaxtyping import Float
from torch import Tensor
from numpy import ndarray
from torch.optim import Optimizer
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


def print_function_name_once(func):
    has_been_called = False

    def wrapper(*args, **kwargs):
        nonlocal has_been_called
        if not has_been_called:
            print(f"Calling function: {func.__name__}")
            has_been_called = True
        return func(*args, **kwargs)

    return wrapper


@dataclass
class ModelResult:
    """
    This is similar to `Ys` dataclass but with addition of deformation gradient for
    each state since we need to compute the plastic deformation gradient later in `validate` function.
    It is also specialized for batch 1 with `__post_init__` check.
    """

    gamma: Float[Tensor, "1 24"] = None
    slip_resistance: Float[Tensor, "1 24"] = None
    defgrad: Float[Tensor, "1 3 3"] = None

    def __post_init__(self):
        assert self.gamma.shape == (1, 24)
        assert self.slip_resistance.shape == (1, 24)
        assert self.defgrad.shape == (1, 3, 3)


def get_model_dtype(model: nn.Module) -> torch.dtype:
    for param in model.parameters():
        return param.dtype
    return None


def simulate_model(model: Model, Fs: Float[ndarray, "n_times + 1 9"], theta: Float[ndarray, "3"]) -> list[ModelResult]:
    """
    Given a trained model, autoregressively feed the model and create a trajectory of predicted quantities.
    """

    model_dtype = get_model_dtype(model)

    # initialize the interval variables
    gamma = torch.zeros(24).to(model_dtype).reshape(1, -1)
    slip_resistance = (consts.s0_F * torch.ones(24)).to(model_dtype).reshape(1, -1)

    # since it's fixed we define it here once
    theta = torch.from_numpy(theta).to(model_dtype).reshape(1, -1)  # batch 1

    # Initialize the results. First entries are the vectors of variables at t = 0,
    # and deformation gradient is identity for t = 0
    model_result_list = [
        ModelResult(
            gamma=gamma.clone(),
            slip_resistance=slip_resistance.clone(),
            defgrad=torch.ones(1, 3, 3).to(model_dtype),
        )
    ]

    model.eval()
    with torch.no_grad():
        for F0, F1 in pairwise(Fs):
            defgrad0 = torch.from_numpy(F0).to(model_dtype).reshape(1, -1)  # batch 1
            defgrad1 = torch.from_numpy(F1).to(model_dtype).reshape(1, -1)  # batch 1
            xs = Xs(
                defgrad0=defgrad0,
                defgrad1=defgrad1,
                theta=theta,
                gamma=gamma,
                slip_resistance=slip_resistance,
            )
            ys_hat = model.forward(xs)

            constrained_gamma = enforce_positive_gamma_increment(ys_hat.gamma, gamma)
            constrained_slip_resistance = clip_slip_resistance(ys_hat.slip_resistance, xs.slip_resistance)
            model_result_list.append(
                ModelResult(
                    gamma=constrained_gamma.clone(),
                    slip_resistance=constrained_slip_resistance.clone(),
                    defgrad=defgrad1.reshape(1, 3, 3),
                )
            )

            gamma = constrained_gamma
            slip_resistance = constrained_slip_resistance

    model.train()

    return model_result_list


def compare_results(stress_umat: Tensor, stress_model: Tensor) -> Tensor:
    relative_error = (stress_umat - stress_model).norm() / stress_umat.norm()
    print(f"Relative difference between umat and model: {relative_error:.4f}")
    return relative_error


def to_Voigt(stress: Float[Tensor, "3 3"]) -> Float[Tensor, "6"]:
    """
    11, 22, 33, 12, 13, 23
    """
    return stress[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]


def validate(circular_val_loader, model: nn.Module, cfg: Config, writer: SummaryWriter, n_step: int):
    model_dtype = get_model_dtype(model)

    data = next(circular_val_loader)
    results_umat: list[UMATResult] = simulate_umat(data, cfg.n_time)

    relative_errors = []
    stress_histories: Float[ndarray, "N 6"] = []

    for result_umat in results_umat:
        result_model = simulate_model(model, result_umat.F, result_umat.theta)
        # Since the model results are batched, we vmap the rest of the functions. we need to
        # make sure that some initializations are then batched. This include plastic_defgrad and
        # theta that we receive from umat model

        # now we can iterate and compute the plastic deformation gradient and stress
        # initialize plastic deformation gradient
        batch_size = 1
        plastic_defgrad = torch.eye(3).to(model_dtype).reshape(batch_size, 3, 3)

        # rotated stiffness and slip systems
        # rm = rotation_matrix(torch.from_numpy(result_umat.theta).to(model_dtype).reshape(1, -1))
        theta = einops.rearrange(torch.from_numpy(result_umat.theta), f"... -> {batch_size} ...").to(model_dtype)
        rm = vmap(rotation_matrix)(theta)
        rotated_slip_system = vmap(rotate_slip_system, in_dims=(None, 0))(SlipSys, rm)
        rotated_elastic_stiffness = vmap(rotate_elastic_stiffness, in_dims=(None, 0))(ElasStif, rm)

        # initial the stress history with zero stress
        stress_hist: list[Float[Tensor, "6"]] = [to_Voigt(torch.zeros(3, 3).to(model_dtype))]

        for model_result0, model_result1 in pairwise(result_model):
            plastic_defgrad = vmap(plastic_def_grad)(
                model_result1.gamma - model_result0.gamma, rotated_slip_system, plastic_defgrad.clone()
            )
            cauchy_stress = vmap(get_cauchy_stress)(model_result1.defgrad, plastic_defgrad, rotated_elastic_stiffness)
            stress_hist.append(to_Voigt(cauchy_stress.squeeze().clone()))

        relative_error = compare_results(
            stress_umat=torch.tensor(result_umat.stress),
            stress_model=torch.stack(stress_hist),
        )
        relative_errors.append(relative_error)
        stress_histories.append(torch.stack(stress_hist).numpy())

    mean_relative_error = torch.tensor(relative_errors).mean()
    writer.add_scalar("validation/ave_relative_error", mean_relative_error, n_step)

    plot_stress(stress_histories, results_umat, relative_errors, writer, n_step)


def get_plot(stress_umat: Float[ndarray, "N 6"], stress_model: Float[ndarray, "N 6"]):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    ts = np.linspace(0, 1, stress_umat.shape[0])
    stress_component_labels = "11 22 33 12 13 23".split()

    for idx, ax in enumerate(axes.ravel()):
        v1 = min(stress_umat.min(), stress_model.min())
        v2 = max(stress_umat.max(), stress_model.max())
        v1 = round(v1 * 10 + 0.5) / 10 + 0.1 if v1 > 0 else round(v1 * 10 - 0.5) / 10 - 0.1
        v2 = round(v2 * 10 + 0.5) / 10 + 0.1 if v2 > 0 else round(v2 * 10 - 0.5) / 10 - 0.1

        ax.set_ylim(v1, v2)
        ax.set_xlim(-0.02, 1)
        ax.set_title(stress_component_labels[idx])
        ax.plot(ts, stress_umat[:, idx])
        ax.plot(ts, stress_model[:, idx])
    plt.tight_layout()

    return fig_to_buffer(fig, dpi=120)


def plot_stress(
    stress_histories: list[Float[ndarray, "N 6"]],
    results_umat: list[UMATResult],
    relative_erros: list[Tensor],
    writer: SummaryWriter,
    n_step: int,
) -> None:

    assert len(stress_histories) == len(results_umat) == len(relative_erros)
    # get the worst case
    max_err_idx = torch.tensor(relative_erros).argmax().item()

    stress_model = stress_histories[max_err_idx]
    stress_umat = results_umat[max_err_idx].stress
    image_buffer = get_plot(stress_umat, stress_model)
    image_tensor = torch.from_numpy(np.array(Image.open(image_buffer))).permute(2, 0, 1).squeeze(0) / 255.0
    writer.add_image(f"Validation/Stress/Step{n_step}", image_tensor, global_step=n_step)


def train(cfg: Config, writer: SummaryWriter) -> None:
    dataset_hdf5 = read_hdf5(cfg.dataset_path)
    dset = UMATDataSet(dataset_hdf5)
    train, val, test = split_dataset(dset, cfg)
    train_loader, val_loader, test_loader = create_data_loaders(train, val, test, cfg)
    circular_train_loader = circular_loader(train_loader)
    circular_val_loader = circular_loader(val_loader)
    # TODO: `n_total_steps` is too small with current setup.
    #        Think how this can be enhanced. Understand how annealing works.
    buffer = PriotorizedReplayBuffer(cfg.buffer_size, cfg.batch_size, 101, n_total_steps=cfg.buffer_n_steps)
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
        # Fill the buffer to `min_buffer_size` first before continuiing with training
        while len(buffer) < cfg.min_buffer_size:
            print(len(buffer))
            update_buffer(circular_train_loader, buffer, cfg)

        for _ in range(1, cfg.N_iteration + 1):
            n_step += 1
            sampled_values: SampledValues = buffer.sample(n_step)
            xs, ys = model.make_batch(sampled_values, cfg)
            ys_hat = model.forward(xs)
            loss, td_error, loss_items_dict = loss_fn.forward(ys, ys_hat, xs, sampled_values.weights, cfg)
            log_losses(loss, loss_items_dict, td_error, optimizer, scheduler, writer, n_step)
            if n_step % 100 == 0:
                print(loss.data)
            train_model(loss, optimizer, scheduler)
            buffer.update_priorities(sampled_values.indices, td_error)

        validate(circular_val_loader, model, cfg, writer, n_step)
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
