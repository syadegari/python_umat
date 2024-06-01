from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import einops
from torch.func import vmap

from .model import Model
from .constants import consts
from .umat import (
    get_ks,
    get_H_matrix,
    get_ws,
    get_beta,
    get_PK2,
    rotation_matrix,
    rotate_slip_system,
    rotate_elastic_stiffness,
    plastic_def_grad,
)
from .trip_ferrite_data import SlipSys, ElasStif
from umat_ggcm.pyUMATLink import pysdvini, fortumat

from typing import Tuple, List
from jaxtyping import Float
from torch import Tensor
from torch._tensor import Tensor
from umat.model import Model


@dataclass
class HistoryResult:
    stress: List[Float[Tensor, "24"]] = field(default_factory=list)
    gamma: list[Float[Tensor, "24"]] = field(default_factory=list)
    slipres: list[Float[Tensor, "24"]] = field(default_factory=list)
    beta: list[Tensor] = field(default_factory=list)
    plastic_defgrad: list[Float[Tensor, "3 3"]] = field(default_factory=list)

    def store_values_returned_from_umat(self, xi: np.ndarray, sigma: np.ndarray) -> None:
        self.stress.append(torch.tensor(sigma.copy()))
        # Fp     0 ... 9
        self.plastic_defgrad.append(torch.tensor(xi.copy()[0:9]).reshape(3, 3))
        # gamma  9 ... 33
        self.gamma.append(torch.tensor(xi.copy()[9:33]))
        # slip resistance 33 ... 57
        self.slipres.append(torch.tensor(xi.copy()[33:57]))
        # beta   57
        self.beta.append(torch.tensor(xi.copy()[57]))

    def store_values_returned_from_model(
        self,
        Fp: Tensor,
        gamma: Tensor,
        slipres: Tensor,
        beta: Tensor,
        stress: Tensor,
    ) -> None:
        self.plastic_defgrad.append(Fp.clone().squeeze())
        self.gamma.append(gamma.clone().squeeze())
        self.slipres.append(slipres.clone().squeeze())
        self.beta.append(beta.clone().squeeze())
        # Voigt notation:
        #
        # 1: (1, 1) -> 0: (0, 0)
        # 2: (2, 2) -> 1: (1, 1)
        # 3: (3, 3) -> 2: (2, 2)
        # 4: (1, 2) -> 3: (0, 1)
        # 5: (1, 3) -> 4: (0, 2)
        # 6: (2, 3) -> 5: (1, 2)
        voigt_stress = torch.tensor(
            [
                stress[0, 0, 0],
                stress[0, 1, 1],
                stress[0, 2, 2],
                stress[0, 0, 1],
                stress[0, 0, 2],
                stress[0, 1, 2],
            ]
        )
        self.stress.append(voigt_stress)

    def vectorize(self) -> dict:
        return {
            "stress": torch.stack(self.stress),
            "gamma": torch.stack(self.gamma),
            "slipres": torch.stack(self.slipres),
            "beta": torch.stack(self.beta),
            "plastic_defgrad": torch.stack(self.plastic_defgrad),
        }


def getF(F_init: int, F_final, t: int):
    assert 0 <= t <= 1
    return (1 - t) * F_init + F_final * t


def get_ts(n_time):
    return np.linspace(0, 1, n_time + 1).round(12)


def pairwise(xs):
    return zip(xs[:-1], xs[1:])


def load_model(path_to_model) -> Model:
    model = Model(nn.Tanh()).to(torch.float64)
    model.load_state_dict(torch.load(path_to_model))
    return model


def inference_get_beta(dgamma: Tensor, beta0, slip_res0, slip_res1):
    """
    NOTE: This should be used onlly for inference purposes!
    """
    # TODO: should we clip slip_res from above and below to be in range?
    ks = get_ks(slip_res1 - slip_res0, slip_res0)
    H_matrix = get_H_matrix(ks)
    ws = get_ws(H_matrix)
    beta = get_beta(dgamma, ws, beta0)
    return beta


def init_internal_variables() -> Tuple[Tensor, ...]:
    # Since we are using this only in the model branch, we should batch
    #  these values before sending them to the main routine
    gamma_init = torch.zeros(1, 24, dtype=torch.float64)
    Fp_init = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)
    beta_init = torch.tensor([0.0], dtype=torch.float64)
    slipres_init = consts.s0_F * torch.ones(1, 24, dtype=torch.float64)

    return Fp_init, gamma_init, slipres_init, beta_init


def get_cauchy_stress(F, Fp, elastic_stiffness):
    """
    Argument list

    F [3x3]: Deformation gradient
    Fp [3x3]: Plastic deformation gradient
    elastic_stiffness [3x3x3x3]: Elastic stiffness tensor

    Returns:
    sigma [3x3]: Cauchy stress tensor at n+1 time step

    Primary usage of this function is at inference to calculate
    the Cauchy stress from the plastic slip.
    """

    Fe = F @ torch.linalg.inv(Fp)
    S = get_PK2(Fe.T @ Fe, elastic_stiffness)
    sigma = 1 / (torch.linalg.det(Fe)) * Fe @ S @ Fe.T

    return sigma


def autoregress(F_final, theta, alpha, path_to_model, n_times):
    """
    F_final: Final deformation gradient.
    theta: Euler angles.
    alpha: value between zero and one (exclusive zero). Determines where we should switch from UMAT to model for autoregression.

    - Start
    - initiate the model: model=load_model(path_to_model)
    - initiate internal variables `gamma`, `slip_resistance`, `stress`, `plastic deformation gradient` and `beta` for time t=0
    - set `defgard F0 = identity`
    - set `def plastic grad Fp0`
    - initiate the vector of psedue time `ts`. this ranges from zero to one. For example ts = [0.0, 0.005, 0.01, ... 1.0]
    - loop
        get current time `t1`.
        F0, F1 = get_defgrads(t0, t1)
        if current time `t` < alpha: # We use UMAT for calculation
            'use UMAT for prediction'
            gamma1, slip_res1, sigmav1 = UMAT(t1, F0, F1)
        else (we have gone beyond alpha threshold and use the train model to get the updated values)
            gamma1, slipres1 = model(theta, F0, F1, gamma0, slipres0)
            Fp1 = Fp(theta, F1, Fp0)
            Fe1 = F1 @ (Fp1) ^ -1
            S1 = getPK2(Fe1, C)
            sigma1 = 1/det(Fe1) Fe1.T @ S @ Fe1
        end
        # Here we store the values of interest like gamma and slip, beta
        ...
    end loop

    Parameters:
    -----------
    F_final : array [3x3]
        Deformation gradient used for running the autoregression

    theta: array [3]
        Euler angles/orientation in radian

    alpha : float in range [0, 1]
        Controls which portion of the autoregression to run with the model
        - alpha = 0; All calculation done by the trained model
        - 0 < alpha < 1; First use UMAT and when `ts` exceeds alpha then switch
          to the trained model. In this mode we should handle the (once) transfer
          of information for time t_n from UMAT computations to the other branch
          to compute with the model
        - alpha = 1; All calculations are done using UMAT.

    path_to_model : str
        Path to the trained model

    n_times: int
        number of time divisions to run the inference with. Must be consistent with
        the trained model delta t
    """

    assert 0.0 <= alpha <= 1.0, "parameter alpha should be between 0.0 and 1.0 (inclusive)"

    if alpha == 0.0:
        # all computation are done using the model and we don't need to bring values
        #  from UMAT branch to model branch
        transfer_once_from_umat_branch = False
    else:
        # this covers 0 < alpha < 1 as well as alpha=1.
        # with alpha=1, we only run the first branch
        transfer_once_from_umat_branch = True

    hist_result = HistoryResult()

    rm = rotation_matrix(torch.tensor(theta).reshape(1, -1))
    rotated_slip_system = rotate_slip_system(SlipSys, rm)
    rotated_elastic_stiffness = rotate_elastic_stiffness(ElasStif, rm)

    model = load_model(path_to_model)
    model.eval()

    F_init = np.eye(3)
    for t0, t1 in pairwise(get_ts(n_times)):
        F0, F1 = getF(F_init, F_final, t0), getF(F_init, F_final, t1)
        if t1 <= alpha:
            if t0 == 0.0:
                # initialize parameters that are used only by UMAT
                c = np.zeros([6, 6], order="F")
                sigma = np.zeros(6)
                xi = np.zeros(93)
                # Use the fortran subroutine to
                pysdvini(xi)
                # store the initial values
                hist_result.store_values_returned_from_umat(xi.copy(), sigma=np.zeros(6))
            # Call the UMAT to update sigma, c and xi
            fortumat(
                f1=F0,
                f2=F1,
                t1=t0,
                t2=t1,
                initemp=consts.Theta0,
                sigma=sigma,
                c=c,
                xi=xi,
                angles=theta,
            )
            # store values after each call to umat
            hist_result.store_values_returned_from_umat(xi.copy(), sigma.copy())
        else:
            # First time we run this branch, we have to bring some of the variables for time n from
            # the previous branch. We need a flag to determine this.
            if transfer_once_from_umat_branch:
                make_batch_1 = lambda x: einops.rearrange(x, "... -> 1 ...")
                Fp0 = make_batch_1(hist_result.plastic_defgrad[-1].clone())
                gamma0 = make_batch_1(hist_result.gamma[-1].clone())
                slip_res0 = make_batch_1(hist_result.slipres[-1].clone())
                beta0 = make_batch_1(hist_result.beta[-1].clone())

                transfer_once_from_umat_branch = False

            if t0 == 0:
                # init variables and store them
                Fp0, gamma0, slip_res0, beta0 = init_internal_variables()
                hist_result.store_values_returned_from_model(
                    Fp0,
                    gamma0,
                    slip_res0,
                    beta0,
                    torch.zeros(1, 3, 3, dtype=torch.float64),
                )

            with torch.no_grad():
                gamma1, slip_res1 = model.forward(
                    theta=torch.tensor(theta).reshape(1, -1),  # [1, 3]
                    defgrad0=torch.tensor(F0).reshape(1, -1),  # [1, 9]
                    defgrad1=torch.tensor(F1).reshape(1, -1),  # [1, 9]
                    gamma0=gamma0.reshape(1, -1),  # [1, 24]
                    slip_res0=slip_res0.reshape(1, -1),  # [1, 24]
                )

            # clip the values of slip resistance to be between two bounds
            slip_res1 = torch.clip(slip_res1, consts.s0_F, consts.sInf_F)

            # only accept nonzero values of delta gamma
            gamma1 = torch.where(gamma1 >= gamma0, gamma1, gamma0)

            Fp1 = vmap(plastic_def_grad)(gamma1 - gamma0, rotated_slip_system, Fp0)
            cauchy_stress = vmap(get_cauchy_stress)(
                torch.tensor(F1).reshape(1, 3, 3),
                Fp1,
                rotated_elastic_stiffness,
            )
            beta1 = vmap(inference_get_beta)(gamma1 - gamma0, beta0, slip_res0, slip_res1)

            hist_result.store_values_returned_from_model(Fp1, gamma1, slip_res1, beta1, cauchy_stress)

            # set the values for the next timestep
            gamma0 = gamma1
            slip_res0 = slip_res1
            beta0 = beta1
            Fp0 = Fp1
    #
    return hist_result.vectorize()
