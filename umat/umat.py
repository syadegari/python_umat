from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor
from torch.func import vmap
import einops

from .constants import consts
from .trip_ferrite_data import SlipSys, ElasStif

from typing import Union, Tuple, TypeVar

# B = TypeVar("B")
# FloatTensor = Union[torch.float32, torch.float64]

# Defining Batch size as a Type Variable
B = TypeVar("B")


# This is a way to represent a tensor with certain shape constraints
class FloatTensor(torch.Tensor):
    @classmethod
    def __class_getitem__(cls, item):
        return cls


# TODO: Decide what to do with these annotations
# BTAngles = FloatTensor[Tuple[B, 3]]
# BTElasticity = FloatTensor[Tuple[B, 3, 3, 3, 3]]
# BTSlipSystem = FloatTensor[Tuple[B, 24, 3, 3]]
# BTVarScalar = FloatTensor[Tuple[B]]
# BTVarVector = FloatTensor[Tuple[B, 24]]
# BTVarMatrix = FloatTensor[Tuple[B, 24, 24]]
# BTVarStrain = FloatTensor[Tuple[B, 3, 3]]
# F32orF64 = Union[torch.float32, torch.float64]


def sdvini():
    # initial plastic deformation is identity
    F_p_init = torch.eye(3)
    # initial plastic slip
    gamma_init = torch.zeros(24)
    #
    #   accumulative plastic strain, slip resistance parameter, dislocation
    #   density parameter
    #   initial slip resistance = 0.158 for original matrix
    #                             0.072 for soft matrix (low yield stress)
    #                             0.216 for hard matrix (high yield stress)
    s_init = consts.s0_F * torch.ones(24)
    # initial dislocation density
    beta_0 = 0.0

    return F_p_init, gamma_init, s_init, beta_0


def non_schmid_stress_bcc(Schmid):
    """
    computes non-Schmid stress according to Bassani's convention
    """
    NonSchmid = torch.zeros_like(Schmid)
    # compute the non-glide stress (non-Schmid stress)
    NonSchmid[0] = Schmid[5]  #      NonSchmid(1)  = NGlide*Schmid(6)
    NonSchmid[1] = Schmid[2]  #      NonSchmid(2)  = NGlide*Schmid(3)
    NonSchmid[2] = Schmid[1]  #      NonSchmid(3)  = NGlide*Schmid(2)
    NonSchmid[3] = Schmid[4]  #      NonSchmid(4)  = NGlide*Schmid(5)
    NonSchmid[4] = Schmid[3]  #      NonSchmid(5)  = NGlide*Schmid(4)
    NonSchmid[5] = Schmid[0]  #      NonSchmid(6)  = NGlide*Schmid(1)
    #
    NonSchmid[6] = Schmid[9]  #      NonSchmid(7)  = NGlide*Schmid(10)
    NonSchmid[7] = Schmid[10]  #      NonSchmid(8)  = NGlide*Schmid(11)
    NonSchmid[8] = Schmid[11]  #      NonSchmid(9)  = NGlide*Schmid(12)
    NonSchmid[9] = Schmid[6]  #      NonSchmid(10) = NGlide*Schmid(7)
    NonSchmid[10] = Schmid[7]  #      NonSchmid(11) = NGlide*Schmid(8)
    NonSchmid[11] = Schmid[8]  #      NonSchmid(12) = NGlide*Schmid(9)
    #
    NonSchmid[12] = Schmid[15]  #      NonSchmid(13) = NGlide*Schmid(16)
    NonSchmid[13] = Schmid[16]  #      NonSchmid(14) = NGlide*Schmid(17)
    NonSchmid[14] = Schmid[17]  #      NonSchmid(15) = NGlide*Schmid(18)
    NonSchmid[15] = Schmid[12]  #      NonSchmid(16) = NGlide*Schmid(13)
    NonSchmid[16] = Schmid[13]  #      NonSchmid(17) = NGlide*Schmid(14)
    NonSchmid[17] = Schmid[14]  #      NonSchmid(18) = NGlide*Schmid(15)
    #
    NonSchmid[18] = Schmid[23]  #      NonSchmid(19) = NGlide*Schmid(24)
    NonSchmid[19] = Schmid[20]  #      NonSchmid(20) = NGlide*Schmid(21)
    NonSchmid[20] = Schmid[19]  #      NonSchmid(21) = NGlide*Schmid(20)
    NonSchmid[21] = Schmid[22]  #      NonSchmid(22) = NGlide*Schmid(23)
    NonSchmid[22] = Schmid[21]  #      NonSchmid(23) = NGlide*Schmid(22)
    NonSchmid[23] = Schmid[18]  #      NonSchmid(24) = NGlide*Schmid(19)
    #
    return consts.NGlide * NonSchmid


def rotation_matrix(
    angles: Union[FloatTensor[3], FloatTensor[B, 3]]
) -> Union[FloatTensor[3, 3], FloatTensor[B, 3, 3]]:
    """
    Defines the rotation matrix using 323 euler rotations.
    This is the replica of the Fortran-version. It has been
    modified to be batch-compatible.
    """

    was_singleton = False
    if angles.ndim == 1:
        angles = angles.unsqueeze(0)
        was_singleton = True

    batch_size = angles.shape[0]

    angle1, angle2, angle3 = angles[:, 0], angles[:, 1], angles[:, 2]

    R1 = torch.zeros([batch_size, 3, 3]).to(angles.dtype)
    R2 = torch.zeros([batch_size, 3, 3]).to(angles.dtype)
    R3 = torch.zeros([batch_size, 3, 3]).to(angles.dtype)

    #  rotation matrix of the second rotation (angle1)
    R1[:, 0, 0] = torch.cos(angle1)
    R1[:, 0, 1] = torch.sin(angle1)
    R1[:, 1, 0] = -torch.sin(angle1)
    R1[:, 1, 1] = torch.cos(angle1)
    R1[:, 2, 2] = 1.0

    #  rotation matrix of the second rotation (angle2)
    R2[:, 0, 0] = torch.cos(angle2)
    R2[:, 0, 2] = -torch.sin(angle2)
    R2[:, 1, 1] = 1.0
    R2[:, 2, 0] = torch.sin(angle2)
    R2[:, 2, 2] = torch.cos(angle2)

    #  rotation matrix of the third rotation (angle3)
    R3[:, 0, 0] = torch.cos(angle3)
    R3[:, 0, 1] = torch.sin(angle3)
    R3[:, 1, 0] = -torch.sin(angle3)
    R3[:, 1, 1] = torch.cos(angle3)
    R3[:, 2, 2] = 1.0

    #  calculate the overall rotation matrix
    RM = R3 @ R2 @ R1

    if was_singleton:
        RM = RM.squeeze(0)

    return RM


def rotate_slip_system(
    slip_sys: FloatTensor[3, 3, 3],
    rotation_matrix: Union[FloatTensor[3, 3], FloatTensor[B, 3, 3]],
) -> Union[FloatTensor[3, 3, 3], FloatTensor[B, 3, 3, 3]]:
    if rotation_matrix.ndim == 2:
        return torch.einsum(
            "kab, ia, jb -> kij",
            slip_sys.to(rotation_matrix.dtype),
            rotation_matrix,
            rotation_matrix,
        )
    elif rotation_matrix.ndim == 3:
        batch_dim = rotation_matrix.shape[0]
        return torch.einsum(
            "Bkab, Bia, Bjb -> Bkij",
            einops.repeat(
                slip_sys.to(rotation_matrix.dtype),
                "... -> b ...",
                b=batch_dim,
            ),
            rotation_matrix,
            rotation_matrix,
        )
    else:
        raise ValueError(
            "Expected input tensor 'rotation_matrix' of rank 2 or 3, but got"
            f" rank {rotation_matrix.ndim}."
        )


def rotate_elastic_stiffness(
    elastic_stiffness: FloatTensor[3, 3, 3, 3],
    rotation_matrix: Union[FloatTensor[3, 3], FloatTensor[B, 3, 3]],
) -> Union[FloatTensor[3, 3, 3, 3], FloatTensor[B, 3, 3, 3, 3]]:
    if rotation_matrix.ndim == 2:
        return torch.einsum(
            "abcd, ia, jb, kc, ld -> ijkl",
            elastic_stiffness.to(rotation_matrix.dtype),
            rotation_matrix,
            rotation_matrix,
            rotation_matrix,
            rotation_matrix,
        )
    elif rotation_matrix.ndim == 3:
        batch_dim = rotation_matrix.shape[0]
        return torch.einsum(
            "Babcd, Bia, Bjb, Bkc, Bld -> Bijkl",
            einops.repeat(
                elastic_stiffness.to(rotation_matrix.dtype),
                "... -> b ...",
                b=batch_dim,
            ),
            rotation_matrix,
            rotation_matrix,
            rotation_matrix,
            rotation_matrix,
        )
    else:
        raise ValueError(
            "Expected input tensor 'rotation_matrix' of rank 2 or 3, but got"
            f" rank {rotation_matrix.ndim}."
        )


def grain_orientation_bcc(ElasStif, SlipSys, angles):
    """rotates the stiffness and slip systems with the calculated rotation matrix"""
    rm = rotation_matrix(angles)
    #
    rotated_slip_system = torch.einsum("kab,ia,jb->kij", SlipSys.to(rm.dtype), rm, rm)
    #
    rotated_elas_stiffness = torch.einsum(
        "abcd,ia,jb,kc,ld->ijkl", ElasStif.to(rm.dtype), rm, rm, rm, rm
    )
    #
    return rotated_slip_system, rotated_elas_stiffness


def material_properties_bcc(angles):
    return grain_orientation_bcc(ElasStif, SlipSys, angles)


def get_ks(delta_s, slip_resist_0):
    return (
        consts.k0_F * (1 - (slip_resist_0 + delta_s) / consts.sInf_F) ** consts.uExp_F
    )


def get_H_matrix(ks):
    N = len(ks)
    dtype = ks.dtype
    return torch.vstack(N * [ks]) * (
        consts.q0_F * (torch.ones([N, N], dtype=dtype) - torch.eye(N, dtype=dtype))
        + torch.eye(N, dtype=dtype)
    )


def get_ws(H):
    N = len(H)
    return (1.0 / (consts.c0_F * consts.mu_F * N)) * H.sum(axis=0)


def get_beta(dgamma, ws, beta_0):
    return beta_0 + torch.dot(ws, dgamma)


def get_gd(beta, ws):
    return -consts.omega_F * consts.mu_F * beta * ws


def plastic_def_grad(dgamma, slip_sys, F_p0):
    I = torch.eye(3, dtype=F_p0.dtype)
    return (
        torch.linalg.inv(I - (dgamma.reshape(-1, 1, 1) * slip_sys).sum(axis=0)) @ F_p0
    )


def get_PK2(C_e, elas_stiff):
    """
    We write this function with `sum` because sum is
    faster than einsum (almost twice). There is a test in the test suite to ensure
    the two operations result in the same value.
    """
    I = torch.eye(3, dtype=C_e.dtype)
    return (elas_stiff * (0.5 * (C_e - I)).reshape(1, 1, 3, 3)).sum(axis=(2, 3))


def get_gm(F_e, slip_sys, elas_stiff):
    C_e = F_e.T @ F_e  # we use this twice
    S = get_PK2(C_e, elas_stiff)
    return ((C_e @ S).reshape(1, 3, 3) * slip_sys).sum(axis=(1, 2))


def get_driving_force(
    slip_resistance0, slip_resistance1, delta_gamma, beta0, Fp0, theta, F1
):
    rm = rotation_matrix(angles=theta)
    rotated_slip_system = rotate_slip_system(SlipSys, rm)
    rotated_elastic_stiffness = rotate_elastic_stiffness(ElasStif, rm)
    #
    gth = consts.g_th
    ks = vmap(get_ks)(
        slip_resistance1 - slip_resistance0,
        slip_resistance0,
    )
    H = vmap(get_H_matrix)(ks)
    ws = vmap(get_ws)(H)
    beta = vmap(get_beta)(delta_gamma, ws, beta0)
    gd = vmap(get_gd)(beta, ws)
    Fp1 = vmap(plastic_def_grad)(delta_gamma, rotated_slip_system, Fp0)
    Fe1 = F1 @ torch.linalg.inv(Fp1)
    gm = vmap(get_gm)(Fe1, rotated_slip_system, rotated_elastic_stiffness)
    non_schmid_stress = vmap(non_schmid_stress_bcc)(gm)
    ### up to this point we have tested all the functions that needed vmap.
    g = gm + gd + gth

    return g, H, non_schmid_stress


def get_cauchy_stress(theta, F1, Fp0, gamma0, gamma1):
    """
    Argument list

    theta [3]: Euler angles
    F1 [3x3]: Deformation gradient at n+1 time step
    Fp0 [3x3]: Plastic deformation gradient at n time step
    gamma0 [24]: Plastic slip at n time step
    gamma1 [24]: Plastic slip at n+1 time step

    Returns:
    sigma [3x3]: Cauchy stress tensor at n+1 time step

    Primary usage of this function is at inference to calculate
    the Cauchy stress from the plastic slip.
    """
    rm = rotation_matrix(theta)
    rotated_slip_system = rotate_slip_system(SlipSys, rm)
    rotated_elastic_stiffness = rotate_elastic_stiffness(ElasStif, rm)

    Fp1 = plastic_def_grad(gamma1 - gamma0, rotated_slip_system, Fp0)
    Fe1 = F1 @ torch.linalg.inv(Fp1)
    S = get_PK2(Fe1.T @ Fe1, rotated_elastic_stiffness)
    sigma = 1 / (torch.linalg.det(Fe1)) * Fe1 @ S @ Fe1.T

    return sigma


def inference_get_beta(dgamma, beta0, slip_res0, slip_res1):
    """
    NOTE: This should be used onlly for inference purposes!
    """
    # TODO: should we clip slip_res from above and below to be in range?
    ks = get_ks(slip_res1 - slip_res0, slip_res0)
    H_matrix = get_H_matrix(ks)
    ws = get_ws(H_matrix)
    beta = get_beta(dgamma, ws, beta0)
    return beta


def getF(F_init, F_final, t):
    assert 0 <= t <= 1
    return (1 - t) * F_init + F_final * t


def get_ts(n_time):
    return np.linspace(0, 1, n_time + 1).round(12)


def pairwise(xs):
    return zip(xs[:-1], xs[1:])


def load_model(path_to_model):
    ...


@dataclass
class HistoryResult:
    stress: list = field(default_factory=list)
    gamma: list = field(default_factory=list)
    slipres: list = field(default_factory=list)
    beta: list = field(default_factory=list)
    plastic_defgrad: list = field(default_factory=list)

    def store_values_returned_from_umat(self, xi, sigma):
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
    ):
        self.plastic_defgrad.append(Fp.clone())
        self.gamma.append(gamma.clone())
        self.slipres.append(slipres.clone())
        self.beta.append(beta.clone())
        # (1, 1) -> (0, 0)
        # (2, 2) -> (1, 1)
        # (3, 3) -> (2, 2)
        # (1, 2) -> (0, 1)
        # (1, 3) -> (0, 2)
        # (2, 3) -> (1, 2)
        voigt_stress = torch.tensor(
            [
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[0, 1],
                stress[0, 2],
                stress[1, 2],
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


def init_internal_variables():
    ...


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
    Fp0, gamma0, slip_res0, beta0 = init_internal_variables()
    model = load_model(path_to_model)
    F_init = np.eye(3)
    for t0, t1 in pairwise(get_ts(n_times)):
        F0, F1 = getF(F_init, F_final, t0), getF(F_init, F_final, t1)
        if t1 <= alpha:
            if t0 == 0.0:
        else:
            print("Use trained model")
            gamma1, slip_res1 = model.forward(
                theta=theta,
                defgrad0=F0,
                defgrad1=F1,
                gamma0=gamma0,
                slip_res0=slip_res0,
            )
            cauchy = get_cauchy_stress(theta, F1, Fp0, gamma0, gamma1)
            beta1 = inference_get_beta(
                dgamma=gamma1 - gamma0,
                beta0=beta0,
                slip_res0=slip_res0,
                slip_res1=slip_res1,
            )
