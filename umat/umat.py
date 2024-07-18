from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch.func import vmap

# from torch._functorch.apis import vmap
import einops

from .constants import consts
from .trip_ferrite_data import SlipSys, ElasStif

from typing import Tuple, Union, List
from jaxtyping import Float
from torch import Tensor
from torch._tensor import Tensor

# TODO: Decide what to do with these annotations
# BTAngles = FloatTensor[Tuple[B, 3]]
# BTElasticity = FloatTensor[Tuple[B, 3, 3, 3, 3]]
# BTSlipSystem = FloatTensor[Tuple[B, 24, 3, 3]]
# BTVarScalar = FloatTensor[Tuple[B]]
# BTVarVector = FloatTensor[Tuple[B, 24]]
# BTVarMatrix = FloatTensor[Tuple[B, 24, 24]]
# BTVarStrain = FloatTensor[Tuple[B, 3, 3]]
# F32orF64 = Union[torch.float32, torch.float64]


def sdvini() -> Tuple[Tensor, Tensor, Tensor, float]:
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


def non_schmid_stress_bcc(Schmid: Tensor) -> Tensor:
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


def rotation_matrix(angles: Float[Tensor, "3"]) -> Float[Tensor, "3 3"]:
    """
    Defines the rotation matrix using 323 Euler rotations.
    This is the replica of the Fortran-version. It has been
    modified to be batch-compatible and usable with vmap.

    See the tests section for further explanation and the testing.
    """
    angle1, angle2, angle3 = angles[0], angles[1], angles[2]

    # Rotation matrix of the first rotation (angle1)
    R1 = torch.stack(
        [
            torch.stack([torch.cos(angle1), torch.sin(angle1), torch.tensor(0.0, dtype=angles.dtype)]),
            torch.stack([-torch.sin(angle1), torch.cos(angle1), torch.tensor(0.0, dtype=angles.dtype)]),
            torch.stack(
                [
                    torch.tensor(0.0, dtype=angles.dtype),
                    torch.tensor(0.0, dtype=angles.dtype),
                    torch.tensor(1.0, dtype=angles.dtype),
                ]
            ),
        ]
    )

    # Rotation matrix of the second rotation (angle2)
    R2 = torch.stack(
        [
            torch.stack([torch.cos(angle2), torch.tensor(0.0, dtype=angles.dtype), -torch.sin(angle2)]),
            torch.stack(
                [
                    torch.tensor(0.0, dtype=angles.dtype),
                    torch.tensor(1.0, dtype=angles.dtype),
                    torch.tensor(0.0, dtype=angles.dtype),
                ]
            ),
            torch.stack([torch.sin(angle2), torch.tensor(0.0, dtype=angles.dtype), torch.cos(angle2)]),
        ]
    )

    # Rotation matrix of the third rotation (angle3)
    R3 = torch.stack(
        [
            torch.stack([torch.cos(angle3), torch.sin(angle3), torch.tensor(0.0, dtype=angles.dtype)]),
            torch.stack([-torch.sin(angle3), torch.cos(angle3), torch.tensor(0.0, dtype=angles.dtype)]),
            torch.stack(
                [
                    torch.tensor(0.0, dtype=angles.dtype),
                    torch.tensor(0.0, dtype=angles.dtype),
                    torch.tensor(1.0, dtype=angles.dtype),
                ]
            ),
        ]
    )

    # Calculate the overall rotation matrix
    RM = R3 @ R2 @ R1

    return RM


def rotate_slip_system(
    slip_sys: Float[Tensor, "3 3 3"], rotation_matrix: Float[Tensor, "3 3"]
) -> Float[Tensor, "3 3 3"]:
    """
    Rotate elastic stiffness tensor according to the given rotation matrix.

    Refer to the note in `rotate_elastic_stiffness_singleton` for usage with `vmap`.
    """
    return torch.einsum(
        "kab, ia, jb -> kij",
        slip_sys.to(rotation_matrix.dtype),
        rotation_matrix,
        rotation_matrix,
    )

def grain_orientation_bcc(ElasStif, SlipSys, angles: Tensor) -> Tuple[Tensor, Tensor]:
    """rotates the stiffness and slip systems with the calculated rotation matrix"""
    rm = rotation_matrix(angles)
    #
    rotated_slip_system = torch.einsum("kab,ia,jb->kij", SlipSys.to(rm.dtype), rm, rm)
    #
    rotated_elas_stiffness = torch.einsum("abcd,ia,jb,kc,ld->ijkl", ElasStif.to(rm.dtype), rm, rm, rm, rm)
    #
    return rotated_slip_system, rotated_elas_stiffness

def rotate_elastic_stiffness(
    elastic_stiffness: Float[Tensor, "3 3 3 3"], rotation_matrix: Float[Tensor, "3 3"]
) -> Float[Tensor, "3 3 3 3"]:
    """
    Rotate elastic stiffness tensor according to the given rotation matrix.

def material_properties_bcc(angles: Tensor) -> Tuple[Tensor, Tensor]:
    return grain_orientation_bcc(ElasStif, SlipSys, angles)
    Note:
    When vmapping this function over a batched rotation matrix with unrotated elastif stiffness
    ElasStif from `trip_ferrite_data.py`, we have two choices:
     - We can call it with `in_dims=(None, 0)`:
       >>> vmap(rotate_elastic_stiffness_singleton, in_dims=(None, 0))(ElasStif, rotation_matrices: Float[Tensor, "b 3 3"])
     - We can call it with ElasStif copied along batch dimension:
       >>> from einops import repeat
       >>> batched_ElasStif = repeat(ElasStif, '3 3 3 3 -> b 3 3 3 3', b=batch_size)
       >>> vmap(rotate_elastic_stiffness_singleton)(batched_ElasStif, rotation_matrices)

    Personally I prefer the first version with the `in_dims` argument.
    """
    return torch.einsum(
        "abcd, ia, jb, kc, ld -> ijkl",
        elastic_stiffness.to(rotation_matrix.dtype),
        rotation_matrix,
        rotation_matrix,
        rotation_matrix,
        rotation_matrix,
    )


def get_ks(delta_s, slip_resist_0) -> float:
    return consts.k0_F * (1 - (slip_resist_0 + delta_s) / consts.sInf_F) ** consts.uExp_F


def get_H_matrix(ks) -> Tensor:
    N = len(ks)
    dtype = ks.dtype
    return torch.vstack(N * [ks]) * (
        consts.q0_F * (torch.ones([N, N], dtype=dtype) - torch.eye(N, dtype=dtype)) + torch.eye(N, dtype=dtype)
    )


def get_ws(H) -> float:
    N = len(H)
    return (1.0 / (consts.c0_F * consts.mu_F * N)) * H.sum(axis=0)


def get_beta(dgamma: Tensor, ws: Tensor, beta_0):
    return beta_0 + torch.dot(ws, dgamma)


def get_gd(beta: float, ws: float) -> float:
    return -consts.omega_F * consts.mu_F * beta * ws


def plastic_def_grad(dgamma, slip_sys, F_p0):
    I = torch.eye(3, dtype=F_p0.dtype)
    return torch.linalg.inv(I - (dgamma.reshape(-1, 1, 1) * slip_sys).sum(axis=0)) @ F_p0


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


def get_driving_force(slip_resistance0, slip_resistance1, delta_gamma, beta0, Fp0, theta: Tensor, F1):
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


def get_rI(s0, s1, gamma0, gamma1, H_matrix):
    return (s1 - s0) - H_matrix @ (gamma1 - gamma0)


def get_rII(g1, s1, non_schmid_stress, gamma0, gamma1, gamma_dot_0, dt, pF) -> Tensor:
    return torch.where(
        g1 > s1,
        g1 - (s1 - non_schmid_stress) * ((gamma1 - gamma0) / (gamma_dot_0 * dt) + 1) ** pF,
        gamma1 - gamma0,
    )


def clip_slip_resistance(slip_resistance: Tensor) -> Tensor:
    """Clips slip resistance to be between min and max"""
    return torch.clip(slip_resistance, consts.s0_F, consts.sInf_F)


def enforce_non_negative_increment(gamma1: Tensor, gamma0: Tensor) -> Tensor:
    """Enforces delta_gamma => 0"""
    return torch.where(gamma1 >= gamma0, gamma1, gamma0)


def get_penalty_delta_gamma(gamma1: Tensor, gamma0: Tensor) -> Tensor:
    """Penalty in case delta gamma is negative"""
    delta_gamma = gamma1 - gamma0
    return torch.where(delta_gamma >= 0, 0.0, -delta_gamma)


def get_penalty_max_slip_resistance(slip_resistance: Tensor) -> Tensor:
    """Penalty in case slip resistance is bigger than s_inf_F"""
    delta_to_upper_value = consts.sInf_F - slip_resistance
    return torch.where(delta_to_upper_value >= 0, 0.0, -delta_to_upper_value)


def get_penalty_min_slip_resistance(slip_resistance: Tensor) -> Tensor:
    """Penalty in case slip resistance is less than s_0_F"""
    delta_to_lower_value = slip_resistance - consts.s0_F
    return torch.where(delta_to_lower_value >= 0, 0.0, -delta_to_lower_value)
