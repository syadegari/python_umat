import h5py
import einops
import numpy as np
import torch
from torch.func import vmap

from umat.constants import consts
from umat.simulate import parse_umat_output
import umat.umat as umat
import umat.simulate as sim
from umat.trip_ferrite_data import SlipSys, ElasStif


pairwise = lambda xs: zip(xs[:-1], xs[1:])


if __name__ == "__main__":
    # read orientation and deformation gradient and convert them to tensor
    with h5py.File("data.h5") as fh:
        orientation = fh["angle"][:]
        defgrad = fh["defgrad"][:]

    orientation = torch.from_numpy(einops.rearrange(orientation, "... -> 1 ..."))
    defgrad = torch.from_numpy(einops.rearrange(defgrad, "... -> 1 ..."))

    # Since we are only having one simulation
    umat_result = sim.simulate_umat({"angle": orientation, "defgrad": defgrad}, n_times=400)[0]
    states = parse_umat_output(umat_result)

    rm = umat.rotation_matrix(orientation.squeeze())
    rotated_slip_system = umat.rotate_slip_system(SlipSys, rm)
    rotated_elastic_stiffness = umat.rotate_elastic_stiffness(ElasStif, rm)

    counter = 0
    for s0, s1 in pairwise(states):
        gamma0 = torch.from_numpy(s0.intvar.gamma)
        gamma1 = torch.from_numpy(s1.intvar.gamma)

        slip_resistance0 = torch.from_numpy(s0.intvar.slip_resistance)
        slip_resistance1 = torch.from_numpy(s1.intvar.slip_resistance)

        Fp0 = torch.from_numpy(s0.intvar.plastic_defgrad).reshape(3, 3)
        F1 = torch.from_numpy(s1.defgrad).reshape(3, 3)
        # import pdb

        # pdb.set_trace()
        g1, H, nonSchmidStress = umat.get_driving_force(
            rotated_slip_system,
            rotated_elastic_stiffness,
            slip_resistance0,
            slip_resistance1,
            delta_gamma=gamma1 - gamma0,
            beta0=s0.intvar.beta,
            Fp0=Fp0,
            F1=F1,
        )

        r_I = umat.get_rI(slip_resistance0, slip_resistance1, gamma0, gamma1, H)

        r_II = umat.get_rII(
            g1, slip_resistance1, nonSchmidStress, gamma0, gamma1, consts.GammaDot0_F, s1.t - s0.t, consts.pExp_F
        )

        Fp1 = umat.plastic_def_grad(gamma1 - gamma0, rotated_slip_system, Fp0)
        sigma = sim.to_Voigt(umat.get_cauchy_stress(F1, Fp1, rotated_elastic_stiffness))

        #
        # predictor values begin
        #
        g1_p, H_p, nonSchmidStress_p = umat.get_driving_force(
            rotated_slip_system,
            rotated_elastic_stiffness,
            slip_resistance0,
            slip_resistance0,
            delta_gamma=gamma0 - gamma0,
            beta0=s0.intvar.beta,
            Fp0=Fp0,
            F1=F1,
        )
        r_II_p = umat.get_rII(
            g1_p, slip_resistance0, nonSchmidStress_p, gamma0, gamma0, consts.GammaDot0_F, s1.t - s0.t, consts.pExp_F
        )
        #
        # predictor section end
        #
        if s0.t == 0.0:
            print("r_I norm, r_II norm, predictive r_II norm, relative stress error")

        relative_stress_error = (sigma - s1.stress).norm() / np.linalg.norm(s1.stress)
        print(
            f"{r_I.norm().item():.6e},",
            " ",
            f"{r_II.norm().item():.6e},",
            " ",
            f"{r_II_p.norm().item():.6e},",
            " ",
            f"{relative_stress_error.item():.6e}",
        )
    # TODOS:
    #       1- [DONE] write the max of r_II as well: We did that initially, but later removed it when we validated that the r_II residual is close to zero.
    #       2- [DONE] write the predictor values of r_II as well (r_II)_p.
    #       3- [DONE] make stress error relative.
    #       4- modify UMAT to report r_II (converged value): Won't do this. We managed to show that residuals are calculated correctly and close to zero, so looking into UMAT is not needed anymore.
    #       5- more than one simulation: Probably a good idea to try another pair of (gardient, orientation). It's very unlikely that a new pair invalidate the current findings, given that the pair was chosen randomly.
