import unittest
import unittest.case
import numpy as np
import torch
from torch import vmap
from einops import rearrange
from umat.constants import consts
from umat.trip_ferrite_data import SlipSys, ElasStif
import umat.umat as umat

# typing
from torch import Tensor
from jaxtyping import Float

"""
This module contains tests with the following objectives:

- **Isolation and Verification**: Each formula is isolated and verified to ensure it functions 
as intended in both standalone (no batch) and batch modes. This guarantees that the composition 
of functions will operate as expected, when we utilize the `vmap`.

- **Constant Validation**: The correctness of constants for each formula is verified. While 
constant tests can be grouped into a single function, they have been separated based on their
 occurrence in the formulation for clarity.

- **Test Categories**: Depending on the specific formula under examination, a test class may 
containt one or more of the following types of tests:
  - `constant`: Verifies that the right constants are used.
  - `construction`: Checks that an alternative construction approach (often different from the one 
  in the `umat` module) produces equivalent results.
  - `batch`: Ensures that the formula functions correctly in batch mode.
"""


def assert_vmapped_results(ts: list[Tensor], vmap_result: Float[Tensor, "b ..."]) -> None:
    """
    Assume we have a function `fn` we wanna test for vmap correctness.
    `ts` is a list of tensors computed by `fn`
    `vmap_result` is the result of same inputs computed using vmap(fn)
    """
    torch.testing.assert_close(rearrange(ts, "b ... -> b ..."), vmap_result)


def generate_random_orientation():
    """Formula is from thesis chapter 3"""
    return torch.tensor(
        [
            2 * np.pi * np.random.uniform(low=0, high=1.0),
            np.arccos(2 * np.random.uniform(low=0, high=1.0) - 1),
            2 * np.pi * np.random.uniform(low=0, high=1.0),
        ]
    )


def rotate_slip_systems(thetas):
    rotation_matrix = umat.rotation_matrix(thetas)
    # NOTE: The sum can be written as either
    # 'kij, ai, bj -> kab' or
    # 'kab, ia, jb -> kij'.
    rotated_slip_system = torch.einsum(
        "kij, ai, bj -> kab",
        SlipSys.to(rotation_matrix.dtype),
        rotation_matrix,
        rotation_matrix,
    )
    return rotated_slip_system


class BaseTest(unittest.TestCase):
    """
    Use this as a base class for tests that are using constants.
    For classes that don't use constants, use either `unittest.TestCase` or `BaseTest`
    """

    def setUp(self) -> None:
        self.consts = consts


class TestRotationMatrix(BaseTest):
    """
    This test is for making sure that the rewrite of rotation matrix works the same
    as the one we have used so far. Up to this point, we had to use a function that
    was doing the vmapping of the function internally. To simplify the usage, for
    both the batched and non-batched quantities, we want to rewrite the rotation matrix
    in a way that it can be vmapped. Problem with this particular function is that
    vmap does not allow for in-place operations such as the followings:
        ```
        R1 = torch.zeros([3, 3]).to(angles.dtype)
        ...
        R1[0, 0] = torch.cos(angle1)
        R1[0, 1] = torch.sin(angle1)
        R1[1, 0] = -torch.sin(angle1)
        R1[1, 1] = torch.cos(angle1)
        R1[2, 2] = 1.0
        ```

    or even this version:
        ```
         R1 = torch.tensor([
        [torch.cos(angle1),  torch.sin(angle1), 0],
        [-torch.sin(angle1), torch.cos(angle1), 0],
        [0, 0, 1.0]], dtype=angles.dtype)
        ```

    this test, therefore, tries to make sure that the new implementation, which
    gives is a bit less readable but vmappable, gives the exact same result as the
    old formulation. We preserve the old formulation here for documentation named
    `_old_non_vmappable_rotation_matrix`. I emphasize once more that this formulation
    only flaw is that it cannot be vmapped.
    """

    def _old_non_vmappable_rotation_matrix(self, angles: Float[Tensor, "3"]) -> Float[Tensor, "3 3"]:
        """
        Readble but non-vmappable version of the rotation matrix.
        """

        angle1, angle2, angle3 = angles[0], angles[1], angles[2]

        R1 = torch.zeros([3, 3]).to(angles.dtype)
        R2 = torch.zeros([3, 3]).to(angles.dtype)
        R3 = torch.zeros([3, 3]).to(angles.dtype)

        #  rotation matrix of the second rotation (angle1)
        R1[0, 0] = torch.cos(angle1)
        R1[0, 1] = torch.sin(angle1)
        R1[1, 0] = -torch.sin(angle1)
        R1[1, 1] = torch.cos(angle1)
        R1[2, 2] = 1.0

        #  rotation matrix of the second rotation (angle2)
        R2[0, 0] = torch.cos(angle2)
        R2[0, 2] = -torch.sin(angle2)
        R2[1, 1] = 1.0
        R2[2, 0] = torch.sin(angle2)
        R2[2, 2] = torch.cos(angle2)

        #  rotation matrix of the third rotation (angle3)
        R3[0, 0] = torch.cos(angle3)
        R3[0, 1] = torch.sin(angle3)
        R3[1, 0] = -torch.sin(angle3)
        R3[1, 1] = torch.cos(angle3)
        R3[2, 2] = 1.0

        #  calculate the overall rotation matrix
        RM = R3 @ R2 @ R1

        return RM

    def test_rotation_matrix_shape(self):
        orientation_1 = generate_random_orientation()
        orientation_2 = generate_random_orientation()
        rm1 = umat.rotation_matrix(orientation_1)
        assert rm1.shape == torch.Size([3, 3]), rm1.shape

        orientations = rearrange([orientation_1, orientation_2], "b ... -> b ...")
        rms = vmap(umat.rotation_matrix)(orientations)
        assert rms.shape == torch.Size([2, 3, 3]), rms.shape

    def test_rotation_matrix_can_be_vmapped(self):
        """
        Make sure that the function works as expected in batch mode under `vmap` for a
        ealistic value of batch_size.
        """
        batch_size = 512
        orientation_list = [generate_random_orientation() for _ in range(batch_size)]
        orientations = rearrange(orientation_list, "b ... -> b ...")

        rm_list = [umat.rotation_matrix(orientation) for orientation in orientation_list]
        rm_vmapped = vmap(umat.rotation_matrix)(orientations)

        assert_vmapped_results(rm_list, rm_vmapped)


class TestSlipSystemRotation(BaseTest):
    """
    Make sure that the function works as expected in batch mode under `vmap`
    """

    def test_slip_system_rotation_shape(self):
        orientation_1 = generate_random_orientation()
        orientation_2 = generate_random_orientation()
        rm1 = umat.rotation_matrix(orientation_1)
        rotated_slip_system = umat.rotate_slip_system(SlipSys, rm1)
        assert rotated_slip_system.shape == torch.Size([24, 3, 3])

        orientations = rearrange([orientation_1, orientation_2], "b ... -> b ...")
        rms_vmapped = vmap(umat.rotation_matrix)(orientations)
        rotated_slip_systems = vmap(umat.rotate_slip_system, in_dims=(None, 0))(SlipSys, rms_vmapped)
        assert rotated_slip_systems.shape == torch.Size([2, 24, 3, 3])

    def test_slip_system_rotation_can_be_vmapped(self):
        """
        Make sure that the function works as expected in batch mode under `vmap` for a
        ealistic value of batch_size.
        """
        batch_size = 512
        orientation_list = [generate_random_orientation() for _ in range(batch_size)]
        rm_list = [umat.rotation_matrix(orientation) for orientation in orientation_list]
        orientations = rearrange(orientation_list, "b ... -> b ...")
        rms_vmapped = vmap(umat.rotation_matrix)(orientations)

        rotated_slip_system_list = [umat.rotate_slip_system(SlipSys, rm) for rm in rm_list]
        rotated_slip_system_vmapped = vmap(umat.rotate_slip_system, in_dims=(None, 0))(SlipSys, rms_vmapped)

        assert_vmapped_results(rotated_slip_system_list, rotated_slip_system_vmapped)


class TestElasticStiffnessRotation(BaseTest):
    """
    Make sure that the function works as expected in batch mode under `vmap`
    """

    def test_elastic_stiffness_rotation_shape(self):
        orientation_1 = generate_random_orientation()
        orientation_2 = generate_random_orientation()
        rm1 = umat.rotation_matrix(orientation_1)
        rotated_elastic_stiffness = umat.rotate_elastic_stiffness(ElasStif, rm1)
        assert rotated_elastic_stiffness.shape == torch.Size([3, 3, 3, 3])

        orientations = rearrange([orientation_1, orientation_2], "b ... -> b ...")
        rms_vmapped = vmap(umat.rotation_matrix)(orientations)
        rotated_elastic_stiffness_vmapped = vmap(umat.rotate_elastic_stiffness, in_dims=(None, 0))(
            ElasStif, rms_vmapped
        )
        assert rotated_elastic_stiffness_vmapped.shape == torch.Size([2, 3, 3, 3, 3])

    def test_elastic_stiffness_rotation_can_be_vmapped(self):
        """
        Make sure that the function works as expected in batch mode under `vmap` for a
        ealistic value of batch_size.
        """
        batch_size = 512
        orientation_list = [generate_random_orientation() for _ in range(batch_size)]
        rm_list = [umat.rotation_matrix(orientation) for orientation in orientation_list]
        orientations = rearrange(orientation_list, "b ... -> b ...")
        rms_vmapped = vmap(umat.rotation_matrix)(orientations)

        rotated_elastic_stiffness_list = [umat.rotate_elastic_stiffness(ElasStif, rm) for rm in rm_list]
        rotated_elastic_stiffness_vmapped = vmap(umat.rotate_elastic_stiffness, in_dims=(None, 0))(
            ElasStif, rms_vmapped
        )
        assert_vmapped_results(rotated_elastic_stiffness_list, rotated_elastic_stiffness_vmapped)


class TestNonSchmidtStress(BaseTest):
    """
    The function that implements the calculation of non-Schmidt stress has a lot of
    inplace operations. Make sure that it works in batch mode with vmap.
    """

    def _non_schmid_stress_bcc_simple(self, Schmid: Float[Tensor, "24"]) -> Float[Tensor, "24"]:
        """
        Computes non-Schmid stress according to Bassani's convention.
        This used to be the original function, but it turned out using an indexing is
        5 times faster than the current implementation. We keep this here for checking
        the validity of the new implementation.
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

    def test_non_schmidt_stress_function_can_be_vmapped(self):
        t1 = torch.rand(24)
        t2 = torch.rand(24)
        ts = rearrange([t1, t2], "b ... -> b ...")

        nonSchmidtStress1 = self._non_schmid_stress_bcc_simple(t1)
        nonSchmidtStress2 = self._non_schmid_stress_bcc_simple(t2)
        nonSchmidtStresses_vmap = vmap(umat.non_schmid_stress_bcc)(ts)

        assert_vmapped_results([nonSchmidtStress1, nonSchmidtStress2], nonSchmidtStresses_vmap)


class TestElasticDefGradient(unittest.TestCase):
    """
    This quantity is a simple matrix multiplication and inversion, since we have the
    following identity for multiplicative decomposition of the deformation gradient.
    Previously this was written as Fe = F @ torch.linalg.inv(Fp) but we make this into
    a lambda for testing purposes.

    \begin{equation}
        {\bf F} = {\bf F}_e {\bf F}_p
    \end{equation}

    Since there are no constants and construction is trivial, we only write the batch test
    to make sure batch computation is working as expected.
    """

    def test_elastic_def_gradient_batch(self):
        # A batch of 2
        F = torch.rand(2, 3, 3)
        Fp = torch.rand(2, 3, 3)

        # decompose the batch. F1 and F2 (as well as Fp1 and Fp2) has nothing to do with
        # time stepping in this particular setting
        F1, F2 = F
        Fp1, Fp2 = Fp

        # define the lambda for testing
        Fe = lambda F, Fp: F @ torch.linalg.inv(Fp)

        torch.testing.assert_close(
            F @ torch.linalg.inv(Fp),
            rearrange([Fe(F1, Fp1), Fe(F2, Fp2)], "b ... -> b ..."),
        )


class TestMechanicalDrivingForce(unittest.TestCase):
    """
    Mechanical driving force is computed using:

    \begin{equation}
        g_m^{(i)} = {\bf F}_e^{\sf T} {\bf F}_e {\bf S} \cdot \left( {\bf m}^{(i)} \otimes {\bf n}^{(i)} \right)
    \end{equation}

    """

    def setUp(self) -> None:
        self.thetas = generate_random_orientation()
        self.slip_systems = rotate_slip_systems(self.thetas)
        self.elastic_stiffness = self._rotate_elastic_stiffness(self.thetas)

        # Generate Fe of the magnitude 1e-3
        self.I = torch.eye(3, dtype=self.thetas.dtype)
        U = 1e-3 * torch.rand(3, 3, dtype=self.thetas.dtype)
        self.Fe = self.I + U

    def _rotate_elastic_stiffness(self, thetas):
        rotation_matrix = umat.rotation_matrix(thetas)
        rotated_elastic_stiffness = torch.einsum(
            "ijkl, ai, bj, ck, dl -> abcd",
            ElasStif.to(rotation_matrix.dtype),
            rotation_matrix,
            rotation_matrix,
            rotation_matrix,
            rotation_matrix,
        )
        return rotated_elastic_stiffness

    def test_PK2_construction(self):
        S = umat.get_PK2(self.Fe.T @ self.Fe, self.elastic_stiffness)
        S_expected = torch.einsum(
            "ijkl,kl->ij",
            self.elastic_stiffness,
            0.5 * (self.Fe.T @ self.Fe - self.I),
        )
        torch.testing.assert_close(S, S_expected)

    def test_mechanical_driving_force_construction(self):
        Ce = self.Fe.T @ self.Fe
        S = torch.einsum(
            "ijkl,kl->ij",
            self.elastic_stiffness,
            0.5 * (Ce - self.I),
        )
        mech_driving_force_expected = torch.einsum(
            "ij, bij -> b", Ce @ S, self.slip_systems
        )
        mech_driving_force = umat.get_gm(
            self.Fe, self.slip_systems, self.elastic_stiffness
        )

        torch.testing.assert_close(mech_driving_force, mech_driving_force_expected)

    def test_mechanical_driving_force_batch(self):
        Fe = torch.rand(2, 3, 3)
        slip_sys = torch.rand(2, 24, 3, 3)
        elastic_stiffness = torch.rand(2, 3, 3, 3, 3)

        Fe1, Fe2 = Fe
        slip_sys1, slip_sys2 = slip_sys
        elastic_stiffness1, elastic_stiffness2 = elastic_stiffness

        torch.testing.assert_close(
            vmap(umat.get_gm)(Fe, slip_sys, elastic_stiffness),
            rearrange(
                [
                    umat.get_gm(Fe1, slip_sys1, elastic_stiffness1),
                    umat.get_gm(Fe2, slip_sys2, elastic_stiffness2),
                ],
                "b ... -> b ...",
            ),
        )


class TestPlasticDefGradient(unittest.TestCase):
    def setUp(self) -> None:
        N = 24
        self.thetas = generate_random_orientation()
        self.slip_systems = rotate_slip_systems(self.thetas)
        assert self.slip_systems.shape == (24, 3, 3)

        self.dgamma = (1e-3 * torch.rand(N)).to(self.thetas.dtype)

        # Diagonal entries of Fp should be close to 1 and
        # off diagonals should be close to zero both vary by magnitude 1e-3.
        # The gradient of Fp0 should be very close 1 (isochoric).
        F_p0 = torch.eye(3, dtype=self.thetas.dtype)
        perturbations = 1e-3 * torch.rand(3, 3) - 5e-4
        F_p0 += perturbations
        F_p0 /= torch.linalg.det(F_p0) ** (1 / 3)
        self.F_p0 = F_p0

    def test_plastic_def_gradient_construction(self):
        I = torch.eye(3)
        temp = torch.zeros_like(self.slip_systems[0])
        for i in range(24):
            temp += self.dgamma[i] * self.slip_systems[i]
        F_p1 = umat.plastic_def_grad(self.dgamma, self.slip_systems, self.F_p0)
        F_p1_expected = torch.linalg.inv(I - temp) @ self.F_p0
        torch.testing.assert_close(F_p1, F_p1_expected)

    def test_plastic_def_gradient_batch(self):
        dgamma = torch.rand(2, 5)
        slip_sys = torch.rand(2, 5, 3, 3)
        Fp0 = torch.rand(2, 3, 3, dtype=torch.float64)

        dgamma1, dgamma2 = dgamma
        slip_sys1, slip_sys2 = slip_sys
        Fp01, Fp02 = Fp0

        torch.testing.assert_close(
            vmap(umat.plastic_def_grad)(dgamma, slip_sys, Fp0),
            rearrange(
                [
                    umat.plastic_def_grad(dgamma1, slip_sys1, Fp01),
                    umat.plastic_def_grad(dgamma2, slip_sys2, Fp02),
                ],
                "b ... -> b ...",
            ),
        )


class TestGammaRateEquation(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.gammadot_0 = self.consts.GammaDot0_F
        self.power_F = self.consts.pExp_F

    def test_GammaRateEquation_constant(self):
        self.assertEqual(self.gammadot_0, 1.0e-3)
        self.assertEqual(self.power_F, 2.0e-2)

    # TODO: Complete the function
    def test_gammarate_eq_construction(self):
        ...


class TestThermalDrivingForce(BaseTest):
    """
    This is a constant for all cases and the values does not
    change since all calculations are isothermal.
    \begin{equation}
        g_{th}^{(i)} = g_{th} = \rho_0 \theta \phi_F
    \end{equation}
    """

    def setUp(self) -> None:
        super().setUp()
        self.g_th = self.consts.g_th

    def test_ThermalDrivingForce_constant(self):
        # Verify if g_th has the expected value.
        self.assertEqual(self.g_th, 0.0099918)

        # Check if g_th is computed correctly using its contributing constants.
        self.assertEqual(
            self.g_th,
            self.consts.Rho_0 * self.consts.Theta0 * self.consts.phi_F,
        )


class TestDamageDrivingForce(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.omega_F = self.consts.omega_F

        # Compute microstrain weights and beta for damage driving force
        ks = torch.rand(consts.N) * consts.sInf_F
        H_matrix = umat.get_H_matrix(ks)
        beta_0 = torch.tensor(0.0012)
        dgamma = 1.3e-3 * torch.rand(consts.N)

        # We need these values for computing g_d
        self.ws = umat.get_ws(H_matrix)
        self.beta = umat.get_beta(dgamma, self.ws, beta_0=beta_0)

    def test_DamageDrivingForce_constant(self):
        self.assertEqual(self.omega_F, 7.0)

    def test_DamageDrivingForce_construction(self):
        """
        Test the construction of damage driving force `g_d`.

        The damage driving force is given by:

        \begin{equation}
            g_d^{(i)} = - \omega_F \mu_F \beta w^{(i)}
        \end{equation}
        """
        gd = umat.get_gd(self.beta, self.ws)
        gd_expected = -consts.omega_F * consts.mu_F * self.beta * self.ws
        torch.testing.assert_close(gd, gd_expected)

    def test_DamageDrivingForce_batch(self):
        ws = torch.rand(2, 3)
        beta = torch.rand(2)

        ws1, ws2 = ws
        beta1, beta2 = beta

        torch.testing.assert_close(
            vmap(umat.get_gd)(beta, ws),
            rearrange(
                [umat.get_gd(beta1, ws1), umat.get_gd(beta2, ws2)],
                "b ... -> b ...",
            ),
        )


class TestEffectiveScalarMicrostrain(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        ks = torch.rand(consts.N) * consts.sInf_F
        self.H_matrix = umat.get_H_matrix(ks)

    def test_EffectiveScalarMicrostrain_construction(self):
        """
        `beta` (effective scalar micro strain) is calculated
         according to the following formula:
        \begin{equation}
            \beta_{n+1} = \beta_n + \Delta \beta = \beta_n + \sum_{i=1}^N w_{n+1}^{(i)} \Delta \gamma^{(i)}
        \end{equation}
        """
        N = 24
        # beta_0 -> beta_n and beta_1 -> beta_{n+1}
        beta_0 = torch.tensor(0.001)

        # scaled to have more realistic values
        dgamma = 1e-3 * torch.rand(N)

        # Calculate weights using the hardening modulus matrix
        ws = umat.get_ws(self.H_matrix)

        beta_1 = umat.get_beta(dgamma, ws, beta_0=beta_0)
        beta_1_expected = beta_0 + (dgamma * ws).sum()
        torch.testing.assert_close(beta_1, beta_1_expected)

    def test_EffectiveScalarMicrostrain_batch(self):
        dgamma = torch.rand(2, 24)
        ws = torch.rand(2, 24)
        beta0 = torch.rand(2)

        dgamma1, dgamma2 = dgamma
        ws1, ws2 = ws
        beta01, beta02 = beta0

        torch.testing.assert_close(
            vmap(umat.get_beta)(dgamma, ws, beta0),
            rearrange(
                [
                    umat.get_beta(dgamma1, ws1, beta01),
                    umat.get_beta(dgamma2, ws2, beta02),
                ],
                "b ... -> b ...",
            ),
        )


class TestMicrostrainWeights(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.cF = self.consts.c0_F
        self.muF = self.consts.mu_F

    def test_MicrostrainWeights_constant(self):
        self.assertEqual(self.cF, 0.5)
        self.assertEqual(self.muF, 55.0)

    def test_MicrostrainWeights_construction(self):
        """
        `ws` are computed from the hardening modulus matrix according to the following formula:

            \begin{equation}
                w^{(i)} = \frac{1}{c_F \mu_F N} \sum_{j=1}^N H^{(j,i)}
            \end{equation}
        """
        N = 24
        # scale ks so it is not bigger than s_inf. See the formula for
        # slip hardening modulus for clarification.
        ks = torch.rand(N) * consts.sInf_F
        H_matrix = umat.get_H_matrix(ks)
        ws = umat.get_ws(H_matrix)
        ws_expected = 1 / (self.cF * self.muF * N) * H_matrix.sum(dim=0)
        torch.testing.assert_close(ws, ws_expected)

    def test_MicrostrainWeights_batch(self):
        H = torch.rand(2, 24, 24)
        H1, H2 = H

        torch.testing.assert_close(
            vmap(umat.get_ws)(H),
            rearrange([umat.get_ws(H1), umat.get_ws(H2)], "b ... -> b ..."),
        )


class TestHardeningModMatrix(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.qF = self.consts.q0_F

    def test_H_constant(self):
        self.assertEqual(self.qF, 1.0)

    def test_H_matrix_construction(self):
        """
        Assume ks = [k1, k2, k3], then H is:

            [[k1       , q_F * k2 , q_F * k3],
             [q_F * k1 , k2       , q_F * k3],
             [q_F * k1 , q_F * k2 , k3      ]]
        """
        qF = self.qF
        ks = torch.rand(3)
        k1, k2, k3 = ks
        H = umat.get_H_matrix(ks=ks)
        H_expected = torch.tensor(
            [
                [k1, qF * k2, qF * k3],
                [qF * k1, k2, qF * k3],
                [qF * k1, qF * k2, k3],
            ]
        )
        torch.testing.assert_close(H, H_expected)


class TestSlipHardeningModulus(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.s_Finf = self.consts.sInf_F
        self.kF0 = self.consts.k0_F
        self.uF = self.consts.uExp_F

    def test_ks_constant(self):
        self.assertEqual(self.s_Finf, 0.412)
        self.assertEqual(self.kF0, 1.9)
        self.assertEqual(self.uF, 2.8)

    def test_zero_slip_resistance(self):
        """Zero slip resistance should result in ks = k_F * [1, ..., 1]"""
        ks = umat.get_ks(delta_s=torch.zeros(24), slip_resist_0=torch.zeros(24))
        ks_expected = self.kF0 * torch.ones(24)
        torch.testing.assert_close(ks, ks_expected)

    def test_sinf_slip_resistance(self):
        """If slip resistance is s_inf then ks should be zero: [0, ..., 0]"""
        ks = umat.get_ks(
            delta_s=torch.zeros(24), slip_resist_0=self.s_Finf * torch.ones(24)
        )
        ks_expected = torch.zeros(24)
        torch.testing.assert_close(ks, ks_expected)

    def test_SlipHardeningModulus_batch(self):
        """
        We should be careful with construction s0 and ds because the sum
        cannot be bigger that s_inf.
        """
        slip_res_0 = 0.8 * self.s_Finf * torch.rand(2, 24)
        ds = 0.2 * self.s_Finf * torch.rand(2, 24)

        #
        ds1, ds2 = ds
        slip_res_01, slip_res_02 = slip_res_0

        torch.testing.assert_close(
            vmap(umat.get_ks)(ds, slip_res_0),
            rearrange(
                [
                    umat.get_ks(ds1, slip_res_01),
                    umat.get_ks(ds2, slip_res_02),
                ],
                "b ... -> b ...",
            ),
        )


class TestMisc(unittest.TestCase):
    """
    Put here all the tests that don't belong to a specific umat quantity.
    """

    def test_einsum_vs_sum(self):
        """
        We want to make sure that the gradient computed two ways are the same.
        The reason for switching to sum from einsum is that sum is almost twice
        as fast as einsum
        """
        # stiffness and strain tensors
        CC = torch.rand(3, 3, 3, 3)
        E = torch.rand(3, 3, requires_grad=True)

        grad_from_einsum = torch.zeros_like(E)
        grad_from_sum = torch.zeros_like(E)

        # compute the gradient using einsum
        l_einsum = torch.einsum("ijkl,kl->ij", CC, E).norm()
        E.grad = None
        l_einsum.backward()
        grad_from_einsum.copy_(E.grad)

        # compute the gradient using sum
        l_sum = (CC * E.reshape(1, 1, 3, 3)).sum(axis=(2, 3)).norm()
        E.grad = None
        l_sum.backward()
        grad_from_sum.copy_(E.grad)

        torch.testing.assert_close(l_einsum, l_sum)
        torch.testing.assert_close(grad_from_einsum, grad_from_sum)


if __name__ == "__main__":
    unittest.main()
