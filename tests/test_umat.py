import unittest
from .umat_imports import umat
import torch

class TestUmat(unittest.TestCase):

    def test_get_ks(self):
        '''
        For delta_s = 0 and s_n = 0 we should get ks = k_0 * [1, ..., 1]
        '''
        ks = umat.get_ks(torch.zeros(24),
                         torch.zeros(24))

        torch.testing.assert_allclose(
            ks,
            torch.ones(24) * umat.consts.k0_F
        )
        
    def test_get_H_matrix(self):
        '''
        Assume k = [1, 2, 3], then H = [[1         2 * q_F    3 * q_F],
                                        [1 * q_F   2          3 * q_F],
                                        [1 * q_F   2 * q_F          3]]
        '''
        H = umat.get_H_matrix(torch.tensor([1, 2, 3]))
        q_F = umat.consts.q0_F
        torch.testing.assert_allclose(
            H,
            torch.tensor(
                [[1     ,    2 * q_F,   3 * q_F],
                 [1 * q_F,   2      ,   3 * q_F],
                 [1 * q_F,   2 * q_F,         3]]
            )
        )

    def test_get_ws(self):
        '''
        w^{(i)} = (1/C) * sum_j H(j, i)      # sum on j index
        where C = c_F * mu_F * N

        Asuume H = [[1 2],
                    [3 4]]
        then ws = 1/C * [4 6]
        '''
        C = umat.consts.c0_F * umat.consts.mu_F * 2
        ws = umat.get_ws(torch.tensor([[1, 2],
                                       [3, 4]]))
        torch.testing.assert_allclose(ws, (1 / C) * torch.tensor([4, 6]))

    def test_get_beta(self):
        '''
        beta_{n+1} = beta_n + dot(ws, dgmamma)
        '''
        beta_0 = 1
        dgamma = torch.zeros(24)
        ws = torch.zeros(24)

        torch.testing.assert_allclose(
            umat.get_beta(dgamma, ws, beta_0),
            1
        )

    def test_gth_constant(self):
        '''
        g_th = Rho_0 * Theta0 * phi_F
        '''
        torch.testing.assert_allclose(
            umat.consts.g_th,
            umat.consts.Rho_0 * umat.consts.Theta0 * umat.consts.phi_F
        )

    def test_get_PK2(self):
        '''
        PK2_{ij} = CC_{ijkl} E_{kl}
        '''
        Fe = torch.rand(3, 3)
        CC = torch.rand(3, 3, 3, 3)
        #
        torch.testing.assert_allclose(
            umat.get_PK2(Fe.T @ Fe, CC),
            torch.einsum('ijkl,kl->ij', CC, 0.5 * (Fe.T @ Fe - torch.eye(3)))
        )

    def test_einsum_vs_sum(self):
        '''
        We want to make sure that the gradient computed two ways are the same
        The reason for switching to sum from einsum is that sum is almost twice
        as fast as einsum
        '''
        # stiffness and strain tensors
        CC = torch.rand(3, 3, 3, 3)
        E = torch.rand(3, 3, requires_grad=True)

        grad_from_einsum = torch.zeros_like(E)
        grad_from_sum = torch.zeros_like(E)

        # compute the gradient using einsum
        l_einsum = torch.einsum('ijkl,kl->ij', CC, E).norm()
        E.grad = None
        l_einsum.backward()
        grad_from_einsum.copy_(E.grad)

        # compute the gradient using sum
        l_sum = (CC * E.reshape(1, 1, 3, 3)).sum(axis=(2, 3)).norm()
        E.grad = None
        l_sum.backward()
        grad_from_sum.copy_(E.grad)

        torch.testing.assert_allclose(l_einsum, l_sum)
        torch.testing.assert_allclose(grad_from_einsum, grad_from_sum)


        
if __name__ == '__main__':
    unittest.main()        
