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

        
if __name__ == '__main__':
    unittest.main()        
