import unittest
import torch.testing as test
from .umat_imports import umat

import torch

from dataclasses import dataclass




class TestUmat(unittest.TestCase):

    def test_get_ks(self):


        ks = umat.get_ks(torch.zeros(24, requires_grad=True),
                         torch.zeros(24),
                         umat.consts.k0_F,
                         umat.consts.w0_F,
                         umat.consts.sInf_F)

        torch.testing.assert_allclose(
            ks,
            torch.ones(24) * umat.consts.k0_F
        )
        
        
        
if __name__ == '__main__':
    unittest.main()        
