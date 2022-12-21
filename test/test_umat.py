import unittest
import torch.testing as test
import _umat_imports_

import torch

import umat


class TestUmat(unittest.TestCase):
    def test_matmul(self):
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.randn(3, 5)

        self.assertTrue(torch.allclose(
            a @ b @ c, (a @ b) @ c
        )    