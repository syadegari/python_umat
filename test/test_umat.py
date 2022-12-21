import pytest
import torch.testing as test
import _umat_imports_

import torch

import umat


def test_matmul():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.randn(5, 6)
    test.assert_allclose(
        a @ (b @ c), (a @ b) @ c
    )
    