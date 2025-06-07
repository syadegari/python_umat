# Overview

This directory contains tests for the python_umat project, which implements a crystal plasticity model. The tests are organized to validate various aspects of the model, including computational correctness, Fortran-to-Python/PyTorch porting, and residual calculations.
Root-Level Tests

The root of the `tests/` directory includes unit tests that verify the correctness of individual computational components of the crystal plasticity model. These tests can be executed from the repository's root directory using:

```bash
python -m unittest tests.test_umat -v
```

These tests ensure that fundamental calculations and algorithms behave as expected.

## [`einsum_test`](./einsum_test) and [`rotation_matrix_test`](./rotation_matrix_test)

These subdirectories contain tests related to porting the model from Fortran to Python/PyTorch. They includes:

   - Validation Tests: Compare outputs between the original Fortran implementation and the Python/PyTorch version to ensure consistency.

   - Performance Benchmarks: Assess the performance differences between implementations.

## [`residual_investigation`](./residual_investigation)

Tests in this folder focus on the correctness of residual calculations within the model. They aim to:

  - Verify Residual Computations: Ensure that residuals are computed accurately via routines that are developed in python/pytorch during simulations.

   - Stability Checks: Test the numerical stability of residual calculations under various conditions.

For detailed information on each test and its purpose, please refer to the README files within the respective subdirectories.
