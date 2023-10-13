# Consistency Test: Computing the Rotation Matrix; Fortran vs Pytorch

## Overview
This test suite validates the consistency between the computation of the rotation matrix in Fortran and in PyTorch. Together with the consistency of summation (see `einsum_test` for details), this test represents one of the few tests that cannot be conducted entirely in Python. As such, we need to compute the results both using the Fortran code and our Python approach. Through this test, it was discovered that the results inside the PyTorch function are truncated to `float32`, as the individual rotation matrices are initiated using the default dtype:

```python
    R1 = torch.zeros([batch_size, 3, 3]) # This defaults to `float32` regardless of input dtype
```

To address this, one can use the dtype of the input when initializing the matrices (this applies to `R2` and `R3` as well):

```python
    R1 = torch.zeros([batch_size, 3, 3]).to(theta.dtype) # This ensures consistency with the input's dtype
```

## Results Comparison

The test shows that the outputs from both Fortran and Python/PyTorch are within the margin of machine epsilon for double precision. 

## Test Workflow
The workflow is largely inspired by the `einsum_test` suite. It involves a Fortran file reading the predefined angles and writing them to a data file, `angles.dat`. This file serves as an input for both `rotation_matrix_fortran.f90` and `rotation_matrix_python.py`, each generating the rotation matrix using their respective methods. The results are saved into `rotation_matrix_fortran.dat` and `rotation_matrix_python.dat`. Subsequently, `compare_results.py` contrasts these results and prints the relative difference between the two. 

## How to Run

The entire test workflow is streamlined using a Makefile. To execute the test:

```bash
    make all
```

This command will:

- Compile the Fortran programs.
- Execute the Fortran and Python rotation calculations.
- Compare their results using `compare_results.py`.
- Display the relative error, showcasing the consistency between the two methods.

To clean up the generated results and artifacts, use:

```bash
    make clean
```
