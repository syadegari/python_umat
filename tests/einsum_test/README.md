# Consistency Test: Fortran Summation vs PyTorch Einsum

## Overview

This test suite validates the consistency between the summation logic used in Fortran and the `einsum` function in PyTorch. By ensuring their results are identical, we can confidently interchange the fortran cumbersome, explicit summation, with einsum for differentiations. We use this as an evidence to interchange other explicit summation with `einsum`, e.g., in rotation of the elasticity tensor or computing the stress from contraction between the elasticity tensor and the Green-Lagrange strain.

## Test Workflow

1. **Data Generation**:
    - Two Fortran files are included in the repository. These files provide:
      - The slip system `slip_system.f90`.
      - A randomly generated orientation `orientation.f90`.
    - From this orientation, we calculate the rotation matrix and save it into `rotation.dat`. We also save the slip systems in a column format that can be consumed later by both python and fortran file, under the name `slipsys.dat`.

2. **Rotation Calculations**:
    - `rotate_fortran.f90` reads the rotation matrix and slip system, and then calculates the rotated slip systems using explicit summation in Fortran.
    - `rotate_python.f90` reads the same data and calculates the rotated slip systems using the `einsum` function from PyTorch.

3. **Results Comparison**:
    - `compare_results.py` compares the rotated slip systems produced by the Fortran and PyTorch routines.
    - The comparison outputs the relative difference between the two results.
    - The test shows a relative difference on the order of \(1 \times 10^{-16}\), approaching the machine epsilon for double precision (or float64) arithmetic, meaning the results are effectively identical within the precision limits of the representation.


## How to Run

The entire test workflow is automated using a Makefile. To execute the test:

```bash
make all
```

This command will:

- Compile the Fortran programs.
- Run the Fortran and Python rotation calculations.
- Compare their results using `compare_results.py`.
- Output the relative error, showing the consistency between the two methods.

To clean the generated results use:

```bash
make clean
```

