import numpy as np


if __name__ == '__main__':
    # Load Fortran and PyTorch results
    fortran_output = np.loadtxt('output_fortran.dat').reshape(24,3,3)
    python_output = np.loadtxt('output_python.dat').reshape(24,3,3)

    # Compute relative norm of the difference
    difference = np.linalg.norm(fortran_output - python_output) / np.linalg.norm(fortran_output)

    print(f"Relative difference between Fortran and PyTorch results for rotated slip systems: {difference}")
