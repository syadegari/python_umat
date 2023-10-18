import numpy as np


if __name__ == "__main__":
    # Load Fortran and PyTorch results
    fortran_output = np.loadtxt("rotation_matrix_fortran.dat")
    python_output = np.loadtxt("rotation_matrix_python.dat")

    # Compute relative norm of the difference
    difference = np.linalg.norm(fortran_output - python_output) / np.linalg.norm(fortran_output)

    print(
        "Relative difference between Fortran and PyTorch results for the rotation matrix:"
        f" {difference}"
    )
