import numpy as np
import torch

import umat.umat as umat


if __name__ == "__main__":
    angles = np.loadtxt("angles.dat")
    print(f"numpy dtype: {angles.dtype}")
    print(f"torch dtype: {torch.tensor(angles).dtype}")
    RM = umat.rotation_matrix(torch.tensor(angles))

    np.savetxt("rotation_matrix_python.dat", RM.numpy(), fmt="%.16e")
