import torch
import numpy as np

def rotate_slip_systems(SlipSys, RM):
    return torch.einsum(
        "kij, ai, bj -> kab", SlipSys, RM, RM
    )

if __name__ == '__main__':
    # Load rotation matrix and slip systems from files
    RM = np.loadtxt('rotation.dat').reshape(3,3)
    SlipSys = np.loadtxt('slipsys.dat').reshape(24,3,3)

    # Rotate using PyTorch
    print(torch.tensor(SlipSys).dtype)
    print(torch.tensor(RM).dtype)
    rotated_slip_system = rotate_slip_systems(torch.tensor(SlipSys), torch.tensor(RM))

    # import pdb; pdb.set_trace()

    # Write results to file
    np.savetxt('output_python.dat', rotated_slip_system.numpy().reshape(72, -1))
