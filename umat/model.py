import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()

        self.f_theta = nn.Sequential(
            nn.Linear(3, 16),
            activation_fn,
            nn.Linear(16, 32),
        )
        self.f_defGrad = nn.Sequential(
            nn.Linear(18, 32),
            activation_fn,
            nn.Linear(32, 64),
            activation_fn,
            nn.Linear(64, 32),
        )
        self.f_intvars = nn.Sequential(
            nn.Linear(48, 64),
            activation_fn,
            nn.Linear(64, 128),
            activation_fn,
            nn.Linear(128, 64),
            activation_fn,
            nn.Linear(64, 32),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.flatten = nn.Linear(640, 48)

    def forward(self, theta, defgrad0, defgrad1, gamma0, slip_res0):
        x1 = self.f_theta(theta)
        x2 = self.f_defGrad(torch.cat((defgrad0, defgrad1), dim=1))
        x3 = self.f_intvars(torch.cat((gamma0, slip_res0), dim=1))

        conv_output = self.conv_layers(torch.stack((x1, x2, x3), dim=1))

        out = self.flatten(conv_output.flatten(start_dim=1, end_dim=2))
        return out[:, :24], out[:, 24:]
