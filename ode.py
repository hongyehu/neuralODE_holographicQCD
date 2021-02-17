import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F

from torchdiffeq import odeint_adjoint as odeint

class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        self.poly_para = nn.Parameter(
            torch.tensor([
                1.2679, 1.7096, 0.8892, 0.8964,1.7631,0.0414,1.1214,0.3398,1.3928
            ]))
#         self.poly_para = nn.Parameter(torch.FloatTensor(9).uniform_(0,2))
    def forward(self, x):
        return (self.poly_para[0] * x**8 + self.poly_para[1] * x**7 +
                self.poly_para[2] * x**6 + self.poly_para[3] * x**5 +
                self.poly_para[4] * x**4 + self.poly_para[5] * x**3 +
                self.poly_para[6] * x**2 + self.poly_para[7] * x**1 +
                self.poly_para[8])

class ODEFunc(nn.Module):
    def __init__(self, metric_module):
        super(ODEFunc, self).__init__()
        self.metric = metric_module
        # lmd > 0
        self.lmd = nn.Parameter(torch.tensor([0.005]))
        self.L = nn.Parameter(torch.tensor([0.7800]))

    def forward(self, t, y):
        """
        dim(y) = [batch_dim, 2 (phi, pi)]
        """
        h = self.metric(t)
        output = torch.stack(
            [-y[:, 1], h * y[:, 1] + 3 * y[:, 0] - self.lmd * y[:, 0]**3],
            dim=1)
        return output

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, z0, t, return_whole_sequence=False):
        z = odeint(self.func, z0, t)
        if return_whole_sequence:
            return z
        else:
            return z[-1]



            