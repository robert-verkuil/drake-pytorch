# High-level imports test
import pydrake
import pydrake.autodiffutils
from pydrake.autodiffutils import AutoDiffXd

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

from NNSystem import NNSystem, NNSystem_
from NNTestSetup import NNTestSetup

# Giving NNSystem a pytorch net.
class FC(nn.Module):
    def __init__(self, layer_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, layer_norm=False):
        super(MLP, self).__init__()
        self.layer_norm = layer_norm

        self.l1 = nn.Linear(2, 64)
        self.ln1 = nn.LayerNorm(64)
        self.tanh1 = F.tanh
        self.l2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.tanh2 = F.tanh
        self.l3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.l1(x)
        if self.layer_norm: x = self.ln1(x)
        x = self.tanh1(x)
        x = self.l2(x)
        if self.layer_norm: x = self.ln2(x)
        x = self.tanh2(x)
        x = self.l3(x)
        return x
    
# nn_test_setup = NNTestSetup(pytorch_nn_object=MLP())
# nn_test_setup.RunSimulation()

from pydrake.systems.framework import (
    AbstractValue,
    BasicVector, BasicVector_
)

net = FC()
net = MLP()
autodiff_in = np.array([
    [AutoDiffXd(1.0, [1.0, 0.0]),],        
    [AutoDiffXd(1.0, [1.0, 0.0]),]
])
print(type(autodiff_in[0]))
nn_system = NNSystem_[AutoDiffXd](pytorch_nn_object=net)
context = nn_system.CreateDefaultContext()
autodiff_in = np.array([AutoDiffXd(x) for x in [1., 2.]])
# autodiff_in = BasicVector([1., 2.])
context.FixInputPort(0, autodiff_in)

output = nn_system.AllocateOutput()
nn_system.CalcOutput(context, output)
# import pdb; pdb.set_trace()
value = output.get_vector_data(0).CopyToVector()
# _get_value_copy() # Might want to use this line above ^
print("output: ", value)




