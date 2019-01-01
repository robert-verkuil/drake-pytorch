import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pydrake.all import (
    AutoDiffXd,
    BasicVector, BasicVector_,
    LeafSystem, LeafSystem_,
    PortDataType,
)
from pydrake.systems.scalar_conversion import TemplateSystem

torch.set_default_tensor_type('torch.DoubleTensor')

class FC(nn.Module):
    def __init__(self, layer_norm=False):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    def __init__(self, layer_norm=False):
        super(MLP, self).__init__()
        self.layer_norm = layer_norm

        self.l1 = nn.Linear(2, 64)
        self.ln1 = nn.LayerNorm(64)
        self.tanh1 = torch.tanh
        self.l2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.tanh2 = torch.tanh
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

def NNInferenceHelper(network, in_list, debug=False):
    '''
    u (inputs) --> NN => y (outputs)
                   ^
                   |
           context.p (parameters)
    '''
    # Ensure that all network weights are doubles (TODO: change this in the future!)
    network = network.double()
#    drake_in = self.EvalVectorInput(context, 0)
#    # p = context.GetParams() #TODO: figure out a way to deal with parameters
#
#    # Convert input to a torch tensor
#    print("drake_in: ", drake_in)
#    # import pdb; pdb.set_trace()

    n_inputs = len(in_list)
    just_values = np.array([item.value() for item in in_list])
    torch_in = torch.tensor(just_values, dtype=torch.double, requires_grad=True)
    if debug: print("torch_in: ", torch_in) 

    # Run the forward pass.
    # We'll do the backward pass(es) when we calculate output and it's gradients.
    torch_out = network.forward(torch_in)
    if debug: print("torch_out: ", torch_out)
    
    # Currently we only support one output
    assert torch_out.shape[0] == 1, "Need just one output for valid cost function"
    
    # Do derivative calculation and pack into the output vector.
    # Because neural network might have multiple outputs, I can't simply use net.backward() with no argument.
    # Instead need to follow the advice here:
    #     https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
    n_outputs = torch_out.shape[0]
    out_list = [0]*n_outputs

    for j in range(n_outputs):
        if debug: print("\niter j: ", j)
        y_j_value = torch_out[j].clone().detach().numpy()
        # Equation: y.derivs = dydu*u.derivs() + dydp*p.derivs()
        # Alternate equation, for each y, y_j.deriv = sum_i  (dy_jdu[i] * u[i].deriv)
        #                          (#y's) (1x#derivs) (#u's)  (1x#u's)    (1x#derivs)

        # Make empty accumulator
        y_j_deriv = np.zeros_like(in_list[0].derivatives())
        
        # https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
        output_selector = torch.zeros(1, n_outputs, dtype=torch.double)
        output_selector[j] = 1.0 # Set the output we want a derivative w.r.t. to 1.
        torch_out.backward(output_selector, retain_graph=True)
        dy_jdu = torch_in.grad.numpy() # From Torch, give us back a numpy object
        for i in range(n_inputs):
            u_i_deriv = in_list[i].derivatives()
            if debug: print("dy_jdu_a[i] * u_i_deriv = ", dy_jdu[i], " * ",  u_i_deriv)
            y_j_deriv += dy_jdu[i] * u_i_deriv;
        if debug: print("putting into output: ", y_j_value, ", ", y_j_deriv)
        tmp = AutoDiffXd(y_j_value, y_j_deriv)
        out_list[j] = tmp

        # output.SetAtIndex(j, tmp)

    return out_list


