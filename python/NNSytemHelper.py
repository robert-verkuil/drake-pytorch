import numpy as np
import sys
import torch

from pydrake.all import (
    AutoDiffXd,
    BasicVector, BasicVector_,
    LeafSystem, LeafSystem_,
    PortDataType,
)
from pydrake.systems.scalar_conversion import TemplateSystem


def NNInferenceHelper(network, in_np_data):
    '''
    u (inputs) --> NN => y (outputs)
                   ^
                   |
           context.p (parameters)
    '''
#    drake_in = self.EvalVectorInput(context, 0)
#    # p = context.GetParams() #TODO: figure out a way to deal with parameters
#
#    # Convert input to a torch tensor
#    print("drake_in: ", drake_in)
#    n_inputs = drake_in.size()
#    # import pdb; pdb.set_trace()

    just_values = np.array([in_np_data.value() for i in range(n_inputs)])
    torch_in = torch.tensor(just_values, requires_grad=True)
    print("torch_in: ", torch_in) 

    # Run the forward pass.
    # We'll do the backward pass(es) when we calculate output and it's gradients.
    torch_out = network.forward(torch_in)
    print("torch_out: ", torch_out)
    
    # Currently we only support one output
    assert torch_out.shape[0] == 1, "Need just one output for valid cost function"
    
    # Do derivative calculation and pack into the output vector.
    # Because neural network might have multiple outputs, I can't simply use net.backward() with no argument.
    # Instead need to follow the advice here:
    #     https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
    n_outputs = torch_out.shape[0]
    out_np_data = [0]*n_outputs

    for j in range(n_outputs):
        print("\niter j: ", j)
        y_j_value = torch_out[j].clone().detach().numpy()
        # Equation: y.derivs = dydu*u.derivs() + dydp*p.derivs()
        # Alternate equation, for each y, y_j.deriv = sum_i  (dy_jdu[i] * u[i].deriv)
        #                          (#y's) (1x#derivs) (#u's)  (1x#u's)    (1x#derivs)

        # Make empty accumulator
        y_j_deriv = np.zeros_like(in_np_data[0].derivatives())
        
        # https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
        output_selector = torch.zeros(1, n_outputs, dtype=torch.double)
        output_selector[j] = 1.0 # Set the output we want a derivative w.r.t. to 1.
        torch_out.backward(output_selector, retain_graph=True)
        dy_jdu = torch_in.grad.numpy() # From Torch, give us back a numpy object
        for i in range(n_inputs):
            u_i_deriv = in_np_data[i].derivatives()
            print("dy_jdu_a[i] * u_i_deriv = ", dy_jdu[i], " * ",  u_i_deriv)
            y_j_deriv += dy_jdu[i] * u_i_deriv;
        print("putting into output: ", y_j_value, ", ", y_j_deriv)
        tmp = AutoDiffXd(y_j_value, y_j_deriv)
        out_np_data[j] = tmp

        # output.SetAtIndex(j, tmp)

    return out_np_data


