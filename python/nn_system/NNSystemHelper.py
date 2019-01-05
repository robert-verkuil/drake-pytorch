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
    def __init__(self, n_inputs=4):
        super(FC, self).__init__()
        self.n_inputs = n_inputs
        self.fc1 = nn.Linear(self.n_inputs, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class FCBIG(nn.Module):
    def __init__(self, n_inputs=4):
        super(FCBIG, self).__init__()
        self.n_inputs = n_inputs
        self.fc2 = nn.Linear(self.n_inputs, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

class MLPSMALL(nn.Module):
    def __init__(self, n_inputs=4, layer_norm=False):
        super(MLPSMALL, self).__init__()
        self.n_inputs = n_inputs
        self.layer_norm = layer_norm

        h_sz = 16
        self.l1 = nn.Linear(self.n_inputs, h_sz)
        self.ln1 = nn.LayerNorm(h_sz)
        self.tanh1 = torch.tanh
        self.l3 = nn.Linear(h_sz, 1)
    
    def forward(self, x):
        x = self.l1(x)
        if self.layer_norm: x = self.ln1(x)
        x = self.tanh1(x)
        x = self.l3(x)
        return x

class MLP(nn.Module):
    def __init__(self, n_inputs=4, layer_norm=False):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.layer_norm = layer_norm

        h_sz = 16
        self.l1 = nn.Linear(self.n_inputs, h_sz)
        self.ln1 = nn.LayerNorm(h_sz)
        self.tanh1 = torch.tanh
        self.l2 = nn.Linear(h_sz, h_sz)
        self.ln2 = nn.LayerNorm(h_sz)
        self.tanh2 = torch.tanh
        self.l3 = nn.Linear(h_sz, 1)
    
    def forward(self, x):
        x = self.l1(x)
        if self.layer_norm: x = self.ln1(x)
        x = self.tanh1(x)
        x = self.l2(x)
        if self.layer_norm: x = self.ln2(x)
        x = self.tanh2(x)
        x = self.l3(x)
        return x

class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 84)
        self.fc2 = nn.Linear(84, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def make_NN_constraint(kNetConstructor, num_inputs, num_states, num_params):
    def constraint(uxT):
    # ##############################
    # #    JUST FOR DEBUGGING!
    # ##############################
    #     for elem in uxT:
    #         print(elem.derivatives())
    #     print()
    #     return uxT
    # prog.AddConstraint(constraint, -np.array([.1]*5), np.array([.1]*5), np.hstack((prog.input(0), prog.state(0))))
    # ##############################

        u = uxT[:num_inputs]
        x = uxT[num_inputs:num_inputs+num_states]
        T = uxT[num_inputs+num_states:]
        
        # We assume that .derivative() at index i 
        # of uxT is a one hot with only derivatives(i) set. Check this here.
        n_derivatives = len(u[0].derivatives())
        assert n_derivatives == sum((num_inputs, num_states, num_params))
        for i, elem in enumerate(uxT): # TODO: is uxT iterable without any reshaping?
            assert n_derivatives == len(elem.derivatives())
            one_hot = np.zeros((n_derivatives))
            one_hot[i] = 1
            np.testing.assert_array_equal(elem.derivatives(), one_hot)
        
        # Construct a model with params T
        net = kNetConstructor()
        params_loaded = 0
        for param in net.parameters():
            T_slice = np.array([T[i].value() for i in range(params_loaded, params_loaded+param.data.nelement())])
            param.data = torch.from_numpy(T_slice.reshape(list(param.data.size())))
            # print("param.data: ", param.data)
            # print("loaded param shape: ", list(param.data.size()))
            params_loaded += param.data.nelement() 
            
        # Do forward pass.
        x_values = np.array([elem.value() for elem in x])
        torch_in = torch.tensor(x_values, requires_grad=True)
        torch_out = net.forward(torch_in)
        y_values = torch_out.clone().detach().numpy()
        
        # Calculate all gradients w.r.t. single output.
        # WARNING: ONLY WORKS IF NET HAS ONLY ONE OUTPUT.
        torch_out.backward(retain_graph=True)
        
        y_derivatives = np.zeros(n_derivatives)
        y_derivatives[:num_inputs] = np.zeros(num_inputs) # No gradient w.r.t. inputs yet.
        # print("torch_in.grad: ", torch_in.grad)
        y_derivatives[num_inputs:num_inputs+num_states] = torch_in.grad
    #     print("hi: ", [(-1 if param.grad is None else param.grad.size()) for param in net.parameters()])
        T_derivatives = []
        for param in net.parameters():
            if param.grad is None:
                # print("got grad of None, making tensor of ", param.data.nelement(), " zeros")
                T_derivatives.append( [0.]*param.data.nelement() )
                # print("extracted (0) data shape: ", list(param.data.size()))
            else:
                # print("got grad, making tensor of size", param.grad.nelement())
		assert param.data.nelement() == param.grad.nelement()
		np.testing.assert_array_equal( list(param.data.size()), list(param.grad.size()) )
                T_derivatives.append( param.grad.numpy().flatten() ) # Flatten will return a copy.
                # print("extracted grad shape: ", list(param.grad.size()))
        assert np.hstack(T_derivatives).size == num_params
        y_derivatives[num_inputs+num_states:] =  np.hstack(T_derivatives)
        
        y = AutoDiffXd(y_values, y_derivatives)
        
        # constraint is Pi(x) == u
        return u - y
    return constraint


def NNInferenceHelper(network, in_list, param_list, debug=False):
    '''
    u (inputs) --> NN => y (outputsres_grad=Truee
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

#    n_params = len(param_list)
#    param_values = np.array([[param.value() for param in param_list])
#    # Now assign the param values to the network!
#    # TODO

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
        # Alternate equation, for each y, y_j.deriv = sum_i  (dy_jdu[i] * u[i].deriv) + sum_k  (dy_jdp[k] * p[k].deriv)
        #                          (#y's) (1x#derivs) (#u's)  (1x#u's)    (1x#derivs)   (#p's)  (1x#p's)    (1x#derivs)

        # Make empty accumulator
        y_j_deriv = np.zeros_like(in_list[0].derivatives())
        
        # Fill the graph with gradients with respect to output y_j.
        # https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
        output_selector = torch.zeros(1, n_outputs, dtype=torch.double)
        output_selector[j] = 1.0 # Set the output we want a derivative w.r.t. to 1.
        torch_out.backward(output_selector, retain_graph=True)

        # Calculate the contribution to y_j.derivs w.r.t all the inputs.
        dy_jdu = torch_in.grad.numpy() # From Torch, give us back a numpy object
        for i in range(n_inputs):
            u_i_deriv = in_list[i].derivatives()
            if debug: print("dy_jdu_a[i] * u_i_deriv = ", dy_jdu[i], " * ",  u_i_deriv)
            y_j_deriv += dy_jdu[i] * u_i_deriv;

        # Calculate the contribution to y_j.derivs w.r.t all the params.
        # Get the gradient
#        for k in range(n_params):
#            p_k_deriv = param_list[k].derivatives()
#            if debug: print("dy_jdp_a[k] * p_k_deriv = ", dy_jdp[k], " * ",  u_i_deriv)
#            y_j_deriv += dy_jdu[i] * u_i_deriv;


        if debug: print("putting into output: ", y_j_value, ", ", y_j_deriv)
        tmp = AutoDiffXd(y_j_value, y_j_deriv)
        out_list[j] = tmp

        # output.SetAtIndex(j, tmp)

    return out_list


