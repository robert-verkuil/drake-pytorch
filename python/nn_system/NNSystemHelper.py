import numpy as np
import sys
import time

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

# Import all the networks definitions we've made in the other file.
from networks import *

torch.set_default_tensor_type('torch.DoubleTensor')


def add_nn_params(prog,
                  kNetConstructor, 
                  h, u, x, # TODO: remove a BUNCH of this!
                  num_inputs, num_states, 
                  num_trajectories,
                  num_samples,
                  initialize_params=True, 
                  reg_type="No", 
                  enable_constraint=True):
    # Determine num_params and add them to the prog.
    dummy_net = kNetConstructor()
    num_params = sum(tensor.nelement() for tensor in kNetConstructor().parameters())
    T = prog.NewContinuousVariables(num_params, 'T')

    if initialize_params:
        ### initalize_T_variables(prog, T, kNetConstructor, num_params):
        # VERY IMPORTANT!!!! - PRELOAD T WITH THE NET'S INITIALIZATION.
        # DEFAULT ZERO INITIALIZATION WILL GIVE YOU ZERO GRADIENTS!!!!
        params_loaded = 0
        initial_guess = [AutoDiffXd]*num_params
        for param in kNetConstructor().parameters(): # Here's where we make a dummy net. Let's seed this?
            param_values = param.data.numpy().flatten()
            for i in range(param.data.nelement()):
                initial_guess[params_loaded + i] = param_values[i]
            params_loaded += param.data.nelement()
        prog.SetInitialGuess(T, np.array(initial_guess))

    # Add No/L1/L2 Regularization to model params T
    ### add_regularization(prog, T, reg_type):
    assert reg_type in ("No", "L1", "L2")
    if reg_type == "No":
        pass
    elif reg_type == "L1":
        def L1Cost(T):
            return sum([t**2 for t in T])
        prog.AddCost(L1Cost, T)
    elif reg_type == "L2":
        prog.AddQuadraticCost(np.eye(len(T)), [0.]*len(T), T)


    if enable_constraint:
        # Add the neural network constraint to all time steps
        for ti in range(num_trajectories):
            for i in range(num_samples):
                u_ti = u[ti][0,i]
                x_ti = x[ti][:,i]
                # Only one output value, so let's have lb and ub of just size one!
                constraint = make_NN_constraint(kNetConstructor, num_inputs, num_states, num_params)
                lb         = np.array([-.1])
                ub         = np.array([.1])
                var_list   = np.hstack((u_ti, x_ti, T))
                prog.AddConstraint(constraint, lb, ub, var_list)
        #         prog.AddCost(lambda x: constraint(x)[0]**2, var_list)

    return T # TODO: remove this

# TODO: use this in place of create_nn_system thing in traj.vis
def create_nn(kNetConstructor, params_list):
    # Construct a model with params T
    net = kNetConstructor()
    params_loaded = 0
    for param in net.parameters():
        param_slice = np.array([params_list[i] for i in range(params_loaded, params_loaded+param.data.nelement())])
        param.data = torch.from_numpy(param_slice.reshape(list(param.data.size())))
        params_loaded += param.data.nelement() 

    return net

import copy
def make_NN_constraint(kNetConstructor, num_inputs, num_states, num_params, do_asserts=False):
    def constraint(uxT):
        # start = time.time()
    # ##############################
    # #    JUST FOR DEBUGGING!
    # ##############################
    #     for elem in uxT:
    #         print(elem.derivatives())
    #     print()
    #     return uxT
    # prog.AddConstraint(constraint, -np.array([.1]*5), np.array([.1]*5), np.hstack((prog.input(0), prog.state(0))))
    # ##############################
        # Force use of AutoDiff Values, so that EvalBinding works (it normally only uses doubles...)
        double_ver = False
        if uxT.dtype != np.object:
            double_ver = True
            uxT = copy.deepcopy(uxT)
            def one_hot(i):
                ret = np.zeros(len(uxT))
                ret[i] = 1
                return ret
            uxT = np.array([AutoDiffXd(val, one_hot(i)) for i, val in enumerate(uxT)])

        u = uxT[:num_inputs]
        x = uxT[num_inputs:num_inputs+num_states]
        T = uxT[num_inputs+num_states:]
        
        # We assume that .derivative() at index i 
        # of uxT is a one hot with only derivatives(i) set. Check this here.
        n_derivatives = len(u[0].derivatives())
        if do_asserts: assert n_derivatives == sum((num_inputs, num_states, num_params))
        for i, elem in enumerate(uxT): # TODO: is uxT iterable without any reshaping?
            if do_asserts: assert n_derivatives == len(elem.derivatives())
            one_hot = np.zeros((n_derivatives))
            one_hot[i] = 1
            if do_asserts: np.testing.assert_array_equal(elem.derivatives(), one_hot)
        
        # Construct a model with params T
        net = kNetConstructor()
        params_loaded = 0
        for param in net.parameters():
            T_slice = np.array([T[i].value() for i in range(params_loaded, params_loaded+param.data.nelement())])
            param.data = torch.from_numpy(T_slice.reshape(list(param.data.size())))
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
        y_derivatives[num_inputs:num_inputs+num_states] = torch_in.grad
        T_derivatives = []
        for param in net.parameters():
            if param.grad is None:
                T_derivatives.append( [0.]*param.data.nelement() )
            else:
                if do_asserts: assert param.data.nelement() == param.grad.nelement()
                if do_asserts: np.testing.assert_array_equal( list(param.data.size()), list(param.grad.size()) )
                T_derivatives.append( param.grad.numpy().flatten() ) # Flatten will return a copy.
        if do_asserts: assert np.hstack(T_derivatives).size == num_params
        y_derivatives[num_inputs+num_states:] =  np.hstack(T_derivatives)
        
        y = AutoDiffXd(y_values, y_derivatives)
        
        # constraint is Pi(x) == u
        # end = time.time()
        # print("constraint eval: ", end - start)
        if double_ver:
            ret = (u-y)[0].value()
            #print("ret: ", ret)
            return [ret]
        return u - y
    return constraint


