from __future__ import print_function, absolute_import

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

import pydrake
import pydrake.autodiffutils
from pydrake.autodiffutils import AutoDiffXd
# from pydrake.systems.framework import (
#     AbstractValue,
#     BasicVector, BasicVector_
# )
from pydrake.all import (
    AutoDiffXd, Expression, Variable,
    MathematicalProgram, SolverType, SolutionResult,
    DirectCollocationConstraint, AddDirectCollocationConstraint,
    PiecewisePolynomial,
    DiagramBuilder, SignalLogger, Simulator, VectorSystem,
)

from NNSystem import NNSystem, NNSystem_, NNInferenceHelper_double, NNInferenceHelper_autodiff, nn_loader
from networks import FC, FCBIG, MLPSMALL, MLP


# Test 1
# Try creating a simple NNSystem and show that it supports AutoDiffXd's.
# Define network, input
torch.set_default_tensor_type('torch.DoubleTensor')
net = FC(2)
net = MLP(2)
autodiff_in = np.array([
    [AutoDiffXd(1., [1., 0.]),],        
    [AutoDiffXd(1., [0., 1.]),]
])

# Make system and give input
nn_system = NNSystem_[AutoDiffXd](pytorch_nn_object=net)
context = nn_system.CreateDefaultContext()
context.FixInputPort(0, autodiff_in)

# Allocate and Eval output
output = nn_system.AllocateOutput()
nn_system.CalcOutput(context, output)
autodiff = output.get_vector_data(0).CopyToVector()[0]
print("autodiff: ", autodiff)
print("derivatives: ", autodiff.derivatives())
np.testing.assert_almost_equal(autodiff.value(), -0.0414410736543)
np.testing.assert_allclose(autodiff.derivatives(), [-0.04370906, -0.03838256])
print()


# Test 2
# Check that only difference between NNInferenceHelper_double and NNInferenceHelper_autodiff is zero gradients.
network = FCBIG(2)
in_list_double = [1., 1.]
in_list_autodiff = [
    AutoDiffXd(1., [1., 0.]),
    AutoDiffXd(1., [0., 1.])
]
out_list_double   = NNInferenceHelper_double(network,   in_list_double)
out_list_autodiff = NNInferenceHelper_autodiff(network, in_list_autodiff)
np.testing.assert_allclose(out_list_double, [elem.value() for elem in out_list_autodiff])


def finite_difference_check_autodiffs(autodiff_params=False, debug=False):
    NUM_INPUTS   = 4
    NUM_OUTPUTS  = 1
    NETS_TO_TEST = [
                    FC,
                    FCBIG,
                    MLPSMALL,
                    #MLP
                   ]

    for kNetConstructor in NETS_TO_TEST:
        if debug: np.random.seed(1), torch.manual_seed(1)

        # Make a net and calculate n_inputs and n_outputs
        network      = kNetConstructor(n_inputs=NUM_INPUTS, n_outputs=NUM_OUTPUTS)
        param_dims   = list(param.size() for param in network.parameters())
        n_params     = np.sum(np.prod(param_dim) for param_dim in param_dims)
        n_inputs     = param_dims[0][-1]
        n_outputs    = param_dims[-1][-1]
        total_params = n_inputs + n_params if autodiff_params else n_inputs
        param_list   = []
        if debug: print("net params: ", list(network.parameters()))

        def one_hot(i, N):
            ret = np.zeros(N)
            ret[i] = 1
            return ret

        # Make total_param number of AutoDiffXd's, with (seeded) random values.
        # Set derivatives array to length total_param with only index i set for ith AutoDiff.
        values = (np.ones(n_inputs) if debug else np.random.randn(n_inputs))
        in_list = [AutoDiffXd(values[i], one_hot(i, total_params)) for i in range(n_inputs)]
        if autodiff_params:
            values = np.hstack(param.clone().detach().numpy().flatten() for param in network.parameters())
            param_list = [AutoDiffXd(values[i], one_hot(n_inputs+i, total_params)) for i in range(n_params)]

        # First, generate all the AutoDiffXd outputs.
        out_list = NNInferenceHelper_autodiff(network, in_list, param_list=param_list, debug=debug)
        if debug: print("param grads: ", [param.grad for param in network.parameters()])

        # f     : function(np.array of AutoDiffXd's) -> array of size one of AutoDiffXd
        # x     : np.array of AutoDiffXd at which to calculate finite_difference
        # idx   : Index of AutoDiffXd in x to perturb
        # delta : magnitude of perturbation of AutoDiffXd at index idx of x
        def finite_difference(f, x, idx, delta=1e-7):
            x_hi = copy.deepcopy(x)
            x_hi[idx] += delta
            x_lo = copy.deepcopy(x)
            x_lo[idx] -= delta
            out_hi = np.array([elem.value() for elem in f(x_hi)])
            out_lo = np.array([elem.value() for elem in f(x_lo)])
            return (out_hi - out_lo) / (2*delta)

        # AutoDiff method returns a list of AutoDiffXd's by output.  It is
        # like a Hessian matrix that is n_output x n_input
        # Repeatedly finite differencing every input will return me vectors of
        # dervivates at each output.  This is like a Hessian that is n_input x n_output.
        # Let's take both hessians, transpose one, and compare them for equality.

        # Hessian will be of size total_params**2, let's not let that get too big.
        assert total_params < 1e3 

        # Make AutoDiffXd Hessian, (n_output x n_input)
        ad_hess = np.array([elem.derivatives() for elem in out_list])

        # Make Finite Difference Hessian, (n_input x n_output)
        def fn(x):
            # Wrapper for NNInferenceHelper_autodiff that accepts a single list of gradients and
            # assigns first n_inputs to in_list, and all the rest to param_list.
            # Also creates a fresh NN using param_list.
            in_list    = x[:n_inputs]
            param_list = x[n_inputs:]
            if param_list:
                nn_loader(param_list, network)
            return NNInferenceHelper_autodiff(network,
                                              in_list,
                                              param_list=param_list,
                                              debug=False)
        fd_hess = np.array([finite_difference(fn, in_list+param_list, inp_idx) for inp_idx in range(total_params)]) 

        # Do our comparison.
        if True:
            print('fd_hess.T: ', fd_hess.T)
            print('ad_hess:   ', ad_hess)
        np.testing.assert_allclose(fd_hess.T, ad_hess, atol=1e-3)
        print("\t ...GOOD")


# Test 3
# Use finite difference method to test AutoDiff flow from input -> output.
finite_difference_check_autodiffs(autodiff_params=False)


# Test 4
# Use finite difference method to test AutoDiff flow from param -> output.
finite_difference_check_autodiffs(autodiff_params=True, debug=False)










        # For each output, finite difference check each input and compare against derivatives vector of corresponding autodiff output.
        # for out_idx in range(n_outputs):
        #     ad_grad_all = out_list[out_idx]
        #     for inp_idx in range(total_params): 
        #         # Do finite difference calculation and compare against gradient
        #         if autodiff_params:
        #             def fn(x):
        #                 in_list = x[:n_inputs]
        #                 param_list = x[n_inputs:]
        #                 nn_loader(param_list, network)
        #                 return NNInferenceHelper_autodiff(network, in_list, param_list=param_list, debug=True)
        #         else:
        #             fn = lambda x: NNInferenceHelper_autodiff(network, x[:n_inputs])
        #         fd_grad = finite_difference(fn, np.hstack((in_list, param_list)), inp_idx)[out_idx]
        #         ad_grad = ad_grad_all.derivatives()[inp_idx]
        #         if debug: print("fd_grad: {} ad_grad: {}".format(fd_grad, ad_grad))
        #         np.testing.assert_allclose(fd_grad, ad_grad, atol=1e-3)
