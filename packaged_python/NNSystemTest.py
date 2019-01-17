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

from NNSystem import NNSystem, NNSystem_, NNInferenceHelper_double, NNInferenceHelper_autodiff
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


# Test 3
# Use finite difference method to test AutoDiff flow from input -> output.
NUM_INPUTS = 4
NETS_TO_TEST = [FC, FCBIG, MLPSMALL, MLP]
DEBUG = False
for kNetConstructor in NETS_TO_TEST:
    if DEBUG: np.random.seed(1), torch.manual_seed(1)

    # Make a net and calculate n_inputs and n_outputs
    network = kNetConstructor()
    param_dims = list(param.size() for param in network.parameters())
    n_inputs  = param_dims[0][-1]
    n_outputs = param_dims[-1][-1]

    # Make total_param number of AutoDiffXd's, with (seeded) random values.
    # Set derivatives array to length total_param with only index i set for ith AutoDiff.
    total_params = NUM_INPUTS # TODO: change this to also include num_params when you start supporting derivatives for those, too.
    values = (np.ones(total_params) if DEBUG else np.random.randn(total_params))
    def one_hot(i, n_params):
        ret = np.zeros(n_params)
        ret[i] = 1
        return ret
    in_list = np.array([AutoDiffXd(values[i], one_hot(i, total_params)) for i in range(total_params)])

    # First, generate all the AutoDiffXd outputs.
    out_list = NNInferenceHelper_autodiff(network, in_list)

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

    # For each output, finite difference check each input and compare against derivatives vector of corresponding autodiff output.
    for out_idx in range(n_outputs):
        ad_grad_all = out_list[out_idx]
        for inp_idx in range(total_params): 
            # Do finite difference calculation and compare against gradient
            fd_grad = finite_difference(lambda x: NNInferenceHelper_autodiff(network, x), in_list, inp_idx)
            ad_grad = ad_grad_all.derivatives()[inp_idx]
            if DEBUG: print("fd_grad: {} ad_grad: {}".format(fd_grad, ad_grad))
            np.testing.assert_allclose(fd_grad, ad_grad, atol=1e-3)
    print("\t ...GOOD")


# Use finite difference method to test AutoDiff flow from param -> output.









