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

@TemplateSystem.define("NNSystem_", T_list=[float, AutoDiffXd])
def NNSystem_(T):

    class Impl(LeafSystem_[T]):
        '''
        Python implementation of a Drake neural network system.
        Powered by PyTorch.
        '''
        def _construct(self, pytorch_nn_object, converter=None):
            '''
                pytorch_nn_object: a function(inputs) -> outputs
            '''
            LeafSystem_[T].__init__(self, converter=converter)

            # Getting relavent qtys out of pytorch_nn_object, for port setup.
            self.network = pytorch_nn_object.double() # Needs to support an Eval method?
            self.n_inputs = list(pytorch_nn_object.parameters())[0].size()[-1]
            print(self.n_inputs)
            self.n_outputs = 1

            # Input Ports
            self.NN_in_input_port = \
                self._DeclareInputPort(
                    "NN_in", PortDataType.kVectorValued, self.n_inputs)

            # Output Ports
            self.NN_out_output_port = \
                self._DeclareVectorOutputPort(
                    "NN_out", BasicVector_[T](self.n_outputs), self.EvalOutput)

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self, other.network, converter=converter)


        def EvalOutput(self, context, output):
            '''
            u (inputs) --> NN => y (outputs)
                           ^
                           |
                   context.p (parameters)
            '''
            drake_in = self.EvalVectorInput(context, 0)
            # p = context.GetParams() #TODO: figure out a way to deal with parameters
            assert drake_in.size() == self.n_inputs
            assert self.n_outputs == 1

            # Pack input
            in_list = []
            for i in range(self.n_inputs):
                in_list.append(drake_in.GetAtIndex(i))

            # Call the helper here to do the heavy lifting.
            out_list = NNInferenceHelper_double(self.network, in_list)

            # Pack output
            for j in range(self.n_outputs):
                output.SetAtIndex(j, out_list[j])

            # Do we even need to return anything here?
            return out_list[0]

    return Impl


# Default instantiations.
NNSystem = NNSystem_[None]

# TODO: Make this a python template instead of two different functions?
def NNInferenceHelper_double(network, in_list, debug=False):
    # Ensure that all network weights are doubles (TODO: change this in the future!)
    network = network.double()

    # Process input
    n_inputs = len(in_list)
    torch_in = torch.tensor(np.array(in_list), dtype=torch.double)
    if debug: print("torch_in: ", torch_in) 

    # Run the forward pass
    torch_out = network.forward(torch_in)
    if debug: print("torch_out: ", torch_out)
    
    # Process output
    n_outputs = torch_out.shape[0]
    assert n_outputs == 1, "Need just one output for valid cost function"
    out = torch_out[0].clone().detach().numpy()

    return [out]


# TODO: CLEAN UP THIS HUGEEEEE MESS...
def NNInferenceHelper_autodiff(network, in_list, debug=False):
    # Ensure that all network weights are doubles (TODO: change this in the future!)
    network = network.double()
#    drake_in = self.EvalVectorInput(context, 0)
#    # p = context.GetParams() #TODO: figure out a way to deal with parameters
#
#    # Convert input to a torch tensor
#    print("drake_in: ", drake_in)
#    # import pdb; pdb.set_trace()

    n_inputs = len(in_list)
    #just_values = np.array([item.value() for item in in_list])
    just_values = np.array([item for item in in_list])
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

    # REMOVE THIS!!!
    out_list[0] = torch_out[0].clone().detach().numpy()
    return out_list
    # REMOVE THIS!!!

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


