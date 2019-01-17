import numpy as np
import sys
import torch

from pydrake.all import (
    AbstractValue,
    AutoDiffXd,
    BasicVector, BasicVector_,
    Context,
    LeafSystem, LeafSystem_,
    PortDataType,
)
from pydrake.systems.scalar_conversion import TemplateSystem

def np_hash(np_obj):
    return hash(np_obj.to_string())

@TemplateSystem.define("NNSystem_", T_list=[float, AutoDiffXd])
def NNSystem_(T):

    class Impl(LeafSystem_[T]):
        '''
        Python implementation of a Drake neural network system.
        Powered by PyTorch.

        PyTorch net parameters can be optonally added to Context 
        as a flat list of float or AutoDiffXd.
        Changes to context parameters will be 'synced' to
        network parameters before each EvalOutput().

        The user can set the context parameters to be AutoDiffXd's to calculate
        gradients w.r.t. network parameters.
        '''
        def _construct(self, pytorch_nn_object, declare_params=False, converter=None):
            '''
                pytorch_nn_object: a function(inputs) -> outputs
            '''
            LeafSystem_[T].__init__(self, converter=converter)

            # Getting relavent qtys out of pytorch_nn_object, for port setup.
            self.network = pytorch_nn_object.double()
            param_dims = list(param.size() for param in pytorch_nn_object.parameters())
            self.n_inputs  = param_dims[0][-1]
            self.n_outputs = param_dims[-1][-1]

            # Optionally expose parameters in Context.
            # TODO(rverkuil): Expose bindings for DeclareNnumericParameter and use that here.
            self.declare_params = declare_params
            self.params = None
            if self.declare_params:
                params = np.hstack([param.clone().detach().numpy().flatten() for param in self.network.parameters()])
                self.set_params(params)

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
                self, other.network, other.declare_params, converter=converter)

        def get_params(self):
            return self.params

        def set_params(self, params):
            self.params = params
            self.param_hash = np_hash(self.params)

        def EvalOutput(self, context, output):
            '''
            u (inputs) --> NN => y (outputs)
                           ^
                           |
                   context.p (parameters)
            '''
            drake_in = self.EvalVectorInput(context, 0)
            assert drake_in.size() == self.n_inputs

            # Possibly sync context.parameters -> self.network.parameters
            if self.declare_params and np_hash(self.params) != self.param_hash:
                params_loaded = 0
                for param in self.network.parameters():
                    T_slice = np.array([self.get_params()[i].value() for i in range(params_loaded, params_loaded+param.data.nelement())])
                    param.data = torch.from_numpy(T_slice.reshape(list(param.data.size())))
                    params_loaded += param.data.nelement() 

            # Pack input
            in_list = []
            for i in range(self.n_inputs):
                in_list.append(drake_in.GetAtIndex(i))

            # Call a helper here to do the heavy lifting.
            # 3 cases:
            #    1) AutoDiffXd, Params in context and AutoDiffXd's  = Can flow derivs of inputs and params.
            #    2) AutoDiffXd, Params are floats or not in context = Can flow derivs of inputs only.
            #    3) Double                                          = No derivs
            if isinstance(in_list[0], AutoDiffXd):
                if self.declare_params and isinstance(self.get_params()[0], AutoDiffXd):
                    # Make sure derivatives vectors have the same size.
                    assert len(in_list[0].derivatives()) == len(self.get_params()[0].derivatives())
                    out_list = NNInferenceHelper_autodiff(self.network, in_list, param_list=self.get_params())
                else:
                    out_list = NNInferenceHelper_autodiff(self.network, in_list)
            else:
                out_list = NNInferenceHelper_double(self.network, in_list)

            # Pack output
            for j in range(self.n_outputs):
                output.SetAtIndex(j, out_list[j])

            # Do we even need to return anything here?
            return out_list[0]

    return Impl

# Default instantiation.
NNSystem = NNSystem_[None]

def NNInferenceHelper_double(network, in_list, debug=False):
    # Ensure that all network weights are doubles.
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


def NNInferenceHelper_autodiff(network, in_list, param_list=None, debug=False):
    # Ensure that all network weights are doubles.
    network = network.double()

    n_inputs = len(in_list)
    just_values = np.array([item.value() for item in in_list])
    torch_in = torch.tensor(just_values, dtype=torch.double, requires_grad=True)
    if debug: print("torch_in: ", torch_in) 

    if param_list is not None:
        assert sum(param.nelement() for param in network.parameters()) == len(param_list)
        assert isinstance(param_list[0], AutoDiffXd)
        n_params = len(param_list)
        # param_values = np.array([param.value() for param in param_list])

    # Run the forward pass.
    # We'll do the backward pass(es) when we calculate output and it's gradients.
    torch_out = network.forward(torch_in)
    if debug: print("torch_out: ", torch_out)
    
    # Do derivative calculation and pack into the output vector.
    # Because neural network might have multiple outputs, I can't simply use net.backward() with no argument.
    # Instead need to follow the advice here:
    #     https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
    n_outputs = torch_out.shape[0]

    # Ready a container for our output AutoDiffXd's
    out_list = [0]*n_outputs

    # Calculate the AutoDiff Vector for each output of the neural network.
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
        output_selector[j] = 1. # Set the output we want a derivative w.r.t. to 1.
        torch_out.backward(output_selector, retain_graph=True) # TODO!! will need to zero some gradients when i test multiple outputs I think!!!

        # Calculate the contribution to y_j.derivs w.r.t all the inputs.
        dy_jdu = torch_in.grad.numpy() # From Torch, give us back a numpy object
        for i in range(n_inputs):
            u_i_deriv = in_list[i].derivatives()
            if debug: print("dy_jdu_a[i] * u_i_deriv = ", dy_jdu[i], " * ",  u_i_deriv)
            y_j_deriv += dy_jdu[i] * u_i_deriv;

        if param_list is not None:
            dy_jdp_list = []
            for param in network.parameters():
                if param.grad is None:
                    dy_jdp_list.append( [0.]*param.data.nelement() )
                else:
                    assert param.data.nelement() == param.grad.nelement()
                    np.testing.assert_array_equal( list(param.data.size()), list(param.grad.size()) )
                    dy_jdp_list.append( param.grad.numpy().flatten() ) # Flatten will return a copy.
            dy_jdp = np.hstack(dy_jdp_list)
            assert len(dy_jdp) == n_params

            # Calculate the contribution to y_j.derivs w.r.t all the params.
            # Get the gradient
            for k in range(n_params):
                p_k_deriv = param_list[k].derivatives()
                if debug: print("dy_jdp[k] * p_k_deriv = ", dy_jdp[k], " * ",  p_k_deriv)
                y_j_deriv += dy_jdp[k] * p_k_deriv;

        if debug: print("putting into output: ", y_j_value, ", ", y_j_deriv)
        tmp = AutoDiffXd(y_j_value, y_j_deriv)
        out_list[j] = tmp

    return out_list


