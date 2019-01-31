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

@TemplateSystem.define("NNSystem_", T_list=[float, AutoDiffXd])
def NNSystem_(T):

    class Impl(LeafSystem_[T]):
        '''
        Python implementation of a Drake neural network system.
        Powered by PyTorch.
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

            # Input Ports
            self.NN_in_input_port = \
                self._DeclareInputPort(
                    "NN_in", PortDataType.kVectorValued, self.n_inputs)

            # Output Ports
            self.NN_out_output_port = \
                self._DeclareVectorOutputPort(
                    "NN_out", BasicVector_[T](self.n_outputs), self.EvalOutput)

        # Necessary for System Scalar Conversion
        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self, other.network, other.declare_params, converter=converter)

        def EvalOutput(self, context, output):
            '''
            u (inputs) --> NN => y (outputs)
                           ^
                           |
                   context.p (parameters)
            '''
            drake_in = self.EvalVectorInput(context, 0)
            assert drake_in.size() == self.n_inputs

            # Ensure that all network weights are doubles.
            network = network.double()

            # Drake input -> Torch input tensor
            # TODO: change this to suit your needs?
            # torch.from_numpy(<np.object>) may be helpful
            in_vec = np.array([drake_in.GetAtIndex(i) for i in range(self.n_inputs)])
            torch_in = torch.tensor(in_vec, dtype=torch.double)

            # Run the forward pass
            torch_out = network.forward(torch_in)
            
            # Process output
            out_vec = np.array([torch_out[j].data.numpy() for j in range(self.n_outputs)])

            # Torch output tensor -> Drake output
            for j in range(self.n_outputs):
                output.SetAtIndex(j, out_vec[j])

    return Impl


# Default instantiation.
NNSystem = NNSystem_[None]


