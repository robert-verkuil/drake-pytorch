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
from NNSystemHelper import NNInferenceHelper

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
            self.n_inputs = 2
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
            out_list = NNInferenceHelper(self.network, in_list)

            # Pack output
            for j in range(self.n_outputs):
                output.SetAtIndex(j, out_list[j])

            # Do we even need to return anything here?
            return out_list[0]

    return Impl


# Default instantiations.
NNSystem = NNSystem_[None]
