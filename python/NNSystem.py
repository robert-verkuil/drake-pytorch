import numpy as np
import sys

from pydrake.all import (
    BasicVector,
    LeafSystem,
    PortDataType,
)

class NNSystem(LeafSystem):
    '''
    Python implementation of a Drake neural network system.
    Powered by PyTorch.
    '''
    def __init__(self, pytorch_nn_object):
        LeafSystem.__init__(self)

        # Getting relavent qtys out of pytorch_nn_object, for port setup.
        self.network = pytorch_nn_object # Needs to support an Eval method?
        self.n_inputs = 4
        self.n_outputs = 1

        # Input Ports
        self.NN_in_input_port = \
            self._DeclareInputPort(
                "NN_in", PortDataType.kVectorValued, self.n_inputs)

        # Output Ports
        self.NN_out_output_port = \
            self._DeclareVectorOutputPort(
                "NN_out", BasicVector(self.n_outputs), self.EvalOutput)


    def EvalOutput(self, context, output):
        '''
        u (inputs) --> NN => y (outputs)
                       ^
                       |
               context.p (parameters)
        '''
        # u = self.EvalVectorInput(
        #     context, self.NN_in_input_port.get_index()).get_value()
        # p = context.GetParams() #TODO: figure out a way to deal with parameters
        # y = self.network.Eval(u) #, p)y
        y = output.get_mutable_value()
        y[:] = 0
        # dydu = self.network.derivatives #TODO
        dydu = 1
        # y.derivs = dydu*u.derivs() # + dydp*p.derivs()
