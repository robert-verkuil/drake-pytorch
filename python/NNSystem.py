import numpy as np
import sys
import torch

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
        '''
            pytorch_nn_object: a function(inputs) -> outputs
        '''
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
        u = self.EvalVectorInput(
            context, self.NN_in_input_port.get_index()).get_value().astype(np.float32)
        # p = context.GetParams() #TODO: figure out a way to deal with parameters
        # y = self.network.Eval(u) #, p)y

        y = output.get_mutable_value()
        if self.network is None:
            y[:] = 0
            assert False
            # y.derivs[:] = 0
        else:
            #TODO This detach here is needed bc we are going to numpy for output, and so can't propagate gradients any further!
            y[:] = self.network.forward(torch.from_numpy(u)).detach()
            # y.derivs[:] = 0

        # dydu = self.network.derivatives
        # dydu = 1
        #TODO
        # y.derivs = dydu*u.derivs() # + dydp*p.derivs()
