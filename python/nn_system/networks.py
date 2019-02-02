import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to("cpu")

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).double().normal_() * scale
            x = x + sampled_noise
        return x 


class FC(nn.Module):
    def __init__(self, n_inputs=4):
        super(FC, self).__init__()
        self.n_inputs = n_inputs
        self.fc1 = nn.Linear(self.n_inputs, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class FCBIG(nn.Module):
    def __init__(self, n_inputs=4, h_sz=8, dropout=False, init=None, input_noise=None, output_noise=None):
        super(FCBIG, self).__init__()
        self.n_inputs = n_inputs
        self.dropout  = dropout
        self.input_noise = input_noise
        self.output_noise = output_noise

        self.fc1 = nn.Linear(self.n_inputs, h_sz)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h_sz, 1)

        if input_noise is not None:
            self.in_noise = GaussianNoise(input_noise)
        if output_noise is not None:
            self.out_noise = GaussianNoise(output_noise)
        if init is not None:
            init(self.fc1.weight)
            init(self.fc2.weight)

    def forward(self, x):
        if self.input_noise:
            x = self.in_noise(x)
        x = F.relu(self.fc1(x))
        if self.dropout: x = self.drop1(x)
        x = self.fc2(x)
        if self.output_noise:
            x = self.out_noise(x)
        return x

class MLPSMALL(nn.Module):
    def __init__(self, n_inputs=4, h_sz=8, layer_norm=False, dropout=False):
        super(MLPSMALL, self).__init__()
        self.n_inputs   = n_inputs
        self.layer_norm = layer_norm
        self.dropout    = dropout

        self.l1 = nn.Linear(self.n_inputs, h_sz)
        self.ln1 = nn.LayerNorm(h_sz)
        self.drop1 = nn.Dropout(0.5)
        self.l3 = nn.Linear(h_sz, 1)
    
    def forward(self, x):
        x = self.l1(x)
        if self.layer_norm: x = self.ln1(x)
        x = F.relu(x)
        if self.dropout: x = self.drop1(x)
        x = self.l3(x)
        return x

class MLP(nn.Module):
    def __init__(self, n_inputs=4, h_sz=256, layer_norm=False, dropout=False, init=None, input_noise=None, output_noise=None):
        super(MLP, self).__init__()
        self.n_inputs   = n_inputs
        self.layer_norm = layer_norm
        self.dropout    = dropout
        self.input_noise = input_noise
        self.output_noise = output_noise

        self.l1 = nn.Linear(self.n_inputs, h_sz)
        self.ln1 = nn.LayerNorm(h_sz)
        self.tanh1 = torch.tanh
        self.drop1 = nn.Dropout(0.1)
        self.l2 = nn.Linear(h_sz, h_sz)
        self.ln2 = nn.LayerNorm(h_sz)
        self.tanh2 = torch.tanh
        self.drop2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(h_sz, 1)

        if input_noise is not None:
            self.in_noise = GaussianNoise(input_noise)
        if output_noise is not None:
            self.out_noise = GaussianNoise(output_noise)
        if init is not None:
            init(self.l1.weight)
            init(self.l2.weight)
            init(self.l3.weight)
    
    def forward(self, x):
        if self.input_noise:
            x = self.in_noise(x)
        x = self.l1(x)
        if self.layer_norm: x = self.ln1(x)
        #x = self.tanh1(x)
        x = F.relu(x)
        if self.dropout: x = self.drop1(x)
        x = self.l2(x)
        if self.layer_norm: x = self.ln2(x)
        #x = self.tanh2(x)
        x = F.relu(x)
        if self.dropout: x = self.drop2(x)
        x = self.l3(x)
        if self.output_noise:
            x = self.out_noise(x)
        return x

class MLPBIG(nn.Module):
    def __init__(self, n_inputs=4, h_sz=256, layer_norm=False, dropout=False):
        super(MLPBIG, self).__init__()
        self.n_inputs   = n_inputs
        self.layer_norm = layer_norm
        self.dropout    = dropout

        self.l1 = nn.Linear(self.n_inputs, h_sz)
        self.ln1 = nn.LayerNorm(h_sz)
        self.drop1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(h_sz, h_sz)
        self.ln2 = nn.LayerNorm(h_sz)
        self.drop2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(h_sz, h_sz)
        self.ln3 = nn.LayerNorm(h_sz)
        self.drop3 = nn.Dropout(0.5)
        self.l4 = nn.Linear(h_sz, 1)
    
    def forward(self, x):
        x = self.l1(x)
        if self.layer_norm: x = self.ln1(x)
        x = F.relu(x)
        if self.dropout: x = self.drop1(x)

        x = self.l2(x)
        if self.layer_norm: x = self.ln2(x)
        x = F.relu(x)
        if self.dropout: x = self.drop2(x)

        x = self.l3(x)
        if self.layer_norm: x = self.ln3(x)
        x = F.relu(x)
        if self.dropout: x = self.drop3(x)

        x = self.l4(x)
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
