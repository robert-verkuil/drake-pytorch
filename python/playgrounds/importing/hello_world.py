import pydrake
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lin = nn.Linear(5, 3)    # maps from R^5 to R^3, parameters A, b
data = torch.randn(2, 5) # data is 2x5.  A maps from 5 to 3... can we map "data" under A?
print(F.relu(lin(data)))

import pydrake
import pydrake.autodiffutils
from pydrake.autodiffutils import AutoDiffXd

x = AutoDiffXd(1.5)
arr = np.ndarray((3,), buffer=np.array([1,2,3]), dtype=np.float64)
y = AutoDiffXd(1.5, arr)
for var in (x, y):
    print(var.value(), var.derivatives())

