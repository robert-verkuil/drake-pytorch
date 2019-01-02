# from IPython.display import HTML
from __future__ import print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    SolutionResult
)
from underactuated import (
    PlanarRigidBodyVisualizer
)
import math
import numpy as np
import random

from pydrake.all import (
    DirectCollocation, 
    FloatingBaseType,
    PiecewisePolynomial, 
    RigidBodyTree, 
    RigidBodyPlant,
    SolutionResult
)
from pydrake.examples.acrobot import AcrobotPlant
from underactuated import (
    PlanarRigidBodyVisualizer
)


def make_real_dircol_mp():
    # expmt = "acrobot" # State: (theta1, theta2, theta1_dot, theta2_dot) Input: Elbow torque

    tree = RigidBodyTree("/opt/underactuated/src/acrobot/acrobot.urdf",
                     FloatingBaseType.kFixed)
    plant = AcrobotPlant()
    context = plant.CreateDefaultContext()
    dircol = DirectCollocation(plant, context, num_time_samples=21,
                           minimum_timestep=0.05, maximum_timestep=0.2)

    dircol.AddEqualTimeIntervalsConstraints()

    # Add input limits.
    torque_limit = 8.0  # N*m.
    u = dircol.input()
    dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
    dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

    initial_state = (0., 0., 0., 0.)
    dircol.AddBoundingBoxConstraint(initial_state, initial_state,
                                    dircol.initial_state())
    final_state = (math.pi, 0., 0., 0.)
    dircol.AddBoundingBoxConstraint(final_state, final_state,
                                    dircol.final_state())

    R = 10  # Cost on input "effort".
    u = dircol.input()
    dircol.AddRunningCost(R*u[0]**2)

    # Add a final cost equal to the total duration.
    dircol.AddFinalCost(dircol.time())

    initial_x_trajectory = \
        PiecewisePolynomial.FirstOrderHold([0., 4.],
                                           np.column_stack((initial_state,
                                                            final_state)))
    dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    print(id(dircol))
    return dircol

dircol = make_real_dircol_mp()

# Explcitly set the cost function here.
# R = 10  # Cost on input "effort".
# u = dircol.input()
# dircol.AddRunningCost(R*u[0]**2)

result = dircol.Solve()
assert(result == SolutionResult.kSolutionFound)
print("used solver: ", dircol.GetSolverId().name())

# The problem
x_trajectory = dircol.ReconstructStateTrajectory()
print("end")

