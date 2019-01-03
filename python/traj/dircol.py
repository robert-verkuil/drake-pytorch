# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

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


tree = None
plant = None
context = None
dircol = None
def make_real_dircol_mp(expmt="cartpole", seed=1776):
    global tree
    global plant
    global context
    global dircol
    # TODO: use the seed in some meaningful way:
    # https://github.com/RobotLocomotion/drake/blob/master/systems/stochastic_systems.h

    assert expmt in ("cartpole", "acrobot")
    # expmt = "cartpole" # State: (x, theta, x_dot, theta_dot) Input: x force
    # expmt = "acrobot" # State: (theta1, theta2, theta1_dot, theta2_dot) Input: Elbow torque

    if expmt == "cartpole":
        tree = RigidBodyTree("/opt/underactuated/src/cartpole/cartpole.urdf",
                             FloatingBaseType.kFixed)
        plant = RigidBodyPlant(tree)
    else:
        tree = RigidBodyTree("/opt/underactuated/src/acrobot/acrobot.urdf",
                         FloatingBaseType.kFixed)
        plant = AcrobotPlant()
        
    context = plant.CreateDefaultContext()

    if expmt == "cartpole":
        dircol = DirectCollocation(plant, context, num_time_samples=21,
                                   minimum_timestep=0.1, maximum_timestep=0.4)
    else:
        dircol = DirectCollocation(plant, context, num_time_samples=21,
                               minimum_timestep=0.05, maximum_timestep=0.2)

    dircol.AddEqualTimeIntervalsConstraints()

    if expmt == "acrobot":
        # Add input limits.
        torque_limit = 8.0  # N*m.
        u = dircol.input()
        dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
        dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

    initial_state = (0., 0., 0., 0.)
    dircol.AddBoundingBoxConstraint(initial_state, initial_state,
                                    dircol.initial_state())
    # More elegant version is blocked on drake #8315:
    # dircol.AddLinearConstraint(dircol.initial_state() == initial_state)

    if expmt == "cartpole":
        final_state = (0., math.pi, 0., 0.)
    else:
        final_state = (math.pi, 0., 0., 0.)
    dircol.AddBoundingBoxConstraint(final_state, final_state,
                                    dircol.final_state())
    # dircol.AddLinearConstraint(dircol.final_state() == final_state)

#    R = 10  # Cost on input "effort".
#    u = dircol.input()
#    dircol.AddRunningCost(R*u[0]**2)

    # Add a final cost equal to the total duration.
    dircol.AddFinalCost(dircol.time())

    initial_x_trajectory = \
        PiecewisePolynomial.FirstOrderHold([0., 4.],
                                           np.column_stack((initial_state,
                                                            final_state)))
    dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    return dircol, tree



def make_pytorch_net_autodiffable(pytorch_net):
    # Take the raw pytorch net reference, and create a policy function that can 
    # support AutoDiffXd input -> AutoDiffXd output, which is needed for any custom
    # PyFunction Cost applied to a Drake Mathematical Program!
    if pytorch_net is None:
        # If not pytorch_net is provided, make a trivial policy that support AD in and out.
        def Pi(x_t):
            return (x_t[0]-2.)*(x_t[1]-2.)
    else:
        # Else, wrap the pytorch_net so that is supports AD in and out.
        def Pi(x_t):
            from nn_system.NNSystemHelper import NNInferenceHelper
            in_list = list(x_t)
            out_list = NNInferenceHelper(pytorch_net, in_list)
            return out_list[0]
    return Pi


# Helper that applies some cost function involving a 
# policy Pi to every timestep of a Mathematical Program.
#
# @param prog A drake mathematical program.
# @param mycost A function that defines the cost at a particular timestep, 
#        as a function of the x_t, and u_t.
def add_running_custom_cost_fn(prog, mycost):
    # Define the custom Cost function that we will be applying to each step.
    # Take care to ensure that AutoDiffXd input -> output is supported.
    # Also ensure that the output is a single AutoDiffXd value.
    # timestep = 0
    def custom_cost_fn(state_concat_inp):
        # global timestep
        # Unpack variables
        x_sz, u_sz = len(prog.state(0)), len(prog.input(0))
        assert len(state_concat_inp) == x_sz + u_sz
        x, u = state_concat_inp[:x_sz], state_concat_inp[-u_sz:]

        cost = mycost(x, u, 0) #TODO: ensure that autodiff's flow through this thing... #TODO: actually use a timestamp here.
        assert len(cost) == 1

        # timestep += 1
        return cost[0]

    # Apply our custom PyFunction cost to every timestep of the mathematical program.
    for t in range(21): #TODO: undo the hard coding here
        prog.AddCost(custom_cost_fn, np.hstack((prog.state(t), prog.input(t))))

