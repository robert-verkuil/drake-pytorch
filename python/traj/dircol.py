# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import math
import numpy as np
import random

import matplotlib.pyplot as plt

from pydrake.all import (
    DiagramBuilder,
    DirectCollocation, 
    FloatingBaseType,
    PiecewisePolynomial, 
    RigidBodyTree, 
    RigidBodyPlant,
    SignalLogger, 
    Simulator, 
    SolutionResult,
    VectorSystem
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

# Currently only set up to make pendulum examples
def make_multiple_dircol_trajectories(num_trajectories, num_samples, initial_conditions=None):
    from pydrake.all import (AutoDiffXd, Expression, Variable,
                         MathematicalProgram, SolverType, SolutionResult,
                         DirectCollocationConstraint, AddDirectCollocationConstraint,
                         PiecewisePolynomial,
                        )
    import pydrake.symbolic as sym
    from pydrake.examples.pendulum import (PendulumPlant)

    # initial_conditions maps (ti) -> [1xnum_states] initial state
    if initial_conditions is not None:
        assert callable(initial_conditions)

    plant = PendulumPlant()
    context = plant.CreateDefaultContext()
    dircol_constraint = DirectCollocationConstraint(plant, context)

    # num_trajectories = 15;
    # num_samples = 15;
    prog = MathematicalProgram()
    # K = prog.NewContinuousVariables(1,7,'K')

    def cos(x):
        if isinstance(x, AutoDiffXd):
            return x.cos()
        elif isinstance(x, Variable):
            return sym.cos(x)
        return math.cos(x)

    def sin(x):
        if isinstance(x, AutoDiffXd):
            return x.sin()
        elif isinstance(x, Variable):
            return sym.sin(x)
        return math.sin(x)

    def final_cost(x):
        return 100.*(cos(.5*x[0])**2 + x[1]**2)   
        
    h = [];
    u = [];
    x = [];
    xf = (math.pi, 0.)
    for ti in range(num_trajectories):
        h.append(prog.NewContinuousVariables(1))
        prog.AddBoundingBoxConstraint(.01, .2, h[ti])
        # prog.AddQuadraticCost([1.], [0.], h[ti]) # Added by me, penalize long timesteps
        u.append(prog.NewContinuousVariables(1, num_samples,'u'+str(ti)))
        x.append(prog.NewContinuousVariables(2, num_samples,'x'+str(ti)))

        # Use Russ's initial conditions, unless I pass in a function myself.
        if initial_conditions is None:
            x0 = (.8 + math.pi - .4*ti, 0.0)    
        else:
            x0 = initial_conditions(ti)
            assert len(x0) == 2 #TODO: undo this hardcoding.
        prog.AddBoundingBoxConstraint(x0, x0, x[ti][:,0]) 

        nudge = np.array([.2, .2])
        prog.AddBoundingBoxConstraint(xf-nudge, xf+nudge, x[ti][:,-1])
        # prog.AddBoundingBoxConstraint(xf, xf, x[ti][:,-1])

        for i in range(num_samples-1):
            AddDirectCollocationConstraint(dircol_constraint, h[ti], x[ti][:,i], x[ti][:,i+1], u[ti][:,i], u[ti][:,i+1], prog)

        for i in range(num_samples):
            prog.AddQuadraticCost([1.], [0.], u[ti][:,i])
    #        prog.AddConstraint(control, [0.], [0.], np.hstack([x[ti][:,i], u[ti][:,i], K.flatten()]))
    #        prog.AddBoundingBoxConstraint([-3.], [3.], u[ti][:,i])
    #        prog.AddConstraint(u[ti][0,i] == (3.*sym.tanh(K.dot(control_basis(x[ti][:,i]))[0])))  # u = 3*tanh(K * m(x))
            
        # prog.AddCost(final_cost, x[ti][:,-1])
        # prog.AddCost(h[ti][0]*100) # Try to penalize using more time than it needs?

    #prog.SetSolverOption(SolverType.kSnopt, 'Verify level', -1)  # Derivative checking disabled. (otherwise it complains on the saturation)
    prog.SetSolverOption(SolverType.kSnopt, 'Print file', "/tmp/snopt.out")

    # result = prog.Solve()
    # print(result)
    # print(prog.GetSolution(K))
    # print(prog.GetSolution(K).dot(control_basis(x[0][:,0])))

    #if (result != SolutionResult.kSolutionFound):
    #    f = open('/tmp/snopt.out', 'r')
    #    print(f.read())
    #    f.close()
    return prog, h, u, x



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


