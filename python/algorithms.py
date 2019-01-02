# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

import pydrake
import pydrake.autodiffutils
from pydrake.autodiffutils import AutoDiffXd
from pydrake.systems.framework import (
    AbstractValue,
    BasicVector, BasicVector_
)
from pydrake.all import (
    DirectTranscription,
    FloatingBaseType,
    LinearSystem,
    LinearQuadraticRegulator,
    PiecewisePolynomial,
    RigidBodyTree, 
    RigidBodyPlant,
    SolutionResult
)
from underactuated import (
    PlanarRigidBodyVisualizer
)
from pydrake.examples.acrobot import AcrobotPlant

from NNSystem import NNSystem, NNSystem_
from NNTestSetup import NNTestSetup
from NNSystemHelper import FC, MLP

def make_trivial_mp():
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(2, "x")
    return prog

def make_linear_system_DirTran_mp():
    sys = LinearSystem(A,B,np.identity(2),np.zeros((2,1)),0.1)
    context = sys.CreateDefaultContext()
    kNumTimeSteps = 41  # Note: When this was 21, it converged on a local minima.

    prog = DirectTranscription(sys, context, kNumTimeSteps)
    K = prog.NewContinuousVariables(1, 2, 'K')
    prog.AddConstraintToAllKnotPoints(prog.input()[0] == (-K.dot(prog.state()))[0])
    prog.AddRunningCost(prog.state().dot(Q.dot(prog.state())) + prog.input().dot(R.dot(prog.input())))
    prog.AddBoundingBoxConstraint([1., 1.], [1., 1.], prog.initial_state())
    return prog

def make_real_dircol_mp(expmt="cartpole", seed=1776):
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

    return dircol


# From the Berkley DeepRL lecture, Guided Policy Search (GPS) has a bit of a canonical form:
# 1) Optimize p(T) w.r.t. some surrogate c'(x_t, u_t)
# 2) Optimize θ w.r.t. some supervised objective.
# 3) Increment or modify dual variables 𝜆.
#
# Choices
#   - Form of p(T) or T (if deterministic)
#   - Optimization method for p(T) or T
#   - Surrogate č(xt, ut)
#   - Supervised objective for πθ(ut|xt)

def solve_constrained_optimization_problem(
        generate_seeds,
        traj_opt_problem,
        N,
        is_deterministic, 
        surrogate_cost_fn,
        supervised_objective,
        dual_update_or_cleanup,
        convergence_fn): 

    if N != 1 or not is_deterministic:
        raise NotImplementedError

    while not convergence_fn():
        # 0) Generate N seeds
        seeds = generate_seeds(N)

        # 1) Optimize p(T) w.r.t. some surrogate c'(x_t, u_t)
        trajectories = [None]*N
        for i, seed in zip(range(N), seeds):
            # TODO: Put some kind of seeing decision making right here?
            trajectory = traj_opt_problem(surrogate_cost_fn, seed)
            trajectories[i] = trajectory

        # 2) Optimize θ w.r.t. some supervised objective.
        optimizer.optimize(trajectories, supervised_objective) # BIG TODO...

        # 3) Increment or modify dual variables 𝜆.
        dual_update_or_cleanup()


# No-op Choices
#   - Form of p(T) or T (if deterministic) : T (deterministic)
#   - Optimization method for p(T) or T    : (Nx?) DirTran/DirCol 
#   - Surrogate č(xt, ut)                  : c(T)
#   - Supervised objective for πθ(ut|xt)   : Supervised learning (SSE?)
def noop_COP(**kwargs):
    def generate_seeds(N):
        # Always return the same dumb seeds
        return list(range(N))
    def traj_opt_problem(surrogate_cost_fn, seed):
        # BIG TODO, make this shared with my dircol notebook?
        prog = make_real_dircol_mp("acrobot", seed=seed)
        return prog
    def surrogate_cost_fn(prog):
        # This is the default cost...
        R = 10  # Cost on input "effort".
        u = prog.input()
        prog.AddRunningCost(R*u[0]**2) # TODO: hopefully this can be done after setting initial trajectory and stuff...

        # No augmentation and no terms related to the policy.
    def supervised_objective():
        pass
    def dual_update_or_cleanup():
        pass # No dual variables to update
    iterations = 0
    def convergence_fn():
        global iterations
        iterations += 1
        return iterations > 100 # Blind 100 iterations

    solve_constrained_optimization_problem(
        traj_opt_problem       = traj_opt_problem,
        is_deterministic       = True,
        surrogate_cost_fn      = surrogate_cost_fn,
        supervised_objective   = supervised_objective,
        dual_update_or_cleanup = dual_update_or_cleanup,
        convergence_fn         = convergence_fn,
        **kwargs)

# GPS Choices
#   - Form of p(T) or T (if deterministic) : T (deterministic)
#   - Optimization method for p(T) or T    : (1x) DirTran/DirCol 
#   - Surrogate č(xt, ut)                  : ℒ(τ, 𝜃,𝜆)
#   - Supervised objective for πθ(ut|xt)   : ℒ(τ, 𝜃,𝜆)
# ℒ ̅(τ, 𝜃, 𝜆) = c(τ) + ∑_(𝑡=1)^𝑇 (𝜆_𝑡 (𝜋_𝜃(𝑥_𝑡) − 𝑢_𝑡) + ∑_(𝑡=1)^𝑇 𝜌_𝑡(𝜋_𝜃(𝑥_𝑡) − 𝑢_𝑡)^2
def GPS_COP(**kwargs):
    pass

# Interactive Control Choices
#   - Form of p(T) or T (if deterministic) : T (deterministic)
#   - Optimization method for p(T) or T    : (Nx) DirTran/DirCol 
#   - Surrogate č(xt, ut)                  : # TODO
#   - Supervised objective for πθ(ut|xt)   : # TODO
def IC_COP(**kwargs):
    pass
