from __future__ import print_function, absolute_import

import numpy as np

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
from pydrake.all import (DirectTranscription, LinearSystem, LinearQuadraticRegulator, 
                         PiecewisePolynomial, SolutionResult)

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


# From the Berkley DeepRL lecture, Guided Policy Search (GPS) has a bit of a canonical form:
# 1) Optimize p(T) w.r.t. some surrogate c'(x_t, u_t)
# 2) Optimize 𝛩 w.r.t. some supervised objective.
# 3) Increment or modify dual variables 𝜆.
#
# Choices
#   - Form of p(T) or T (if deterministic)
#   - Optimization method for p(T) or T
#   - Surrogate č(xt, ut)
#   - Supervised objective for π𝛩(ut|xt)

def solve_constrained_optimization_problem(
        traj_opt_problem,
        N=1,
        is_deterministic=True,
        surrogate_cost_fn,
        supervised_objective,
        dual_update_or_cleanup,
        convergence_fn): 

    if not is_deterministic:
        raise NotImplementedError

    while not convergence_fn():
        # 1) Optimize p(T) w.r.t. some surrogate c'(x_t, u_t)
        trajectories = [None]*N
        for i in range(N):
            # TODO: Put some kind of seeing decision making right here?
            trajectory = traj_opt_problem(surrogate_cost_fn)
            trajectories[i] = trajectory

        # 2) Optimize 𝛩 w.r.t. some supervised objective.
        optimizer.optimize(trajectories, supervised_objective) # BIG TODO...

        # 3) Increment or modify dual variables 𝜆.
        dual_update_or_cleanup()


# No-op Choices
#   - Form of p(T) or T (if deterministic) : T (deterministic)
#   - Optimization method for p(T) or T    : (Nx?) DirTran/DirCol 
#   - Surrogate č(xt, ut)                  : c(T)
#   - Supervised objective for π𝛩(ut|xt)   : Supervised learning (SSE?)
def noop_COP(**kwargs):
    def traj_opt_problem(surrogate_cost_fn):
        pass
    def surrogate_cost_fn():
        pass
    def supervised_objective():
        pass
    def dual_update_or_cleanup():
        pass
    def convergence_fn():
        pass

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
#   - Supervised objective for π𝛩(ut|xt)   : ℒ(τ, 𝜃,𝜆)
# ℒ ̅(τ, 𝜃, 𝜆) = c(τ) + ∑_(𝑡=1)^𝑇 (𝜆_𝑡 (𝜋_𝜃(𝑥_𝑡) − 𝑢_𝑡) + ∑_(𝑡=1)^𝑇 𝜌_𝑡(𝜋_𝜃(𝑥_𝑡) − 𝑢_𝑡)^2
def noop_GPS(**kwargs):
    pass

# Interactive Control Choices
#   - Form of p(T) or T (if deterministic) : T (deterministic)
#   - Optimization method for p(T) or T    : (Nx) DirTran/DirCol 
#   - Surrogate č(xt, ut)                  : # TODO
#   - Supervised objective for π𝛩(ut|xt)   : # TODO
def noop_IC(**kwargs):
    pass
