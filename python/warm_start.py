from __future__ import print_function, absolute_import
import math
import matplotlib.pyplot as plt
import numpy as np

from multiple_traj_opt import (
    make_mto,
    MultipleTrajOpt,
    initial_conditions_Russ,
    initial_conditions_grid,
    initial_conditions_random,
)
from nn_system.networks import *

import torch


# These warm-starting schemes should all follow some very common interface so that I use them
# for warm starting:
# 1) Naive
# 2) MTO/Russ
# 3) Igor

# Format should be:
# warm_start_trajs = warm_start_method(ic_list, traj_bank, network, most_recent_dircol)

# Big List of pseudocode
# method1, Try using the exact previous answer as a warm start          NEEDS INTERFACE?
#     most_recent_dircol.GetSolution() -> ret
#
# method2: Use limited-wallclock policy rollouts
#     ic_list.rollout(network) -> ret
#
# method3: Nearby trajectory interpolations
#     interpolate(ic_list, traj_bank) -> ret
#
# method4: Restart with fresh unconstrained traj solves                 NEEDS INTERFACE
#     mto.solve(ic_list) -> ret
#
# method5: Restart with cost-penalized traj solves                      NEEDS INTERFACE
#     mto.solve(ic_list, cost_factor=<>) -> ret
#
# method6: Restart with constrained traj solves                         NEEDS INTERFACE
#     mto.solve(ic_list, use_constraint=True) -> ret
#
# method7: bonus, carry over more solver state than just decision var's NEEDS INTERFACE
#     mto.prog.GetFullState() -> ret, extra_ret (RETURNS MORE!)
#


# 1) SIMPLE RESTART WITH POTENTIALLY DIFFERENT SETTINGS, full huxT
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method1(mto):
    old_mto = mto
    mto = make_mto()

    # Warm start...
    old_mto_dec_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
    mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)

    mto.Solve()
    return old_mto, mto


# 2) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, use limited wallclock time policy rollouts?
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method2(mto):
    old_mto = mto
    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.T)
    mto = make_mto()

    ic_list = #TODO
    assert len(ic_list) == num_trajectories
    for ic in ic_list:
        t_samples, x_samples, u_samples, logger = old_mto.__rollout_policy_at_solution(ti_or_ic=ic) # Be careful about this taking forever!!!!
        # Add a return for u_samples!!
        warm_start = #TODO: assemble a new batch of h, u, x?
        mto.prog.SetInitialGuess(mto.h[ti], ) #TODO
        mto.prog.SetInitialGuess(mto.u[ti], ) #TODO
        mto.prog.SetInitialGuess(mto.x[ti], ) #TODO
    mto.prog.SetInitialGuess(mto.T, old_mto_T_vals)
    mto.Solve()
    return old_mto, mto


# 3) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, nearby traj. interpolations?
# Begs the question, will I want a history of trajectories??
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
# Keep a "BANK" of trajectories, that we can optionally use for warm starting?
# No consistency here!
if not trajectories:
    trajectories = []
def method3(mto, ic_list):
    global trajectories
    old_mto = mto

    for ti in range(num_trajectories):
        trajectory = np.hstack([old_mto.GetSolution(var) for var in (old_mto.h[ti], old_mto.u[ti], old_mto.x[ti])]) # (h, u_t's, x_t's)
        trajectories.append(trajectory)
    if len(trajectories) >mto = mto
    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.T)
    mto = make_mto()

    warm_mto = MultipleTrajOpt("pendulum", 16, 16, 0.2, 0.5, ic_list=ic_list, warm_start=True, seed=old_mto.seed)
    warm_mto.add_nn_params(old_mto.kNetConstructor,
                      use_constraint    = False,
                      cost_factor       = 1.0,
                      initialize_params = True, 
                      reg_type          = old_mto.reg_type)
    warm_mto.add_cost_and_constraint_printing_callback(1)
    warm_mto.prog.SetInitialGuess(warm_mto.T, old_mto_T_vals) # "Warm start the warm start with the old network?"
    warm_mto.Solve()

    # Warm start paths with fresh solves, but carry over NN
    warm_mto_hux_vals = np.hstack([warm_mto.prog.GetSolution(var) for var in (warm_mto.h, warm_mto.u, warm_mto.x)])
    mto.prog.SetInitialGuessForAllVariables(np.hstack([warm_mto_hux_vals, old_mto_T_vals]))

    mto.Solve() CUTOFF:
        trajectories = trajectories[-CUTOFF:] # Keep bank trim and hopefully filter out the old crappy trajectories...
    mto = make_mto()

    # Warm start... but use a different warm start scheme
    ic_list = #TODO
    interpolants = []
    for ic in ic_list:
        # TODO, should have format of (h, u_t's, x_t's). Should be graded by ic and fc (and path?) proximity??
        # Can come from the previous solution, or the bank
        nearest = #TODO
        interpolants.append()
    assert len(interpolants) == len(ic_list)
    old_mto_dec_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
    mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)

    mto.Solve()
    return old_mto, mto


# 4) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, fresh traj. solves? - should i split the traj solves?
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method4(mto, ic_list):
    old_mto = mto
    mto = make_mto()

    warm_mto = MultipleTrajOpt("pendulum", 16, 16, 0.2, 0.5, ic_list=ic_list, warm_start=True, seed=old_mto.seed)
    warm_mto.add_cost_and_constraint_printing_callback(1)
    warm_mto.Solve()

    # Warm start paths with fresh solves, but carry over NN
    warm_mto_hux_vals = np.hstack([warm_mto.prog.GetSolution(var) for var in (warm_mto.h, warm_mto.u, warm_mto.x)])
    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
    mto.prog.SetInitialGuessForAllVariables(mto.prog.decision_variables(), np.hstack([warm_mto_hux_vals, old_mto_T_vals]))

    mto.Solve()
    return old_mto, mto


# 5) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, fresh traj. solves with policy violation cost?
# Is this even cheaper?
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method5(mto, ic_list):
    old_mto = mto
    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.T)
    mto = make_mto()

    warm_mto = MultipleTrajOpt("pendulum", 16, 16, 0.2, 0.5, ic_list=ic_list, warm_start=True, seed=old_mto.seed)
    warm_mto.add_nn_params(old_mto.kNetConstructor,
                      use_constraint    = False,
                      cost_factor       = 1.0,
                      initialize_params = True, 
                      reg_type          = old_mto.reg_type)
    warm_mto.add_cost_and_constraint_printing_callback(1)
    warm_mto.prog.SetInitialGuess(warm_mto.T, old_mto_T_vals) # "Warm start the warm start with the old network?"
    warm_mto.Solve()

    # Warm start paths with fresh solves, but carry over NN
    warm_mto_hux_vals = np.hstack([warm_mto.prog.GetSolution(var) for var in (warm_mto.h, warm_mto.u, warm_mto.x)])
    mto.prog.SetInitialGuessForAllVariables(np.hstack([warm_mto_hux_vals, old_mto_T_vals]))

    mto.Solve()
    return old_mto, mto


# 6) EXPERIMENT WITH CARRYING OVER MORE THAN DECISION VARIABLE INITIAL GUESSES!!!!
# THIS WILL BE THE HARDEST EXPERIMENT... IT WILL REQUIRE RECOMPILING A NEW VERSION OF DRAKE????
# <TODO>
# https://github.com/RobotLocomotion/drake/blob/7c513516620b1e6001fd487e076c39f716027a79/solvers/snopt_solver.cc#L682
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method6(mto, ic_list):
    old_mto = mto
    mto = make_mto()

    # Warm start... but use a different warm start scheme
    old_mto_dec_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
    mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)

    mto.Solve()
    return old_mto, mto


