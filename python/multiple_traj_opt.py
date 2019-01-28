from __future__ import print_function, absolute_import

import copy
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np


# pydrake stuff
from pydrake.all import (AutoDiffXd, Expression, Variable,
                     MathematicalProgram, SolverType, SolutionResult,
                     DirectCollocationConstraint, AddDirectCollocationConstraint,
                     PiecewisePolynomial,
                    )
import pydrake.symbolic as sym
from pydrake.examples.pendulum import (PendulumPlant)

# Weirddddddd, apparently pydrake MUST be imported before torch???
import torch
from torch.nn.init import * # Here's where I can specify which Torch NN inits I want...

# My stuff
from nn_system.NNSystem import NNInferenceHelper_double
from nn_system.NNSystemHelper import (
    create_nn,
    create_nn_policy_system,
    make_NN_constraint,
    FCBIG,
)
from traj.vis import (
    simulate_and_log_policy_system,
    plot_trajectory,
    render_trajectory,
)

#########################################################
# Benchmarking utils
#########################################################
import time
current_milli_time = lambda: int(round(time.time() * 1000))
mylog = []
def start(name):
    ret = {
        "name": name,
        "cat": "PERF",
        "ph": "B",
        "pid": 0,
        #"tid": 22630,
        "ts": current_milli_time()
    }
    mylog.append(ret)
def end(name):
    ret = {
        "name": name,
        "cat": "PERF",
        "ph": "E",
        "pid": 0,
        #"tid": 22630,
        "ts": current_milli_time()
    }
    mylog.append(ret)
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

#########################################################
# Big Batch of Initial Conditions
#########################################################
theta_bounds     = 0, math.pi
theta_dot_bounds = -5., 5.
def initial_conditions_Russ(num_trajectories):
    return np.array([(.8 + math.pi - .4*ti, 0.0) for ti in range(num_trajectories)])
# TODO: use sqrt(num_trajectories) to determine n_theta/_dot
def initial_conditions_grid(num_trajectories, theta_bounds=theta_bounds, theta_dot_bounds=theta_dot_bounds):
    # Have ti index into a grid over some state space bounds
    sqrt = int(math.sqrt(num_trajectories))#+1
    n_theta, n_theta_dot = (sqrt, sqrt)
    theta_range     = np.linspace(theta_bounds[0], theta_bounds[1], n_theta)
    theta_dot_range = np.linspace(theta_dot_bounds[0], theta_dot_bounds[1], n_theta_dot)
    ret = []
    for ti in range(num_trajectories):
        r = int(ti / n_theta)
        c = ti % n_theta
        ret.append( (theta_range[c], theta_dot_range[r]) )
    return ret
def initial_conditions_random(num_trajectories, theta_bounds=theta_bounds, theta_dot_bounds=theta_dot_bounds):
    # random state over some state-space bounds
    ret = []
    for ti in range(num_trajectories):
        rand1, rand2 =  np.random.random(2)
        theta     =  (theta_bounds[1] - theta_bounds[0]) * rand1 + theta_bounds[0]
        theta_dot =  (theta_dot_bounds[1] - theta_dot_bounds[0]) * rand2 + theta_dot_bounds[0]
        ret.append( (theta, theta_dot) )
    return ret
def initial_conditions_random_all_dims(num_trajectories, bounds_list):
    # random state over some state-space bounds
    N = len(bounds_list)
    ret = []
    for ti in range(num_trajectories):
        coord = tuple(np.random.uniform(*bound) for bound in bounds_list)
        ret.append( coord )
    return ret
# intial_cond_dict = {
#     "Russ": initial_conditions_Russ,
#     "grid": initial_conditions_grid,
#     "random": initial_conditions_random,
# }

#########################################################
# Warm Start Methods
#########################################################
# Helper to get solution else initial guess
def get_solution_else_initial_guess(prog, variables):
    ret = prog.GetSolution(variables)
    if np.isnan(ret.flat[0]):
        ret = prog.GetInitialGuess(variables)
    return ret
# 0) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS
# Simplest possible version, (using the EXACT previous answer as a warm start)
def method0(mto, ic_list, **kwargs):
    old_mto = mto
    mto = make_mto(ic_list=ic_list, **kwargs) # Give the new ics here.

    # Warm start...
#     if old_mto is not None:
#         old_mto_dec_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
#         mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)

    mto.Solve()
    return old_mto, mto
# 1) SIMPLE RESTART WITH POTENTIALLY DIFFERENT SETTINGS, full huxT
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method1(mto):
    old_mto = mto
    mto = make_mto()

    # Warm start...
    old_mto_dec_vals = get_solution_else_initial_guess(old_mto.prog, old_mto.prog.decision_variables())
    mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)

    mto.Solve()
    return old_mto, mto
# 2) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, use limited wallclock time policy rollouts?
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
#def method2(mto):
#    old_mto = mto
#    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.T)
#    mto = make_mto()
#
#    ic_list = #TODO
#    assert len(ic_list) == num_trajectories
#    for ic in ic_list:
#        t_samples, x_samples, u_samples, logger = old_mto.__rollout_policy_at_solution(ti_or_ic=ic) # Be careful about this taking forever!!!!
#        # Add a return for u_samples!!
#        warm_start = #TODO: assemble a new batch of h, u, x?
#        mto.prog.SetInitialGuess(mto.h[ti], ) #TODO
#        mto.prog.SetInitialGuess(mto.u[ti], ) #TODO
#        mto.prog.SetInitialGuess(mto.x[ti], ) #TODO
#    mto.prog.SetInitialGuess(mto.T, old_mto_T_vals)
#    mto.Solve()
#    return old_mto, mto
# 3) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, nearby traj. interpolations?
# Begs the question, will I want a history of trajectories??
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
# Keep a "BANK" of trajectories, that we can optionally use for warm starting?
# No consistency here!
#if not trajectories:
#    trajectories = []
#def method3(mto, ic_list):
#    global trajectories
#    old_mto = mto
#
#    for ti in range(num_trajectories):
#        trajectory = np.hstack([old_mto.GetSolution(var) for var in (old_mto.h[ti], old_mto.u[ti], old_mto.x[ti])]) # (h, u_t's, x_t's)
#        trajectories.append(trajectory)
#    if len(trajectories) >mto = mto
#    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.T)
#    mto = make_mto()
#
#    warm_mto = MultipleTrajOpt("pendulum", 16, 16, 0.2, 0.5, ic_list=ic_list, warm_start=True, seed=old_mto.seed)
#    warm_mto.add_nn_params(old_mto.kNetConstructor,
#                      use_constraint    = False,
#                      cost_factor       = 1.0,
#                      initialize_params = True, 
#                      reg_type          = old_mto.reg_type)
#    warm_mto.add_cost_and_constraint_printing_callback(1)
#    warm_mto.prog.SetInitialGuess(warm_mto.T, old_mto_T_vals) # "Warm start the warm start with the old network?"
#    warm_mto.Solve()
#
#    # Warm start paths with fresh solves, but carry over NN
#    warm_mto_hux_vals = np.hstack([warm_mto.prog.GetSolution(var) for var in (warm_mto.h, warm_mto.u, warm_mto.x)])
#    mto.prog.SetInitialGuessForAllVariables(np.hstack([warm_mto_hux_vals, old_mto_T_vals]))
#
#    mto.Solve() CUTOFF:
#        trajectories = trajectories[-CUTOFF:] # Keep bank trim and hopefully filter out the old crappy trajectories...
#    mto = make_mto()
#
#    # Warm start... but use a different warm start scheme
#    ic_list = #TODO
#    interpolants = []
#    for ic in ic_list:
#        # TODO, should have format of (h, u_t's, x_t's). Should be graded by ic and fc (and path?) proximity??
#        # Can come from the previous solution, or the bank
#        nearest = #TODO
#        interpolants.append()
#    assert len(interpolants) == len(ic_list)
#    old_mto_dec_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
#    mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)
#
#    mto.Solve()
#    return old_mto, mto

# 4) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, fresh traj. solves? - should i split the traj solves?
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
def method4(mto, ic_list, **kwargs):
    old_mto = mto
    mto = make_mto(ic_list=ic_list, **kwargs)

#    print("\n\n BEGIN WARM START")
    warm_kwargs = copy.deepcopy(kwargs)
    warm_kwargs["kNetConstructor"] = None # Turn off NN, just for warm start
    warm_kwargs["vis_cb_every_nth"] = None
    warm_kwargs["cost_cb_every_nth"] = None
#     warm_kwargs["snopt_overrides"] = [('Major iterations limit',  1e9)]
    warm_mto = make_mto(ic_list=ic_list, **warm_kwargs)
    warm_mto.Solve()
#    print("END WARM START \n\n")

    # Warm start paths with fresh solves, but carry over NN
    warm_mto_hux_vals = warm_mto.prog.GetSolution(warm_mto.prog.decision_variables())
    if old_mto is not None:
        old_mto_T_vals = get_solution_else_initial_guess(old_mto.prog, old_mto.T)
    else:
        old_mto_T_vals = mto.prog.GetInitialGuess(mto.T)
    mto.prog.SetInitialGuessForAllVariables(np.hstack([warm_mto_hux_vals, old_mto_T_vals]))
        
#    print("\n\n BEGIN REAL SOLVE")
    mto.Solve()
#    print("END REAL SOLVE \n\n")
    return old_mto, mto
# 5) RESTART WITH A DIFFERENT MINIBATCH OF INITIAL CONDITIONS, fresh traj. solves with policy violation cost?
# Is this even cheaper?
# Then resolve again with (potentially different settings, but using the previous answer as a warm start)
#def method5(mto, ic_list):
#    old_mto = mto
#    old_mto_T_vals = old_mto.prog.GetSolution(old_mto.T)
#    mto = make_mto()
#
#    warm_mto = MultipleTrajOpt("pendulum", 16, 16, 0.2, 0.5, ic_list=ic_list, warm_start=True, seed=old_mto.seed)
#    warm_mto.add_nn_params(old_mto.kNetConstructor,
#                      use_constraint    = False,
#                      cost_factor       = 1.0,
#                      initialize_params = True, 
#                      reg_type          = old_mto.reg_type)
#    warm_mto.add_cost_and_constraint_printing_callback(1)
#    warm_mto.prog.SetInitialGuess(warm_mto.T, old_mto_T_vals) # "Warm start the warm start with the old network?"
#    warm_mto.Solve()
#
#    # Warm start paths with fresh solves, but carry over NN
#    warm_mto_hux_vals = np.hstack([warm_mto.prog.GetSolution(var) for var in (warm_mto.h, warm_mto.u, warm_mto.x)])
#    mto.prog.SetInitialGuessForAllVariables(np.hstack([warm_mto_hux_vals, old_mto_T_vals]))
#
#    mto.Solve()
#    return old_mto, mto
## 6) EXPERIMENT WITH CARRYING OVER MORE THAN DECISION VARIABLE INITIAL GUESSES!!!!
## THIS WILL BE THE HARDEST EXPERIMENT... IT WILL REQUIRE RECOMPILING A NEW VERSION OF DRAKE????
## <TODO>
## https://github.com/RobotLocomotion/drake/blob/7c513516620b1e6001fd487e076c39f716027a79/solvers/snopt_solver.cc#L682
#
## Then resolve again with (potentially different settings, but using the previous answer as a warm start)
#def method6(mto, ic_list):
#    old_mto = mto
#    mto = make_mto()
#
#    # Warm start... but use a different warm start scheme
#    old_mto_dec_vals = old_mto.prog.GetSolution(old_mto.prog.decision_variables())
#    mto.prog.SetInitialGuessForAllVariables(old_mto_dec_vals)
#
#    mto.Solve()
#    return old_mto, mto

def make_mto(
             # Settings for just the trajectory optimization.
             expmt="pendulum",
             num_trajectories=16,
             num_samples=16,
             kMinimumTimeStep=0.2,
             kMaximumTimeStep=0.5,
             ic_list=None,
             warm_start=True,
             seed=1338,

             # Below are the NN-centric init options.
             use_dropout=True,
             nn_init=kaiming_uniform,
             nn_noise=1e-2,
             kNetConstructor=lambda: FCBIG(2, 32),
             use_constraint=True,
             cost_factor=None,
             initialize_params=True,
             reg_type="No",

             # Callback display settings.
             vis_cb_every_nth=None,
             vis_cb_display_rollouts=False,
             cost_cb_every_nth=None,

             i=0,
             snopt_overrides=[]):
    if ic_list is None:
        if expmt == "pendulum":
            ic_list=initial_conditions_grid(num_trajectories, (0, 2*math.pi), (-5., 5.))
        elif expmt == "cartpole":
            ic_list=initial_conditions_random_all_dims(num_trajectories, ((-3., 3.), (0., 2*math.pi), (-1., 1.), (-1., 1.)) )
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ic_list = None
    # ic_list = [(0., 0.)]
    # ic_list = np.array([(0. + .1*ti, 0.0) for ti in range(num_trajectories)])
    # ic_list = initial_conditions_grid(num_trajectories)
    # seed = 1776
    # seed = None
    # seed = np.random.randint(0, 2000); print("seed: {}".format(seed))
    # seed = 1338
    mto = MultipleTrajOpt(expmt,
                          num_trajectories, 
                          num_samples,
                          kMinimumTimeStep,
                          kMaximumTimeStep,
                          ic_list=ic_list,
                          warm_start=warm_start)

    ###############################################
    # Add a neural network!
    ###############################################
    # kNetConstructor = None
    # kNetConstructor = lambda: FC(2)
    # kNetConstructor = lambda: FCBIG(2)
    # kNetConstructor = lambda: MLPSMALL(2)
    # kNetConstructor = lambda: MLP(2)
    # reg_type = "No"
    # reg_type = "L1"
    # reg_type = "L2"
    if kNetConstructor is not None:
        mto.add_nn_params(kNetConstructor,
                          use_constraint    = use_constraint,
                          cost_factor       = cost_factor,
                          initialize_params = initialize_params, 
                          reg_type          = reg_type,
                          use_dropout       = True,
                          nn_init           = kaiming_uniform,
                          nn_noise          = 1e-2)
        
    if vis_cb_every_nth is not None:
        if vis_cb_display_rollouts:
            mto.add_multiple_trajectories_visualization_callback(vis_cb_every_nth, vis_ic_list=None)
        else:
            mto.add_multiple_trajectories_visualization_callback(vis_cb_every_nth, vis_ic_list=[])
    if cost_cb_every_nth is not None:
        mto.add_cost_and_constraint_printing_callback(cost_cb_every_nth)
    
    # Add in some SNOPT settings changes here!!!
    # Looks like we are getting good enough solutions!!!
    from pydrake.all import (SolverType)
    # mto.prog.SetSolverOption(SolverType.kSnopt, 'Verify level', -1)
    if expmt == "pendulum":
        mto.prog.SetSolverOption(SolverType.kSnopt, 'Print file', "/tmp/snopt.out")

#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Major feasibility tolerance', 2.0e-2) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Major optimality tolerance',  2.0e-2) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Minor feasibility tolerance', 2.0e-3) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Minor optimality tolerance',  2.0e-3) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Major feasibility tolerance', 1.0e-6) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Major optimality tolerance',  1.0e-6) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Minor feasibility tolerance', 1.0e-6) # default="1.0e-6"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Minor optimality tolerance',  1.0e-6) # default="1.0e-6"

        # Lower if nonlinear constraint are cheap to evaluate, else higher...
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Linesearch tolerance',  0.1) # default="0.9"

        mto.prog.SetSolverOption(SolverType.kSnopt, 'Major step limit',  0.5) # default="2.0e+0"
        mto.prog.SetSolverOption(SolverType.kSnopt, 'Time limit (secs)',  30.0) # default="9999999.0"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Reduced Hessian dimension',  10000) # Default="min{2000, n1 + 1}"
        mto.prog.SetSolverOption(SolverType.kSnopt, 'Hessian',  "full memory") # Default="10"
#        mto.prog.SetSolverOption(SolverType.kSnopt, 'Hessian updates',  30) # Default="10"
        mto.prog.SetSolverOption(SolverType.kSnopt, 'Major iterations limit',  9300000) # Default="9300"
        mto.prog.SetSolverOption(SolverType.kSnopt, 'Minor iterations limit',  50000) # Default="500"
        mto.prog.SetSolverOption(SolverType.kSnopt, 'Iterations limit',  50*10000) # Default="10000"

        # Factoriztion?
        mto.prog.SetSolverOption(SolverType.kSnopt, 'QPSolver Cholesky', True) # Default="*Cholesky/CG/QN"
        # mto.prog.SetSolverOption(SolverType.kSnopt, 'QPSolver CG', True) # Default="*Cholesky/CG/QN"
        # mto.prog.SetSolverOption(SolverType.kSnopt, 'QPSolver QN', True) # Default="*Cholesky/CG/QN"
    elif expmt == "cartpole":
        dircol.SetSolverOption(SolverType.kSnopt, 'Major step limit',  0.1) # default="2.0e+0"
        # TODO: add more here

    for setting_name, setting in snopt_overrides:
        print("Overrode {} = {}".format(setting_name, setting))
        mto.prog.SetSolverOption(SolverType.kSnopt, setting_name, setting)

    return mto

# The culprits of non-packability are:
# Decision Variables
# MathematicalProgram
# Bindings...
#
# Packer -----------------------------> Unpacker
# 1) Save decision varibales
# 2) Strip mto
# 3) Send dec_vals 
#      and stripped mto---->
#                               ----> 4) Recv
#                                     5) Recall mto.GenerateProg
#                                     6) Load in decision variables as initial guesses
#                                     7) Rely on the fact that my warm starters will pull from solution
#                                            if solved, else initial guess...
# SOOOO HACKYYYYYY!!!
def pack_mto(mto):
    if mto is None:
        return (None, None)
    dec_vals = get_solution_else_initial_guess(mto.prog, mto.prog.decision_variables())
    mto.clear_prog()
    return (mto, dec_vals)
def unpack_mto(mto, dec_vals):
    if mto is None:
        return None
    mto.generate_prog()
    #print("adding nn params with knetconstructor = ", mto.kNetConstructor)
    mto.add_nn_params(mto.kNetConstructor) # To make T decision variables...
    mto.prog.SetInitialGuessForAllVariables(dec_vals)
    #print("CHECK unpack_mto ", mto.prog.GetInitialGuess(mto.T))
    return mto

class MultipleTrajOpt(object):
    # Currently only set up to make pendulum examples
    def __init__(self, 
                 expmt, 
                 num_trajectories, num_samples,
                 kMinimumTimeStep, kMaximumTimeStep,
                 ic_list=None,
                 warm_start=True):#,
                 #seed=None):
        # assert expmt == "pendulum" # Disabling this as we bring cartpole into the fold...
        self.expmt = expmt
        self.num_trajectories = num_trajectories
        self.num_samples = num_samples
        self.kMinimumTimeStep = kMinimumTimeStep
        self.kMaximumTimeStep = kMaximumTimeStep
        self.num_inputs = 1
        self.num_states = 2
        self.ic_list = ic_list
        self.warm_start = warm_start
        self.cbs = [] # This list will contain which visualization cb's to call
        self.vis_cb_counter = 0
        self.kNetConstructor = None
        # For tracking NN costs and constraints
        # Don't seed here, seed in make_mto
        # self.seed = seed
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        #     torch.manual_seed(self.seed)

        # initial_conditions return a list of [num_trajectories x num_states] initial states
        # Use Russ's initial conditions, unless I pass in a function myself.
        if self.ic_list is None:
            self.ic_list = initial_conditions_Russ(self.num_trajectories)
        assert len(self.ic_list) == self.num_trajectories
        assert np.all( [len(ic) == self.num_states for ic in self.ic_list] )

        # Make the program
        self.generate_prog()

    # clear_prog and generate_prog below are used to wipe and reload
    # state that can not be copied/pickled when using mutliprocessing library!
    # (results in weird pybind11 error...)
    def clear_prog(self):
        #self.nn_cons = []
        #self.nn_costs = []
        #self.prog = None
        #self.h = None
        #self.u = None
        #self.x = None
        #self.T = None
        delattr(self, "nn_conss")
        delattr(self, "nn_costs")
        delattr(self, "prog")
        delattr(self, "h")
        delattr(self, "u")
        delattr(self, "x")
        delattr(self, "T")
            
    def generate_prog(self):
        self.nn_conss = []
        self.nn_costs = []

        plant = PendulumPlant()
        context = plant.CreateDefaultContext()
        dircol_constraint = DirectCollocationConstraint(plant, context)

        prog = MathematicalProgram()
        # K = prog.NewContinuousVariables(1,7,'K')
        def final_cost(x):
            return 100.*(cos(.5*x[0])**2 + x[1]**2)   

        h = [];
        u = [];
        x = [];
        xf = np.array([math.pi, 0.])
        # Declare the MathematicalProgram vars up front in a good order, so that 
        # prog.decision_variables will be result of np.hstack(h.flatten(), u.flatten(), x.flatten(), T.flatten())
        # for the following shapes:                          unfortunately, prog.decision_variables() will have these shapes:
        #   h = (num_trajectories, 1)                       | h = (num_trajectories, 1)
        #   u = (num_trajectories, num_inputs, num_samples) | u = (num_trajectories, num_samples, num_inputs)
        #   x = (num_trajectories, num_states, num_samples) | x = (num_trajectories, num_samples, num_states)
        #   T = (num_params)                                | T = (num_params)
        for ti in range(self.num_trajectories):
            h.append(prog.NewContinuousVariables(1,'h'+str(ti)))
        for ti in range(self.num_trajectories):
            u.append(prog.NewContinuousVariables(1, self.num_samples,'u'+str(ti)))
        for ti in range(self.num_trajectories):
            x.append(prog.NewContinuousVariables(2, self.num_samples,'x'+str(ti)))

        # Add in constraints
        for ti in range(self.num_trajectories):
            prog.AddBoundingBoxConstraint(self.kMinimumTimeStep, self.kMaximumTimeStep, h[ti])
            # prog.AddQuadraticCost([1.], [0.], h[ti]) # Added by me, penalize long timesteps

            x0 = np.array(self.ic_list[ti]) # TODO: hopefully this isn't subtley bad...
            prog.AddBoundingBoxConstraint(x0, x0, x[ti][:,0]) 

            # nudge = np.array([.2, .2])
            # prog.AddBoundingBoxConstraint(xf-nudge, xf+nudge, x[ti][:,-1])
            prog.AddBoundingBoxConstraint(xf, xf, x[ti][:,-1])

            # Do optional warm start here
            if self.warm_start:
                prog.SetInitialGuess(h[ti], [(self.kMinimumTimeStep+self.kMaximumTimeStep)/2])
                for i in range(self.num_samples):
                    prog.SetInitialGuess(u[ti][:,i], [0.])
                    x_interp = (xf-x0)*i/self.num_samples + x0
                    prog.SetInitialGuess(x[ti][:,i], x_interp)
                    # prog.SetInitialGuess(u[ti][:,i], np.array(1.0))

            for i in range(self.num_samples-1):
                AddDirectCollocationConstraint(dircol_constraint, h[ti], x[ti][:,i], x[ti][:,i+1], u[ti][:,i], u[ti][:,i+1], prog)

            for i in range(self.num_samples):
                prog.AddQuadraticCost([[2., 0.], [0., 2.]], [0., 0.], x[ti][:,i])
                prog.AddQuadraticCost([25.], [0.], u[ti][:,i])
                #u_var = u[ti][:i]
                #x_var = x[ti][:0]
                #prog.AddCost(2*x_var.dot(x_var))
                kTorqueLimit = 5
                prog.AddBoundingBoxConstraint([-kTorqueLimit], [kTorqueLimit], u[ti][:,i])
                # prog.AddConstraint(control, [0.], [0.], np.hstack([x[ti][:,i], u[ti][:,i], K.flatten()]))
                # prog.AddConstraint(u[ti][0,i] == (3.*sym.tanh(K.dot(control_basis(x[ti][:,i]))[0])))  # u = 3*tanh(K * m(x))
                
            # prog.AddCost(final_cost, x[ti][:,-1])
            # prog.AddCost(h[ti][0]*100) # Try to penalize using more time than it needs?

        # Setting solver options
        #prog.SetSolverOption(SolverType.kSnopt, 'Verify level', -1)  # Derivative checking disabled. (otherwise it complains on the saturation)
        #prog.SetSolverOption(SolverType.kSnopt, 'Print file', "/tmp/snopt.out")

        # Save references
        self.prog = prog
        self.h = h
        self.u = u
        self.x = x
        self.T = []

    def add_nn_params(self,
                      kNetConstructor, 
                      use_constraint    = True,
                      cost_factor       = None,
                      initialize_params = True, 
                      reg_type          = "No",
                      use_dropout       = True,
                      nn_init           = kaiming_uniform,
                      nn_noise          = 1e-2):

        self.kNetConstructor = kNetConstructor

        # Determine num_params and add them to the prog.
        dummy_net = self.kNetConstructor()
        self.num_params = sum(tensor.nelement() for tensor in dummy_net.parameters())
        self.T = self.prog.NewContinuousVariables(self.num_params, 'T')

        if initialize_params:
            # VERY IMPORTANT!!!! - PRELOAD T WITH THE NET'S INITIALIZATION.
            # DEFAULT ZERO INITIALIZATION WILL GIVE YOU ZERO GRADIENTS!!!!
            params_loaded = 0
            initial_guess = [AutoDiffXd]*self.num_params
            for param in dummy_net.parameters(): # Here's where we make a dummy net. Let's seed this?
                param_values = param.data.numpy().flatten()
                for i in range(param.data.nelement()):
                    initial_guess[params_loaded + i] = param_values[i]
                params_loaded += param.data.nelement()
            self.prog.SetInitialGuess(self.T, np.array(initial_guess))

        # Add No/L1/L2 Regularization to model params T
        assert reg_type in ("No", "L1", "L2")
        if reg_type == "No":
            pass
        elif reg_type == "L1":
            def L1Cost(T):
                return sum([t**2 for t in T])
            self.prog.AddCost(L1Cost, self.T)
        elif reg_type == "L2":
            self.prog.AddQuadraticCost(np.eye(len(self.T)), [0.]*len(self.T), self.T)


        # Add the neural network constraint to all time steps
        for ti in range(self.num_trajectories):
            for i in range(self.num_samples):
                u_ti = self.u[ti][0,i]
                x_ti = self.x[ti][:,i]
                # Only one output value, so let's have lb and ub of just size one!
                constraint = make_NN_constraint(self.kNetConstructor, self.num_inputs, self.num_states, self.num_params)
                lb         = np.array([-.0001])
                ub         = np.array([.0001])
                var_list   = np.hstack((u_ti, x_ti, self.T))
                if use_constraint:
                    cons = self.prog.AddConstraint(lambda x: [constraint(x)], lb, ub, var_list)
                    self.nn_conss.append(cons)
                if cost_factor is not None:
                    cost = self.prog.AddCost(lambda x: cost_factor*constraint(x)**2, var_list)
                    self.nn_costs.append(cost)

        
    def add_multiple_trajectories_visualization_callback(self, every_nth, vis_ic_list=None):
        # initial_conditions return a list of [<any> x num_states] initial states
        # Use Russ's initial conditions, unless I pass in a function myself.
        self.vis_ic_list = vis_ic_list
        if vis_ic_list is None:
            vis_ic_list = initial_conditions_Russ(self.num_trajectories)
        assert np.all( [len(ic) == self.num_states for ic in vis_ic_list] )

        def cb(huxT):
            print(" {}".format(self.vis_cb_counter), end='')
            if (self.vis_cb_counter) % every_nth != 0:
                return
            #print()
            
            # Unpack the serialized variables
            num_h = self.num_trajectories
            num_u = self.num_trajectories*self.num_samples*self.num_inputs
            num_x = self.num_trajectories*self.num_samples*self.num_states
            h = huxT[                  : num_h            ].reshape((self.num_trajectories, 1))
            u = huxT[num_h             : num_h+num_u      ].reshape((self.num_trajectories, self.num_samples, self.num_inputs))
            x = huxT[num_h+num_u       : num_h+num_u+num_x].reshape((self.num_trajectories, self.num_samples, self.num_states))
            # Swap last two axes here to get it into the format we're used to above
            u = np.swapaxes(u, 1, 2)
            x = np.swapaxes(x, 1, 2)
            T = huxT[num_h+num_u+num_x :                  ]

            # Visualize the trajectories
            for ti in range(self.num_trajectories):
                h_sol = h[ti][0]
                if h_sol <= 0:
                    print("bad h_sol")
                    continue
                breaks = [h_sol*i for i in range(self.num_samples)]
                knots = x[ti]
                x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
                t_samples = np.linspace(breaks[0], breaks[-1], self.num_samples*3)
                x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])

                # 1) Visualize the trajectories
                plot_trajectory(x_samples, "state_scatter", self.expmt, create_figure=False) 

            if len(T) != 0:
                for ic in vis_ic_list:
                    # 2) Then visualize what the policy would say to do from (possibly the same) initial conditions.
                    h_sol = (0.01+0.2)/2 # TODO: improve this hard coding?
                    _, pi_x_samples, _ = self.__rollout_policy_given_params(h_sol, T, ic)
                    plot_trajectory(pi_x_samples, "state_scatter", self.expmt, create_figure=False, symbol=':') 
                
            plt.show()
            
        self.cbs.append(cb)

    def add_cost_and_constraint_printing_callback(self, every_nth):
        def cb(decision_vars):
            if (self.vis_cb_counter) % every_nth != 0:
                return

            # Get the total cost
            all_costs = self.prog.EvalBindings(self.prog.GetAllCosts(), decision_vars)
            nn_costs = self.prog.EvalBindings(self.nn_costs, decision_vars)

            # Get the total cost of the constraints.
            # Additionally, the number and extent of any constraint violations.
            violated_constraint_count = 0
            violated_constraint_cost  = 0
            constraint_cost           = 0
            nn_violated_constraint_cost = 0
            for constraint in self.prog.GetAllConstraints():
                val = self.prog.EvalBinding(constraint, decision_vars)

                # Consider switching to DoCheckSatisfied if you can find the binding...
                nudge = 1e-1 # This much constraint violation is not considered bad...
                lb = constraint.evaluator().lower_bound()
                ub = constraint.evaluator().upper_bound()
                good_lb = np.all( np.less_equal(lb, val+nudge) )
                good_ub = np.all( np.greater_equal(ub, val-nudge) )
                if not good_lb or not good_ub:
                    # print("{} <= {} <= {}".format(lb, val, ub))
                    violated_constraint_count += 1
                    violated_constraint_cost += np.sum(np.abs(val))
                constraint_cost += np.sum(np.abs(val))

            # TODO: dedup!
            for constraint in self.nn_conss:
                val = self.prog.EvalBinding(constraint, decision_vars)

                # Consider switching to DoCheckSatisfied if you can find the binding...
                nudge = 1e-1 # This much constraint violation is not considered bad...
                lb = constraint.evaluator().lower_bound()
                ub = constraint.evaluator().upper_bound()
                good_lb = np.all( np.less_equal(lb, val+nudge) )
                good_ub = np.all( np.greater_equal(ub, val-nudge) )
                if not good_lb or not good_ub:
                    nn_violated_constraint_cost += np.sum(np.abs(val))

            print("total cost: {: .2f} + ({: .2f}) | \tconstraint {: .2f} \tbad {}, {: .2f} + ({: .2f})".format(
                sum(all_costs)-sum(nn_costs), sum(nn_costs), 
                constraint_cost, 
                violated_constraint_count, violated_constraint_cost - nn_violated_constraint_cost, nn_violated_constraint_cost))
        self.cbs.append(cb)

    def Solve(self):
        if self.cbs:
            def cb_all(huxT):
                for i, cb in enumerate(self.cbs):
                    cb(huxT)
                self.vis_cb_counter +=1
            self.prog.AddVisualizationCallback(cb_all, self.prog.decision_variables())
        start = time.time()
        result = self.prog.Solve()
        dur = time.time() - start
        print("RESULT: {} TOTAL ELAPSED TIME: {}".format(result, dur))
        return result

    def PrintFinalCostAndConstraint(self):
        sol_costs = np.hstack([self.prog.EvalBindingAtSolution(cost) for cost in self.prog.GetAllCosts()])
        sol_constraints = np.hstack([self.prog.EvalBindingAtSolution(constraint) for constraint in self.prog.GetAllConstraints()])
        print("TOTAL cost: {:.2f} | constraint {:.2f}".format(sum(sol_costs), sum(sol_constraints)))
        print()

    def create_net(self, from_initial_guess=False):
        if from_initial_guess:
            nn = create_nn(self.kNetConstructor, self.prog.GetInitialGuess(self.T))
        else:
            nn = create_nn(self.kNetConstructor, self.prog.GetSolution(self.T))
        return nn

    def print_pi_divergence(self, ti):
        #nn = create_nn(self.kNetConstructor, self.prog.GetSolution(self.T))
        nn = self.create_net()
        u_vals = self.prog.GetSolution(self.u[ti])
        x_vals = self.prog.GetSolution(self.x[ti])

        print("u_val-Pi(x_val)= diff")
        for i in range(self.num_samples):
            u_val = u_vals[0,i]
            x_val = x_vals[:,i]
            u_pi = NNInferenceHelper_double(nn, x_val)[0]
            print( "({: .2f})-({: .2f})= {: .2f}".format(u_val, u_pi, u_val - u_pi) )

    def __rollout_policy_given_params(self, h_sol, params_list, ic=None, WALLCLOCK_TIME_LIMIT=6):
        nn_policy = create_nn_policy_system(self.kNetConstructor, params_list)
        simulator, _, logger = simulate_and_log_policy_system(nn_policy, self.expmt, ic)
        # simulator.get_integrator().set_target_accuracy(1e-1)

        start = time.time()
        while simulator.get_context().get_time() < h_sol*self.num_samples:
            if time.time() - start > WALLCLOCK_TIME_LIMIT:
                print("quit simulation early at {}/{} due to exceeding time limit".format(
                    simulator.get_context().get_time(), h_sol*self.num_samples))
                break
            simulator.StepTo(simulator.get_context().get_time()+0.001)
        # simulator.StepTo(h_sol*self.num_samples)

        t_samples = logger.sample_times()
        x_samples = logger.data()
        return t_samples, x_samples, logger

    def __rollout_policy_at_solution(self, ti_or_ic=None):
        if isinstance(ti_or_ic, int):
            ic = self.prog.GetSolution(self.x[ti_or_ic])[:,0]
            h_sol = self.prog.GetSolution(self.h[ti_or_ic])[0]
        else:
            ic = ti_or_ic
            h_sol = (0.01+0.2)/2
        params_list = self.prog.GetSolution(self.T)
        return self.__rollout_policy_given_params(h_sol, params_list, ic=ic)

    def plot_policy(self, plot_type, ti_or_ic, create_figure=True):
        _, x_samples, _ = self.__rollout_policy_at_solution(ti_or_ic)

        plot_trajectory(x_samples, plot_type, expmt=self.expmt, create_figure=create_figure)
        

    def plot_all_policies(self, plot_type, ti_or_ics=None):
        if ti_or_ics is None:
            ti_or_ics = list(range(self.num_trajectories))
        ax = plt.figure()
        plt.title('Pendulum policy rollouts')
        for ti_or_ic in ti_or_ics:
            self.plot_policy(plot_type, ti_or_ic, create_figure=False)
        plt.xlim((-math.pi*1.5, math.pi*1.5))
        plt.ylim((-7, 7))

    def render_policy(self, ti_or_ic):
        _, _, logger = self.__rollout_policy_at_solution(ti_or_ic)

        return render_trajectory(logger)

#    def render_all_policies(self, ti_or_ics=None):
#        if ti_or_ics is None:
#            ti_or_ics = list(range(self.num_trajectories))
#        ret = []
#        for ti_or_ic in ti_or_ics:
#            ret.append( self.render_policy(ti_or_ic) )
#        return ret

    def get_trajectory_data(self, ti):
        h_sol = self.prog.GetSolution(self.h[ti])[0]
        breaks = [h_sol*i for i in range(self.num_samples)]
        knots = self.prog.GetSolution(self.x[ti])
        x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
        # t_samples = np.linspace(breaks[0], breaks[-1], 45)
        t_samples = np.linspace(breaks[0], breaks[-1], 100)
        x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])
        return x_trajectory, t_samples, x_samples

    # Visualize with a static arrows plot.
    def plot_single_trajectory(self, ti, plot_type, create_figure=True):
        _, _, x_samples = self.get_trajectory_data(ti)

        plot_trajectory(x_samples, plot_type, expmt=self.expmt, create_figure=create_figure)

    def plot_all_trajectories(self, plot_type):
        plt.figure()
        plt.title('Pendulum trajectories')
        for ti in range(self.num_trajectories):
            self.plot_single_trajectory(ti, plot_type, create_figure=False)

    # Visualize the result as a video.
    def render_single_trajectory(self, ti):
        from traj.visualizer import PendulumVisualizer
        x_trajectory, _, _ = self.get_trajectory_data(ti)
        
        return render_trajectory(x_trajectory)







