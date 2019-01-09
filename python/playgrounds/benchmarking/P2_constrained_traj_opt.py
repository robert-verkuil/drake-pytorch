from __future__ import print_function, absolute_import
import math
import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    AutoDiffXd, Expression, Variable,
    MathematicalProgram, SolverType, SolutionResult,
    DirectCollocationConstraint, AddDirectCollocationConstraint,
    PiecewisePolynomial,
    DiagramBuilder, SignalLogger, Simulator, VectorSystem,
)

from nn_system.NNSystemHelper import (
    make_NN_constraint,
)
from nn_system.networks import *

from traj.dircol import (
    make_multiple_dircol_trajectories,
    make_real_dircol_mp,
)
from traj.vis import (
    add_multiple_trajectories_visualization_callback,
    add_visualization_callback,
    create_nn_policy_system,
    plot_multiple_dircol_trajectories,
    simulate_and_log_policy_system,
    visualize_trajectory,
)


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

cb_counter = 0
def do(num_trajectories):
    start("do")
    #num_trajectories = 50
    num_samples      = 15
    n_theta, n_theta_dot = (5, 3)
    theta_bounds     = -math.pi, math.pi
    theta_dot_bounds = -5, 5
    def initial_conditions(ti):
        # Russ's
        return (.8 + math.pi - .4*ti, 0.0)
    def initial_conditions2(ti):
        # Have ti index into a grid over some state space bounds
        theta_range     = np.linspace(theta_bounds[0], theta_bounds[1], n_theta)
        theta_dot_range = np.linspace(theta_dot_bounds[0], theta_dot_bounds[1], n_theta_dot)
        r = int(ti / n_theta)
        c = ti % n_theta
        return (theta_range[c], theta_dot_range[r])
    def initial_conditions3(ti):
        # random state over some state-space bounds
        rand1, rand2 =  np.random.random(2)
        theta     =  (theta_bounds[1] - theta_bounds[0]) * rand1 + theta_bounds[0]
        theta_dot =  (theta_dot_bounds[1] - theta_dot_bounds[0]) * rand2 + theta_dot_bounds[0]
        return theta, theta_dot


    start("make_multiple_dircol_trajectories")
    prog, h, u, x = make_multiple_dircol_trajectories(
                        num_trajectories, 
                        num_samples)#,
    #                     initial_conditions=initial_conditions2)
    end("make_multiple_dircol_trajectories")


    start("net init")
    if (False):
        # kNetConstructor = lambda: FC(2)
        kNetConstructor = lambda: FCBIG(2)
        # kNetConstructor = lambda: MLPSMALL(2)
        # kNetConstructor = lambda: MLP(2)
        num_params = sum(tensor.nelement() for tensor in kNetConstructor().parameters())
        num_inputs = 1
        num_states = 2
        total_params = sum((num_inputs, num_states, num_params))
        T = prog.NewContinuousVariables(num_params, 'T')


        # Add No/L1/L2 Regularization to model params T
        reg_type= "No"
        # reg_type= "L1"
        # reg_type= "L2"
        if reg_type == "No":
            pass
        elif reg_type == "L1":
            def L1Cost(T):
                return sum([t**2 for t in T])
            prog.AddCost(L1Cost, T)
        elif reg_type == "L2":
            prog.AddQuadraticCost(np.eye(len(T)), [0.]*len(T), T)


        # VERY IMPORTANT!!!! - PRELOAD T WITH THE NET'S INITIALIZATION.
        # DEFAULT ZERO INITIALIZATION WILL GIVE YOU ZERO GRADIENTS!!!!
        params_loaded = 0
        initial_guess = [AutoDiffXd]*num_params
        for param in kNetConstructor().parameters():
            param_values = param.data.numpy().flatten()
            for i in range(param.data.nelement()):
                initial_guess[params_loaded + i] = param_values[i]
            params_loaded += param.data.nelement()
        prog.SetInitialGuess(T, np.array(initial_guess))


        for ti in range(num_trajectories):
            for i in range(num_samples):
                u_ti = u[ti][0,i]
                x_ti = x[ti][:,i]
                # Only one output value, so let's have lb and ub of just size one!
                constraint_inner = make_NN_constraint(kNetConstructor, num_inputs, num_states, num_params)
                def constraint(x):
                    end("SNOPT?")
                    start("constraint")
                    ret = constraint_inner(x)
                    end("constraint")
                    start("SNOPT?")
                    return ret
                lb         = np.array([-.1])
                ub         = np.array([.1])
                var_list   = np.hstack((u_ti, x_ti, T))
                prog.AddConstraint(constraint, lb, ub, var_list)
        #        prog.AddCost(lambda x: constraint(x)[0]**2, var_list)
    end("net init")


    def print_start():
        print("calling the solver")
    def print_end():
        print("finished solve")

    # Add total cost and constraint value reporting
    def cb(decision_vars):
        global cb_counter
        cb_counter += 1
        if cb_counter % 1 != 1:
            pass
            #return

        all_costs = prog.EvalBindings(prog.GetAllCosts(), decision_vars)
        all_constraints = prog.EvalBindings(prog.GetAllConstraints(), decision_vars)
        print("total cost: {:.2f} | constraint {:.2f}".format(sum(all_costs), sum(all_constraints)))

    prog.AddVisualizationCallback(cb, np.array(prog.decision_variables()))
    print_start()
    start("Solve")
    start("SNOPT?")
    result = prog.Solve()
    end("SNOPT?")
    end("Solve")
    print_end()
    end("do")
    sol_costs = np.hstack([prog.EvalBindingAtSolution(cost) for cost in prog.GetAllCosts()])
    sol_constraints = np.hstack([prog.EvalBindingAtSolution(constraint) for constraint in prog.GetAllConstraints()])
    print("TOTAL cost: {:.2f} | constraint {:.2f}".format(sum(sol_costs), sum(sol_constraints)))
    print()
    
    import io, json
    with io.open('/home/rverkuil/Desktop/data.cpuprofile', 'w', encoding='utf-8') as f:
        f.write(unicode("["))
        all_data = ",".join([json.dumps(data, ensure_ascii=False) for data in mylog])
        f.write(unicode(all_data))
        f.write(unicode("]"))


#if __name__ == "__main__":
#    do()
#
#    from pytracing import TraceProfiler
#    tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
#    with tp.traced():
#        do()



import sys
import time

should_init=True
iters = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]
#iters = [25]
print("should_init: {}".format(should_init))
for i in iters:
    start_t = time.time()
    do(i)
    dur = time.time() - start_t
    print("{}: | dur: {:.2} | avg: {:.2}".format(i, dur, dur/i))

