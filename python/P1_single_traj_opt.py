from __future__ import print_function, absolute_import

import math
import numpy as np
#%matplotlib notebook  
#import matplotlib.pyplot as plt

from pydrake.examples.pendulum import (PendulumPlant, PendulumState)
from pydrake.all import ( DirectCollocation, PiecewisePolynomial, SolutionResult )

cb_counter = 0
def do_single_basic_pend_traj_opt(should_init):
    plant = PendulumPlant()
    context = plant.CreateDefaultContext()

    kNumTimeSamples = 15
    kMinimumTimeStep = 0.01
    kMaximumTimeStep = 0.2

    dircol = DirectCollocation(plant, context, kNumTimeSamples, kMinimumTimeStep, kMaximumTimeStep)

    dircol.AddEqualTimeIntervalsConstraints()

    kTorqueLimit = 3.0  # N*m.
    u = dircol.input()
    dircol.AddConstraintToAllKnotPoints(-kTorqueLimit <= u[0])
    dircol.AddConstraintToAllKnotPoints(u[0] <= kTorqueLimit)

    initial_state = PendulumState()
    initial_state.set_theta(0.0)
    initial_state.set_thetadot(0.0)
    dircol.AddBoundingBoxConstraint(initial_state.get_value(), initial_state.get_value(), dircol.initial_state())
    # dircol.AddLinearConstraint(dircol.initial_state() == initial_state.get_value())

    final_state = PendulumState()
    final_state.set_theta(math.pi)
    final_state.set_thetadot(0.0)
    dircol.AddBoundingBoxConstraint(final_state.get_value(), final_state.get_value(), dircol.final_state())
    # dircol.AddLinearConstraint(dircol.final_state() == final_state.get_value())

    R = 10  # Cost on input "effort".
    dircol.AddRunningCost(R*u[0]**2)

    if should_init:
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold([0., 4.], [initial_state.get_value(), final_state.get_value()])
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    def cb(decision_vars):
        global cb_counter
        cb_counter += 1
        if cb_counter % 10 != 1:
            return

        # Get the total cost
        all_costs = dircol.EvalBindings(dircol.GetAllCosts(), decision_vars)
        # :all_constraints = dircol.EvalBindings(dircol.GetAllConstraints(), decision_vars)

        # Get the total cost of the constraints.
        # Additionally, the number and extent of any constraint violations.
        violated_constraint_count = 0
        violated_constraint_cost  = 0
        constraint_cost           = 0
        for constraint in dircol.GetAllConstraints():
            val = dircol.EvalBinding(constraint, decision_vars)

            # TODO: switch to DoCheckSatisfied...
            nudge = 1e-1 # This much constraint violation is not considered bad...
            lb = constraint.evaluator().lower_bound()
            ub = constraint.evaluator().upper_bound()
            good_lb = np.all( np.less_equal(lb, val+nudge) )
            good_ub = np.all( np.greater_equal(ub, val-nudge) )
            if not good_lb or not good_ub:
                # print("val,lb,ub: ", val, lb, ub)
                violated_constraint_count += 1
                violated_constraint_cost += np.sum(val)
            constraint_cost += np.sum(val)
        print("total cost: {:.2f} | constraint {:.2f}, bad {}, {:.2f}".format(
            sum(all_costs), constraint_cost, violated_constraint_count, violated_constraint_cost))
#    dircol.AddVisualizationCallback(cb, np.array(dircol.decision_variables()))
    result = dircol.Solve()
    assert(result == SolutionResult.kSolutionFound)

    sol_costs = np.hstack([dircol.EvalBindingAtSolution(cost) for cost in dircol.GetAllCosts()])
    sol_constraints = np.hstack([dircol.EvalBindingAtSolution(constraint) for constraint in dircol.GetAllConstraints()])
#    print("TOTAL cost: {:.2f} | constraint {:.2f}".format(sum(sol_costs), sum(sol_constraints)))
#    print()
 
# x_trajectory = dircol.ReconstructStateTrajectory()
# x_knots = np.hstack([x_trajectory.value(t) for t in np.linspace(x_trajectory.start_time(),x_trajectory.end_time(),100)])
# print x_knots[-1]
# 
# # plt.plot(x_knots[0,:], x_knots[1,:])
 
import sys
import time

should_init=True
iters = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]
print("should_init: {}".format(should_init))
for i in iters:
    start = time.time()
    for _ in range(i):
        do_single_basic_pend_traj_opt(should_init)
    dur = time.time() - start
    print("{}: | dur: {:.2} | avg: {:.2}".format(i, dur, dur/i))

