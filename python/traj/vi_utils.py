from __future__ import print_function, absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from pydrake.examples.pendulum import PendulumPlant
from pydrake.all import (
    DiagramBuilder,
    DirectCollocation, 
    DynamicProgrammingOptions,
    FittedValueIteration,
    PeriodicBoundaryCondition,
    PiecewisePolynomial, 
    SignalLogger,
    Simulator,
    SolutionResult,
    VectorSystem
)

from traj.visualizer import PendulumVisualizer # <- weird!

##################################
# PENDULUM
##################################
vis_cb_counter = 0
dircol = None
plant = None
context = None
def do_dircol(ic=(-1., 0.), num_samples=16, min_timestep=0.2, max_timestep=0.5, warm_start="linear", seed=1776, should_vis=False):
    global dircol
    global plant
    global context
    plant = PendulumPlant()
    context = plant.CreateDefaultContext()
    dircol = DirectCollocation(plant, context, num_time_samples=num_samples,
                               # minimum_timestep=0.01, maximum_timestep=0.01)
                               minimum_timestep=min_timestep, maximum_timestep=max_timestep)

#     dircol.AddEqualTimeIntervalsConstraints()

#     torque_limit = input_limit  # N*m.
    torque_limit = 5.
    u = dircol.input()
    dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
    dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

    initial_state = ic
    dircol.AddBoundingBoxConstraint(initial_state, initial_state,
                                    dircol.initial_state())
    final_state = (math.pi, 0.)
    dircol.AddBoundingBoxConstraint(final_state, final_state,
                                    dircol.final_state())

#     R = 100  # Cost on input "effort".
    u = dircol.input()
    x = dircol.state()
#     print(x)
    dircol.AddRunningCost(2*((x[0]-math.pi)*(x[0]-math.pi) + x[1]*x[1]) + 25*u.dot(u))

    # Add a final cost equal to the total duration.
#     dircol.AddFinalCost(dircol.time())

    if warm_start == "linear":
        initial_u_trajectory = PiecewisePolynomial()
        initial_x_trajectory = \
            PiecewisePolynomial.FirstOrderHold([0., 4.],
                                           np.column_stack((initial_state,
                                                            final_state)))
        dircol.SetInitialTrajectory(initial_u_trajectory, initial_x_trajectory)

#     elif warm_start == "random":
#         assert isinstance(seed, int)
#         np.random.seed(seed)
#         breaks = np.linspace(0, 4, 21).reshape((-1, 1))  # using num_time_samples
#         u_knots = np.random.rand(1, 21)*2*input_limit-input_limit # num_inputs vs num_samples? 
#         x_knots = np.random.rand(2, 21)*2*3-3 # num_states vs num_samples? 
#         initial_u_trajectory = PiecewisePolynomial.Cubic(breaks, u_knots, False)
#         initial_x_trajectory = PiecewisePolynomial.Cubic(breaks, x_knots, False)
#         dircol.SetInitialTrajectory(initial_u_trajectory, initial_x_trajectory)

    
    def MyVisualization(sample_times, values):
        global vis_cb_counter

        vis_cb_counter += 1
        if vis_cb_counter % 10 != 0:
            return

        x, x_dot = values[0], values[1]
        plt.plot(x, x_dot, '-o', label=vis_cb_counter)
    
    if should_vis:
        plt.figure()
        plt.title('Tip trajectories')
        plt.xlabel('x')
        plt.ylabel('x_dot')
        dircol.AddStateTrajectoryCallback(MyVisualization)

    from pydrake.all import (SolverType)
    dircol.SetSolverOption(SolverType.kSnopt, 'Major feasibility tolerance', 1.0e-6) # default="1.0e-6"
    dircol.SetSolverOption(SolverType.kSnopt, 'Major optimality tolerance',  1.0e-6) # default="1.0e-6"
    dircol.SetSolverOption(SolverType.kSnopt, 'Minor feasibility tolerance', 1.0e-6) # default="1.0e-6"
    dircol.SetSolverOption(SolverType.kSnopt, 'Minor optimality tolerance',  1.0e-6) # default="1.0e-6"

    result = dircol.Solve()
    if result != SolutionResult.kSolutionFound:
        print("result={}".format(result))

    return dircol, result


# Either ics or trajs should be not not None
# so that we have something to rollout or directly plot
def graph_vi_policy_vs_traj_knot_scatter(vi_policy, 
        ics_or_dircols,
        combine_vi_policy_and_scatter=True,
        plot_residual=True):
    # using_ics = not isinstance(ics_or_dircols[0], DirectCollocation) and not isinstance(ics_or_dircols[0], FakeDircol) 
    using_ics = 'numpy' in str(type(ics_or_dircols[0]))
    print("using_ics= ", using_ics)

    def eval_policy(x):
        mesh = vi_policy.get_mesh()
        ovs  = vi_policy.get_output_values()
        return mesh.Eval(ovs, x)

    # Do dircol from them
    # Compare policy at each knot point of the dircol solution!!!
    knot_ics, knot_expected_us, knot_found_us, knot_SSE = [], [], [], 0
    traj_ics, traj_expected_us, traj_found_us, traj_SSE = [], [], [], 0
    for ic_or_dircol in ics_or_dircols:
        if using_ics:
            ic = ic_or_dircol
            dircol, result = do_dircol(ic=ic, warm_start="linear", seed=1776, should_vis=False)
        else:
            dircol = ic_or_dircol
        # Could compare just the knot points...
        times   = dircol.GetSampleTimes().T
        x_knots = dircol.GetStateSamples().T
        u_knots = dircol.GetInputSamples().T
    #     print(times.shape, x_knots.shape, u_knots.shape)
        assert len(times) == len(x_knots) and len(x_knots) == len(u_knots)
        for t, x, u in zip(times, x_knots, u_knots):
            expected_u = eval_policy(x)
            found_u    = u
    #         print(expected_u, found_u)
            knot_ics.append(x)
            knot_expected_us.append(expected_u)
            knot_found_us.append(found_u)
            knot_SSE += (found_u - expected_u)**2
        
        # ...and the interpolation points too!
        x_trajectory = dircol.ReconstructStateTrajectory()
        u_trajectory = dircol.ReconstructInputTrajectory()
        for t in np.linspace(times[0], times[-1], 100): # Pick uniform 100 times along the trajectory!
            x = x_trajectory.value(t)
            u = u_trajectory.value(t)
            expected_u = eval_policy(x)
            found_u    = u[0] # To go from shape (1, 1) -> (1,)
            traj_ics.append(x)
            traj_expected_us.append(expected_u)
            traj_found_us.append(found_u)
            traj_SSE += (found_u - expected_u)**2

    knot_MSE = knot_SSE / len(ics_or_dircols)
    traj_MSE = traj_SSE / len(ics_or_dircols)

    print("knot_MSE: {}, traj_MSE: {}".format(knot_MSE, traj_MSE))

    ics, expected_us, found_us = knot_ics, knot_expected_us, knot_found_us
    # ics, expected_us, found_us = traj_ics, traj_expected_us, traj_found_us
    print(list((np.array(l).shape for l in (ics, expected_us, found_us))))

    # Extract Q, Qdot from the policy
    sets = vi_policy.get_mesh().get_input_grid()
    lists = [sorted(list(s)) for s in sets]
    [Q, Qdot] = np.meshgrid(*lists)

    # Plot expected vs actual, in a 3d graph, since we have a 2d state space...
    # Here's the reference graph, in all of it's glory...
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # TODO: Clean up this mess!
    if not combine_vi_policy_and_scatter:
        if plot_residual:
            plot_codes = (131, 132, 133)
        else:
            plot_codes = (121, 122)
    else:
        if plot_residual:
            plot_codes = (121, 420, 122)
        else:
            plot_codes = (111,)
    ax = fig.add_subplot(plot_codes[0], projection='3d')
    Pi = np.reshape(vi_policy.get_output_values(), Q.shape)
    surf = ax.plot_surface(Q, Qdot, Pi, rstride=1, cstride=1,
                            cmap=cm.jet)
    ax.set_xlim((0., 2*math.pi))
    ax.set_ylim((-10., 10.))

    # Plot expected vs actual, in a 3d graph, since we have a 2d state space...
    # 3d scatter for the actuals?
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    xs, ys = zip(*ics)
    zs = found_us
    if not combine_vi_policy_and_scatter:
        ax2 = fig.add_subplot(plot_codes[1], projection='3d')
        ax2.scatter(xs, ys, zs, c='b', marker='^', s=1, alpha=0.25)
    else:
        ax.scatter(xs, ys, zs, c='b', marker='^', s=1, alpha=0.25)

    # Plot the residuals here!
    if plot_residual:
        ax3 = fig.add_subplot(plot_codes[2], projection='3d')
        xs, ys = zip(*ics)
        zs = np.array(expected_us) - np.array(found_us)
        ax3.scatter(xs, ys, zs, c='r', marker='^')

