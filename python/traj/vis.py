# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import copy
import math
import numpy as np
import random
import torch

import matplotlib.pyplot as plt

from pydrake.all import (
    DirectCollocation, 
    DiagramBuilder,
    FloatingBaseType,
    PiecewisePolynomial, 
    RigidBodyTree, 
    RigidBodyPlant,
    SignalLogger,
    Simulator,
    SolutionResult,
    VectorSystem,
)
from pydrake.examples.acrobot import AcrobotPlant
from pydrake.examples.pendulum import PendulumPlant
from underactuated import (
    PlanarRigidBodyVisualizer
)

from nn_system.NNSystem import NNSystem, NNSystem_ # How does this work????
from traj.visualizer import PendulumVisualizer

from IPython.display import HTML

# The purpose of this file is to house a bunch of utilities for
# visualizing SINGLE trajectories.  It should NOT know anything 
# about multiple trajectory optimization problems.

def state_to_tip_coord_pendulum(state_vec):
    # State: (x, theta, x_dot, theta_dot)
    theta, theta_dot = state_vec
    pole_length = 0.5 # manually looked this up
    return (pole_length*np.sin(theta), pole_length*(-np.cos(theta)))
def state_to_tip_coord_cartpole(state_vec):
    # State: (x, theta, x_dot, theta_dot)
    x, theta, _, _ = state_vec
    pole_length = 0.5 # manually looked this up
    return (x-pole_length*np.sin(theta), pole_length*(-np.cos(theta)))
def state_to_tip_coord_acrobot(state_vec):
    # State: (theta1, theta2, theta1_dot, theta2_dot)
    theta1, theta2, _, _ = state_vec
    link1_length = 1
    link2_length = 2
    return (-link1_length*np.sin(theta1)  -link2_length*np.sin(theta1+theta2), 
            -link1_length*np.cos(theta1)  -link2_length*np.cos(theta1+theta2))
state_to_tip_coord_fns = {
    "pendulum": state_to_tip_coord_pendulum,
    "cartpole": state_to_tip_coord_cartpole,
    "acrobot": state_to_tip_coord_acrobot,
}
# This function can either plot the (x,y) of the tip of the pendulum, or the 
# (theta, theta_dot) in the actual state space, with a scatter or quiver plot.
def plot_trajectory(x_samples, plot_type, expmt, create_figure=True, symbol='-'):
    assert expmt in state_to_tip_coord_fns.keys()
    assert plot_type in ("tip_scatter", "tip_quiver", "state_scatter", "state_quiver")
    if create_figure:
        plt.figure()
        plt.title(plot_type)
    if "tip" in plot_type:
        plt.xlabel('x')
        plt.ylabel('y')
    elif "state" in plot_type:
        plt.xlabel('theta')
        plt.ylabel('theta_dot')

    if "tip" in plot_type:
        coords = [state_to_tip_coord_fns[expmt](state) for state in x_samples.T]
    elif "state" in plot_type:
        coords = [state for state in x_samples.T]

    x, y = zip(*coords)
    if "scatter" in plot_type:
        plt.plot(x, y, symbol+'o')
    elif "quiver" in plot_type:
        x = np.array(x)
        y = np.array(y)
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.plot(x[0], y[0], 'go')
    plt.plot(x[-1], y[-1], 'ro')

# Visualize the result as a video.
def render_trajectory(x_trajectory_or_logger):
    vis = PendulumVisualizer()
    ani = vis.animate(x_trajectory_or_logger, repeat=True)
    plt.close(vis.fig)
    return HTML(ani.to_html5_video())

# Given a policy system, (e.g. created by FittedValueIteraton or nn_system.NNSystem(pytorch_net_object))
# creates a simulator that can be used to visualize the policy.
tree = None
logger = None
def simulate_and_log_policy_system(policy_system, expmt, ic=None):
    global tree
    global logger
    expmt_settings = {
        "pendulum": {
            'constructor_or_path': PendulumPlant,
            'state_dim': 2,
            'twoPI_boundary_condition_state_idxs': (0,),
            'initial_state': [0.1, 0.0],
        },
        "acrobot": {
            'constructor_or_path': "/opt/underactuated/src/acrobot/acrobot.urdf",
            'state_dim': 4,
            'twoPI_boundary_condition_state_idxs': (0, 1),
            'initial_state': [0.5, 0.5, 0.0, 0.0],
        },
        "cartpole": {
            'constructor_or_path': "/opt/underactuated/src/cartpole/cartpole.urdf",
            'state_dim': 4,
            'twoPI_boundary_condition_state_idxs': (1,),
            'initial_state': [0.5, 0.5, 0.0, 0.0],
        },
    }
    assert expmt in expmt_settings.keys()

    # Animate the resulting policy.
    settings = expmt_settings[expmt]
    builder = DiagramBuilder()
    constructor_or_path = settings['constructor_or_path']
    if not isinstance(constructor_or_path, str):
        plant = constructor_or_path()
    else:
        tree = RigidBodyTree(constructor_or_path, FloatingBaseType.kFixed)
        plant = RigidBodyPlant(tree)
    plant_system = builder.AddSystem(plant)

    # TODO(russt): add wrap-around logic to barycentric mesh
    # (so the policy has it, too)
    class WrapTheta(VectorSystem):
        def __init__(self):
            VectorSystem.__init__(self, settings['state_dim'], settings['state_dim'])

        def _DoCalcVectorOutput(self, context, input, state, output):
            output[:] = input
            twoPI = 2.*math.pi
            for idx in settings['twoPI_boundary_condition_state_idxs']:
                output[idx] = output[idx] - twoPI * math.floor(output[idx] / twoPI)

    wrap = builder.AddSystem(WrapTheta())
    builder.Connect(plant_system.get_output_port(0), wrap.get_input_port(0))
    vi_policy = builder.AddSystem(policy_system)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0), plant_system.get_input_port(0))

    logger = builder.AddSystem(SignalLogger(settings['state_dim']))
    logger._DeclarePeriodicPublish(0.033333, 0.0)
    builder.Connect(plant_system.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)

    state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
    if ic is None:
        ic = settings['initial_state']
    state.SetFromVector(ic)

    return simulator, tree, logger





##########################################################
# OLD STUFF BELOW HERE
##########################################################


# # Get rid of need for global vars here, if possible?
# vis_cb_counter = 0
# def add_visualization_callback(prog, expmt):
#     assert expmt in state_to_tip_coord_fns.keys()
# 
#     plt.figure()
#     plt.title('Tip trajectories')
#     plt.xlabel('x')
#     plt.ylabel('y')
# 
#     def MyVisualization(sample_times, values):
#         global vis_cb_counter
# 
#         vis_cb_counter += 1
#         if vis_cb_counter % 50 != 0:
#             return
#         
#         visualize_trajectory(sample_times, values, expmt, create_figure=False)
# 
#     print(id(dircol))
#     prog.AddStateTrajectoryCallback(MyVisualization)

