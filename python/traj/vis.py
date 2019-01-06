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


def plot_multiple_dircol_trajectories(prog, h, u, x, num_trajectories, num_samples):
    plt.title('Pendulum trajectories')
    plt.xlabel('theta')
    plt.ylabel('theta_dot')
    for ti in range(num_trajectories):
        h_sol = prog.GetSolution(h[ti])[0]    
        breaks = [h_sol*i for i in range(num_samples)]
        knots = prog.GetSolution(x[ti])
        x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
        t_samples = np.linspace(breaks[0], breaks[-1], 3*num_samples)
        x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])
        plt.plot(x_samples[0,:], x_samples[1,:])
        plt.plot(x_samples[0,-1], x_samples[1,-1], 'ro')



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


def visualize_trajectory(sample_times, values, expmt, create_figure=True):
    assert expmt in state_to_tip_coord_fns.keys()
    if create_figure:
        plt.figure()
        plt.title('Tip trajectories')
        plt.xlabel('x')
        plt.ylabel('y')

    coords = [state_to_tip_coord_fns[expmt](state) for state in values.T]
    x, y = zip(*coords)
    # plt.plot(x, y, '-o', label=vis_cb_counter)
    x = np.array(x)
    y = np.array(y)
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.plot(x[0], y[0], 'go')
    plt.plot(x[-1], y[-1], 'ro')

# Get rid of need for global vars here, if possible?
vis_cb_counter = 0
def add_visualization_callback(prog, expmt):
    assert expmt in state_to_tip_coord_fns.keys()

    plt.figure()
    plt.title('Tip trajectories')
    plt.xlabel('x')
    plt.ylabel('y')

    def MyVisualization(sample_times, values):
        global vis_cb_counter

        vis_cb_counter += 1
        if vis_cb_counter % 50 != 0:
            return
        
        visualize_trajectory(sample_times, values, expmt, create_figure=False)

    print(id(dircol))
    prog.AddStateTrajectoryCallback(MyVisualization)


# Given a policy system, (e.g. created by FittedValueIteraton or nn_system.NNSystem(pytorch_net_object))
# creates a simulator that can be used to visualize the policy.
tree = None
logger = None
def simulate_and_log_policy_system(policy_system, expmt, initial_conditions=None):
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
    if initial_conditions is None:
        initial_conditions = settings['initial_state']
    state.SetFromVector(initial_conditions)

    return simulator, tree, logger


def create_nn_policy_system(kNetConstructor, params_list):
    # Construct a model with params T
    net = kNetConstructor()
    params_loaded = 0
    for param in net.parameters():
        param_slice = np.array([params_list[i] for i in range(params_loaded, params_loaded+param.data.nelement())])
        param.data = torch.from_numpy(param_slice.reshape(list(param.data.size())))
        params_loaded += param.data.nelement() 

    nn_policy = NNSystem(pytorch_nn_object=net)
    return nn_policy


vis_cb_counter = 0
def add_multiple_trajectories_visualization_callback(prog, h, u, x, T, num_trajectories, num_samples, kNetConstructor, expmt):
    num_inputs = len(u[0])
    num_states = len(x[0])
    print(num_inputs, num_states)

    def cb(huxT):
        global vis_cb_counter
        vis_cb_counter += 1
        print(vis_cb_counter, end='')
        if (vis_cb_counter+1) % 50 != 0:
            return
        
        # Unpack the serialized variables
        num_h = num_trajectories
        num_u = num_trajectories*num_samples*num_inputs
        num_x = num_trajectories*num_samples*num_states

        cb_h = huxT[:num_trajectories].reshape((num_trajectories, 1))
        cb_u = huxT[num_h:num_h+num_u].reshape((num_trajectories, num_inputs, num_samples))
        cb_x = huxT[num_h+num_u:num_h+num_u+num_x].reshape((num_trajectories, num_states, num_samples))
        cb_T = huxT[num_h+num_u+num_x:]

        # Visualize the trajectories
        for ti in range(num_trajectories):
            h_sol = cb_h[ti][0]
            breaks = [h_sol*i for i in range(num_samples)]
            knots = cb_x[ti]
            x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
            t_samples = np.linspace(breaks[0], breaks[-1], num_samples*3)
            x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])

            # 1) Visualize the trajectories
            plt.plot(x_samples[0,:], x_samples[1,:])
            plt.plot(x_samples[0,0], x_samples[1,0], 'go')
            plt.plot(x_samples[0,-1], x_samples[1,-1], 'ro')

            # 2) Then visualize what the policy would say to do from those initial conditions
            nn_policy = create_nn_policy_system(kNetConstructor, cb_T)
            initial_conditions = cb_x[ti][:,0]
            simulator, _, logger = simulate_and_log_policy_system(nn_policy, expmt, initial_conditions=initial_conditions)
            simulator.StepTo(h_sol*num_samples)
            t_samples = logger.sample_times()
            x_samples = logger.data()
            plt.plot(x_samples[0,:], x_samples[1,:], ':')
            # plt.plot(x_samples[0,0], x_samples[1,0], 'go')
            # plt.plot(x_samples[0,-1], x_samples[1,-1], 'ro')
            
        plt.show()
        
    flat_h = np.hstack(elem.flatten() for elem in h)
    flat_u = np.hstack(elem.flatten() for elem in u)
    flat_x = np.hstack(elem.flatten() for elem in x)
    prog.AddVisualizationCallback(cb, np.hstack([flat_h, flat_u, flat_x, T]))

