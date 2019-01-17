from __future__ import print_function, absolute_import
from IPython import display
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

from traj.vis import (
    plot_trajectory
)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nn_system.networks import FC, FCBIG, MLPSMALL, MLP
import copy
import time

# SUPER HACK - because for some reason you can't keep around a reference to a dircol?
class FakeDircol():
    def __init__(self, dircol):
        self.sample_times     = copy.deepcopy(dircol.GetSampleTimes())
        self.state_samples    = copy.deepcopy(dircol.GetStateSamples())
        self.input_samples    = copy.deepcopy(dircol.GetInputSamples())
#             self.state_trajectory = copy.deepcopy(dircol.ReconstructStateTrajectory())
#             self.input_trajectory = copy.deepcopy(dircol.ReconstructInputTrajectory())
        self.state_trajectory = dircol.ReconstructStateTrajectory()
        self.input_trajectory = dircol.ReconstructInputTrajectory()
    def GetSampleTimes(self):
        return self.sample_times
    def GetStateSamples(self):
        return self.state_samples
    def GetInputSamples(self):
        return self.input_samples
    def ReconstructStateTrajectory(self):
        return self.state_trajectory
    def ReconstructInputTrajectory(self):
        return self.input_trajectory

def igor_traj_opt_serial(do_dircol_fn, ic_list, **kwargs):
    optimized_trajs, dircols, results = [], [], []
    for i, ic in enumerate(ic_list):
        start = time.time()
        dircol, result = do_dircol_fn(
                            ic           = ic,
                            warm_start   = "linear",
                            seed         = 1776,
                            **kwargs)
        print("{} took {}".format(i, time.time() - start))
        dircols.append(FakeDircol(dircol))
        results.append(result)

        times   = dircol.GetSampleTimes().T
        x_knots = dircol.GetStateSamples().T
        u_knots = dircol.GetInputSamples().T
        optimized_traj = (times, x_knots, u_knots)
        optimized_trajs.append(optimized_traj)
        if (i+1) % 10 == 0:
            print("completed {} trajectories".format(i+1))
    assert len(optimized_trajs) == len(ic_list)
    return optimized_trajs, dircols, results

# Will be used below in igor_traj_opt_parallel()
def f(inp):
    (do_dircol_fn, i, ic, kwargs) = inp
    # print(kwargs)
    start = time.time()
    dircol, result = do_dircol_fn(
                        ic           = ic,
                        warm_start   = "linear",
                        seed         = 1776,
                        **kwargs) # <- will this work?
    print("{} took {}".format(i, time.time() - start))

    times   = dircol.GetSampleTimes().T
    x_knots = dircol.GetStateSamples().T
    u_knots = dircol.GetInputSamples().T
    optimized_traj = (times, x_knots, u_knots)

    return optimized_traj, FakeDircol(dircol), result

def igor_traj_opt_parallel(do_dircol_fn, ic_list, **kwargs):
    import multiprocessing
    from multiprocessing import Pool

    p = Pool(multiprocessing.cpu_count() - 1)
    inputs = [(do_dircol_fn, i, ic, kwargs) for i, ic in enumerate(ic_list)]
    results = p.map(f, inputs)
    p.close() # good?
    optimized_trajs, dircols, results = zip(*results)
    assert len(optimized_trajs) == len(ic_list)
    
    return optimized_trajs, dircols, results


def igor_supervised_learning(trajectories, net, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha, lam, eta = 10., 10.**2, 10.**-2
    frozen_parameters = [param.clone() for param in net.parameters()]

    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # My data
    all_inputs = np.vstack([np.expand_dims(traj[1][:-1], axis=0) for traj in trajectories])
    all_labels = np.vstack([np.expand_dims(traj[2][:-1], axis=0) for traj in trajectories])
    all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    print(all_inputs.shape)
    print(all_labels.shape)
    def my_gen():
        for _ in range(iter_repeat):
            yield all_inputs, all_labels

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(my_gen(), 0):
            # Unpack data
            inputs, labels = data
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            # Forward pass = THE SAUCE!
            outputs = net(inputs)
            loss = alpha/2*criterion1(outputs, labels)
            if use_prox:
                for param, ref in zip(net.parameters(), frozen_parameters):
                    loss += eta/2*criterion2(param, ref)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % iter_repeat == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print('Finished Training')

def igor_supervised_learning_cuda(trajectories, net, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    alpha, lam, eta = 10., 10.**2, 10.**-2
    frozen_parameters = [param.clone() for param in net.parameters()]

    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # My data, and drop the last point
    all_inputs = np.vstack([np.expand_dims(traj[1][:-1], axis=0) for traj in trajectories])
    all_labels = np.vstack([np.expand_dims(traj[2][:-1], axis=0) for traj in trajectories])
    all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    print(all_inputs.shape)
    print(all_labels.shape)

    inputs = torch.tensor(all_inputs)
    labels = torch.tensor(all_labels)
    inputs, labels = inputs.to(device), labels.to(device)
    #def my_gen():
    #    for _ in range(iter_repeat):
    #        yield all_inputs, all_labels

    for epoch in range(EPOCHS):
        running_loss = 0.0
        # for i, data in enumerate(my_gen(), 0):
        for i, _ in enumerate(range(iter_repeat)):
            # Unpack data
            #inputs, labels = data
            # inputs = torch.tensor(inputs)
            # labels = torch.tensor(labels)

            # Forward pass = THE SAUCE!
            outputs = net(inputs)
            loss = alpha/2*criterion1(outputs, labels)
            if use_prox:
                for param, ref in zip(net.parameters(), frozen_parameters):
                    loss += eta/2*criterion2(param, ref)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % iter_repeat == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print('Finished Training')
    
def visualize_intermediate_results(trajectories, 
                                   num_trajectories,
                                   num_samples,
                                   network=None, 
                                   ic_list=None,  
                                   ic_scale=1., 
                                   WALLCLOCK_TIME_LIMIT=1,
                                   constructor=lambda: FCBIG(2, 128),
                                   expmt="pendulum",
                                   plot_type="state_scatter"):
    vis_ic_list = ic_list
    vis_trajectories = trajectories
    if len(ic_list) > 25:
        print("truncating")
        idcs = np.random.choice(len(ic_list), 25, replace=False)
        vis_ic_list = list((ic_list[idx] for idx in idcs))
        if vis_trajectories:
            vis_trajectories = list((trajectories[idx] for idx in idcs))
        
    plt.figure()
    for (times, x_knots, u_knots) in vis_trajectories:
        plot_trajectory(x_knots.T, plot_type, expmt, create_figure=False, symbol='-')
    
    if network is not None:
        dummy_mto = MultipleTrajOpt(expmt, num_trajectories, num_samples, 1, 1)
        dummy_mto.kNetConstructor = constructor # a hack!
        params_list = np.hstack([param.clone().detach().numpy().flatten() for param in network.parameters()])
    
        if vis_ic_list is None:
            vis_ic_list = [x_knots[0] for (_, x_knots, _) in vis_trajectories] # Scary indexing get's first x_knot of each traj.
        for ic in vis_ic_list:
#             h_sol = (0.2+0.5)/2 # TODO: control this better
            h_sol = 0.5
            _, x_knots, _ = dummy_mto._MultipleTrajOpt__rollout_policy_given_params(h_sol, params_list, ic=np.array(ic)*ic_scale, WALLCLOCK_TIME_LIMIT=WALLCLOCK_TIME_LIMIT)
            print("last x_knot: ", x_knots.T[-1])
            plot_trajectory(x_knots, plot_type, expmt, create_figure=False, symbol=':')
    plt.show()
    # Enable this to clear each plot on the next draw() call.
#     display.display(plt.gcf())
#     display.clear_output(wait=True)


def do_igor_optimization():
    net = FCBIG(2, 128)
    # net = MLPSMALL(2)
    # net = MLP(2, 16)

    ##### IGOR'S BLOCK-ALTERNATING METHOD
    # Either A) you pick one huge block of initial conditions and stick to them throughout the optimization process
    # OR B) Keep bouncing around randomly chosen trajectories? (could be similarly huge or smaller...)
    # ^ Let's make the above easily flag switchable!!!

    # TODO...
    # trajs = igor_traj_opt(ic_list, net)  <- yeah should we think about this now??

    # Minibatch optimization... Let's not do any warm starting for Igor's...
    # plt.figure()
    total_iterations = 1
    num_trajectories = 961
    while total_iterations > 0:
        total_iterations -= 1
    #     ic_list = initial_conditions_grid(num_trajectories, (-math.pi, math.pi), (-5., 5.))
        ic_list = initial_conditions_grid(num_trajectories, (0, 2*math.pi), (-5., 5.))

        # Basically exactly what I have now EXCEPT, DON'T Give the NN parameters to the optimizer!!!!!
        # So, actually might want to solve all the N trajectories in parallel/simultaneously!
        # Just add proximity cost on their change from the last iteration...
        # ^ will this get 
        trajectories = igor_traj_opt(ic_list, net) 
        
        # Will need to have access to the current state of the knot points, here...
        # Then will just do a fitting, (can even add in regularization for, like, free!)
        # With an additional knot penalty term and a proximity cost on difference in parameters from the last iteration...
        igor_supervised_learning(trajectories, net, use_prox=False, iter_repeat=1000, EPOCHS=10, lr=1e-2)
        
        # Is this even needed?
        visualize_intermediate_results(vis_trajectories, network=net, ic_list=vis_ic_list, ic_scale=0.5)
    #     visualize_intermediate_results(trajectories)#, network=net, ic_list=ic_list)
        
        
