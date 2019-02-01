from __future__ import print_function, absolute_import
from IPython import display
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import subprocess

from multiple_traj_opt import (
    make_mto,
    MultipleTrajOpt,
    initial_conditions_Russ,
    initial_conditions_grid,
    initial_conditions_random,
    initial_conditions_random_all_dims,
)
from nn_system.networks import *

from traj.vis import (
    plot_trajectory
)
from traj.vi_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nn_system.networks import FC, FCBIG, MLPSMALL, MLP
import copy
import time

from pydrake.all import (
    DiagramBuilder,
    DirectCollocation, 
    DynamicProgrammingOptions,
    FittedValueIteration,
    FloatingBaseType,
    PeriodicBoundaryCondition,
    PiecewisePolynomial, 
    RigidBodyTree,
    RigidBodyPlant,
    SignalLogger,
    Simulator,
    SolutionResult,
    VectorSystem
)

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
        if 'target_trajs' in kwargs:
            target_traj = kwargs['target_trajs'][i]
        else:
            target_traj = []
            if kwargs['warm_start'] == "target":
                kwargs['warm_start'] = 'linear'
        dircol, result = do_dircol_fn(
                            ic           = ic,
                            seed         = 1776,
                            target_traj  = target_traj,
                            **kwargs)
        #print("{} took {:.2f}s".format(i, time.time() - start))
        dircols.append(FakeDircol(dircol))
        results.append(result)

        times   = dircol.GetSampleTimes().T
        x_knots = dircol.GetStateSamples().T
        u_knots = dircol.GetInputSamples().T
        # TODO: remove
        # if i == 0:
        #     print("0 u_knots = ", u_knots)
        #     if target_traj != []:
        #         print("target_traj u_knots = ", target_traj[2])
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
    if 'target_trajs' in kwargs:
        target_traj = kwargs['target_trajs'][i]
    else:
        target_traj = []
        if kwargs['warm_start'] == "target":
            kwargs['warm_start'] = 'linear'
    dircol, result = do_dircol_fn(
                        ic           = ic,
                        seed         = 1776,
                        target_traj  = target_traj,
                        **kwargs) # <- will this work?
    if i % 10 == 0:
        print("{} took {:.2f}s".format(i, time.time() - start))

    times   = dircol.GetSampleTimes().T
    x_knots = dircol.GetStateSamples().T
    u_knots = dircol.GetInputSamples().T
    optimized_traj = (times, x_knots, u_knots)

    return optimized_traj, FakeDircol(dircol), result

def igor_traj_opt_parallel(do_dircol_fn, ic_list, **kwargs):
    import multiprocessing
    from multiprocessing import Pool

    p = Pool(multiprocessing.cpu_count() - 10)
    inputs = [(do_dircol_fn, i, ic, kwargs) for i, ic in enumerate(ic_list)]
    results = p.map(f, inputs)
    p.close() # good?
    optimized_trajs, dircols, results = zip(*results)
    assert len(optimized_trajs) == len(ic_list)
    
    return optimized_trajs, dircols, results


def igor_supervised_learning(trajectories, net, kNetConstructor, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
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
    return net

def igor_supervised_learning_cuda(trajectories, net, kNetConstructor, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
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
    return net


def igor_supervised_learning_remote(trajectories, net, kNetConstructor, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
    print()
    # First pickle/npsave the data to std location, overwrite possible old files.
    dir_name = 'remote_GPU'
    all_times, all_x_knots, all_u_knots = zip(*trajectories)
    np.save(dir_name+'/GPU_all_times',   np.array(all_times))
    np.save(dir_name+'/GPU_all_x_knots', np.array(all_x_knots))
    np.save(dir_name+'/GPU_all_u_knots', np.array(all_u_knots))

    #os.remove(dir_name+'/new_GPU_model.pt')
    # Then save the torch model, using a copy.
#    net_copy = kNetConstructor()
#    #net_copy = FCBIG()
#    for (param, param_copy) in zip(net.parameters(), net_copy.parameters()):
#        param_copy.data = copy.deepcopy(param.data)
#    net_copy.eval()
#    torch.save(net_copy, dir_name+'/GPU_model.pt')
    print("saving torch state dict -> GPU_model.pt")
    torch.save(net.state_dict(), dir_name+'/GPU_model.pt')

    # Then scp those files over
    print("rsync-ing files to remote")
    cmd = "rsync -zaP remote_GPU RLG:/home/rverkuil/integration/drake-pytorch/python"
    subprocess.call(cmd.split(' '))

    # Remotely run the training code with arguments
    # (remotely, progress is printed and new weights are saved to a file)
    print("remotely training!")
    python_path = "/home/rverkuil/integration/integration/bin/python"
    script_path = "/home/rverkuil/integration/drake-pytorch/python/remote_train.py"
    sub_cmd = python_path+" "+script_path+" {} {} {} {}".format(int(use_prox), iter_repeat, EPOCHS, lr)
    p = subprocess.Popen(['ssh','RLG',sub_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while p.poll() is None:
        l = p.stdout.readline() # This blocks until it receives a newline.
        print(l, end='')
    # When the subprocess terminates there might be unconsumed output 
    # that still needs to be processed.
    print(p.stdout.read())
    #while True:
    #    # out = p.stderr.read(1)
    #    out = p.stdout.read(1)
    #    if out == '' and p.poll() != None:
    #        break
    #    if out != '':
    #        sys.stdout.write(out)
    #        sys.stdout.flush()

    # SCP the new weights back
    print("rsyncing files back to local...")
    cmd = "rsync -zaP RLG:/home/rverkuil/integration/drake-pytorch/python/remote_GPU ."
    subprocess.call(cmd.split(' '))

    # Load the new weights back
    print("loading the new state dict...")
    net = kNetConstructor()
    net.load_state_dict(torch.load(dir_name+'/new_GPU_model.pt', map_location='cpu'))
    #net.eval()

    # Do optional cleanup or files that were used!
    print()
    return net
    

def parallel_rollout_helper(inp):
    (dummy_mto, h_sol, params_list, ic, ic_scale, WALLCLOCK_TIME_LIMIT) = inp
    _, x_knots, _ = dummy_mto._MultipleTrajOpt__rollout_policy_given_params(h_sol,
                                                                            params_list,
                                                                            ic=np.array(ic)*ic_scale,
                                                                            WALLCLOCK_TIME_LIMIT=WALLCLOCK_TIME_LIMIT)
    return x_knots

def visualize_intermediate_results(trajectories, 
                                   num_trajectories,
                                   num_samples,
                                   expmt="pendulum",
                                   network=None, 
                                   ic_list=None,  
                                   ic_scale=1., 
                                   WALLCLOCK_TIME_LIMIT=1,
                                   constructor=lambda: FCBIG(2, 128),
                                   plot_type="state_scatter",
                                   # parallel=False):
                                   parallel=True):
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
    
    plt.figure()
    if network is not None:
        dummy_mto = MultipleTrajOpt(expmt, num_trajectories, num_samples, 1, 1)
        dummy_mto.kNetConstructor = constructor # a hack!
        params_list = np.hstack([param.clone().detach().numpy().flatten() for param in network.parameters()])
    
        if vis_ic_list is None:
            vis_ic_list = [x_knots[0] for (_, x_knots, _) in vis_trajectories] # Scary indexing get's first x_knot of each traj.

#       h_sol = (0.2+0.5)/2 # TODO: control this better
        h_sol = 0.5

        # Do rollouts in serial or in parallel
        if not parallel:
            for ic in vis_ic_list:
                _, x_knots, _ = dummy_mto._MultipleTrajOpt__rollout_policy_given_params(h_sol,
                                                                                        params_list,
                                                                                        ic=np.array(ic)*ic_scale,
                                                                                        WALLCLOCK_TIME_LIMIT=WALLCLOCK_TIME_LIMIT)
                print("last x_knot: ", x_knots.T[-1])
                plot_trajectory(x_knots, plot_type, expmt, create_figure=False, symbol=':')
        else:
            all_x_knots = []
            import multiprocessing
            from multiprocessing import Pool

            p = Pool(multiprocessing.cpu_count() - 10)
            # Can dummy_mto be serialized if it contains a network?
            inputs = [(dummy_mto, h_sol, params_list, ic, ic_scale, WALLCLOCK_TIME_LIMIT) for i, ic in enumerate(ic_list)]
            all_x_knots = p.map(parallel_rollout_helper, inputs)
            p.close() # good?
            assert len(all_x_knots) == len(ic_list)

            for x_knots in all_x_knots:
                plot_trajectory(x_knots, plot_type, expmt, create_figure=False, symbol=':')

    plt.show()
    # Enable this to clear each plot on the next draw() call.
#     display.display(plt.gcf())
#     display.clear_output(wait=True)


from traj.vi_utils import (
    make_dircol_pendulum,
    do_dircol_pendulum,
    make_dircol_cartpole,
    do_dircol_cartpole,
)
from nn_system.NNSystemHelper import (
    create_nn,
    create_nn_policy_system,
    make_NN_constraint,
    FCBIG,
)
def nn_loader(param_list, network):
    params_loaded = 0
    for param in network.parameters():
        T_slice = np.array([param_list[i] for i in range(params_loaded, params_loaded+param.data.nelement())])
        param.data = torch.from_numpy(T_slice.reshape(list(param.data.size())))
        params_loaded += param.data.nelement()

def do_igor_dircol_fn(dircol_gen_fn=None, **kwargs): # TODO: somehow give/specify the target trajectory
    alpha, lam, eta = 3.e-4, 10.**2, 10.**0
    # alpha, lam, eta = 10.**4, 0., 0.
    # alpha, lam, eta = 0., 10.**4, 0.
    # alpha, lam, eta = 0., 0., 10.**6
    # alpha, lam, eta = 10.**4, 10.**4, 10.**4
    dircol = dircol_gen_fn(**kwargs) # Will be using all the default settings in this thing
    
    # Create a network
    net = kwargs['kNetConstructor']()
    nn_loader(kwargs['net_params'], net)

    target_traj = kwargs['target_traj']
    if not kwargs['naive']:
        for i in range(kwargs['num_samples']):
            # Now let's add on the policy deviation cost... 
            num_params = 0
            pi_cost = make_NN_constraint(lambda: 0, kwargs['num_inputs'], kwargs['num_states'], num_params, network=net, do_asserts=False, L2=True)
            dircol.AddCost(lambda(ux): alpha/2.*pi_cost(ux), np.hstack([dircol.input(i), dircol.state(i)]))

            # ...and the trajectory proximity cost.
            x = dircol.state(i)
            target = target_traj[1][i] # Get i'th knot's state vector
            #print('target: ', target)
            #dircol.AddCost(eta/2. * (x - target).dot(x - target)) # L2 penalty on staying close to the last traj's state here.
            #dircol.AddConstraint( x == target ) # L2 penalty on staying close to the last traj's state here.
            #dircol.AddBoundingBoxConstraint(target, target, x)

    # Can maybe put special settings here that will make pendulum 
    # and cartpole dircols train faster in Igor mode?
    if kwargs['expmt'] == 'pendulum':
        pass
    elif kwargs['expmt'] == 'cartpole':
        pass

    # TODO: Add a retry here to try and get success
    result = dircol.Solve()
    #if result != SolutionResult.kSolutionFound:
    if False:
        if result != SolutionResult.kSolutionFound:
            #print("result={}".format(result))
            if result == SolutionResult.kInfeasibleConstraints:
                #print("result={}, retrying".format(result))
                # Up the accuracy.
                from pydrake.all import (SolverType)
                dircol.SetSolverOption(SolverType.kSnopt, 'Major feasibility tolerance', 1.0e-6) # default="1.0e-6"
                dircol.SetSolverOption(SolverType.kSnopt, 'Major optimality tolerance',  1.0e-3) # default="1.0e-6" was 5.0e-1
                dircol.SetSolverOption(SolverType.kSnopt, 'Minor feasibility tolerance', 1.0e-6) # default="1.0e-6"
                dircol.SetSolverOption(SolverType.kSnopt, 'Minor optimality tolerance',  1.0e-3) # default="1.0e-6" was 5.0e-1
                result = dircol.Solve()
                if result != SolutionResult.kSolutionFound:
                    print("retry result={}".format(result))
    return dircol, result

# If naive is true, it just does parallel trajectory optimization and supervised learning.
# Else, it does Igor's method.
def do_igor_optimization(net, kNetConstructor, expmt, ic_list, naive=True, **kwargs):
    assert expmt in ("pendulum", "cartpole")
    if expmt == "pendulum":
        do_dircol_fn  = do_dircol_pendulum
        dircol_gen_fn = make_dircol_pendulum
        num_inputs  = 1
        num_states  = 2
        num_samples = 32
        iter_repeat = 100
        EPOCHS      = 5 #50
        lr          = 1e-2
        plot_type   = "state_scatter"
        WALLCLOCK_TIME_LIMIT = 15
        if ic_list == None:
            num_trajectories = 144 #50**2
            ic_list = initial_conditions_random(num_trajectories, (0., 2*math.pi), (-5., 5.))
        vis_ic_list = initial_conditions_random(16, (0., 2*math.pi), (-5., 5.))
        total_iterations = 30
    elif expmt == "cartpole":
        do_dircol_fn = do_dircol_cartpole
        dircol_gen_fn = make_dircol_cartpole
        num_inputs  = 1
        num_states  = 4
        num_samples = 21
        iter_repeat = 10#00
        EPOCHS      = 15 #50
        lr          = 1e-2
        plot_type   = "tip_scatter"
        WALLCLOCK_TIME_LIMIT = 30
        if ic_list == None:
            num_trajectories = 400  #**2
            ic_list = initial_conditions_random_all_dims(num_trajectories, ((-5., 5.), (0., 2*math.pi), (-10., 10.), (-10., 10.)) )
        vis_ic_list = initial_conditions_random_all_dims(16, ((-1., 1.), (0., 2*math.pi), (-1., 1.), (-1., 1.)) )
        total_iterations = 30


    ##### IGOR'S BLOCK-ALTERNATING METHOD
    # Either A) you pick one huge block of initial conditions and stick to them throughout the optimization process
    # OR B) Keep bouncing around randomly chosen trajectories? (could be similarly huge or smaller...)
    # ^ accomplish the above via an outer loop over this function, so that we can use that outer loop for the Russ method, too.

    # Do a warm start via an unconstrained optimization is nothing is given to us
    if not naive:
        print("doing warm start", time.time())
        #optimized_trajs, dircols, results = igor_traj_opt_serial(do_dircol_fn, ic_list, **kwargs)
        optimized_trajs, dircols, results = igor_traj_opt_parallel(do_dircol_fn, ic_list, **kwargs)
        print("finished warm start", time.time())
        kwargs['target_trajs']    = optimized_trajs
        vis_trajs = optimized_trajs
        #vis_trajs = []
        visualize_intermediate_results(vis_trajs,
                                       len(ic_list),
                                       num_samples,
                                       expmt=expmt,
                                       plot_type=plot_type,
                                       network=net.cpu(),
                                       #network=None,
                                       #ic_list=ic_list,
                                       #ic_list=ic_list[:multiprocessing.cpu_count()-10],
                                       #ic_list=ic_list[:8],
                                       ic_list = vis_ic_list,
                                       ic_scale=1.,
                                       constructor=kNetConstructor,
                                       WALLCLOCK_TIME_LIMIT=WALLCLOCK_TIME_LIMIT)

    vi_policy, _ = load_policy("good", "pendulum")

    first_iter = True
    iters = 0
    while total_iterations > 0:
        total_iterations -= 1
        
        # Do periodic visualization and file writing here...
        if kwargs['write_row']:
            # Print learned policy vs vi_policy here (only possible for state dim == 2)
            #fig = plt.figure()
            #ax1 = fig.add_subplot(131, projection='3d')
            #ax2 = fig.add_subplot(132, projection='3d')
            #ax3 = fig.add_subplot(133)
            if expmt == "pendulum":
                vis_vi_policy(vi_policy)#, ax1)
                vis_nn_policy_like_vi_policy(net, vi_policy)#, ax2)

            # Print Divergence metrics between the two policies
            vis_results = []
            if expmt == "pendulum":
                test_coords = initial_conditions_random(100000, (0, 2*math.pi), (-5, 5))
            elif expmt == "cartpole":
                test_coords = initial_conditions_random_all_dims(100000, ((-3., 3.), (0., 2*math.pi), (-1., 1.), (-1., 1.)) )
            test_coords = ic_list
            for coord in test_coords:
                pair = (eval_vi_policy(coord, vi_policy), eval_nn_policy(coord, net))
                vis_results.append(pair)
            diffs = [result[1] - result[0] for result in vis_results]
            avg, std, MSE, MAE = plot_and_print_statistics(diffs, "nn - vi policy deviations", bins=500, xlim=None)#, ax=ax3)
            
            kwargs['write_row'](time.time(), iters, avg, std, MSE, MAE)



        # Basically exactly what I have now EXCEPT, DON'T Give the NN parameters to the optimizer!!!!!
        # So, actually might want to solve all the N trajectories in parallel/simultaneously!
        # Just add proximity cost on their change from the last iteration...
        # ^ will this get 
        kwargs['num_samples']     = num_samples
        kwargs['dircol_gen_fn']   = dircol_gen_fn
        kwargs['num_inputs']      = num_inputs
        kwargs['num_states']      = num_states
        # I have to send the network constructor and weights separately because only pickalable things are sendable
        # with Python's multiprocessing package...
        kwargs['kNetConstructor'] = kNetConstructor
        kwargs['net_params']      = np.hstack([param.data.numpy().flatten() for param in net.parameters()])
        kwargs['naive'] = naive
        kwargs['expmt'] = expmt
        if not naive and first_iter:
            first_iter = False
        else:
            #optimized_trajs, dircols, results = igor_traj_opt_serial(do_igor_dircol_fn, ic_list, **kwargs)
            optimized_trajs, dircols, results = igor_traj_opt_parallel(do_igor_dircol_fn, ic_list, **kwargs)
        
        import pickle
        f = open('test.pkl', 'wb')
        pickle.dump((optimized_trajs, dircols, results), f)
        # Will need to have access to the current state of the knot points, here...
        # Then will just do a fitting, (can even add in regularization for, like, free!)
        # With an additional knot penalty term and a proximity cost on difference in parameters from the last iteration...
        trajs_to_fit = []
        for traj, status in zip(optimized_trajs, results):
            if status != SolutionResult.kInfeasibleConstraints: # Can even try to filter out all but successes here!
            # if result == SolutionResult.kSolutionFound: # Can even try to filter out all but successes here!
                trajs_to_fit.append(traj)
        print("Training on {}/{} trajs".format(len(trajs_to_fit), len(optimized_trajs)))
        print(time.time())
        net.train(True)
        # sl_fn = igor_supervised_learning
        # sl_fn = igor_supervised_learning_cuda
        sl_fn = igor_supervised_learning_remote
        net = sl_fn(trajs_to_fit, net, kNetConstructor, use_prox=not naive, iter_repeat=iter_repeat, EPOCHS=EPOCHS, lr=lr)
        print("local net params hash: ", hash(np.hstack([param.data.flatten() for param in net.parameters()]).tostring() ))
        print(time.time())
        net.cpu()
        net.eval()

        # Is this even needed? TODO: get the visualization working as well.
        # vis_trajs = optimized_trajs
        vis_trajs = trajs_to_fit
        # vis_trajs = []
        visualize_intermediate_results(vis_trajs,
                                       len(ic_list),
                                       num_samples,
                                       expmt=expmt,
                                       plot_type=plot_type,
                                       network=net.cpu(),
                                       #network=None,
                                       #ic_list=ic_list,
                                       #ic_list=ic_list[:multiprocessing.cpu_count()-10],
                                       #ic_list=ic_list[:8],
                                       ic_list = vis_ic_list,
                                       ic_scale=1.,
                                       constructor=kNetConstructor,
                                       WALLCLOCK_TIME_LIMIT=WALLCLOCK_TIME_LIMIT)
        iters += 1
        

# Will be using this function to try the minibatching + warm-starting approach 
# over any functions that satisfy the interfaces laid out below.
def mini_batch_and_warm_start(ic_gen, n_iters, inner_fn, network, warm_start_method, traj_bank):
    for i in range(n_iters):
        # Generate initial conditions
        ic_list = ic_gen()
        
        # Create some warm start trajectories, via some method
        warm_start_trajs = warm_start_method(ic_list, traj_bank, network)

        # Run the inner loop optimizer
        trajectories = inner_fn(ic_list, network, warm_start_trajs)

        # Store results
        traj_bank.append(trajectories)








