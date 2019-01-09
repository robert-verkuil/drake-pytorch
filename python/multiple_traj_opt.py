from __future__ import print_function, absolute_import

import matplotlib.pyplot as plt
import math
import numpy as np

# pydrake stuff
from pydrake.all import (AutoDiffXd, Expression, Variable,
                     MathematicalProgram, SolverType, SolutionResult,
                     DirectCollocationConstraint, AddDirectCollocationConstraint,
                     PiecewisePolynomial,
                    )
import pydrake.symbolic as sym
from pydrake.examples.pendulum import (PendulumPlant)

# My stuff
from nn_system.NNSystemHelper import (
    create_nn_policy_system,
    make_NN_constraint,
)
from traj.vis import (
    simulate_and_log_policy_system,
    plot_trajectory,
    render_trajectory,
)

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

class MultipleTrajOpt(object):
    # Currently only set up to make pendulum examples
    def __init__(self, 
                 expmt, 
                 num_trajectories, num_samples,
                 initial_conditions=None):
        assert expmt == "pendulum"
        self.expmt = expmt
        self.num_inputs = 1
        self.num_states = 2
        self.num_trajectories = num_trajectories
        self.num_samples = num_samples

        # initial_conditions maps (ti) -> [1xnum_states] initial state
        if initial_conditions is not None:
            initial_conditions = intial_cond_dict[initial_conditions]
            assert callable(initial_conditions)

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
        xf = (math.pi, 0.)
        for ti in range(num_trajectories):
            h.append(prog.NewContinuousVariables(1))
            prog.AddBoundingBoxConstraint(.01, .2, h[ti])
            # prog.AddQuadraticCost([1.], [0.], h[ti]) # Added by me, penalize long timesteps
            u.append(prog.NewContinuousVariables(1, num_samples,'u'+str(ti)))
            x.append(prog.NewContinuousVariables(2, num_samples,'x'+str(ti)))

            # Use Russ's initial conditions, unless I pass in a function myself.
            if initial_conditions is None:
                x0 = (.8 + math.pi - .4*ti, 0.0)    
            else:
                x0 = initial_conditions(ti)
                assert len(x0) == 2 #TODO: undo this hardcoding.
            prog.AddBoundingBoxConstraint(x0, x0, x[ti][:,0]) 

            nudge = np.array([.2, .2])
            prog.AddBoundingBoxConstraint(xf-nudge, xf+nudge, x[ti][:,-1])
            # prog.AddBoundingBoxConstraint(xf, xf, x[ti][:,-1])

            for i in range(num_samples-1):
                AddDirectCollocationConstraint(dircol_constraint, h[ti], x[ti][:,i], x[ti][:,i+1], u[ti][:,i], u[ti][:,i+1], prog)

            for i in range(num_samples):
                prog.AddQuadraticCost([1.], [0.], u[ti][:,i])
                # prog.AddConstraint(control, [0.], [0.], np.hstack([x[ti][:,i], u[ti][:,i], K.flatten()]))
                # prog.AddBoundingBoxConstraint([-3.], [3.], u[ti][:,i])
                # prog.AddConstraint(u[ti][0,i] == (3.*sym.tanh(K.dot(control_basis(x[ti][:,i]))[0])))  # u = 3*tanh(K * m(x))
                
            # prog.AddCost(final_cost, x[ti][:,-1])
            # prog.AddCost(h[ti][0]*100) # Try to penalize using more time than it needs?

        # Setting solver options
        #prog.SetSolverOption(SolverType.kSnopt, 'Verify level', -1)  # Derivative checking disabled. (otherwise it complains on the saturation)
        prog.SetSolverOption(SolverType.kSnopt, 'Print file', "/tmp/snopt.out")

        # Save references, TODO: think about doing this right at the start?
        self.prog = prog
        self.h = h
        self.u = u
        self.x = x

    def add_nn_params(self,
                      kNetConstructor, 
                      initialize_params=True, 
                      reg_type="No", 
                      enable_constraint=True):
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
            self.prog.AddCost(L1Cost, T)
        elif reg_type == "L2":
            self.prog.AddQuadraticCost(np.eye(len(T)), [0.]*len(T), T)


        if enable_constraint:
            # Add the neural network constraint to all time steps
            for ti in range(self.num_trajectories):
                for i in range(self.num_samples):
                    u_ti = self.u[ti][0,i]
                    x_ti = self.x[ti][:,i]
                    # Only one output value, so let's have lb and ub of just size one!
                    constraint = make_NN_constraint(self.kNetConstructor, self.num_inputs, self.num_states, self.num_params)
                    lb         = np.array([-.1])
                    ub         = np.array([.1])
                    var_list   = np.hstack((u_ti, x_ti, self.T))
                    self.prog.AddConstraint(constraint, lb, ub, var_list)
            #         self.prog.AddCost(lambda x: constraint(x)[0]**2, var_list)

        
    def add_multiple_trajectories_visualization_callback(self,
            expmt,
            initial_conditions=None):

        self.vis_cb_counter = 0
        # Exposing the variables that the callback will need to bind to...
        # TODO: is this needed?
        kNetConstructor = self.kNetConstructor
        num_inputs = self.num_inputs
        num_states = self.num_states
        num_trajectories = self.num_trajectories
        num_samples = self.num_samples
        def cb(huxT):
            # global vis_cb_counter
            self.vis_cb_counter += 1
            print(" {}".format(self.vis_cb_counter), end='')
            if (self.vis_cb_counter) % 17 != 1:
                return
            
            # Unpack the serialized variables
            num_h = num_trajectories
            num_u = num_trajectories*num_samples*num_inputs
            num_x = num_trajectories*num_samples*num_states
            h = huxT[                  : num_h            ].reshape((num_trajectories, 1))
            u = huxT[num_h             : num_h+num_u      ].reshape((num_trajectories, num_inputs, num_samples))
            x = huxT[num_h+num_u       : num_h+num_u+num_x].reshape((num_trajectories, num_states, num_samples))
            T = huxT[num_h+num_u+num_x :                  ]

            # Visualize the trajectories
            for ti in range(num_trajectories):
                h_sol = h[ti][0]
                breaks = [h_sol*i for i in range(num_samples)]
                knots = x[ti]
                x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
                t_samples = np.linspace(breaks[0], breaks[-1], num_samples*3)
                x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])

                # 1) Visualize the trajectories
                plot_trajectory(x_samples, "state_scatter", self.expmt, create_figure=False) 

                # 2) Then visualize what the policy would say to do from those initial conditions
                if ti != 0: #TOOD: remove this!!!
                    continue
                _, x_samples, _ = self.__rollout_policy(h_sol, T)#, intial_conditions=initial_conditions)
                #TODO: replace the below with a call to plot_trajectory("state_scatter")
                plt.plot(x_samples[0,:], x_samples[1,:], ':')
                plt.plot(x_samples[0,0], x_samples[1,0], 'go')
                plt.plot(x_samples[0,-1], x_samples[1,-1], 'ro')
                
            plt.show()
            
        flat_h = np.hstack(elem.flatten() for elem in self.h)
        flat_u = np.hstack(elem.flatten() for elem in self.u)
        flat_x = np.hstack(elem.flatten() for elem in self.x)
        self.prog.AddVisualizationCallback(cb, np.hstack([flat_h, flat_u, flat_x, self.T]))

    def Solve(self):
        return self.prog.Solve()

    def print_pi_divergence(self, ti):
        from nn_system.NNSystem import NNInferenceHelper_double
        from nn_system.NNSystemHelper import create_nn

        ti = 0
        nn = create_nn(self.kNetConstructor, self.prog.GetSolution(self.T))
        u_vals = self.prog.GetSolution(self.u[ti])
        x_vals = self.prog.GetSolution(self.x[ti])
        print("u_val-Pi(x_val)= diff")
        for i in range(self.num_samples):
            u_val = u_vals[0,i]
            x_val = x_vals[:,i]
            u_pi = NNInferenceHelper_double(nn, x_val)[0]
            print( "({: .2f})-({: .2f})= {: .2f}".format(u_val, u_pi, u_val - u_pi) )

    def __rollout_policy(self, h_sol, params_list):
        nn_policy = create_nn_policy_system(self.kNetConstructor, params_list)
#        if initial_conditions is None:
#            inits = x[ti][:,0]           # Use the start of the corresponding trajectory.
#        else:
        # inits = initial_conditions(ti)  # Else do the user's selected inits.
        simulator, _, logger = simulate_and_log_policy_system(nn_policy, self.expmt)#, initial_conditions=inits)
        # TODO: overwrite h_sol here?
        simulator.StepTo(h_sol*self.num_samples)
        t_samples = logger.sample_times()
        x_samples = logger.data()
        return t_samples, x_samples, logger

    # TODO: add ability to see policy graphs in state and physical space!
    def plot_policy(self, plot_type):#, initial_conditions=None):
        h_sol = (0.01+0.2)/2 #TODO: Figure out a good way to set timestep for policy rollouts...
        params_list = self.prog.GetSolution(self.T)
        _, x_samples, _ = self.__rollout_policy(h_sol, params_list)#, initial_conditions=initial_conditions)

        plot_trajectory(x_samples, plot_type, expmt=self.expmt, create_figure=True)

    def render_policy(self, ti):
        # initial_conditions = self.prog.GetSolution(self.x[ti])[:,0]
        h_sol = self.prog.GetSolution(self.h[ti])[0]
        params_list = self.prog.GetSolution(self.T)
        _, _, logger = self.__rollout_policy(h_sol, params_list)#, initial_conditions=initial_conditions)

        return render_trajectory(logger)

    # TODO: have this be more intelligent!
    def get_trajectory_data(self, ti):
        h_sol = self.prog.GetSolution(self.h[ti])[0]
        breaks = [h_sol*i for i in range(self.num_samples)]
        knots = self.prog.GetSolution(self.x[ti])
        x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
        t_samples = np.linspace(breaks[0], breaks[-1], 45)
        x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])
        return x_trajectory, t_samples, x_samples

    def plot_single_trajectory(self, ti, plot_type, create_figure=True):
        # Visualize with a static arrows plot.
        _, _, x_samples = self.get_trajectory_data(ti)

        plot_trajectory(x_samples, plot_type, expmt=self.expmt, create_figure=create_figure)

    def plot_all_trajectories(self, plot_type):
        plt.figure()
        plt.title('Pendulum trajectories')
        plt.xlabel('theta')
        plt.ylabel('theta_dot')
        for ti in range(self.num_trajectories):
            _, _, x_samples = self.get_trajectory_data(ti)

            self.plot_single_trajectory(ti, plot_type, create_figure=False)
            # plt.plot(x_samples[0,:], x_samples[1,:]) # TODO: remove this stuff
            # plt.plot(x_samples[0,0], x_samples[1,0],   'go')
            # plt.plot(x_samples[0,-1], x_samples[1,-1], 'ro')

    def render_single_trajectory(self, ti):
        from traj.visualizer import PendulumVisualizer
        # Visualize the result as a video.
        x_trajectory, _, _ = self.get_trajectory_data(ti)
        
        return render_trajectory(x_trajectory)







