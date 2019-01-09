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
from nn_system.NNSystem import NNInferenceHelper_double
from nn_system.NNSystemHelper import (
    create_nn_policy_system,
    make_NN_constraint,
    create_nn,
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

#########################################################
# Big Batch of Initial Conditions
#########################################################
n_theta, n_theta_dot = (5, 3)
theta_bounds     = -math.pi, math.pi
theta_dot_bounds = -5, 5
def initial_conditions_Russ(num_trajectories):
    return np.array([(.8 + math.pi - .4*ti, 0.0) for ti in range(num_trajectories)])
def initial_conditions_grid(num_trajectories):
    # Have ti index into a grid over some state space bounds
    theta_range     = np.linspace(theta_bounds[0], theta_bounds[1], n_theta)
    theta_dot_range = np.linspace(theta_dot_bounds[0], theta_dot_bounds[1], n_theta_dot)
    ret = []
    for ti in range(num_trajectories):
        r = int(ti / n_theta)
        c = ti % n_theta
        ret.append( (theta_range[c], theta_dot_range[r]) )
    return ret
def initial_conditions_random(num_trajectories):
    # random state over some state-space bounds
    ret = []
    for ti in range(num_trajectories):
        rand1, rand2 =  np.random.random(2)
        theta     =  (theta_bounds[1] - theta_bounds[0]) * rand1 + theta_bounds[0]
        theta_dot =  (theta_dot_bounds[1] - theta_dot_bounds[0]) * rand2 + theta_dot_bounds[0]
        ret.append( (theta, theta_dot) )
    return ret
# intial_cond_dict = {
#     "Russ": initial_conditions_Russ,
#     "grid": initial_conditions_grid,
#     "random": initial_conditions_random,
# }

class MultipleTrajOpt(object):
    # Currently only set up to make pendulum examples
    def __init__(self, 
                 expmt, 
                 num_trajectories, num_samples,
                 ic_list=None):
        assert expmt == "pendulum"
        self.expmt = expmt
        self.num_inputs = 1
        self.num_states = 2
        self.num_trajectories = num_trajectories
        self.num_samples = num_samples

        # initial_conditions return a list of [num_trajectories x num_states] initial states
        # Use Russ's initial conditions, unless I pass in a function myself.
        if ic_list is None:
            ic_list = initial_conditions_Russ(self.num_trajectories)
        assert len(ic_list) == self.num_trajectories
        assert np.all( [len(ic) == self.num_states for ic in ic_list] )
            # assert callable(ic_list)

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

            x0 = ic_list[ti]
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

        # Save references
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

        
    def add_multiple_trajectories_visualization_callback(self, ic_list=None):
        # initial_conditions return a list of [<any> x num_states] initial states
        # Use Russ's initial conditions, unless I pass in a function myself.
        if ic_list is None:
            ic_list = initial_conditions_Russ(self.num_trajectories)
        assert np.all( [len(ic) == self.num_states for ic in ic_list] )

        self.vis_cb_counter = 0
        def cb(huxT):
            self.vis_cb_counter += 1
            print(" {}".format(self.vis_cb_counter), end='')
            if (self.vis_cb_counter) % 17 != 1:
                return
            
            # Unpack the serialized variables
            num_h = self.num_trajectories
            num_u = self.num_trajectories*self.num_samples*self.num_inputs
            num_x = self.num_trajectories*self.num_samples*self.num_states
            h = huxT[                  : num_h            ].reshape((self.num_trajectories, 1))
            u = huxT[num_h             : num_h+num_u      ].reshape((self.num_trajectories, self.num_inputs, self.num_samples))
            x = huxT[num_h+num_u       : num_h+num_u+num_x].reshape((self.num_trajectories, self.num_states, self.num_samples))
            T = huxT[num_h+num_u+num_x :                  ]

            # Visualize the trajectories
            for ti in range(self.num_trajectories):
                h_sol = h[ti][0]
                breaks = [h_sol*i for i in range(self.num_samples)]
                knots = x[ti]
                x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
                t_samples = np.linspace(breaks[0], breaks[-1], self.num_samples*3)
                x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])

                # 1) Visualize the trajectories
                plot_trajectory(x_samples, "state_scatter", self.expmt, create_figure=False) 

            for ic in ic_list:
                # 2) Then visualize what the policy would say to do from (possibly the same) initial conditions.
                h_sol = (0.01+0.2)/2 # TODO: improve this hard coding?
                _, pi_x_samples, _ = self.__rollout_policy_given_params(h_sol, T, ic)
                plot_trajectory(pi_x_samples, "state_scatter", self.expmt, create_figure=False, symbol=':') 
                
            plt.show()
            
        flat_h = np.hstack(elem.flatten() for elem in self.h)
        flat_u = np.hstack(elem.flatten() for elem in self.u)
        flat_x = np.hstack(elem.flatten() for elem in self.x)
        self.prog.AddVisualizationCallback(cb, np.hstack([flat_h, flat_u, flat_x, self.T]))

    def Solve(self):
        return self.prog.Solve()

    def print_pi_divergence(self, ti):
        nn = create_nn(self.kNetConstructor, self.prog.GetSolution(self.T))
        u_vals = self.prog.GetSolution(self.u[ti])
        x_vals = self.prog.GetSolution(self.x[ti])

        print("u_val-Pi(x_val)= diff")
        for i in range(self.num_samples):
            u_val = u_vals[0,i]
            x_val = x_vals[:,i]
            u_pi = NNInferenceHelper_double(nn, x_val)[0]
            print( "({: .2f})-({: .2f})= {: .2f}".format(u_val, u_pi, u_val - u_pi) )

    def __rollout_policy_given_params(self, h_sol, params_list, ic=None):
        nn_policy = create_nn_policy_system(self.kNetConstructor, params_list)
        simulator, _, logger = simulate_and_log_policy_system(nn_policy, self.expmt, ic)
        simulator.StepTo(h_sol*self.num_samples)
        t_samples = logger.sample_times()
        x_samples = logger.data()
        return t_samples, x_samples, logger

    def __rollout_policy_at_solution(self, ic=None):
        if ic is None:
            ic = self.prog.GetSolution(self.x[ti])[:,0]
            h_sol = self.prog.GetSolution(self.h[ti])[0]
        else:
            h_sol = (0.01+0.2)/2
        params_list = self.prog.GetSolution(self.T)
        return self.__rollout_policy_given_params(h_sol, params_list, ic=ic)

    def plot_policy(self, plot_type, ic=None):
        _, x_samples, _ = __rollout_policy_at_solution(ic=ic)

        plot_trajectory(x_samples, plot_type, expmt=self.expmt, create_figure=True)

    def render_policy(self, ti, ic=None):
        _, _, logger = __rollout_policy_at_solution(ic=ic)

        return render_trajectory(logger)

    def get_trajectory_data(self, ti):
        h_sol = self.prog.GetSolution(self.h[ti])[0]
        breaks = [h_sol*i for i in range(self.num_samples)]
        knots = self.prog.GetSolution(self.x[ti])
        x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
        t_samples = np.linspace(breaks[0], breaks[-1], 45)
        x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])
        return x_trajectory, t_samples, x_samples

    # Visualize with a static arrows plot.
    def plot_single_trajectory(self, ti, plot_type, create_figure=True):
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

    # Visualize the result as a video.
    def render_single_trajectory(self, ti):
        from traj.visualizer import PendulumVisualizer
        x_trajectory, _, _ = self.get_trajectory_data(ti)
        
        return render_trajectory(x_trajectory)







