from __future__ import print_function, absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from pydrake.all import (
    DiagramBuilder,
    FloatingBaseType,
    RigidBodyTree,
    RigidBodyPlant,
    SignalLogger, 
    Simulator, 
    VectorSystem
)
# from pydrake.examples.pendulum import PendulumPlant
from pydrake.systems.controllers import (
    DynamicProgrammingOptions, FittedValueIteration, PeriodicBoundaryCondition)
from traj.visualizer import PendulumVisualizer
# from pydrake.examples.acrobot import AcrobotPlant
from underactuated import (
    PlanarRigidBodyVisualizer
)




from pydrake.all import (BarycentricMesh, BarycentricMeshSystem)
def save_policy(name, policy, cost_to_go, state_grid): # binds to policy, state_grid, and cost_to_go
    output_values = policy.get_output_values()
    np.save('numpy_saves/pi_b_mesh_init__cartpole_'+name, state_grid)
    np.save('numpy_saves/pi_output_values__cartpole_'+name, output_values)
    np.save('numpy_saves/ctg__cartpole_'+name, cost_to_go)
def load_policy(name):
    b_mesh_init = np.load('numpy_saves/pi_b_mesh_init__cartpole_'+name+'.npy').tolist()
    output_values = np.load('numpy_saves/pi_output_values__cartpole_'+name+'.npy')
    b_mesh = BarycentricMesh(b_mesh_init)
    ctg = np.load('numpy_saves/ctg__cartpole_'+name+'.npy')
    return BarycentricMeshSystem(b_mesh, output_values), ctg
# save_policy('test_stabilize_top7_min_cost')
# policy = load_policy('test_stabilize_top7') # pre-emptively typed this <--




# State: (theta1, theta2, theta1_dot, theta2_dot) Input: Elbow torque
def VI(u_cost=180.**2):
    tree = RigidBodyTree("/opt/underactuated/src/cartpole/cartpole.urdf",
                         FloatingBaseType.kFixed)
    plant = RigidBodyPlant(tree)
    simulator = Simulator(plant)
    options = DynamicProgrammingOptions()

    def min_time_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        u = plant.EvalVectorInput(context, 0).CopyToVector()
        x[1] = x[1] - math.pi
        if x.dot(x) < .1: # seeks get x to (math.pi, 0., 0., 0.)
            return 0.
        return 1. + 2*x.dot(x)/(10**2+math.pi**2+10**2+math.pi**2) + u.dot(u)/(u_cost)

    def quadratic_regulator_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        x[1] = x[1] - math.pi
        u = plant.EvalVectorInput(context, 0).CopyToVector()
        return 2*x.dot(x)/(10**2+math.pi**2+10**2+math.pi**2) + u.dot(u)/(u_cost)

    if (True):
        cost_function = min_time_cost
        input_limit = 360.
        options.convergence_tol = 0.001
        state_steps = 19
        input_steps = 19
    else:
        cost_function = quadratic_regulator_cost
        input_limit = 250.
        options.convergence_tol = 0.01
        state_steps = 19
        input_steps = 19

    ####### SETTINGS ####### My cartpole linspaces are off??????
    # State: (x, theta, x_dot, theta_dot)
    # Previous Best... (min. time) (3)
    xbins = np.linspace(-10., 10., state_steps)
    thetabins = np.hstack((np.linspace(0., math.pi-0.2, 8), np.linspace(math.pi-0.2, math.pi+0.2, 11), np.linspace(math.pi+0.2, 8, 2*math.pi)))
    xdotbins = np.linspace(-10., 10., state_steps)
    thetadotbins = np.linspace(-10., 10., state_steps)
    timestep = 0.01

    # Test 1 (4)
    xbins = np.linspace(-10., 10., state_steps)
    thetabins = np.hstack((np.linspace(0., math.pi-0.12, 8), np.linspace(math.pi-0.12, math.pi+0.12, 11), np.linspace(math.pi+0.12, 8, 2*math.pi)))
    xdotbins = np.linspace(-10., 10., state_steps+2)
    thetadotbins = np.hstack((np.linspace(-10., -1.5, 9), np.linspace(-1.5, 1.5, 11), np.linspace(1.5, 10., 9)))
    # timestep = 0.001 <- wasn't active...

    # Test 2 - Test 1 was worse? WOW I HAD A BUG! - in my last np.linspace  (5) SWEET!!!
    xbins = np.linspace(-10., 10., state_steps)
    thetabins = np.hstack((np.linspace(0., math.pi-0.2, 10), np.linspace(math.pi-0.2, math.pi+0.2, 9), np.linspace(math.pi+0.2, 2*math.pi, 10)))
    xdotbins = np.linspace(-10., 10., state_steps+2)
    thetadotbins = np.linspace(-10., 10., state_steps)
    timestep = 0.01
    input_limit = 1000. # test_stabilize_top7 for the higher input_limit version


    options.periodic_boundary_conditions = [
        PeriodicBoundaryCondition(1, 0., 2.*math.pi),
    ]
    state_grid = [set(xbins), set(thetabins), set(xdotbins), set(thetadotbins)]
    input_grid = [set(np.linspace(-input_limit, input_limit, input_steps))] # Input: x force

    print("VI with u_cost={} beginning".format(u_cost))
    policy, cost_to_go = FittedValueIteration(simulator, cost_function,
                                              state_grid, input_grid,
                                              timestep, options)
    print("VI with u_cost={} completed!".format(u_cost))

    save_policy("u_cost={:.0f}_torque_limit={:.0f}".format(u_cost, input_limit), policy, cost_to_go, state_grid)
    return policy, cost_to_go




# Run from inside an iPython notebook!
def animate_cartpole(policy, duration=10.):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    tree = RigidBodyTree("/opt/underactuated/src/cartpole/cartpole.urdf",
                         FloatingBaseType.kFixed)
    plant = RigidBodyPlant(tree)
    plant_system = builder.AddSystem(plant)

    # TODO(russt): add wrap-around logic to barycentric mesh
    # (so the policy has it, too)
    class WrapTheta(VectorSystem):
        def __init__(self):
            VectorSystem.__init__(self, 4, 4)

        def _DoCalcVectorOutput(self, context, input, state, output):
            output[:] = input
            twoPI = 2.*math.pi
            output[1] = output[1] - twoPI * math.floor(output[1] / twoPI)


    wrap = builder.AddSystem(WrapTheta())
    builder.Connect(plant_system.get_output_port(0), wrap.get_input_port(0))
    vi_policy = builder.AddSystem(policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0), plant_system.get_input_port(0))

    logger = builder.AddSystem(SignalLogger(4))
    logger._DeclarePeriodicPublish(0.033333, 0.0)
    builder.Connect(plant_system.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)

    state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
    state.SetFromVector([-1., math.pi-1, 1., -1.])

    # Do the sim.
    simulator.StepTo(duration)

    # Visualize the result as a video.
    vis = PlanarRigidBodyVisualizer(tree, xlim=[-12.5, 12.5], ylim=[-1, 2.5])
    ani = vis.animate(logger, repeat=True)

    # plt.show()
    # Things added to get visualizations in an ipynb
    plt.close(vis.fig)
    HTML(ani.to_html5_video())


if __name__ == "__main__":
    u_costs = [32400, 20e3, 10e3, 5e3, 2.5e3, 1e3, 500, 250, 100]

    import multiprocessing
    from multiprocessing import Pool

    p = Pool(multiprocessing.cpu_count() - 2)
    print("Running with {} cpus".format(multiprocessing.cpu_count() - 2))
    results = p.map(VI, u_costs)
    p.close()
    policies, cost_to_gos = zip(*results)
    assert len(policies) == len(u_costs)
    assert len(cost_to_gos) == len(u_costs)





