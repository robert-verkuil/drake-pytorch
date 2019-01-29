from multiple_traj_opt import *
from nn_system.NNSystemHelper import *


def kNetConstructor():
    #return FCBIG(n_inputs=2, h_sz=8)
    return FCBIG(n_inputs=2, h_sz=32)
    #return MLP(n_inputs=2, h_sz=8)
    #return MLP(n_inputs=2, h_sz=32)

# def make_mto(
#              # Settings for just the trajectory optimization.
#              expmt="pendulum",
#              num_trajectories=16,
#              num_samples=16,
#              #kMinimumTimeStep=0.2,
#              #kMaximumTimeStep=0.5,
#              kMinimumTimeStep=0.0001,
#              kMaximumTimeStep=1.,
#              ic_list=None,
#              warm_start=True,
#              seed=1338,
# 
#              # Below are the NN-centric init options.
#              use_dropout=True,
#              nn_init=kaiming_uniform,
#              nn_noise=1e-2,
#              kNetConstructor=lambda: FCBIG(2, 32),
#              use_constraint=True,
#              cost_factor=None,
#              initialize_params=True,
#              reg_type="No",
# 
#              # Callback display settings.
#              vis_cb_every_nth=None,
#              vis_cb_display_rollouts=False,
#              cost_cb_every_nth=None,
# 
#              i=0,
#              snopt_overrides=[]):

# dircol.SetSolverOption(SolverType.kSnopt, 'Major iterations limit',  9300000) # Default="9300"
# dircol.SetSolverOption(SolverType.kSnopt, 'Minor iterations limit',  50000) # Default="500"
# dircol.SetSolverOption(SolverType.kSnopt, 'Iterations limit',  50*10000) # Default="10000"

if __name__ == "__main__":
    mto = make_mto(expmt="pendulum",
                   num_trajectories=1,
                   kNetConstructor=kNetConstructor,
                   snopt_overrides=[
                       ['Major iterations limit', 3],
                       ['Minor iterations limit', 3],
                       ['Iterations limit', 3],
                   ])
    mto.Solve()
    write_log() # Hopefully will use the big one? if not let's experiment with python binding of atexit()...

