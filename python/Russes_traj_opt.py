import math
import numpy as np
from pydrake.all import (AutoDiffXd, Expression, Variable,
                         MathematicalProgram, SolverType, SolutionResult,
                         DirectCollocationConstraint, AddDirectCollocationConstraint,
                         PiecewisePolynomial,
                        )
import pydrake.symbolic as sym
from pydrake.examples.pendulum import (PendulumPlant)
from pydrake.examples.acrobot import (AcrobotPlant)

# plant = PendulumPlant(); num_states=2
plant = AcrobotPlant(); num_states=4

context = plant.CreateDefaultContext()
dircol_constraint = DirectCollocationConstraint(plant, context)

num_trajectories = 5;
print("num_trajectories: ", num_trajectories)
num_samples = 15;
prog = MathematicalProgram()
K = prog.NewContinuousVariables(1,7,'K')

# TODO: Move this into autodiffutils.py
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
    

def control_basis(x):
    return [1, cos(x[0]), sin(x[0]), x[1], x[1]*cos(x[0]), x[1]*sin(x[0]), x[1]**3 ]


def final_cost(x):
    return 100.*(cos(.5*x[0])**2 + x[1]**2)

def control(xuK):
    x = xuK[[0, 1]]
    u = xuK[2]
    K = xuK[3:]
    uc = K.dot(control_basis(x))
    # TODO: snopt doesn't like this.  perhaps a softmax?    
    if (uc.value() > 3.0):
        uc = 3.0
    elif(uc.value() < -3.0):
        uc = -3.0
    return [u - uc]
    
h = [];
u = [];
x = [];
for ti in range(num_trajectories):
    h.append(prog.NewContinuousVariables(1))
    prog.AddBoundingBoxConstraint(.01, .2, h[ti])
    u.append(prog.NewContinuousVariables(1, num_samples,'u'+str(ti)))
    x.append(prog.NewContinuousVariables(num_states, num_samples,'x'+str(ti)))

    x0 = (.8 + math.pi - .4*ti, 0.0)    
    xf = (math.pi, 0.)
    if num_states == 4:
        x0 += (0., 0.)
        xf += (0., 0.)

    prog.AddBoundingBoxConstraint(x0, x0, x[ti][:,0]) 

    prog.AddBoundingBoxConstraint(xf, xf, x[ti][:,-1])

    for i in range(num_samples-1):
        AddDirectCollocationConstraint(dircol_constraint, h[ti], x[ti][:,i], x[ti][:,i+1], u[ti][:,i], u[ti][:,i+1], prog)

    for i in range(num_samples):
        prog.AddQuadraticCost([1.], [0.], u[ti][:,i])
#        prog.AddConstraint(control, [0.], [0.], np.hstack([x[ti][:,i], u[ti][:,i], K.flatten()]))
#        prog.AddBoundingBoxConstraint([0.], [0.], u[ti][:,i])
        # prog.AddConstraint(u[ti][0,i] == (3.*sym.tanh(K.dot(control_basis(x[ti][:,i]))[0])))  # u = 3*tanh(K * m(x))
        
#     prog.AddCost(final_cost, x[ti][:,-1])

#prog.SetSolverOption(SolverType.kSnopt, 'Verify level', -1)  # Derivative checking disabled. (otherwise it complains on the saturation)
prog.SetSolverOption(SolverType.kSnopt, 'Print file', "/tmp/snopt.out")
result = prog.Solve()
print(result)
print(prog.GetSolution(K))
print(prog.GetSolution(K).dot(control_basis(x[0][:,0])))

#if (result != SolutionResult.kSolutionFound):
#    f = open('/tmp/snopt.out', 'r')
#    print(f.read())
#    f.close()
    
#for ti in range(num_trajectories):
#    h_sol = prog.GetSolution(h[ti])[0]    
#    breaks = [h_sol*i for i in range(num_samples)]
#    knots = prog.GetSolution(x[ti])
#    x_trajectory = PiecewisePolynomial.Cubic(breaks, knots, False)
#    t_samples = np.linspace(breaks[0], breaks[-1], 45)
#    x_samples = np.hstack([x_trajectory.value(t) for t in t_samples])
#    plt.plot(x_samples[0,:], x_samples[1,:])
