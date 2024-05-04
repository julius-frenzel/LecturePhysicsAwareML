import numpy as np
import time
from scipy.integrate import ode
from scipy.optimize import root
import matplotlib.pyplot as plt
from math import floor
from helper import poly_eval, poly_der, quad
from galerkin_approximation import GalerkinApproximation
from beam_model import get_mass_matrix, get_force_vector, get_rhs

boundary_conds = [0, 1] # what kind of boundary conditions to use at the ends of the beam (0: Dirichlet, 1: Neumann, 2: mixed)
dirichlet_vals = [0, 1] # values for the Dirichlet boundary conditions
neumann_vals = [0, 1] # values for the Neumann boundary conditions
q = lambda x: 1 # specific lateral force as a function of the position along the beam
f = lambda x: 1 # specific axial force as a function of the position along the beam

# define mesh (describes the domain, in which the problem is to be solved)
l = 1 # length of the beam
num_elements = 10
mesh = {"num_elements": num_elements,
        "element_boundaries": np.linspace(0, l, num_elements+1),
        "boundary_conds": boundary_conds,
        "dirichlet_vals": dirichlet_vals,
        "neumann_vals": neumann_vals,
        "q": q,
        "f": f}

# define polynomial basis functions (will be evaluated between 0 and 1 / equal to test functions, because the Galerkin method is used)
poly_basis = np.array([[1, 0, -3, 2],
                       [0, 1, -2, 1],
                       [0, 0, 3, -2],
                       [0, 0, -1, 1]])
# define object, which holds an approximate solution to the problem
approx = GalerkinApproximation(mesh, poly_basis)

# mass matrix and right hand side don't depend on coefficients
M = get_mass_matrix(approx, mesh)
b = get_rhs(approx, mesh)

#print(M)
#eigvals = np.linalg.eigvals(M)
#plt.scatter(np.real(eigvals), np.imag(eigvals))
#plt.grid(True)
#plt.show()

approx.solve_static(b, get_force_vector)

approx.solve_dynamic(M, b, get_force_vector)

num_eval_points = 1000
eval_points = np.linspace(0, l, num_eval_points)

approx.visualize(eval_points)





