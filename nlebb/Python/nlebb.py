import numpy as np
import time
from galerkin_approximation import GalerkinApproximation
from beam_model import get_mass_matrix_u, get_mass_matrix_w, get_force_vector_u, get_force_vector_w, get_rhs_u, get_rhs_w
import matplotlib.pyplot as plt
import logging
import sys

#np.set_printoptions(precision=3)

# ----------findings----------
# 


# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(filename)25s | %(levelname)7s | %(message)s', datefmt='%H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# !!!!!!!!!!!!!!!!!!!!activate coupling and nonlinearities!!!!!!!!!!!!!!!!!!!!!!!
# define mesh (describes the domain, in which the problem is to be solved)
l = 1 # length of the beam
num_elements = 10
mesh = {"num_elements": num_elements,
        "element_boundaries": np.linspace(0, l, num_elements+1),
        "boundary_conds": [2, 0], # what kind of boundary conditions to use at the ends of the beam (0: free, 1:pinned, 2: clamped)
        "load_points": [0, 1], # points, at which external loads are applied
        "N": [0, 0], # values for free boundary conditions (applying to internal points would leed to a solution with a discontinuous derivative, which can't be represented exactly using Hermite splines)
        "Q": [0, 0],
        "M": [0, 1],
        "u": [0, 0], # values for pinned and clamped boundary conditions (u and w are used for both)
        "w": [0, 0],
        "w_x": [0, 0], # additional slopes for clamped boundary conditions
        "f": lambda x: 1, # specific axial force as a function of the position along the beam
        "q": lambda x: 0} # specific lateral force as a function of the position along the beam

# define polynomial basis functions (will be evaluated between 0 and 1 / equal to test functions, because the Galerkin method is used)
# the coefficients must be exactly representable using machine numbers, so the polynomials can be evaluated exactly at the boundaries
poly_basis = np.array([[1, 0, -3, 2],
                       [0, 1, -2, 1],
                       [0, 0, 3, -2],
                       [0, 0, -1, 1]])
# define object, which holds an approximate solution to the problem
approx = GalerkinApproximation(mesh, poly_basis)

def print_matrix(a):
    for row in a:
        for element in row:
            print(f"{element:.2e}", end=",  ")
        print("")

# mass matrix and right hand side don't depend on coefficients
M_u = get_mass_matrix_u(approx, mesh)
M_w = get_mass_matrix_w(approx, mesh)
#print(M_u)
#print_matrix(M_w)
#time.sleep(100)
M = np.block([[M_u,                     np.zeros_like(M_u)],
              [np.zeros_like(M_w),      M_w]])
b = np.concatenate([get_rhs_u(approx, mesh),
                    get_rhs_w(approx, mesh)])

#print(M)
#eigvals = np.linalg.eigvals(M)
#plt.scatter(np.real(eigvals), np.imag(eigvals))
#plt.grid(True)
#plt.show()

num_eval_points = 1000
eval_points = np.linspace(0, l, num_eval_points)

force_vector = lambda coeffs_u, coeffs_w: np.concatenate([get_force_vector_u(coeffs_u, coeffs_w, approx, approx.mesh),
                                                          get_force_vector_w(coeffs_u, coeffs_w, approx, approx.mesh)])

approx.solve_static(b, force_vector)

#approx.visualize(eval_points, blocking=True)

t = np.linspace(0,3,15)
approx.solve_dynamic(M, b, force_vector, t)

approx.visualize(eval_points, time_scaling_factor=0.5)





