import numpy as np
import time
from galerkin_approximation import GalerkinApproximation
from visualization import BeamVisualization
import matplotlib.pyplot as plt
import logging
import sys

#np.set_printoptions(precision=3)

# ----------findings----------
# Hermite splines can only accurately approximate solutions, which are twice continuously differentiable, otherwise the approximation will be "too smooth" at some points


# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(filename)25s | %(levelname)7s | %(message)s', datefmt='%H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# !!!!!!!!!!!!!!!!!!!!visualize derivatives!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!integrate energies into visualization!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!visualize static solution as reference in dynamic visualization!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!activate coupling and nonlinearities!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!introduce check, whether the beam with given boundary conditions is statically detterminate (depends on nonlinearities...)!!!!!!!!!!!!!!!!!!!!!

# define domain (describes the domain, in which the problem is to be solved)
l = 1 # length of the beam
num_elements = 10
domain = {"num_elements": num_elements,
        "element_boundaries": np.linspace(0, l, num_elements+1),
        "boundary_conds": [0, 4], # what kind of boundary conditions to use at the ends of the beam (0: free, 1: pinned in vertical direction, 2: pinned in horizontal direction, 3: pinned in both directions, 4: clamped)
        "load_points": [0, 0.5, 1], # points, at which discrete external loads are applied
        "N": [0, 0, 0], # axial force at load points (applying to internal points leeds to solutions with discontinuous derivatives, which can't be represented exactly using Hermite splines)
        "Q": [0, 0, 0], # lateral force at load points
        "M": [0, 0, 0], # moment at load points
        "u": [0, 0], # axial displacement for pinned and clamped boundary conditions
        "w": [0, 0], # lateral displacement for pinned and clamped boundary conditions
        "w_x": [0, 0], # derivatiive of lateral displacement for clamped boundary conditions
        "f": lambda x: 0, # specific axial force as a function of the position along the beam
        "q": lambda x: 1} # specific lateral force as a function of the position along the beam

# define polynomial basis functions (will be evaluated between 0 and 1 / equal to test functions, because the Galerkin method is used)
# the coefficients must be exactly representable using machine numbers, so that the polynomials can be evaluated without rounding errorss at the boundaries
poly_basis = np.array([[1, 0, -3, 2],
                       [0, 1, -2, 1],
                       [0, 0, 3, -2],
                       [0, 0, -1, 1]])
# define object, which holds an approximate solution to the problem
approx = GalerkinApproximation(domain, poly_basis)

#def print_matrix(a):
#    for row in a:
#        for element in row:
#            print(f"{element:.2e}", end=",  ")
#        print("")



num_eval_points = 1000
eval_points = np.linspace(0, l, num_eval_points)

beamVisualization = BeamVisualization(time_scaling_factor=0.2)

approx.solve_static()

#beamVisualization.visualize(approx, eval_points, blocking=True)

t = np.linspace(0,1,15)
approx.solve_dynamic(t)

#w_e, e_k, e_i = postprocessing(approx.results["t"], approx.results["u"], approx.results["w"], M, b)

beamVisualization.visualize(approx, eval_points)





