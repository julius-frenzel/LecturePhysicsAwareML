import numpy as np
import time
from galerkin_approximation import GalerkinApproximation
from visualization import BeamVisualization
from helper import check_determinacy
from beam_model import get_force_vector_u, get_force_vector_w, get_rhs_u
import matplotlib.pyplot as plt
import logging
import sys

#np.set_printoptions(precision=3)

# ----------findings----------
# Hermite splines can only accurately approximate solutions, which are twice continuously differentiable, otherwise the approximation will be "too smooth" at some points
# (dictionary lookups might be inefficient)

# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(filename)25s | %(levelname)7s | %(message)s', datefmt='%H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# !!!!!!!!!!!!!!!!!!!implement MOR based on observations!!!!!!!!!!!!!!!!!!!!!

# define domain (describes the domain, in which the problem is to be solved)
l = 1 # length of the beam
num_elements = 8
domain = {"l": l,
        "num_elements": num_elements,
        "element_boundaries": np.linspace(0, l, num_elements+1),
        "boundary_conds": [4, 0], # what kind of boundary conditions to use at the ends of the beam (0: free, 1: pinned in vertical direction, 2: pinned in horizontal direction, 3: pinned in both directions, 4: clamped)
        "load_points": lambda t: [0, l/2, l], # points, at which discrete external loads are applied
        "N": lambda t: [0, 1, 1e-2], # axial force at load points in N (applying to internal points leeds to solutions with discontinuous derivatives, which can't be represented exactly using Hermite splines)
        "Q": lambda t: [0, 0, 1e-2], # lateral force at load points in N
        "M": lambda t: [0, 0, 1e-2], # moment at load points in Nm
        "u": [0, 0], # axial displacement for pinned and clamped boundary conditions
        "w": [0, 0], # lateral displacement for pinned and clamped boundary conditions
        "w_x": [0, 0], # derivatiive of lateral displacement for clamped boundary conditions
        "f": lambda t, x: 1e-2, # specific axial force as a function of the position along the beam
        "q": lambda t, x: 1e-2} # specific lateral force as a function of the position along the beam (gravity: q = -A*g)

# check whether the beam is statically determinate with the given boundary conditions
check_determinacy(domain["boundary_conds"])

# define polynomial basis functions (will be evaluated between 0 and 1 / equal to test functions, because the Galerkin method is used)
# the coefficients must be exactly representable using machine numbers, so that the polynomials can be evaluated without rounding errors at the boundaries
poly_basis = np.array([[1, 0, -3, 2],
                       [0, 1, -2, 1],
                       [0, 0, 3, -2],
                       [0, 0, -1, 1]])

# object, which holds an approximate solution to the problem
approx = GalerkinApproximation(domain, poly_basis)


num_eval_points = 100
eval_points = np.linspace(0, domain["l"], num_eval_points)

beamVisualization = BeamVisualization(time_scaling_factor=0.1)

results_static = approx.solve_static(t=0)

beamVisualization.visualize(approx, eval_points, results_static, blocking=True)

observation_points = np.array([domain["l"]/2, domain["l"]])
t = np.linspace(0,1,15)
results_dynamic, results_static = approx.solve_dynamic(t, static_reference=True)

beamVisualization.visualize(approx, eval_points, results_dynamic, results_static, observation_points)





