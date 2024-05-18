import numpy as np
import time
from galerkin_approximation import GalerkinApproximation
from visualization import BeamVisualization
from helper import check_determinacy
import matplotlib.pyplot as plt
from math import pi
import logging
import sys

#np.set_printoptions(precision=3)

# ----------findings----------
# If u and w are combined into a single vector for the SVD, their magnitudes should be similar. Otherwise the smaller component will not be represented well by the resulting basis vectors.
# Hermite splines can only accurately approximate solutions, which are twice continuously differentiable, otherwise the approximation will be "too smooth" at some points


# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(filename)25s | %(levelname)7s | %(message)s', datefmt='%H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# define domain (describes the domain, in which the problem is to be solved)
l = 1 # length of the beam
num_elements = 16
domain = {"l": l,
        "num_elements": num_elements,
        "element_boundaries": np.linspace(0, l, num_elements+1),
        "boundary_conds": [4, 0], # what kind of boundary conditions to use at the ends of the beam (0: free, 1: pinned in vertical direction, 2: pinned in horizontal direction, 3: pinned in both directions, 4: clamped)
        "load_points": lambda t: [0, l/2, l], # points, at which discrete external loads are applied
        "N": lambda t: [0, 0, 1e-2*np.sin(2*pi*(t+1)**2)], # axial force at load points in N (applying to internal points leeds to solutions with discontinuous derivatives, which can't be represented exactly using Hermite splines)
        "Q": lambda t: [0, 0, 1e-2*np.sin(2*pi*(t+1)**2)], # lateral force at load points in N
        "M": lambda t: [0, 0, 0], # moment at load points in Nm
        "u": [0, 0], # axial displacement for pinned and clamped boundary conditions
        "w": [0, 0], # lateral displacement for pinned and clamped boundary conditions
        "w_x": [0, 0], # derivatiive of lateral displacement for clamped boundary conditions
        "f": lambda t, x: 0, # specific axial force as a function of the position along the beam
        "q": lambda t, x: 0} # specific lateral force as a function of the position along the beam (gravity: q = -A*g)

# check whether the beam is statically determinate with the given boundary conditions
check_determinacy(domain["boundary_conds"])

observation_points = np.array([domain["l"]/2, domain["l"]]) # points along the beam, at which the solution is plotted over time

# object, which allows computing approximate solutions to the problem
approx = GalerkinApproximation(domain)

# object, which allows visualizing the solution
beamVisualization = BeamVisualization(time_scaling_factor=0.1)

# solve static problem and visualize the solution
results_static = approx.solve_static(t=0)
#beamVisualization.visualize(approx, results_static, blocking=False)

# solve dynamic problem and visualize the solution
t_sim = np.linspace(0,2,50)
results_dynamic, results_static = approx.solve_dynamic(t_sim, static_reference=True)
beamVisualization.visualize(approx, results_dynamic, results_static, observation_points, blocking=False)

# ---------- perform Proper Orthogonal Decomposition----------

num_snapshots = np.inf # maximum number of snapshots, which are chosen equidistantly from the available time steps
sv_num = 5 # number of singular values to use

def snaphsots_from_results(results):
    # get snapshot matrix from simulation results
    snapshots = {"u": [], "w": []}
    for time_index in np.round(np.linspace(0, len(results["t"])-1, min(num_snapshots, len(results["t"])-1))).astype(int):
        snapshots["u"].append(results["u"][time_index,:])
        snapshots["w"].append(results["w"][time_index,:])
    snapshots["u"] = np.stack(snapshots["u"])
    snapshots["w"] = np.stack(snapshots["w"])
    
    snapshot_matrix = np.concatenate([snapshots["u"], snapshots["w"]], axis=1).transpose()
    return snapshots, snapshot_matrix

snapshots, snapshot_matrix = snaphsots_from_results(results_dynamic)

# get basis vectors using Singular Value Decomposition
U, S, _ = np.linalg.svd(snapshot_matrix) # U contains the singular vectors as columns
S /= S.sum() # normalize, so the relative error can be computed

basis_pod = U[:, :sv_num]
logger.info(f"sum of normalized singular values: {sum(S[:sv_num]):.4f}")
logger.info(f"absoute error on snapshots in coefficient space: error={np.linalg.norm(snapshot_matrix - basis_pod@basis_pod.transpose()@snapshot_matrix):.6f}")
#print(snapshot_matrix[:,-1])
#print(reduced_bases[component]@reduced_bases[component].transpose()@snapshot_matrix[:,-1])

# ----------perform simulation using the reduced basis----------
# different loads could be defined here by redefining the domain and approximation object

results_dynamic_pod, results_static_pod = approx.solve_dynamic(t_sim, Q=basis_pod, static_reference=True)
beamVisualization.visualize(approx, results_dynamic_pod, results_static_pod, observation_points, blocking=False)


# ----------plot snapshots with and without POD----------

snapshots_pod, _ = snaphsots_from_results(results_dynamic_pod)
    
fig, (ax_displacements, ax_centerline) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"hspace": 0.35, "top": 0.90, "bottom": 0.1})
fig.suptitle("snapshots wth and without POD")

# plot snapshots
eval_points = np.linspace(0, domain["l"], 100)
for snapshot_u, snapshot_u_pod, snapshot_w, snapshot_w_pod in zip(snapshots["u"], snapshots_pod["u"], snapshots["w"], snapshots_pod["w"]):
    l1 = ax_displacements.plot(eval_points, approx.eval_solution(snapshot_u, eval_points, "f"), "-b")[0]
    l2 = ax_displacements.plot(eval_points, approx.eval_solution(snapshot_u_pod, eval_points, "f"), "--b")[0]
    l3 = ax_displacements.plot(eval_points, approx.eval_solution(snapshot_w, eval_points, "f"), "-r")[0]
    l4 = ax_displacements.plot(eval_points, approx.eval_solution(snapshot_w_pod, eval_points, "f"), "--r")[0]
    ax_displacements.set_xlabel("x in m")
    ax_displacements.set_ylabel("z in m")
    ax_displacements.set_title("displacements")
    ax_displacements.grid(True)
    
    l5 = ax_centerline.plot(eval_points + approx.eval_solution(snapshot_u, eval_points, "f"), approx.eval_solution(snapshot_w, eval_points, "f"), "-b")[0]
    l6 = ax_centerline.plot(eval_points + approx.eval_solution(snapshot_u_pod, eval_points, "f"), approx.eval_solution(snapshot_w_pod, eval_points, "f"), "--b")[0]
    ax_centerline.set_xlabel("x in m")
    ax_centerline.set_ylabel("z in m")
    ax_centerline.set_title("displacements")
    ax_centerline.grid(True)
    
ax_displacements.legend([l1, l2, l3, l4], ["u", "u (POD)", "w", "w (POD)"])
ax_centerline.legend([l5, l6], ["centerline", "centerline (POD)"])

plt.show()





