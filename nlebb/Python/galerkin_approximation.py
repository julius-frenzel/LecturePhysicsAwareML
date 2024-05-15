import numpy as np
from helper import poly_eval, poly_der, quad, quad_points_untransformed
from newtons_method import newton
from newmark_method import newmark
from beam_model import get_mass_matrix_u, get_mass_matrix_w, get_force_vector_u, get_force_vector_w, get_rhs_u, get_rhs_w
import scipy
from scipy.optimize import root
import logging
import scipy
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.raise_window"] = False # disable raising windows, when "show()" or "pause()" is called

logger = logging.getLogger(__name__)

class GalerkinApproximation():
    def __init__(self, domain, poly_basis):
        self.domain = domain
        
        # precompute coefficients for the derivatives of the basis functions
        self.basis_coeffs = [{"f": poly_basis[i], "f_x": poly_der(poly_basis[i], 1), "f_xx": poly_der(poly_basis[i], 2)} for i in range(len(poly_basis))]
        self.basis_evaluated = []
        for basis_coeffs in self.basis_coeffs:
            self.basis_evaluated.append(dict())
            for derivative in basis_coeffs:
                self.basis_evaluated[-1][derivative] = np.array([poly_eval(basis_coeffs[derivative], point) for point in quad_points_untransformed])
        
        self.coeffs_per_element = len(self.basis_coeffs)
        self.coeffs_len = (domain["num_elements"] + 1)*int(len(self.basis_coeffs)/2) # length of the vector of coefficients, which describes the solution
        
    
    def solve_static(self, t=0):
        t_start = time.time()
        # assemble right hand side
        b = np.concatenate([get_rhs_u(self, self.domain, t),
                            get_rhs_w(self, self.domain, t)])
        # function for assembling force vector
        force_vector = lambda coeffs_u, coeffs_w: np.concatenate([get_force_vector_u(coeffs_u, coeffs_w, self, self.domain),
                                                                  get_force_vector_w(coeffs_u, coeffs_w, self, self.domain)])
        # define residual
        residual = lambda coeffs: force_vector(coeffs[:self.coeffs_len], coeffs[self.coeffs_len:]) - b
        # solve problem
        #result = newton(residual, np.zeros((2*self.coeffs_len,)), tol=1e-5, maxiter=10) # own implementation (slower)
        result = root(residual, np.zeros((2*self.coeffs_len,)), method="hybr", tol=1e-10, options={"maxfev": 1000})["x"] # solver from scipy (faster)
        # fomat results
        coeffs_u = result[:self.coeffs_len]
        coeffs_w = result[self.coeffs_len:]
        results = {"t": np.array([t]), "u": np.stack([coeffs_u]), "w": np.stack([coeffs_w])}
        
        logger.info(f"time for solving static problem: {time.time() - t_start:.2f}s")
        
        result = result.reshape(1,-1) # add time dimension for postprocessing
        results.update(self.postprocessing({"x": result, "x_d": np.zeros_like(result), "x_dd": np.zeros_like(result)}, lambda t: b, np.array([t]), silent=True))
        
        return results
    
    
    def solve_dynamic(self, t, static_reference=False):
        t_start_dynamic_total = time.time()
        # assemble mass matrix
        M_u = get_mass_matrix_u(self, self.domain)
        M_w = get_mass_matrix_w(self, self.domain)
        M = np.block([[M_u,                     np.zeros_like(M_u)],
                      [np.zeros_like(M_w),      M_w]])
        # assemble right hand side
        b = lambda t: np.concatenate([get_rhs_u(self, self.domain, t),
                            get_rhs_w(self, self.domain, t)])
        # function for assembling force vector
        force_vector = lambda coeffs_u, coeffs_w: np.concatenate([get_force_vector_u(coeffs_u, coeffs_w, self, self.domain),
                                                                  get_force_vector_w(coeffs_u, coeffs_w, self, self.domain)])
        # define residual
        residual = lambda x_np1, x_d_np1, x_dd_np1, t: M@x_dd_np1 + force_vector(x_np1[:self.coeffs_len], x_np1[self.coeffs_len:]) - b(t)
        
        # perform time integration
        newmark_results = newmark(residual, t, x_0=np.zeros(2*self.coeffs_len), x_d_0=np.zeros(2*self.coeffs_len), x_dd_0=np.zeros(2*self.coeffs_len))
        # format results (split into u and w)
        results_dynamic = {"t": t,
                           "u": newmark_results["x"][:,:self.coeffs_len],
                           "u_d": newmark_results["x_d"][:,:self.coeffs_len],
                           "u_dd": newmark_results["x_dd"][:,:self.coeffs_len],
                           "w": newmark_results["x"][:,self.coeffs_len:],
                           "w_d": newmark_results["x_d"][:,self.coeffs_len:],
                           "w_dd": newmark_results["x_dd"][:,self.coeffs_len:]}

        logger.info(f"time for solving dynamic problem: {time.time() - t_start_dynamic_total:.2f}s")
        
        # postprocessing for dynamic results
        results_dynamic.update(self.postprocessing(newmark_results, b, t))
        
        logger.info("computing static reference solutions")
        results_static = dict()
        for t_i in t:
            result_static = self.solve_static(t_i)
            for key in result_static:
                if not key in results_static:
                    results_static[key] = [result_static[key]]
                else:
                    results_static[key].append(result_static[key])
        results_static = {key: np.stack(results_static[key]) for key in results_static}
        
        
        
        return results_dynamic, results_static
    
    
    def eval_solution(self, coeffs, eval_points):
        y = np.zeros_like(eval_points)
        coeffs = coeffs.squeeze()
        
        for e in range(self.domain["num_elements"]): # loop over elements
            left_boundary = self.domain["element_boundaries"][e]
            right_boundary = self.domain["element_boundaries"][e+1]
            point_mask = np.argwhere(np.logical_and(eval_points >= left_boundary, eval_points <= right_boundary))
            for point_index in point_mask: # loop over points on the element
                y[point_index] = 0 # points on the boundaries between elements are evaluated twice to avoid more complex logic, therefore the elements have to be reset to zero before (re-)evaluation
                point_local = (eval_points[point_index] - left_boundary)/(right_boundary - left_boundary) # local coordinate of the point
                index_global_left = e*int(self.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
                for basis_index_local in range(self.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    y[point_index] += coeffs[basis_index_global].item()*poly_eval(self.basis_coeffs[basis_index_local]["f"], point_local)
        
        return y
    
    def postprocessing(self, results, b, t, silent=False):
        """compute external work, kinetic energies and internal energies from given trajectories, the mass matrix and the right hand side"""
        t_start = time.time()
        
        results_post = dict()
        #is_dynamic = len(results["u"]) > 1 # determine, whether there is more than one set of coefficients in the results
        
        # assemble mass matrix
        M_u = get_mass_matrix_u(self, self.domain)
        M_w = get_mass_matrix_w(self, self.domain)
        M = np.block([[M_u,                     np.zeros_like(M_u)],
                      [np.zeros_like(M_w),      M_w]])
        
        if not silent:
            logger.info("performing postprocessing")
        
        # external work
        states = results["x"]
        states_d = results["x_d"]
        states_dd = results["x_dd"]
        ext_work = [states[0].transpose()@b(t[0])] # assume constant load starting from zero state before the start of the simulation
        printed_info = False
        for i in range(1, len(states)):
            t_prev = t[i - 1]
            t_next = t[i]
            x_d_prev = states_d[i-1]
            x_d_next = states_d[i]
            x_dd_prev = states_dd[i-1]
            x_dd_next = states_dd[i]
            # derivatives assuming linear acceleratios
            x_d = lambda t: x_d_prev + (x_dd_prev - (x_dd_next - x_dd_prev)*t_prev/(t_next - t_prev))*(t - t_prev) + (x_dd_next - x_dd_prev)/2*(t**2 - t_prev**2)/(t_next - t_prev)
            #check whether results are consistent using x_d_next and fall back to constant accelerations, if necessary
            der_diff = x_d_next - x_d(t_next)
            if (np.abs(der_diff) > 1e-10).any():
                if not printed_info:
                    logger.info("Inconsistent results with assumption of linear accelerations. Falling back to constant accelerations for computing external work.")
                    printed_info = True
                x_d = lambda t: x_d_prev + (x_d_next - x_d_prev)*(t - t_prev)/(t_next - t_prev)
            ext_work.append(ext_work[-1] + quad(lambda t: x_d(t).transpose()@b(t), [t_prev, t_next]))
        results_post["W_ext"] = np.stack(ext_work)
        
        
        # kinetic energies
        kin_energy = []
        for state_d in states_d:
            kin_energy.append(1/2*state_d.transpose()@M@state_d)
        results_post["E_k"] = np.stack(kin_energy)
        
        # internal energies
        results_post["W_i"] = results_post["W_ext"] - results_post["E_k"]
        
        if not silent:
            logger.info(f"postprocessing took {time.time() - t_start:.2f}s")
        
        return results_post






