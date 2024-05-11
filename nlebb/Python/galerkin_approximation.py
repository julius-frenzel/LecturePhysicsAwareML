import numpy as np
from helper import poly_eval, poly_der
from newtons_method import newton
from newmark_method import newmark
from beam_model import get_mass_matrix_u, get_mass_matrix_w, get_force_vector_u, get_force_vector_w, get_rhs_u, get_rhs_w
import scipy
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
        self.basis = [{"f": poly_basis[i], "f_x": poly_der(poly_basis[i], 1), "f_xx": poly_der(poly_basis[i], 2)} for i in range(len(poly_basis))]
        self.coeffs_per_element = len(self.basis)
        
        self.coeffs_len = (domain["num_elements"] + 1)*int(len(self.basis)/2) # length of the vector of coefficients, which describes the solution
        self.results = {"t": np.zeros((1,)), "u": np.zeros((1, self.coeffs_len))}
        
    
    def solve_static(self):
        t_start = time.time()
        # assemble right hand side
        b = np.concatenate([get_rhs_u(self, self.domain),
                            get_rhs_w(self, self.domain)])
        # function for assembling force vector
        force_vector = lambda coeffs_u, coeffs_w: np.concatenate([get_force_vector_u(coeffs_u, coeffs_w, self, self.domain),
                                                                  get_force_vector_w(coeffs_u, coeffs_w, self, self.domain)])
        # define residual
        residual = lambda coeffs: force_vector(coeffs[:self.coeffs_len], coeffs[self.coeffs_len:]) - b
        # solve problem
        result = newton(residual, np.zeros((2*self.coeffs_len,)), tol=1e-5, maxiter=10) # own implementation (slower)
        #result = root(residual, np.zeros((2*self.coeffs_len,)), method="hybr", tol=1e-5, options={"maxfev": 1000})["x"] # solver from scipy (faster)
        # fomat results
        coeffs_u = result[:self.coeffs_len]
        coeffs_w = result[self.coeffs_len:]
        self.results = {"t": np.zeros((1,)), "u": np.stack([coeffs_u]), "w": np.stack([coeffs_w])}
        
        logger.info(f"time for solving static problem: {time.time() - t_start:.2f}s")
        
        return self.results
    
    def solve_dynamic(self, t):
        t_start_dynamic_total = time.time()
        # assemble mass matrix
        M_u = get_mass_matrix_u(self, self.domain)
        M_w = get_mass_matrix_w(self, self.domain)
        M = np.block([[M_u,                     np.zeros_like(M_u)],
                      [np.zeros_like(M_w),      M_w]])
        # assemble right hand side
        b = np.concatenate([get_rhs_u(self, self.domain),
                            get_rhs_w(self, self.domain)])
        # function for assembling force vector
        force_vector = lambda coeffs_u, coeffs_w: np.concatenate([get_force_vector_u(coeffs_u, coeffs_w, self, self.domain),
                                                                  get_force_vector_w(coeffs_u, coeffs_w, self, self.domain)])
        # define residual
        residual = lambda x_np1, x_d_np1, x_dd_np1: M@x_dd_np1 + force_vector(x_np1[:self.coeffs_len], x_np1[self.coeffs_len:]) - b
        
        # perform time integration
        newmark_results = newmark(residual, t, x_0=np.zeros(2*self.coeffs_len), x_d_0=np.zeros(2*self.coeffs_len), x_dd_0=np.zeros(2*self.coeffs_len))
        # format results (split into u and w)
        self.results = {"t": t, "u": newmark_results["x"][:,:self.coeffs_len], "w": newmark_results["x"][:,self.coeffs_len:]}

        logger.info(f"time for solving dynamic problem: {time.time() - t_start_dynamic_total:.2f}s")
        
        self.postprocessing()
        
        return self.results
    
    
    
    def eval_solution(self, coeffs, eval_points):
        y = np.zeros_like(eval_points)
        
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
                    y[point_index] += coeffs[basis_index_global].item()*poly_eval(self.basis[basis_index_local]["f"], point_local)
        
        return y
    
    
    def postprocessing(self):
        """compute external work, kinetic energies and internal energies from given trajectories, the mass matrix and the right hand side"""
        t_start = time.time()
        
        u = self.results["u"]
        w = self.results["w"]
        t = self.results["t"]
        
        # assemble mass matrix
        M_u = get_mass_matrix_u(self, self.domain)
        M_w = get_mass_matrix_w(self, self.domain)
        M = np.block([[M_u,                     np.zeros_like(M_u)],
                      [np.zeros_like(M_w),      M_w]])
        # assemble right hand side
        b = np.concatenate([get_rhs_u(self, self.domain),
                            get_rhs_w(self, self.domain)])
        
        
        logger.info("performing postprocessing")
        
        # external work
        states = np.concatenate([u, w], axis=1)
        ext_work = []
        for state in states:
            ext_work.append(state.transpose()@b)
        ext_work = np.stack(ext_work)
        
        
        # kinetic energies
        u_d = np.gradient(u, np.diff(t)[0], axis=0)
        w_d = np.gradient(w, np.diff(t)[0], axis=0)
        states_d = np.concatenate([u_d, w_d], axis=1)
        kin_energy = []
        for state_d in states_d:
            kin_energy.append(1/2*state_d.transpose()@M@state_d)
        kin_energy = np.stack(kin_energy)
        
        
        # internal energies
        int_energy = ext_work - kin_energy
        
        logger.info(f"postprocessing took {time.time() - t_start}")
        
        # visualize external work
        fig, ax = plt.subplots()
        ax.plot(t, ext_work)
        ax.set_xlabel("time in s")
        ax.set_ylabel("external work in J")
        ax.grid(True)
        plt.pause(1e-1)
        
        # visualize kinetic energies
        fig, ax = plt.subplots()
        ax.plot(t, kin_energy)
        ax.set_xlabel("time in s")
        ax.set_ylabel("kinetic energy in J")
        ax.grid(True)
        plt.pause(1e-1)
        
        # visualize internal energies
        fig, ax = plt.subplots()
        ax.plot(t, int_energy)
        ax.set_xlabel("time in s")
        ax.set_ylabel("internal energy in J")
        ax.grid(True)
        plt.pause(1e-1)
        
        return ext_work, kin_energy, int_energy






