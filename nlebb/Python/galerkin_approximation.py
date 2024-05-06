import numpy as np
from helper import poly_eval, poly_der
from newtons_method import newton
from scipy.optimize import root,fsolve
import scipy
import logging
import scipy
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
matplotlib.rcParams["figure.raise_window"] = False # disable raising windows, when "show()" or "pause()" is called

logger = logging.getLogger(__name__)

class GalerkinApproximation():
    def __init__(self, mesh, poly_basis):
        self.mesh = mesh
        
        # precompute coefficients for the derivatives of the basis functions
        self.basis = [{"f": poly_basis[i], "f_x": poly_der(poly_basis[i], 1), "f_xx": poly_der(poly_basis[i], 2)} for i in range(len(poly_basis))]
        self.coeffs_per_element = len(self.basis)
        
        self.coeffs_len = (mesh["num_elements"] + 1)*int(len(self.basis)/2) # length of the vector of coefficients, which describes the solution
        self.results = {"t": np.zeros((1,)), "u": np.zeros((1, self.coeffs_len))}
        
        self.fig_index = 100
        self.max_deflection = 0 # maximum recorded deflection for visualization
    
    def solve_static(self, b, force_vector):
        t_start = time.time()
        
        res = lambda coeffs: force_vector(coeffs[:self.coeffs_len], coeffs[self.coeffs_len:]) - b # define residual
        
        result = newton(res, np.zeros((2*self.coeffs_len,)), tol=1e-5, maxiter=1000000000) # own implementation
        #result = root(res, np.zeros((2*self.coeffs_len,)), method="hybr", tol=1e-5, options={"maxfev": 1000})
        #result = result["x"]
        coeffs_u = result[:self.coeffs_len]
        coeffs_w = result[self.coeffs_len:]
        self.results = {"t": np.zeros((1,)), "u": np.stack([coeffs_u]), "w": np.stack([coeffs_w])}
        logger.info(f"time for solving static problem: {time.time() - t_start:.2f}s")
    
    def solve_dynamic(self, M, b, force_vector, t):
        t_start_dynamic_total = time.time()
        
        state_prev = np.zeros((6*self.coeffs_len,))
        
        gamma = 1/2 # + 2e-1 # add a small constant for numerical damping
        beta = max(gamma/2, 1/4) # for achieving unconditional stability
        
        # define residual
        def residual(state_prev, state_next):
            # parse state_prev
            x_n = state_prev[:2*self.coeffs_len]
            x_d_n = state_prev[2*self.coeffs_len:4*self.coeffs_len]
            x_dd_n = state_prev[4*self.coeffs_len:]
            # parse state_next
            x_np1 = state_next[:2*self.coeffs_len]
            x_d_np1 = state_next[2*self.coeffs_len:4*self.coeffs_len]
            x_dd_np1 = state_next[4*self.coeffs_len:]
            
            residual = np.concatenate([M@x_dd_np1 + force_vector(x_np1[:self.coeffs_len], x_np1[self.coeffs_len:]) - b, # equation of motion
                                       1/(beta*del_t**2)*(x_np1 - x_n) - 1/(beta*del_t)*x_d_n - (0.5 - beta)/beta*x_dd_n - x_dd_np1, # equations for Newmark method
                                       gamma/(beta*del_t)*(x_np1 - x_n) + (1 - gamma/beta)*x_d_n + del_t*(beta - 0.5*gamma)/beta*x_dd_n - x_d_np1], axis=0)
            
            
            return residual
        
        self.results = {"t": [t[0]], "u": [state_prev[:self.coeffs_len]], "w": [state_prev[self.coeffs_len:2*self.coeffs_len]]}
        
        for iter_index in range(len(t) - 1):
            t_start_dynamic_step = time.time()
            logger.info(f"starting Newmark iteration {iter_index+1}")
            
            del_t = t[iter_index + 1] - t[iter_index]
            #state_next = newton(lambda state_next: residual(state_prev, state_next), state_prev, tol=1e-5, maxiter=1000000000) # own implementation
            state_next = root(lambda state_next: residual(state_prev, state_next), state_prev, method="hybr", tol=1e-5, options={"maxfev": 1000})["x"] # solver from scipy (about twice as fast)
            state_prev = state_next.copy()
            
            self.results["t"].append(t[iter_index+1])
            self.results["u"].append(state_next[:self.coeffs_len])
            self.results["w"].append(state_next[self.coeffs_len:2*self.coeffs_len])
            #print(force_vector(self.results["u"][-1], self.results["w"][-1]))
            
            logger.info(f"time for Newmark iteration: {time.time() - t_start_dynamic_step:.2f}s")
            
        self.results = {key: np.stack(self.results[key]) for key in self.results}
        
        logger.info(f"time for solving dynamic problem: {time.time() - t_start_dynamic_total:.2f}s")
    
    def eval_solution(self, coeffs, eval_points):
        y = np.zeros_like(eval_points)
        
        for e in range(self.mesh["num_elements"]): # loop over elements
            left_boundary = self.mesh["element_boundaries"][e]
            right_boundary = self.mesh["element_boundaries"][e+1]
            point_mask = np.argwhere(np.logical_and(eval_points >= left_boundary, eval_points <= right_boundary))
            for point_index in point_mask: # loop over points on the element
                y[point_index] = 0 # points on the boundaries between elements are evaluated twice to avoid more complex logic, therefore the elements have to be reset to zero before (re-)evaluation
                point_local = (eval_points[point_index] - left_boundary)/(right_boundary - left_boundary) # local coordinate of the point
                index_global_left = e*int(self.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
                for basis_index_local in range(self.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    y[point_index] += coeffs[basis_index_global].item()*poly_eval(self.basis[basis_index_local]["f"], point_local)
        
        self.max_deflection = max(max(abs(y)), self.max_deflection)
        return y

    def visualize(self, eval_points, time_scaling_factor=1, blocking=True):
        """animation runs faster than real-time, if time_saling_factor > 1"""
        
        is_dynamic = len(self.results["u"]) > 1 # determine, whether there is more than one set of coefficients in the results
        
        x = eval_points
        #u = self.eval_solution(coeffs_u, x)
        
        spec = matplotlib.gridspec.GridSpec(nrows=2, ncols=1,
                                            width_ratios=[1], wspace=0.5,
                                            hspace=0.15, height_ratios=[4, 1],
                                            bottom=0)
        fig = plt.figure(self.fig_index)
        fig.clear()
        
        ax_values = fig.add_subplot(spec[0])
        
        t_min = self.results["t"].min()
        t_max = self.results["t"].max()
        if is_dynamic:
            ax_slider = fig.add_subplot(spec[1])
            slider = Slider(ax_slider, 'time in s', t_min, t_max, valinit=t_max)
        
        # function for interpolating solutions at time steps
        if is_dynamic:
            interp_func_u = scipy.interpolate.interp1d(self.results["t"], self.results["u"], kind="linear", axis=0)
            interp_func_w = scipy.interpolate.interp1d(self.results["t"], self.results["w"], kind="linear", axis=0)
        else:
            interp_func_u = lambda t: self.results["u"][0]
            interp_func_w = lambda t: self.results["w"][0]
        # define the function to update the plot (callback for the slider / for manually updating the displayed time)
        def update(time):
            coeffs_u = interp_func_u(time)
            coeffs_w = interp_func_w(time)
            u = self.eval_solution(coeffs_u, x)
            w = self.eval_solution(coeffs_w, x)
            
            # plot values of u and w
            ax_values.clear()
            ax_values.plot(x, u)
            ax_values.plot(x, w)
            ax_values.set_ylim([-1.1*self.max_deflection, 1.1*self.max_deflection])
            ax_values.set_xlabel("x in m")
            ax_values.set_ylabel("u, w in m")
            ax_values.legend(["u", "w"])
            ax_values.grid(True)
            
            # plot centerline of the beam
            
            fig.canvas.draw_idle()
        
        if is_dynamic:
            #update(0)
            #plt.pause(2e-1)
            t_start = time.time()
            while t_min + (time.time() - t_start)*time_scaling_factor < t_max:
                t_scaled = (time.time() - t_start)*time_scaling_factor
                update(t_scaled)
                slider.set_val(t_scaled)
                plt.pause(5e-2)
            
            slider.set_val(t_max)
            slider.on_changed(update) # attach the callback function to the slider
            
        update(t_max) # make sure that the last point in time is displayed at the end
        
        if blocking:
            plt.show()
        else:
            plt.pause(2)