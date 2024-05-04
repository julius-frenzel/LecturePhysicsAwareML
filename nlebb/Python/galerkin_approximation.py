import numpy as np
import matplotlib.pyplot as plt
from helper import poly_eval, poly_der
from newtons_method import newton
from scipy.optimize import root,fsolve
import scipy
import time
import matplotlib
matplotlib.rcParams["figure.raise_window"] = False # disable raising windows, when "show()" or "pause()" is called

class GalerkinApproximation():
    def __init__(self, mesh, poly_basis):
        self.mesh = mesh
        
        # precompute coefficients for the derivatives of the basis functions
        self.basis = [{"f": poly_basis[i], "f_x": poly_der(poly_basis[i], 1), "f_xx": poly_der(poly_basis[i], 2)} for i in range(len(poly_basis))]
        self.coeffs_per_element = len(self.basis)
        
        # !!!!!!!!!!rename coeffs_len to num_coeffs?!!!!!!!!!
        self.coeffs_len = (mesh["num_elements"] + 1)*int(len(self.basis)/2) # length of vector of coefficients, which describes the solution
        self.coeffs = np.zeros((self.coeffs_len,))
        
        self.fig_index = 100
    
    def solve_static(self, b, force_vector):
        t_start = time.time()
        
        res = lambda x: force_vector(x, self, self.mesh) - b # define residual

        result = root(res, np.zeros((self.coeffs_len,)))
        coeffs = result["x"]
        #print(coeffs)
        self.coeffs = coeffs
        print(f"solving static problem took {time.time() - t_start:.2f}s")
    
    def solve_dynamic(self, M, b, force_vector):
        t_start = time.time()
        
        x_n = np.zeros((self.coeffs_len,))
        x_d_n = np.zeros((self.coeffs_len,))
        x_dd_n = np.zeros((self.coeffs_len,))
        x_prev = np.concatenate([x_n, x_d_n, x_dd_n], axis=0).copy()
        #x_prev += 1*np.random.rand(x_prev.shape[0])
        #x_prev = x_n
        
        gamma = 1/2 + 2e-1 # add a small constant for numerical damping
        beta = max(gamma/2, 1/4)
        del_t = 1e-1
        
        # define residual
        def res(x_prev, x_next):
            # parse x_prev
            x_n = x_prev[:self.coeffs_len]
            x_d_n = x_prev[self.coeffs_len:2*self.coeffs_len]
            x_dd_n = x_prev[2*self.coeffs_len:]
            # parse x_next
            x_np1 = x_next[:self.coeffs_len]
            x_d_np1 = x_next[self.coeffs_len:2*self.coeffs_len]
            x_dd_np1 = x_next[2*self.coeffs_len:]
            
            #res = force_vector(x_np1, self, self.mesh) - b
            res = np.concatenate([M@x_dd_np1 + force_vector(x_np1, self, self.mesh) - b,
                             1/(beta*del_t**2)*(x_np1 - x_n) - 1/(beta*del_t)*x_d_n - (0.5 - beta)/beta*x_dd_n - x_dd_np1,
                             gamma/(beta*del_t)*(x_np1 - x_n) + (1 - gamma/beta)*x_d_n + del_t*(beta - 0.5*gamma)/beta*x_dd_n - x_d_np1], axis=0)
            #print(np.linalg.norm(res))
            #time.sleep(0.1)
            return res
        
        for iter in range(100000):
            print(f"iteration {iter+1}")
            
            #scipy.optimize.show_options(solver="root")
            #time.sleep(100)
            result = root(lambda x_next: res(x_prev, x_next), x_prev, method="hybr", tol=1e-5, options={"maxfev": 1000})
            #result = fsolve(lambda x_next: res(x_prev, x_next), x_prev, xtol=1e-3, maxfev=1000, factor=0.1)
            #result_test = newton(lambda x: np.array([x[0]**2 + (x[1] - 1)**3, x[1]]), np.array([1., -1.]), tol=1e-5, maxiter=100)
            #print(f"test result: {result_test}")
            #print("done")
            #time.sleep(100)
            
            #t_start = time.time()
            #result = newton(lambda x_next: res(x_prev, x_next), x_prev, tol=1e-5, maxiter=1000000000)
            #print(f"solving dynamic problem took {time.time() - t_start:.2f}s")
            #print(f"result: {result}")
            #self.coeffs = result
            #self.visualize(np.linspace(0, 1, 1000))
            #time.sleep(100)
            
            # parse results
            x_n = result["x"][:self.coeffs_len]
            x_d_n = result["x"][self.coeffs_len:2*self.coeffs_len]
            x_dd_n = result["x"][2*self.coeffs_len:]
            x_prev = result["x"].copy()
            
            self.coeffs = x_n
            print(self.coeffs)
            self.visualize(np.linspace(0, 1, 1000))
            
            if not result["success"]:
                time.sleep(100)
            
        print("done")
        time.sleep(100)
        

        print(f"solving dynamic problem took {time.time() - t_start:.2f}s")

    def visualize(self, eval_points):
        coeffs_u = self.coeffs
        
        x = eval_points
        u = np.zeros_like(eval_points)
        
        for e in range(self.mesh["num_elements"]): # loop over elements
            left_boundary = self.mesh["element_boundaries"][e]
            right_boundary = self.mesh["element_boundaries"][e+1]
            point_mask = np.argwhere(np.logical_and(eval_points >= left_boundary, eval_points <= right_boundary))
            for point_index in point_mask: # loop over points on the element
                u[point_index] = 0 # points on the boundaries between elements are evaluated twice to avoid more complex logic, therefore the elements have to be reset to zero before (re-)evaluation
                point_local = (eval_points[point_index] - left_boundary)/(right_boundary - left_boundary) # local coordinate of the point
                index_global_left = e*int(self.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
                for basis_index_local in range(self.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    u[point_index] += coeffs_u[basis_index_global].item()*poly_eval(self.basis[basis_index_local]["f"], point_local)
        
        fig = plt.figure(self.fig_index)
        fig.clear()
        
        plt.plot(x, u)
        plt.grid(True)
    
        plt.pause(1e-5)