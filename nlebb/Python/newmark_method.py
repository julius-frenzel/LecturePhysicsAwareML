import numpy as np
from newtons_method import newton
from scipy.optimize import root
import time
import logging

logger = logging.getLogger(__name__)


def newmark(f, t, x_0, x_d_0, x_dd_0=None):
    logger.info(f"starting Newmark method with t_start={t.min()}, t_end={t.max()}, delta_t={np.diff(t)[0]:.2e}s")
    
    # set initial accelerations to zero, if no other values are given
    if x_dd_0 is None:
        x_dd_0 = np.zeros_like(x_0)
    
    x_n, x_d_n, x_dd_n = x_0, x_d_0, x_dd_0
    len_x = len(x_0)
    
    # parameters for Newmark method
    gamma = 1/2 # + 1e-1 # add small constant for numerical damping
    beta = max(gamma/2, 1/4) # ensure unconditional stability
    
    residual = lambda x_n, x_d_n, x_dd_n, x_np1, x_d_np1, x_dd_np1: np.concatenate([f(x_np1, x_d_np1, x_dd_np1), # equation of motion
                                                                                    1/(beta*del_t**2)*(x_np1 - x_n) - 1/(beta*del_t)*x_d_n - (0.5 - beta)/beta*x_dd_n - x_dd_np1, # equations for Newmark method
                                                                                    gamma/(beta*del_t)*(x_np1 - x_n) + (1 - gamma/beta)*x_d_n + del_t*(beta - 0.5*gamma)/beta*x_dd_n - x_d_np1], axis=0)
    
    results = {"t": [t[0]], "x": [x_n]}
    
    for iter_index in range(len(t) - 1):
        t_start_newmark_iter = time.time()
        #logger.info(f"starting Newmark iteration {iter_index+1}")
        
        del_t = t[iter_index + 1] - t[iter_index]
        # own implementation (slower)
        #state_np1 = newton(lambda state_np1: residual(x_n, x_d_n, x_dd_n, state_np1[:len_x], state_np1[len_x:2*len_x], state_np1[2*len_x:]),
        #                   np.concatenate([x_n, x_d_n, x_dd_n]),
        #                   tol=1e-5,
        #                   maxiter=10)
        # solver from scipy (faster)
        state_np1 = root(lambda state_np1: residual(x_n, x_d_n, x_dd_n, state_np1[:len_x], state_np1[len_x:2*len_x], state_np1[2*len_x:]),
                          np.concatenate([x_n, x_d_n, x_dd_n]),
                          method="hybr",
                          tol=1e-5,
                          options={"maxfev": 1000})["x"]
        x_n, x_d_n, x_dd_n = state_np1[:len_x], state_np1[len_x:2*len_x], state_np1[2*len_x:]
        
        results["t"].append(t[iter_index+1])
        results["x"].append(x_n)
        
        logger.info(f"Newmark iteration {iter_index + 1}: {time.time() - t_start_newmark_iter:.2f}s")
        
    results = {key: np.stack(results[key]) for key in results}
    
    return results
    
    
    
    
    