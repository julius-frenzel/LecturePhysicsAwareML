import numpy as np
from newtons_method import newton
from scipy.optimize import root
import time
import logging

logger = logging.getLogger(__name__)


def newmark(f, t, x_0, x_d_0, x_dd_0=None):
    logger.info(f"starting Newmark method with t_start={t.min()}, t_end={t.max()}, delta_t={np.diff(t)[0]:.2e}s")
    
    # Set initial accelerations to zero, if no other values are given
    if x_dd_0 is None:
        x_dd_0 = np.zeros_like(x_0)
    
    x_n, x_d_n, x_dd_n = x_0, x_d_0, x_dd_0
    
    # Parameters for Newmark method
    gamma = 1/2 # + 1e-1 # add small constant for numerical damping
    beta = max(gamma/2, 1/4) # ensure unconditional stability
    
    results = {"t": [t[0]], "x": [x_n], "x_d": [x_d_n], "x_dd": [x_dd_n]}
    
    for iter_index in range(len(t) - 1):
        t_start_newmark_iter = time.time()
        #logger.info(f"starting Newmark iteration {iter_index+1}")
        
        t_np1 = t[iter_index + 1]
        del_t = t_np1 - t[iter_index]
        
        # Lambda function for derivatives in time step n+1
        x_d_np1 = lambda x_np1: (1 - gamma/beta) * x_d_n + del_t * (1 - gamma/(2*beta)) * x_dd_n + gamma/(beta*del_t) * (x_np1 - x_n)
        x_dd_np1 = lambda x_np1: 1/(beta*del_t**2) * (x_np1 - x_n - del_t * x_d_n) + (1 - 1/(2*beta)) * x_dd_n
        
        # own implementation (slower)
        #state_np1 = newton(lambda state_np1: residual(x_n, x_d_n, x_dd_n, state_np1[:len_x], state_np1[len_x:2*len_x], state_np1[2*len_x:], t_np1),
        #                   x_0=np.concatenate([x_n, x_d_n, x_dd_n]),
        #                   tol=1e-5,
        #                   maxiter=10)
        # Solver from scipy (faster)
        x_np1 = root(lambda x_np1: f(x_np1, x_d_np1(x_np1), x_dd_np1(x_np1), t_np1),
                          x0=x_n,
                          method="hybr",
                          tol=1e-5,
                          options={"maxfev": 1000})["x"]
        
        # Update variables
        x_n, x_d_n, x_dd_n = (x_np1, x_d_np1(x_np1), x_dd_np1(x_np1))
        
        results["t"].append(t[iter_index+1])
        results["x"].append(x_n)
        results["x_d"].append(x_d_n)
        results["x_dd"].append(x_dd_n)
        
        logger.info(f"time for Newmark iteration {iter_index + 1}: {time.time() - t_start_newmark_iter:.2f}s")
        
    results = {key: np.stack(results[key]) for key in results}
    
    return results
    
    
    
    
    