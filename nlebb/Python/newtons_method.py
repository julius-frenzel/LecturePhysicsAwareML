import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

def newton(func, x_0, tol=1e-5, maxiter=100):
    logger.info(f"starting Newton's method with tol={tol}, maxiter={maxiter}")
    
    num_vars = len(x_0)
    del_x = np.sqrt(np.finfo(x_0.dtype).eps)
    x_n = x_0.copy() # copying x_0 this is critical, otherwise data can be altered unexpectedly in the outer scope
    y_n = func(x_n)
    
    for iter_index in range(maxiter):
        
        # assemble Jacobian matrix
        J = np.zeros((num_vars,num_vars))
        for der_index in range(num_vars):
            x_m = x_n.copy()
            x_p = x_n.copy()
            x_m[der_index] -= del_x
            x_p[der_index] += del_x
            y_m = func(x_m)
            y_p = func(x_p)
            J[:,der_index] = (y_p - y_m)/(2*del_x)
        
        # execute iteration using pseudoinverse
        J_pinv = np.linalg.pinv(J, rcond=1e-10)
        x_n -= J_pinv@y_n
        #x_n -= np.linalg.solve(J, y_n)
        y_n = func(x_n)
        
        
        
        # check convergence
        #print(f"iteration {iter_index+1}: norm of residual={np.linalg.norm(y_n):.2f}, rank of jacobian matrix={np.linalg.matrix_rank(J, tol=1e-10)}, condition of jacobian matrix={np.linalg.cond(J):.2f}") # computing the condition is expensive, so shouldn't be done carelessly
        if np.all(abs(y_n) < tol):
            logger.info(f"Newton's method converged after {iter_index + 1} iteration(s)")
            break
    
    return x_n


