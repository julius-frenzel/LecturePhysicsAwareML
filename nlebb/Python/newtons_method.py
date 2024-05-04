import numpy as np
import time



def newton(func, x_0, tol=1e-5, maxiter=100):
    num_vars = len(x_0)
    del_x = np.sqrt(np.finfo(float).eps)
    x_n = x_0
    
    for iter_index in range(maxiter):
        # assemble Jacobian matrix
        J = np.zeros((num_vars,num_vars))
        for test_index in range(num_vars):
            x_m = x_n.copy()
            x_p = x_n.copy()
            x_m[test_index] -= del_x
            x_p[test_index] += del_x
            #print(x_m[test_index+1])
            #print(x_p[test_index+1])
            #time.sleep(100)
            y_m = func(x_m)
            y_p = func(x_p)
            J[:, test_index] = (y_p - y_m)/(2*del_x)
        #print("")
        #for i in J:
        #    for j in i:
        #        print(f"{j:.2f}", end=",  ")
        #    print("")
        
        J_pinv = np.linalg.pinv(J, rcond=1e-10)
            
        #print(f"rank of jacobian matrix: {np.linalg.matrix_rank(J, tol=1e-10)}")
        #print(f"condition of jacobian matrix: {np.linalg.cond(J)}")
        #time.sleep(1000000)
        
        
        # execute iteration
        y = func(x_n)
        #x_n -= np.linalg.solve(J, y)
        x_n -= J_pinv@y #*(1-1e-5)
        
        
        
        # check convergence
        print(f"iteration {iter_index+1}: norm of residual={np.linalg.norm(y):.2f}, rank of jacobian matrix={np.linalg.matrix_rank(J, tol=1e-10)}, condition of jacobian matrix={np.linalg.cond(J):.2f}")
        #time.sleep(1)
        if np.all(abs(y) < tol):
            #print(max(abs(y)))
            print(f"converged after {iter_index + 1} iterations")
            break
    
    return x_n


