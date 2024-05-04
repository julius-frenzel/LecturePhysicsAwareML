import numpy as np
import time
from scipy.integrate import ode
import matplotlib.pyplot as plt


# define basis and test functions (Galerkin method)
H = np.array([[1, 0, -3, 2],
              [0, 1, -2, 1],
              [0, 0, 3, -2],
              [0, 0, -1, 1]])

# evaluate polynomial at a given point
def poly_eval(coeffs, point):
    #print(coeffs)
    val = 0
    mon_val = 1 # value of monomial
    for coeff in coeffs:
        val += coeff*mon_val
        mon_val *= point
    return val

# compute the coefficients for the derivation of a polynomial
def poly_der(coeffs, order=1):
    coeffs_der = coeffs.copy()
    for _ in range(order):
        if len(coeffs_der) > 1:
            coeffs_der = np.array([coeff*(i+1) for i, coeff in enumerate(coeffs_der[1:])])
        else:
            coeffs_der = np.zeros((1,))
    return coeffs_der

# compute the integral over a given integrant
def quad(integrant, bounds, num_points=5):
    points, weights = np.polynomial.legendre.leggauss(deg=num_points) # points and weights for Gauss-Legendre quadrature
    # scale points and weights
    points = bounds[0] + 0.5*(points + 1)*(bounds[1] - bounds[0])
    weights = 0.5*weights*(bounds[1] - bounds[0])
    # compute approximation of the integral
    approx = 0
    for point, weight in zip(points, weights):
        #print(f"{point}: {weight*integrant(point)}")
        approx += weight*integrant(point)
    return approx

#for coeffs in H:
#    print(poly_eval(poly_der(coeffs), 1))

#for coeffs in H:
#    print(quad(lambda point: poly_eval(coeffs, point), [0, 1]))

l = 1 # length of the beam
boundary_vals = [0, 1] # values for the boundary conditions
boundary_conds = [0, 0] # what kind of boundary conditions to use at the ends of the beam (0: Dirichlet, 1: Neumann, 2: mixed)
num_elements = 1
num_coeffs = len(H[0]) # number of coefficients per element
element_boundaries = np.linspace(0, l, num_elements+1)

e = 0 # element index
# element mass matrix
M = np.zeros((num_coeffs,)*2)
for i in range(num_coeffs): # loop over rows of matrix
    H_test = H[i]
    for j in range(num_coeffs): # loop over columns of matrix
        if e == 0 and i == 0 and boundary_conds[0] == 0:
            pass
        elif e == num_elements - 1 and i == num_coeffs - 1 and boundary_conds[1] == 0:
            pass
        else:
            H_basis = H[j]
            #print(H_basis_dd)
            #time.sleep(100)
            print(f"{i}, {j}")
            # integrate over the element using test function i and basis function j
            M[i, j] = quad(lambda point: poly_eval(H_basis, point)*poly_eval(H_test, point), [0, 1])
            M[i, j] *= (element_boundaries[e + 1] - element_boundaries[e])

print(f"M={M}")

# element stiffness matrix
K = np.zeros((num_coeffs,)*2)
is_left_dirichlet_element = e == 0 and boundary_conds[0] == 0
is_right_dirichlet_element = e == num_elements - 1 and boundary_conds[1] == 0
for i in range(num_coeffs): # loop over rows of matrix (test functions)
    # check, if row should be used for left Dirichlet boundary condition
    if is_left_dirichlet_element:
        for j in range(num_coeffs): # loop over columns of matrix (basis functions)
            H_test = H[i]
            H_basis = H[j]
            K[i, j] = poly_eval(H_basis, 0)*poly_eval(H_test, 0)
    # check, if row should be used for right Dirichlet boundary condition
    # the sum is needed in case there is only a single elements
    if is_right_dirichlet_element and K[i, :].sum() < 1e-10:
        for j in range(num_coeffs): # loop over columns of matrix (basis functions)
            H_test = H[i]
            H_basis = H[j]
            K[i, j] = poly_eval(H_basis, 1)*poly_eval(H_test, 1)
    # check, if PDE condotion should be applied in this row
    # could done more elegantly, but this way should be efficient, because the sum is only actually evaluated for the boundary elements
    if not (is_left_dirichlet_element or is_right_dirichlet_element) or K[i, :].sum() < 1e-10:
        for j in range(num_coeffs): # loop over columns of matrix (basis functions)
            H_test_d = poly_der(H[i], 1)
            H_basis_d = poly_der(H[j], 1)
            # integrate over the element using test function i and basis function j
            K[i, j] = quad(lambda point: poly_eval(H_basis_d, point)*poly_eval(H_test_d, point), [0, 1])
            K[i, j] *= (element_boundaries[e + 1] - element_boundaries[e])

print(f"K={K}")

# right hand side
b = np.zeros((num_coeffs,1))
for i in range(num_coeffs): # loop over test functions
    if e == 0: # left element
        # neumann boundary condition
        if boundary_conds[0] == 0:
            b[i] = boundary_vals[0]*poly_eval(H[i], 0)
        elif boundary_conds[0] == 1:
            b[i] = boundary_vals[0]*poly_eval(H[i], 0)
    if e == num_elements - 1: # right element
        # neumann boundary condition
        if boundary_conds[1] == 0:
            b[i] += boundary_vals[1]*poly_eval(H[i], 1)
        elif boundary_conds[1] == 1:
            b[i] = boundary_vals[1]*poly_eval(H[i], 1)
    if e > 0 and e < num_elements - 1: # inner element
        pass

print(f"b={b}")
    

# force vector from coefficients
f = lambda x: K@x

coeffs = np.linalg.solve(K, b)
print(coeffs)

num_eval_points = 1000
x = np.linspace(0, l, num_eval_points)
u = np.zeros_like(x)
for i in range(num_eval_points):
    for coeff_index, coeff in enumerate(coeffs):
        u[i] += coeff.item()*poly_eval(H[coeff_index], x[i])

plt.plot(x, u)
plt.grid(True)

#A = np.block([[np.zeros_like(M), np.eye(M.shape[0])],
#              [-np.linalg.solve(M, K), np.zeros_like(M)]])

#eigvals, eigvecs = np.linalg.eig(A)
#print(eigvals)



#plt.scatter(np.real(eigvals), np.imag(eigvals))
#plt.xlim([-1,1])
#plt.grid(True)
plt.show()





