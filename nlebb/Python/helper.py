import numpy as np


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


# cached parameters for Gauss-Legendre quadrature
cached_points = cached_weights = None
cached_degree = None
cached_bounds = None

# compute the integral of a given integrant over given bounds using Gauss-Legendre quadrature
def quad(integrant, bounds, degree=3):
    # computing the points and weights for Gauss-Legendre quadrature takes too long to do every time the function is called, therefore they are cached
    global cached_points, cached_weights, cached_degree, cached_bounds
    if not degree == cached_degree:
        cached_points, cached_weights = np.polynomial.legendre.leggauss(deg=degree)
    if (not degree == cached_degree) or (not bounds == cached_bounds):
        cached_degree = degree
        cached_bounds = bounds
        # scale points and weights
        cached_points = bounds[0] + 0.5*(cached_points + 1)*(bounds[1] - bounds[0])
        cached_weights = 0.5*cached_weights*(bounds[1] - bounds[0])
        
    # compute approximation of the integral
    int_val = 0
    for point, weight in zip(cached_points, cached_weights):
        int_val += weight*integrant(point)
    return int_val