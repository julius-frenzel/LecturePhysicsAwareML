import numpy as np
import time



# evaluate polynomial at a given point using Horner's method
def poly_eval(coeffs, point):
    result = coeffs[-1]
    for i in range(len(coeffs)-2,-1,-1):
        result = result * point + coeffs[i]
    return result


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
cached_points_untransformed = cached_points = cached_weights_untransformed = cached_weights = cached_degree = cached_bounds = None

# compute the integral of a given integrant over given bounds using Gauss-Legendre quadrature
def quad(integrant, bounds, degree=5):
    # computing the points and weights for Gauss-Legendre quadrature takes too long to do every time the function is called, therefore they are cached
    global cached_points_untransformed, cached_weights_untransformed, cached_points, cached_weights, cached_degree, cached_bounds
    if not degree == cached_degree:
        cached_points_untransformed, cached_weights_untransformed = np.polynomial.legendre.leggauss(deg=degree)
        cached_degree = degree
    if (not degree == cached_degree) or (not bounds == cached_bounds):
        cached_bounds = bounds
        # scale points and weights
        cached_points = bounds[0] + 0.5*(cached_points_untransformed + 1)*(bounds[1] - bounds[0])
        cached_weights = 0.5*cached_weights_untransformed*(bounds[1] - bounds[0])
        
    # compute approximation of the integral
    result = 0
    for point, weight in zip(cached_points, cached_weights):
        result += weight*integrant(point)

    return result