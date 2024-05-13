import numpy as np
from helper import poly_eval, quad
import matplotlib.pyplot as plt
#from scipy.integrate import quad as quad_scipy
import time
import logging

logger = logging.getLogger(__name__)

# parameters of the beam
params = {"E": 1, #210e9, # Young's modulus in Pa
          "rho": 1, # density in kg/m^3
          "A": 1, #1e-4, # cross-sectional area in m^2
          "I": 1, # moment of inertia in m^4
          }


# mass matrix for u
def get_mass_matrix_u(approx, domain):
    # element mass matrix
    M = np.zeros((approx.coeffs_len,)*2)
    for e in range(domain["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            
            replace_row = False
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] == 0: # free
                    pass
                elif domain["boundary_conds"][boundary_index] == 1: # pinned in vertical direction (not relevant for u)
                    pass
                elif domain["boundary_conds"][boundary_index] == 2: # pinned in horizontal direction
                    H_test = approx.basis[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 3: # pinned in both directions
                    H_test = approx.basis[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 4: # clamped (equivalent to pinned in horizontal direction for u)
                    H_test = approx.basis[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
            
            if not replace_row:
                H_test = approx.basis[test_index_local]["f"]
                for basis_index_local in range(approx.coeffs_per_element): # loop over relevant columns for element e
                    basis_index_global = index_global_left + basis_index_local
                    H_basis = approx.basis[basis_index_local]["f"]
                    # integrate over the element
                    M[test_index_global, basis_index_global] += quad(lambda point:
                                                                     params["rho"]*params["A"]*poly_eval(H_basis, point)*poly_eval(H_test, point),
                                                                     [0, 1])*element_scale
    
    logger.debug(f"M={M}")
    return M


# mass matrix for w
def get_mass_matrix_w(approx, domain):
    # element mass matrix
    M = np.zeros((approx.coeffs_len, approx.coeffs_len))
    for e in range(domain["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            
            replace_row = False
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] == 0: # free
                    pass
                elif domain["boundary_conds"][boundary_index] == 1: # pinned in vertical direction
                    H_test = approx.basis[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 2: # pinned in horizontal direction (not relevant for w)
                    pass
                elif domain["boundary_conds"][boundary_index] == 3: # pinned in both directions
                    H_test = approx.basis[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 4: # clamped
                    H_test = approx.basis[test_index_local]["f"]
                    # constrain displacement
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
                    # constrain derivative
                    H_test_x = approx.basis[test_index_local]["f_x"]
                    bc_test_val = poly_eval(H_test_x, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                    replace_row |= abs(bc_test_val) > 0
            
            if not replace_row:
                H_test = approx.basis[test_index_local]["f"]
                for basis_index_local in range(approx.coeffs_per_element): # loop over relevant columns for element e
                    basis_index_global = index_global_left + basis_index_local
                    H_basis = approx.basis[basis_index_local]["f"]
                    # integrate over the element
                    M[test_index_global, basis_index_global] += quad(lambda point:
                                                                     params["rho"]*params["A"]*poly_eval(H_basis, point)*poly_eval(H_test, point),
                                                                     [0, 1])*element_scale
    
    logger.debug(f"M={M}")
    return M


# force vector for u
def get_force_vector_u(coeffs_u, coeffs_w, approx, domain):
    #print("assembling force vector")
    f = np.zeros((approx.coeffs_len,))
    for e in range(domain["num_elements"]):
        #print(f"e={e}")
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            #print(f"row={test_index_local}")
            test_index_global = index_global_left + test_index_local
            
            # check, if row should be used for left pinned or clamped boundary condition
            replace_row = False
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] == 0: # free
                    pass
                elif domain["boundary_conds"][boundary_index] == 1: # pinned in vertical direction (not relevant for u)
                    pass
                elif domain["boundary_conds"][boundary_index] == 2: # pinned in horizontal direction
                    H_test = approx.basis[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_u[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 3: # pinned in both directions
                    H_test = approx.basis[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_u[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 4: # clamped (equivalent to pinned in horizontal direction for u)
                    H_test = approx.basis[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_u[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                        
            # check, if PDE condition should be applied in this row
            if not replace_row:
                H_test_x = approx.basis[test_index_local]["f_x"]/element_scale
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_basis_u_x = coeffs_u[basis_index_global]*approx.basis[basis_index_local]["f_x"]/element_scale
                    H_basis_w_x = coeffs_w[basis_index_global]*approx.basis[basis_index_local]["f_x"]/element_scale
                    # integrate over the element
                    f[test_index_global] += quad(lambda point:
                                                 params["E"]*params["A"]*(poly_eval(H_basis_u_x, point) + 0*0.5*poly_eval(H_basis_w_x, point)**2)*poly_eval(H_test_x, point),
                                                 [0, 1])*element_scale
                #print("normal or free")

    logger.debug(f"f={f}")
    return f


# force vector for w
def get_force_vector_w(coeffs_u, coeffs_w, approx, domain):
    #print("assembling force vector")
    f = np.zeros((approx.coeffs_len,))
    for e in range(domain["num_elements"]):
        #print(f"e={e}")
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            #print(f"row={test_index_local}")
            test_index_global = index_global_left + test_index_local
            #print(f"test_index_local: {test_index_local}")
            # check, if row should be used for left Dirichlet boundary condition
            replace_row = False
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] == 0: # free
                    pass
                elif domain["boundary_conds"][boundary_index] == 1: # pinned in vertical direction
                    H_test = approx.basis[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 2: # pinned in horizontal direction (not relevant for w)
                    pass
                elif domain["boundary_conds"][boundary_index] == 3: # pinned in both directions
                    H_test = approx.basis[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 4: # clamped (equivalent to pinned in horizontal direction for u)
                    # constrain displacement
                    H_test = approx.basis[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                    # constrain derivative
                    H_test_x = approx.basis[test_index_local]["f_x"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis_x = approx.basis[basis_index_local]["f_x"]
                        bc_test_val = poly_eval(H_basis_x, boundary_pos)*poly_eval(H_test_x, boundary_pos) # must be either exactly zero or sufficiently large (is this acceptable?)
                        f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                        replace_row |= abs(bc_test_val) > 0
                        
            # check, if PDE condition should be applied in this row
            if not replace_row:
                H_test_x = approx.basis[test_index_local]["f_x"]/element_scale
                H_test_xx = approx.basis[test_index_local]["f_xx"]/element_scale**2
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_basis_u_x = coeffs_u[basis_index_global]*approx.basis[basis_index_local]["f_x"]/element_scale
                    H_basis_w_x = coeffs_w[basis_index_global]*approx.basis[basis_index_local]["f_x"]/element_scale
                    H_basis_w_xx = coeffs_w[basis_index_global]*approx.basis[basis_index_local]["f_xx"]/element_scale**2
                    # integrate over the element
                    f[test_index_global] += quad(lambda point:
                                                 0*params["E"]*params["A"]*poly_eval(H_basis_w_x, point)*(poly_eval(H_basis_u_x, point) + 0.5*poly_eval(H_basis_w_x, point)**2)*poly_eval(H_test_x, point) + params["E"]*params["I"]*poly_eval(H_basis_w_xx, point)*poly_eval(H_test_xx, point),
                                                 [0, 1])*element_scale
                #print("normal or free")
    
    #print(f"coeffs after: {coeffs_w}")
    #time.sleep(1)
    
    logger.debug(f"f={f}")
    return f


# right hand side for u
def get_rhs_u(approx, domain, t=0):
    #print("assembling rhs")
    b = np.zeros((approx.coeffs_len,))
    for e in range(domain["num_elements"]):
        #print(f"e={e}")
        coeffs_per_element = len(approx.basis)
        index_global_left = e*int(coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        left_boundary = domain["element_boundaries"][e]
        right_boundary = domain["element_boundaries"][e + 1]
        x_local = lambda x_global: (x_global - left_boundary)/(right_boundary - left_boundary)
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            basis_index_global = index_global_left + test_index_local
            H_test = approx.basis[test_index_local]["f"]
            # evaluating the Dirichlet boundary condition for all test functions on the boundary elements is fine,
            # because the result will be zero, except for the relevant test function
            # an element can be left and right element, if there is only a single element
            replace_row = False
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] == 0: # free
                    pass
                elif domain["boundary_conds"][boundary_index] == 1: # pinned in vertical direction (not relevant for u)
                    pass
                elif domain["boundary_conds"][boundary_index] == 2: # pinned in horizontal direction
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    b[basis_index_global] = domain["u"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 3: # pinned in both directions
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    b[basis_index_global] = domain["u"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
                    #if abs(bc_test_val) > 0:
                    #    print(f"element {e+1}, pinned, boundary_index={boundary_index}")
                elif domain["boundary_conds"][boundary_index] == 4: # clamped (equivalent to pinned in horizontal direction for u)
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    b[basis_index_global] = domain["u"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
                    
            # inner element
            if not replace_row:
                # apply discrete loads
                load_points = domain["load_points"](t)
                axial_forces = domain["N"](t)
                specific_axial_force = lambda x: domain["f"](t, x)
                for load_index, load_point in enumerate(load_points):
                    if (load_point > left_boundary or boundary_index == 0) and load_point <= right_boundary:
                        b[basis_index_global] += axial_forces[load_index]*poly_eval(H_test, x_local(load_point))
                # apply specific axial force
                element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
                point_global = lambda point_local: domain["element_boundaries"][e] + point_local*element_scale
                b[basis_index_global] += quad(lambda point: specific_axial_force(point_global(point))*poly_eval(H_test, point), [0, 1])*element_scale
                #print("normal or free")
                
    #print(f"b={b}")
    return b


# right hand side for w
def get_rhs_w(approx, domain, t=0):
    #print("assembling rhs")
    b = np.zeros((approx.coeffs_len,))
    for e in range(domain["num_elements"]):
        #print(f"e={e}")
        coeffs_per_element = len(approx.basis)
        index_global_left = e*int(coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        left_boundary = domain["element_boundaries"][e]
        right_boundary = domain["element_boundaries"][e + 1]
        element_scale = right_boundary - left_boundary
        x_local = lambda x_global: (x_global - left_boundary)/(right_boundary - left_boundary)
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            basis_index_global = index_global_left + test_index_local
            H_test = approx.basis[test_index_local]["f"]
            # evaluating the Dirichlet boundary condition for all test functions on the boundary elements is fine,
            # because the result will be zero, except for the relevant test function
            # an element can be left and right element, if there is only a single element
            replace_row = False
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] == 0: # free
                    pass
                elif domain["boundary_conds"][boundary_index] == 1: # pinned in vertical direction
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    b[basis_index_global] = domain["w"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 2: # pinned in horizontal direction (not relevant for w)
                    pass
                elif domain["boundary_conds"][boundary_index] == 3: # pinned in both directions
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    b[basis_index_global] = domain["w"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
                elif domain["boundary_conds"][boundary_index] == 4: # clamped
                    # constrain displacement
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    b[basis_index_global] = domain["w"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
                    # constrain derivative
                    bc_test_val = poly_eval(approx.basis[test_index_local]["f_x"], boundary_pos)
                    b[basis_index_global] = domain["w_x"][0]*bc_test_val
                    replace_row |= abs(bc_test_val) > 0
            
            # inner element
            if not replace_row:
                # apply discrete loads
                load_points = domain["load_points"](t)
                lateral_forces = domain["Q"](t)
                #print(f"lateral forces: {lateral_forces}")
                moments = domain["M"](t)
                specific_lateral_force = lambda x: domain["q"](t, x)
                for load_index, load_point in enumerate(load_points):
                    if (load_point > left_boundary or boundary_index == 0) and load_point <= right_boundary:
                        b[basis_index_global] += lateral_forces[load_index]*poly_eval(H_test, x_local(load_point))
                        H_test_x = approx.basis[test_index_local]["f_x"]
                        b[basis_index_global] -= moments[load_index]*poly_eval(H_test_x, x_local(load_point))/element_scale
                # apply specific lateral force
                element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
                point_global = lambda point_local: domain["element_boundaries"][e] + point_local*element_scale
                b[basis_index_global] += quad(lambda point: specific_lateral_force(point_global(point))*poly_eval(H_test, point), [0, 1])*element_scale
                #print("normal or free")
                
    logger.debug(f"b={b}")
    return b














    
    
    
    