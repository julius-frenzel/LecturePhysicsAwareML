import numpy as np
from helper import poly_eval, quad, quad_unit
import matplotlib.pyplot as plt
import time
import logging

logger = logging.getLogger(__name__)

# parameters of the beam
params = {"E": 1, #50e6, # Young's modulus in Pa
          "rho": 1, #1100 # density in kg/m^3
          "A": 1, #1e-4, # cross-sectional area in m^2
          "I": 1/12, #(0.02**4)/12 # moment of inertia in m^4
          }

# mass matrix for u
def get_mass_matrix_u(approx, domain):
    M = np.zeros((approx.coeffs_len, approx.coeffs_len))
    replace_row_mask = np.zeros((approx.coeffs_len,), dtype=np.bool_)
    for e in range(domain["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            
            # check, if row should be used for boundary conditions
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] in [2, 3, 4]: # pinned in horizontal direction, pinned in both directions or clamped (all equivalent for u)
                    H_test = approx.basis_coeffs[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large
                    replace_row_mask[test_index_global] |= abs(bc_test_val) > 0
            
            if not replace_row_mask[test_index_global]:
                test = approx.basis_evaluated[test_index_local]["f"]
                for basis_index_local in range(approx.coeffs_per_element): # loop over relevant columns for element e
                    basis_index_global = index_global_left + basis_index_local
                    basis = approx.basis_evaluated[basis_index_local]["f"]
                    # integrate over the element
                    M[test_index_global, basis_index_global] += quad_unit(params["rho"]*params["A"]*basis*test)*element_scale
    
    #logger.debug(f"M={M}") # logging takes time, even if the message isn't actually displayed
    return M


# mass matrix for w
def get_mass_matrix_w(approx, domain):
    M = np.zeros((approx.coeffs_len, approx.coeffs_len))
    replace_row_mask = np.zeros((approx.coeffs_len,), dtype=np.bool_)
    for e in range(domain["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            
            # check, if row should be used for boundary conditions
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] in [1, 3, 4]: # pinned in vertical direction, pinned in both directions or clamped (deflection constrained)
                    H_test = approx.basis_coeffs[test_index_local]["f"]
                    bc_test_val = poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large
                    replace_row_mask[test_index_global] |= abs(bc_test_val) > 0
                    
                    if domain["boundary_conds"][boundary_index] == 4: # clamped (derivative constrained as well)
                        # constrain derivative
                        H_test_x = approx.basis_coeffs[test_index_local]["f_x"]
                        bc_test_val = poly_eval(H_test_x, boundary_pos) # must be either exactly zero or sufficiently large
                        replace_row_mask[test_index_global] |= abs(bc_test_val) > 0
            
            if not replace_row_mask[test_index_global]:
                test = approx.basis_evaluated[test_index_local]["f"]
                for basis_index_local in range(approx.coeffs_per_element): # loop over relevant columns for element e
                    basis_index_global = index_global_left + basis_index_local
                    basis = approx.basis_evaluated[basis_index_local]["f"]
                    # integrate over the element
                    M[test_index_global, basis_index_global] += quad_unit(params["rho"]*params["A"]*basis*test)*element_scale
    
    #logger.debug(f"M={M}") # logging takes time, even if the message isn't actually displayed
    return M


# force vector for u
def get_force_vector_u(coeffs_u, coeffs_w, approx, domain):
    
    f = np.zeros((approx.coeffs_len,))
    replace_row_mask = np.zeros_like(f, dtype=np.bool_)
    for e in range(domain["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            test_index_global = index_global_left + test_index_local
            
            # check, if row should be used for boundary conditions
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                f_temp = 0
                if domain["boundary_conds"][boundary_index] in [2, 3, 4]: # pinned in horizontal direction, pinned in both directions or clamped (all equivalent for u)
                    H_test = approx.basis_coeffs[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis_coeffs[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large
                        if abs(bc_test_val) > 0:
                            f_temp += bc_test_val*coeffs_u[basis_index_global]
                            replace_row_mask[test_index_global] = True
                if replace_row_mask[test_index_global]:
                    f[test_index_global] = f_temp
                        
            if not replace_row_mask[test_index_global]:
                # apply PDE condition
                test_x = approx.basis_evaluated[test_index_local]["f_x"]/element_scale
                u_x = w_x = 0
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    u_x += coeffs_u[basis_index_global]*approx.basis_evaluated[basis_index_local]["f_x"]/element_scale
                    w_x += coeffs_w[basis_index_global]*approx.basis_evaluated[basis_index_local]["f_x"]/element_scale

                f[test_index_global] += quad_unit(params["E"]*params["A"]*(u_x + 0.5*w_x**2)*test_x)*element_scale

    #logger.debug(f"f={f}") # logging takes time, even if the message isn't actually displayed
    return f


# force vector for w
def get_force_vector_w(coeffs_u, coeffs_w, approx, domain):
    
    #print("assembling force vector")
    f = np.zeros((approx.coeffs_len,))
    replace_row_mask = np.zeros_like(f, dtype=np.bool_)
    for e in range(domain["num_elements"]):
        #print(f"e={e}")
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (domain["element_boundaries"][e + 1] - domain["element_boundaries"][e])
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            test_index_global = index_global_left + test_index_local
            
            # check, if row should be used for boundary conditions
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                f_temp = 0
                if domain["boundary_conds"][boundary_index] in [1, 3, 4]: # pinned in vertical direction, pinned in both directions or clamped (deflection constrained)
                    H_test = approx.basis_coeffs[test_index_local]["f"]
                    for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                        basis_index_global = index_global_left + basis_index_local
                        H_basis = approx.basis_coeffs[basis_index_local]["f"]
                        bc_test_val = poly_eval(H_basis, boundary_pos)*poly_eval(H_test, boundary_pos) # must be either exactly zero or sufficiently large
                        if abs(bc_test_val) > 0:
                            f_temp += bc_test_val*coeffs_w[basis_index_global]
                            replace_row_mask[test_index_global] = True
                            
                    if domain["boundary_conds"][boundary_index] == 4: # clamped (derivative constrained as well)
                        # constrain derivative
                        H_test_x = approx.basis_coeffs[test_index_local]["f_x"]
                        for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                            basis_index_global = index_global_left + basis_index_local
                            H_basis_x = approx.basis_coeffs[basis_index_local]["f_x"]
                            bc_test_val = poly_eval(H_basis_x, boundary_pos)*poly_eval(H_test_x, boundary_pos) # must be either exactly zero or sufficiently large
                            if abs(bc_test_val) > 0:
                                f_temp += bc_test_val*coeffs_w[basis_index_global]
                                replace_row_mask[test_index_global] = True
                if replace_row_mask[test_index_global]:
                    f[test_index_global] = f_temp
                        
            if not replace_row_mask[test_index_global]:
                # apply PDE condition
                test_x = approx.basis_evaluated[test_index_local]["f_x"]/element_scale
                test_xx = approx.basis_evaluated[test_index_local]["f_xx"]/element_scale**2
                u_x = w_x = w_xx = 0
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    u_x += coeffs_u[basis_index_global]*approx.basis_evaluated[basis_index_local]["f_x"]/element_scale
                    w_x += coeffs_w[basis_index_global]*approx.basis_evaluated[basis_index_local]["f_x"]/element_scale
                    w_xx += coeffs_w[basis_index_global]*approx.basis_evaluated[basis_index_local]["f_xx"]/element_scale**2

                f[test_index_global] += quad_unit(params["E"]*params["A"]*w_x*(u_x + 0.5*w_x**2)*test_x + params["E"]*params["I"]*w_xx*test_xx)*element_scale
    
    #logger.debug(f"f={f}") # logging takes time, even if the message isn't actually displayed
    return f

# right hand side for u
def get_rhs_u(coeffs_u, coeffs_w, approx, domain, t=0):
    b = np.zeros((approx.coeffs_len,))
    replace_row_mask = np.zeros_like(b, dtype=np.bool_)
    for e in range(domain["num_elements"]):
        #print(f"e={e}")
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        left_boundary = domain["element_boundaries"][e]
        right_boundary = domain["element_boundaries"][e + 1]
        element_scale = right_boundary - left_boundary
        global2local = lambda x_global: (x_global - left_boundary)/element_scale
        local2global = lambda x_local: left_boundary + x_local*element_scale
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            test_index_global = index_global_left + test_index_local
            H_test = approx.basis_coeffs[test_index_local]["f"]
            
            # check, if row should be used for boundary conditions
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] in [2, 3, 4]: # pinned in horizontal direction, pinned in both directions or clamped (all equivalent for u)
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    if abs(bc_test_val) > 0:
                        b[test_index_global] = domain["u"][boundary_index]*bc_test_val
                        replace_row_mask[test_index_global] = True
                    
            if not replace_row_mask[test_index_global]:
                # discrete loads
                load_points = domain["load_points"](t)
                axial_forces = domain["N"](t)
                specific_axial_force = lambda x: domain["f"](t, x)
                for load_index, load_point in enumerate(load_points):
                    if (load_point > left_boundary or boundary_index == 0) and load_point <= right_boundary:
                        b[test_index_global] += axial_forces[load_index]*poly_eval(H_test, global2local(load_point))
                # specific axial force
                b[test_index_global] += quad(lambda point: specific_axial_force(local2global(point))*poly_eval(H_test, point), [0, 1])*element_scale

                
    #logging.debug(f"b={b}") # logging takes time, even if the message isn't actually displayed
    return b


# right hand side for w
def get_rhs_w(coeffs_u, coeffs_w, approx, domain, t=0):
    b = np.zeros((approx.coeffs_len,))
    replace_row_mask = np.zeros_like(b, dtype=np.bool_)
    for e in range(domain["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        left_boundary = domain["element_boundaries"][e]
        right_boundary = domain["element_boundaries"][e + 1]
        element_scale = right_boundary - left_boundary
        global2local = lambda x_global: (x_global - left_boundary)/element_scale
        local2global = lambda x_local: left_boundary + x_local*element_scale
        boundary_index = 0 if e == 0 else 1 if e == domain["num_elements"] - 1 else -1 # shows, whether the element is a boundary element
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            test_index_global = index_global_left + test_index_local
            H_test = approx.basis_coeffs[test_index_local]["f"]
            
            # check, if row should be used for boundary conditions
            if boundary_index > -1:
                boundary_pos = 0. if boundary_index == 0 else 1. # numerically equivalent to boundary index, but represents the local position of the boundary on the element
                if domain["boundary_conds"][boundary_index] in [1, 3, 4]: # pinned in vertical direction, pinned in both directions or clamped (deflection constrained)
                    bc_test_val = poly_eval(H_test, boundary_pos)
                    if abs(bc_test_val) > 0:
                        replace_row_mask[test_index_global] = True
                        b[test_index_global] = domain["w"][boundary_index]*bc_test_val
                        
                    if domain["boundary_conds"][boundary_index] == 4: # clamped (derivative constrained as well)
                        # constrain derivative
                        bc_test_val = poly_eval(approx.basis_coeffs[test_index_local]["f_x"], boundary_pos)
                        if abs(bc_test_val) > 0:
                            b[test_index_global] = domain["w_x"][boundary_index]*bc_test_val
                            replace_row_mask[test_index_global] = True
            
            if not replace_row_mask[test_index_global]:
                # discrete loads
                load_points = domain["load_points"](t)
                axial_forces = domain["N"](t)
                lateral_forces = domain["Q"](t)
                moments = domain["M"](t)
                specific_lateral_force = lambda x: domain["q"](t, x)
                for load_index, load_point in enumerate(load_points):
                    if (load_point > left_boundary or boundary_index == 0) and load_point <= right_boundary:
                        # normal boundary terms
                        H_test_x = approx.basis_coeffs[test_index_local]["f_x"]/element_scale
                        b[test_index_global] += lateral_forces[load_index]*poly_eval(H_test, global2local(load_point))
                        b[test_index_global] -= moments[load_index]*poly_eval(H_test_x, global2local(load_point))
                        # mysterious boundary term (has significant influence, if there are large lateral displacements and discrete axial loads)
                        # probably best to neglect this term, since there is no counterpart in the rhs for u
                        continue # neglect mysterious boundary term
                        w_x = 0
                        for basis_index_local in range(approx.coeffs_per_element):
                            basis_index_global = index_global_left + basis_index_local
                            H_basis_x = approx.basis_coeffs[basis_index_local]["f_x"]/element_scale
                            w_x += coeffs_w[basis_index_global]*poly_eval(H_basis_x, global2local(load_point))
                        b[test_index_global] += w_x*axial_forces[load_index]*poly_eval(H_test, global2local(load_point))
                # specific lateral force
                b[test_index_global] += quad(lambda point: specific_lateral_force(local2global(point))*poly_eval(H_test, point), [0, 1])*element_scale
                
    # logger.debug(f"b={b}") # logging takes time, even if the message isn't actually displayed
    return b














    
    
    
    