import numpy as np
from helper import poly_eval, quad
#from scipy.integrate import quad as quad_scipy
import time
import logging

logger = logging.getLogger(__name__)

# parameters of the beam
params = {"E": 1, #210e9, # Young's modulus
          "rho": 1, # density
          "A": 1, #1e-4, # cross-sectional area
          "I": 1, # moment of inertia
          }


# mass matrix for u
def get_mass_matrix_u(approx, mesh):
    # element mass matrix
    M = np.zeros((approx.coeffs_len,)*2)
    for e in range(mesh["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
        # flags for boundary elements
        is_left_pinned = e == 0 and mesh["boundary_conds"][0] == 1
        is_right_pinned = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 1
        is_left_clamped = e == 0 and mesh["boundary_conds"][0] == 2
        is_right_clamped = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 2
        #print(f"e_{e}: {is_pinned or is_clamped}")
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            H_test = approx.basis[test_index_local]["f"]
            
            replace_row = False
            if is_left_pinned or is_left_clamped:
                bc_test_val = poly_eval(H_test, 0) # must be either exactly zero or sufficiently large (is this acceptable?)
                replace_row |= abs(bc_test_val) > 0
            
            if is_right_pinned or is_right_clamped:
                bc_test_val = poly_eval(H_test, 1) # must be either exactly zero or sufficiently large (is this acceptable?)
                replace_row |= abs(bc_test_val) > 0
            
            if not replace_row:
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
def get_mass_matrix_w(approx, mesh):
    # element mass matrix
    M = np.zeros((approx.coeffs_len, approx.coeffs_len))
    for e in range(mesh["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
        # flags for boundary elements
        is_left_pinned = e == 0 and mesh["boundary_conds"][0] == 1
        is_right_pinned = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 1
        is_left_clamped = e == 0 and mesh["boundary_conds"][0] == 2
        is_right_clamped = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 2
        #print(f"e_{e}: {is_pinned or is_clamped}")
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            H_test = approx.basis[test_index_local]["f"]
            
            replace_row = False
            if is_left_pinned or is_left_clamped:
                bc_test_val = poly_eval(H_test, 0) # must be either exactly zero or sufficiently large (is this acceptable?)
                replace_row |= abs(bc_test_val) > 0
            if is_left_clamped:
                H_test_x = approx.basis[test_index_local]["f_x"]
                bc_test_val = poly_eval(H_test_x, 0) # must be either exactly zero or sufficiently large (is this acceptable?)
                replace_row |= abs(bc_test_val) > 0
            
            if is_right_pinned or is_right_clamped:
                bc_test_val = poly_eval(H_test, 1) # must be either exactly zero or sufficiently large (is this acceptable?)
                replace_row |= abs(bc_test_val) > 0
            if is_right_clamped:
                H_test_x = approx.basis[test_index_local]["f_x"]
                bc_test_val = poly_eval(H_test_x, 1) # must be either exactly zero or sufficiently large (is this acceptable?)
                replace_row |= abs(bc_test_val) > 0
            
            if not replace_row:
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
def get_force_vector_u(coeffs_u, coeffs_w, approx, mesh):
    #print("assembling force vector")
    f = np.zeros((approx.coeffs_len,))
    for e in range(mesh["num_elements"]):
        #print(f"e={e}")
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
        # flags for boundary elements
        is_left_pinned = e == 0 and mesh["boundary_conds"][0] == 1
        is_right_pinned = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 1
        is_left_clamped = e == 0 and mesh["boundary_conds"][0] == 2
        is_right_clamped = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 2
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            #print(f"row={test_index_local}")
            test_index_global = index_global_left + test_index_local
            H_test = approx.basis[test_index_local]["f"]
            H_test_x = approx.basis[test_index_local]["f_x"]/element_scale
            
            # check, if row should be used for left pinned or clamped boundary condition
            replace_row = False
            if is_left_pinned or is_left_clamped:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                    basis_index_global = index_global_left + basis_index_local
                    H_basis = approx.basis[basis_index_local]["f"]
                    bc_test_val = poly_eval(H_basis, 0)*poly_eval(H_test, 0) # must be either exactly zero or sufficiently large (is this acceptable?)
                    f[test_index_global] += bc_test_val*coeffs_u[basis_index_global]
                    replace_row |= abs(bc_test_val) > 0
                    #if abs(bc_test_val) > 0:
                    #    print("left pinned or clamped (u)")

            # check, if row should be used for right pinned or clamped boundary condition
            if is_right_pinned or is_right_clamped:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_basis = approx.basis[basis_index_local]["f"]
                    bc_test_val = poly_eval(H_basis, 1)*poly_eval(H_test, 1) # must be either exactly zero or sufficiently large (is this acceptable?)
                    f[test_index_global] += bc_test_val*coeffs_u[basis_index_global]
                    replace_row |= abs(bc_test_val) > 0
                        
            # check, if PDE condition should be applied in this row
            if not replace_row:
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
def get_force_vector_w(coeffs_u, coeffs_w, approx, mesh):
    #print("assembling force vector")
    f = np.zeros((approx.coeffs_len,))
    for e in range(mesh["num_elements"]):
        #print(f"e={e}")
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
        # flags for boundary elements (free boundary conditions require no changes here)
        is_left_pinned = e == 0 and mesh["boundary_conds"][0] == 1
        is_right_pinned = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 1
        is_left_clamped = e == 0 and mesh["boundary_conds"][0] == 2
        is_right_clamped = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 2
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            #print(f"row={test_index_local}")
            test_index_global = index_global_left + test_index_local
            #print(f"test_index_local: {test_index_local}")
            # check, if row should be used for left Dirichlet boundary condition
            replace_row = False
            if is_left_pinned or is_left_clamped:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                    basis_index_global = index_global_left + basis_index_local
                    H_test = approx.basis[test_index_local]["f"]
                    H_basis = approx.basis[basis_index_local]["f"]
                    bc_test_val = poly_eval(H_basis, 0)*poly_eval(H_test, 0) # must be either exactly zero or sufficiently large (is this acceptable?)
                    f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                    replace_row |= abs(bc_test_val) > 0
                    #if abs(bc_test_val) > 0:
                    #    print("left pinned or clamped (w)")
            if is_left_clamped:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                    basis_index_global = index_global_left + basis_index_local
                    H_test_x = approx.basis[test_index_local]["f_x"]
                    H_basis_x = approx.basis[basis_index_local]["f_x"]
                    bc_test_val = poly_eval(H_basis_x, 0)*poly_eval(H_test_x, 0) # must be either exactly zero or sufficiently large (is this acceptable?)
                    f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                    replace_row |= abs(bc_test_val) > 0
                    #if abs(bc_test_val) > 0:
                    #    print("left clamped (w_x)")

            # check, if row should be used for right Dirichlet boundary condition
            if is_right_pinned or is_right_clamped:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_test = approx.basis[test_index_local]["f"]
                    H_basis = approx.basis[basis_index_local]["f"]
                    bc_test_val = poly_eval(H_basis, 1)*poly_eval(H_test, 1) # must be either exactly zero or sufficiently large (is this acceptable?)
                    f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                    replace_row |= abs(bc_test_val) > 0
                    #if abs(bc_test_val) > 0:
                    #    print("right pinned or clamped (w)")
            if is_right_clamped:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_test_x = approx.basis[test_index_local]["f_x"]
                    H_basis_x = approx.basis[basis_index_local]["f_x"]
                    bc_test_val = poly_eval(H_basis_x, 1)*poly_eval(H_test_x, 1) # must be either exactly zero or sufficiently large (is this acceptable?)
                    f[test_index_global] += bc_test_val*coeffs_w[basis_index_global]
                    replace_row |= abs(bc_test_val) > 0
                    #if abs(bc_test_val) > 0:
                    #    print("right clamped (w_x)")
                        
            # check, if PDE condition should be applied in this row
            if not replace_row:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_test_x = approx.basis[test_index_local]["f_x"]/element_scale
                    H_test_xx = approx.basis[test_index_local]["f_xx"]/element_scale**2
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
def get_rhs_u(approx, mesh):
    #print("assembling rhs")
    b = np.zeros((approx.coeffs_len,))
    for e in range(mesh["num_elements"]):
        #print(f"e={e}")
        coeffs_per_element = len(approx.basis)
        index_global_left = e*int(coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        left_boundary = mesh["element_boundaries"][e]
        right_boundary = mesh["element_boundaries"][e + 1]
        x_local = lambda x_global: (x_global - left_boundary)/(right_boundary - left_boundary)
        # flags for boundary elements
        is_left = e == 0
        is_right = e == mesh["num_elements"] - 1
        is_left_pinned = is_left and mesh["boundary_conds"][0] == 1
        is_right_pinned = is_right and mesh["boundary_conds"][1] == 1
        is_left_clamped = is_left and mesh["boundary_conds"][0] == 2
        is_right_clamped = is_right and mesh["boundary_conds"][1] == 2
        for test_index_local in range(coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            basis_index_global = index_global_left + test_index_local
            # evaluating the Dirichlet boundary condition for all test functions on the boundary elements is fine,
            # because the result will be zero, except for the relevant test function
            # an element can be left and right element, if there is only a single element
            replace_row = False

            # left element
            if is_left_pinned or is_left_clamped: # pinned and clamped is the same for axial displacement
                # pinned or clamped boundary condition (same for axial displacement)
                bc_test_val = poly_eval(approx.basis[test_index_local]["f"], 0)
                b[basis_index_global] = mesh["u"][0]*bc_test_val
                replace_row |= abs(bc_test_val) > 0
                #if abs(bc_test_val) > 0:
                #    print("left pinned or clamped (u)")
                
            # right element
            if is_right_pinned or is_right_clamped:
                # pinned or clamped boundary condition (same for axial displacement)
                bc_test_val = poly_eval(approx.basis[test_index_local]["f"], 1)
                b[basis_index_global] = mesh["u"][1]*bc_test_val
                replace_row |= abs(bc_test_val) > 0
                    
            # inner element
            if not replace_row:
                # apply discrete loads
                for load_index, load_point in enumerate(mesh["load_points"]):
                    if (load_point > left_boundary or is_left) and load_point <= right_boundary:
                        b[basis_index_global] += mesh["N"][load_index]*poly_eval(approx.basis[test_index_local]["f"], x_local(load_point))
                # apply specific axial force
                element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
                point_global = lambda point_local: mesh["element_boundaries"][e] + point_local*element_scale
                b[basis_index_global] += quad(lambda point: mesh["f"](point_global(point))*poly_eval(approx.basis[test_index_local]["f"], point), [0, 1])*element_scale
                #print("normal or free")
                
    #print(f"b={b}")
    return b


# right hand side for w
def get_rhs_w(approx, mesh):
    #print("assembling rhs")
    b = np.zeros((approx.coeffs_len,))
    for e in range(mesh["num_elements"]):
        #print(f"e={e}")
        coeffs_per_element = len(approx.basis)
        index_global_left = e*int(coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        left_boundary = mesh["element_boundaries"][e]
        right_boundary = mesh["element_boundaries"][e + 1]
        element_scale = right_boundary - left_boundary
        x_local = lambda x_global: (x_global - left_boundary)/(right_boundary - left_boundary)
        # flags for boundary elements
        is_left = e == 0
        is_right = e == mesh["num_elements"] - 1
        is_left_pinned = is_left and mesh["boundary_conds"][0] == 1
        is_right_pinned = is_right and mesh["boundary_conds"][1] == 1
        is_left_clamped = is_left and mesh["boundary_conds"][0] == 2
        is_right_clamped = is_right and mesh["boundary_conds"][1] == 2
        for test_index_local in range(coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            basis_index_global = index_global_left + test_index_local
            # evaluating the Dirichlet boundary condition for all test functions on the boundary elements is fine,
            # because the result will be zero, except for the relevant test function
            # an element can be left and right element, if there is only a single element
            replace_row = False

            # left element
            if is_left_pinned or is_left_clamped:
                # set displacement for pinned or clamped  boundary condition
                bc_test_val = poly_eval(approx.basis[test_index_local]["f"], 0)
                b[basis_index_global] = mesh["w"][0]*bc_test_val
                replace_row |= abs(bc_test_val) > 0
                #if abs(bc_test_val) > 0:
                #    print("left pinned or clamped (w)")
            if is_left_clamped:
                # set slope for clamped boundary condition
                bc_test_val = poly_eval(approx.basis[test_index_local]["f_x"], 0)
                b[basis_index_global] = mesh["w_x"][0]*bc_test_val
                replace_row |= abs(bc_test_val) > 0
                #if abs(bc_test_val) > 0:
                #    print("left clamped (w_x)")
                

            # right element
            if is_right_pinned or is_right_clamped:
                # set displacement for pinned or clamped  boundary condition
                bc_test_val = poly_eval(approx.basis[test_index_local]["f"], 1)
                b[basis_index_global] = mesh["w"][1]*bc_test_val
                replace_row |= abs(bc_test_val) > 0
            if is_right_clamped:
                # set slope for clamped boundary condition
                bc_test_val = poly_eval(approx.basis[test_index_local]["f_x"], 1)
                b[basis_index_global] = mesh["w_x"][1]*bc_test_val
                replace_row |= abs(bc_test_val) > 0
            
            # inner element
            if not replace_row:
                # apply discrete loads
                for load_index, load_point in enumerate(mesh["load_points"]):
                    if (load_point > left_boundary or is_left) and load_point <= right_boundary:
                        b[basis_index_global] += mesh["Q"][load_index]*poly_eval(approx.basis[test_index_local]["f"], x_local(load_point))
                        b[basis_index_global] += mesh["M"][load_index]*poly_eval(approx.basis[test_index_local]["f_x"], x_local(load_point))/element_scale
                # apply specific lateral force
                element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
                point_global = lambda point_local: mesh["element_boundaries"][e] + point_local*element_scale
                b[basis_index_global] += quad(lambda point: mesh["q"](point_global(point))*poly_eval(approx.basis[test_index_local]["f"], point), [0, 1])*element_scale
                #print("normal or free")
                
    logger.debug(f"b={b}")
    return b

