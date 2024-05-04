import numpy as np
from helper import poly_eval, quad

# parameters of the beam
params = {"E": 1, #210e9, # Young's modulus
          "A": 1, #1e-4, # cross-sectional area
          "I": 1, # moment of inertia
          "rho": 1, # density
          }


# mass matrix
def get_mass_matrix(approx, mesh):
    # element mass matrix
    M = np.zeros((approx.coeffs_len,)*2)
    for e in range(mesh["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
        # flags for boundary elements
        is_left_dirichlet_element = e == 0 and mesh["boundary_conds"][0] == 0
        is_right_dirichlet_element = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 0
        for test_index_local in range(approx.coeffs_per_element): # loop over relevant rows for element e
            test_index_global = index_global_left + test_index_local
            H_test = approx.basis[test_index_local]["f"]
            for basis_index_local in range(approx.coeffs_per_element): # loop over relevant columns for element e
                basis_index_global = index_global_left + basis_index_local
                if not is_left_dirichlet_element or is_right_dirichlet_element:
                    H_basis = approx.basis[basis_index_local]["f"]
                    #print(H_basis_xx)
                    #time.sleep(100)
                    #print(f"{i}, {j}")
                    # integrate over the element using test function i and basis function j
                    M[test_index_global, basis_index_global] += quad(lambda point: poly_eval(H_basis, point)*poly_eval(H_test, point), [0, 1])*element_scale
    
    print(f"M={M}")
    return M

# force vector
def get_force_vector(coeffs, approx, mesh):
    #print("-----")
    f = np.zeros((approx.coeffs_len,))
    for e in range(mesh["num_elements"]):
        index_global_left = e*int(approx.coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
        # flags for boundary elements
        is_left_dirichlet_element = e == 0 and mesh["boundary_conds"][0] == 0
        is_right_dirichlet_element = e == mesh["num_elements"] - 1 and mesh["boundary_conds"][1] == 0
        for test_index_local in range(approx.coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            #test_index_local = i - left_index
            test_index_global = index_global_left + test_index_local
            #print(f"test_index_local: {test_index_local}")
            # check, if row should be used for left Dirichlet boundary condition
            dirichlet_evaluated = False
            if is_left_dirichlet_element:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions (basis functions only have non-zero values on one element)
                    basis_index_global = index_global_left + basis_index_local
                    H_test = approx.basis[test_index_local]["f"]
                    H_basis = approx.basis[basis_index_local]["f"]
                    dirichlet_toggle = poly_eval(H_basis, 0)*poly_eval(H_test, 0) # must be either zero, or sufficiently large !!!!!!!!improve this!!!!!!!
                    f[test_index_global] += dirichlet_toggle*coeffs[basis_index_global]
                    dirichlet_evaluated |= abs(dirichlet_toggle) > 0
                    #if abs(dirichlet_value) > 1e-10:
                    #    print("left bc")
                    if abs(dirichlet_toggle) > 0 and abs(dirichlet_toggle) < 1e-10:
                        print(f"danger: {dirichlet_toggle}")

            # check, if row should be used for right Dirichlet boundary condition
            if is_right_dirichlet_element:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_test = approx.basis[test_index_local]["f"]
                    H_basis = approx.basis[basis_index_local]["f"]
                    dirichlet_toggle = poly_eval(H_basis, 1)*poly_eval(H_test, 1) # must be either zero, or sufficiently large !!!!!!!!improve this!!!!!!!
                    f[test_index_global] += dirichlet_toggle*coeffs[basis_index_global]
                    dirichlet_toggle |= abs(dirichlet_toggle) > 0
                    #if abs(dirichlet_toggle) > 1e-10:
                    #    print("right bc")
                        
            # check, if PDE condotion should be applied in this row
            if not dirichlet_evaluated:
                for basis_index_local in range(approx.coeffs_per_element): # loop over basis functions
                    basis_index_global = index_global_left + basis_index_local
                    H_test_x = approx.basis[test_index_local]["f_x"]/element_scale
                    H_basis_x = coeffs[basis_index_global]*approx.basis[basis_index_local]["f_x"]/element_scale
                    # integrate over the element using test function i and basis function j
                    f[test_index_global] += quad(lambda point: poly_eval(H_basis_x, point)*poly_eval(H_test_x, point), [0, 1])*element_scale
                #print("normal")

    #print(f"f={f}")
    return f


# right hand side
def get_rhs(approx, mesh):
    b = np.zeros((approx.coeffs_len,))
    for e in range(mesh["num_elements"]):
        coeffs_per_element = len(approx.basis)
        index_global_left = e*int(coeffs_per_element/2) # global index for basis / test functions at the left side of the element
        for test_index_local in range(coeffs_per_element): # loop over test functions on element e (equal to basis functions)
            basis_index_global = index_global_left + test_index_local
            # evaluating the Dirichlet boundary condition for all test functions on the boundary elements is fine,
            # because the result will be zero, except for the relevant test function
            # an element can be left and right element, if there is only a single element
            dirichlet_evaluated = False
            if e == 0:
                # left element
                if mesh["boundary_conds"][0] == 0 or mesh["boundary_conds"][0] == 2:
                    # Dirichlet boundary condition
                    dirichlet_test_value = poly_eval(approx.basis[test_index_local]["f"], 0)
                    b[basis_index_global] += mesh["dirichlet_vals"][0]*dirichlet_test_value
                    dirichlet_evaluated |= abs(dirichlet_test_value) > 1e-10
                elif mesh["boundary_conds"][0] == 1 or mesh["boundary_conds"][0] == 2:
                    # Neumann boundary condition
                    b[basis_index_global] += mesh["neumann_vals"][0]*poly_eval(approx.basis[test_index_local]["f"], 0)
                    
            if e == mesh["num_elements"] - 1:
                # right element        # indices in the force vector
                if mesh["boundary_conds"][1] == 0 or mesh["boundary_conds"][0] == 2:
                    # Dirichlet boundary condition
                    dirichlet_test_value = poly_eval(approx.basis[test_index_local]["f"], 1)
                    b[basis_index_global] += mesh["dirichlet_vals"][1]*dirichlet_test_value
                    dirichlet_evaluated |= abs(dirichlet_test_value) > 1e-10
                elif mesh["boundary_conds"][1] == 1 or mesh["boundary_conds"][0] == 2:
                    # Neumann boundary condition
                    b[basis_index_global] += mesh["neumann_vals"][1]*poly_eval(approx.basis[test_index_local]["f"], 1)
                    
            if not dirichlet_evaluated:
                # inner element
                element_scale = (mesh["element_boundaries"][e + 1] - mesh["element_boundaries"][e])
                point_global = lambda point_local: mesh["element_boundaries"][e] + point_local*element_scale
                b[basis_index_global] += quad(lambda point: mesh["q"](point_global(point))*poly_eval(approx.basis[test_index_local]["f"], point), [0, 1])*element_scale
    
    print(f"b={b}")
    return b



