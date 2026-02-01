
import sympy as sp
from itertools import product

def solve_stabilizer(u_expr, v_expr, max_degree=2):
    """
    Computes the basis of the Stabilizer Stab(K[u,v]) inside W_2(K)
    by solving the Fundamental Congruences up to a specific degree.
    """
    # 1. Setup Variables
    x, y = sp.symbols('x y')
    # Coefficients for the generic polynomials f and g
    # We will solve for these unknowns.
    coeffs_f = []
    coeffs_g = []

    # 2. Compute Derivatives and Jacobian
    ux = sp.diff(u_expr, x)
    uy = sp.diff(u_expr, y)
    vx = sp.diff(v_expr, x)
    vy = sp.diff(v_expr, y)

    J = sp.simplify(ux*vy - uy*vx)
    print(f"--- Analysis for degree <= {max_degree} ---")
    print(f"Jacobian J = {J}")

    if J == 0:
        print("Error: u and v are algebraically dependent (Jacobian is 0).")
        return

    # 3. Construct Generic Polynomials f(u,v) and g(u,v) (The Ansatz)
    # f = sum( c_ij * u^i * v^j )
    # g = sum( k_ij * u^i * v^j )
    f_poly = 0
    g_poly = 0

    unknowns_map = {} # Map symbol -> (type, power_u, power_v) to reconstruct later

    # Generate terms u^i * v^j such that total degree in u,v is <= max_degree
    # Note: 'degree' here refers to the "weight" in terms of u,v.
    # Adjust logic if you want 'weighted' degree based on x,y.

    for i in range(max_degree + 1):
        for j in range(max_degree + 1 - i):
            # Create unknown coefficients c_ij and k_ij
            c_sym = sp.Symbol(f'c_{i}_{j}')
            k_sym = sp.Symbol(f'k_{i}_{j}')

            coeffs_f.append(c_sym)
            coeffs_g.append(k_sym)

            unknowns_map[c_sym] = ('f', i, j)
            unknowns_map[k_sym] = ('g', i, j)

            term = (u_expr**i) * (v_expr**j)
            f_poly += c_sym * term
            g_poly += k_sym * term

    # 4. Formulate the Numerators for the Extension Formula
    # The coefficients A and B of the derivation D = A*dx + B*dy are:
    # A = (f * vy - g * uy) / J
    # B = (-f * vx + g * ux) / J

    num_A = sp.expand(f_poly * vy - g_poly * uy)
    num_B = sp.expand(-f_poly * vx + g_poly * ux)

    # 5. Extract Linear Equations (The "Modulo J" Condition)
    # We need num_A and num_B to be divisible by J.
    # This means rem(num_A, J) == 0.

    # Note: sp.div or sp.rem for multivariate polynomials can be tricky.
    # We use a robust check: Polynomial Remainder.

    # Function to get remainder of poly P divided by divisor D w.r.t variables x,y
    def get_remainder_coeffs(numerator, divisor):
        # We perform pseudo-division or check divisibility directly
        # Since we are solving for constants, we can treat this as:
        # Numerator = Q * J + R. We want R = 0.
        # But Q is unknown.

        # SIMPLER APPROACH for this script:
        # If J is a monomial (common in testing), division is easy.
        # If J is a polynomial, we use `pdiv` or `rem`.
        # For general robustness in this script, we assume division must be exact.

        q, r = sp.div(numerator, divisor, domain='QQ')
        return r

    rem_A = get_remainder_coeffs(num_A, J)
    rem_B = get_remainder_coeffs(num_B, J)

    # The remainder is a polynomial in x,y with coefficients being linear combos of c_ij, k_ij.
    # We must force ALL coefficients of x^a y^b in the remainder to be 0.

    system_equations = []

    # Extract coeffs of x,y from the remainders
    # We treat the remainders as polynomials in x, y
    full_coeffs_A = sp.Poly(rem_A, x, y).coeffs()
    full_coeffs_B = sp.Poly(rem_B, x, y).coeffs()

    system_equations.extend(full_coeffs_A)
    system_equations.extend(full_coeffs_B)

    # 6. Solve the Linear System
    all_unknowns = coeffs_f + coeffs_g
    # We pass the list of equations (expressions that must equal 0)
    solutions = sp.solve(system_equations, all_unknowns)

    # 7. Interpret Results (Basis Construction)
    # The solution is a dictionary like {c_0_0: 0, c_1_0: -k_0_1, ...}
    # We need to find the free variables (parameters).

    # Substitute solutions back into f and g
    f_sol = f_poly.subs(solutions)
    g_sol = g_poly.subs(solutions)

    # Collect terms by free variables
    # We identify which symbols remain in the expressions
    free_symbols = f_sol.free_symbols.union(g_sol.free_symbols)
    free_symbols = free_symbols.intersection(set(all_unknowns))

    print(f"\nFound Stabilizer Dimension (at deg {max_degree}): {len(free_symbols)}")

    if len(free_symbols) == 0:
        print("Stabilizer is trivial (Zero algebra) at this degree.")
    else:
        print("Basis Derivations (D(u), D(v)):")
        for i, sym in enumerate(free_symbols):
            # To get the basis vector, set this symbol to 1 and others to 0
            subs_map = {s: 0 for s in free_symbols}
            subs_map[sym] = 1

            basis_f = sp.simplify(f_sol.subs(subs_map).subs(solutions))
            basis_g = sp.simplify(g_sol.subs(subs_map).subs(solutions))

            # Check if this basis vector is non-zero
            if basis_f == 0 and basis_g == 0:
                continue

            # Re-verify the derivation on x,y
            # A = (f * vy - g * uy) / J
            basis_A = sp.simplify((basis_f * vy - basis_g * uy) / J)
            basis_B = sp.simplify((-basis_f * vx + basis_g * ux) / J)

            print(f"\nBasis Element #{i+1} (Variable {sym}):")
            print(f"  D(u) = {basis_f}")
            print(f"  D(v) = {basis_g}")
            print(f"  Geometric Form: D = ({basis_A}) * dx + ({basis_B}) * dy")


# ==========================================
# TEST CASE: The "Cusp" Example
# u = x^2, v = x^3 (Algebraically dependent? No, wait. y is missing?)
# Let's use a standard "Monomial Curve" example:
# u = x^2
# v = y^2
# ==========================================

if __name__ == "__main__":
    x, y = sp.symbols('x y')

    # CASE 1: u=x^2, v=y^2 (Stabilizer should be u*du, v*dv -> rank 2)
    # J = 4xy. f,g must be multiples of u,v?
    print("=== TEST CASE 1: u=x^2, v=y^2 ===")
    solve_stabilizer(x**2, y**2, max_degree=2)

    print("\n" + "="*40 + "\n")

    # CASE 2: Mixed (External Jacobian)
    # u = x + y
    # v = x*y
    print("=== TEST CASE 2: u=x+y, v=xy ===")
    solve_stabilizer(x+y, x*y, max_degree=2)
