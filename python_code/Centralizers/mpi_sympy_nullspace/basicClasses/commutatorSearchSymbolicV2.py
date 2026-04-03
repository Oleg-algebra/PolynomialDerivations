from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sympy import symbols, diff, expand, solve, Poly, Matrix, nsimplify, simplify


@dataclass
class Derivation:
    """Represents a derivation D = P1*d/dx1 + P2*d/dx2."""
    polynomials: List[any]  # SymPy Expressions
    variables: List[any]

    def apply(self, expression) -> any:
        """Applies the derivation D(f) = sum(Pi * df/dxi)."""
        return expand(sum(p * diff(expression, v) for p, v in zip(self.polynomials, self.variables)))

    def is_zero(self) -> bool:
        return all(p.equals(0) for p in self.polynomials)


class CommutatorFinder:
    def __init__(self, derivation: Derivation, max_k: int = 10):
        self.der = derivation
        self.vars = derivation.variables
        self.max_k = max_k
        self.k = 0

    def _get_search_degree(self) -> int:
        """Calculates the max degree for the unknown polynomials."""
        degrees = []
        for p in self.der.polynomials:
            poly_obj = Poly(p, self.vars)
            degrees.append(sum(poly_obj.total_degree() for _ in [1]) if not p.equals(0) else 0)
        return max(degrees) + self.k

    def _generate_unknown_derivation(self, degree: int) -> Tuple[Derivation, List]:
        """Creates a derivation with unknown coefficients up to a total degree."""
        unknown_polys = []
        all_coeffs = []
        prefixes = ['a', 'b']

        for k in range(len(self.vars)):
            poly_expr = 0
            # Generate all monomials x^i * y^j where i+j <= degree
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    c = symbols(f"{prefixes[k]}_{i}_{j}")
                    all_coeffs.append(c)
                    poly_expr += c * self.vars[0] ** i * self.vars[1] ** j
            unknown_polys.append(poly_expr)

        return Derivation(unknown_polys, self.vars), all_coeffs

    def find_commutator(self) -> Tuple[Derivation, bool]:
        """Search for a derivation D_u such that [D, D_u] = 0."""
        last_result = None
        while self.k < self.max_k:
            print(f"solving for k = {self.k}")
            degree = self._get_search_degree()
            unknown_der, coeffs = self._generate_unknown_derivation(degree)

            # Lie Bracket [D1, D2] = 0 means D1(D2_i) - D2(D1_i) = 0 for each component i
            equations = []
            for i in range(len(self.vars)):
                bracket_component = self.der.apply(unknown_der.polynomials[i]) - \
                                    unknown_der.apply(self.der.polynomials[i])

                # Extract coefficients of every monomial to set to zero
                poly_eq = Poly(bracket_component, self.vars)
                equations.extend(poly_eq.coeffs())

            solutions = solve(equations, coeffs)

            # 1. Apply the solutions found by the solver
            partially_substituted = [p.subs(solutions) for p in unknown_der.polynomials]

            # 2. Identify all remaining free variables across ALL polynomials
            all_free_vars = set()
            for p in partially_substituted:
                all_free_vars.update(p.free_symbols - set(self.vars))

            # 3. Create one consistent mapping for all free variables
            # This ensures the same symbol is replaced by the same integer everywhere
            random_map = {fv: np.random.randint(1, 10) for fv in all_free_vars}

            # 4. Finalize the polynomials
            final_polys = []
            for p in partially_substituted:
                final_p = p.subs(random_map)
                final_polys.append(nsimplify(final_p, rational=True))


            result_der = Derivation(final_polys, self.vars)
            last_result = result_der

            if not self.check_proportionality(self.der, result_der):
                return result_der, self.check_proportionality(self.der, result_der)

            self.k += 1

        return last_result, self.check_proportionality(self.der, last_result)

    @staticmethod
    def check_proportionality(d1: Derivation, d2: Derivation) -> bool:
        """Checks if D1 = f * D2 for some scalar function f."""
        # Cross product of 2D vector fields: P1*Q2 - P2*Q1 == 0
        cross_prod = d1.polynomials[0] * d2.polynomials[1] - d1.polynomials[1] * d2.polynomials[0]
        return simplify(cross_prod).equals(0)