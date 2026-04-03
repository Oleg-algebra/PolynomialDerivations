from sympy import symbols, diff, expand, Poly, Matrix, nsimplify, linear_eq_to_matrix, simplify
from typing import List, Tuple
from dataclasses import dataclass


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




class FastCommutatorFinder:
    def __init__(self, derivation: Derivation, max_k: int = 5):
        print("initializing fast commutator finder....")
        self.der = derivation
        self.vars = derivation.variables
        self.max_k = max_k

    def find_commutator(self):

        best_solution = None
        is_proportional = False
        """Швидкий пошук через nullspace матриці."""
        for k in range(self.max_k + 1):
            print(f'k = {k}')
            # Визначаємо степінь шуканого диференціювання
            # (степінь вихідного + k)
            max_p_deg = max((Poly(p, self.vars).total_degree() if not p.is_zero else 0)
                            for p in self.der.polynomials)
            search_degree = max_p_deg + k

            # 1. Генеруємо невідомі коефіцієнти
            unknown_der, coeffs = self._generate_unknown_derivation(search_degree)

            # 2. Формуємо рівняння [D, Du] = 0
            equations = []
            for i in range(len(self.vars)):
                # Компонента i дужки Лі: D(Du_i) - Du(D_i)
                bracket_comp = self.der.apply(unknown_der.polynomials[i]) - \
                               unknown_der.apply(self.der.polynomials[i])

                # Кожен коефіцієнт полінома має дорівнювати нулю
                poly_eq = Poly(bracket_comp, self.vars)
                equations.extend(poly_eq.coeffs())

            if not equations:
                continue

            # 3. Перетворюємо систему в матрицю A * x = 0
            # Це ключовий етап рефакторингу: замість solve() використовуємо матрицю
            A, _ = linear_eq_to_matrix(equations, coeffs)

            # 4. Знаходимо ядро матриці (nullspace) - це всі точні розв'язки
            ns = A.nullspace()

            for sol_vector in ns:
                # Підставляємо знайдені коефіцієнти
                sol_dict = dict(zip(coeffs, sol_vector))
                res_polys = [nsimplify(p.subs(sol_dict), rational=True)
                             for p in unknown_der.polynomials]

                result_der = Derivation(res_polys, self.vars)

                # Перевіряємо, чи не є результат тривіальним (нульовим)
                # та чи не є він пропорційним вихідному D
                if not result_der.is_zero():
                    is_prop = self.check_proportionality(self.der, result_der)
                    if best_solution == None:
                        best_solution = result_der
                        is_proportional = is_prop
                    if not is_prop:
                        return result_der, False # Знайдено незалежне диференціювання

        return best_solution,is_proportional

    def _generate_unknown_derivation(self, degree):
        """Створює Du з символьними коефіцієнтами."""
        all_coeffs = []
        polys = []
        for v_idx in range(len(self.vars)):
            poly_expr = 0
            # Створюємо повний базис мономів до заданого степеня
            for d in range(degree + 1):
                for i in range(d + 1):
                    j = d - i
                    c = symbols(f"c_{v_idx}_{i}_{j}")
                    all_coeffs.append(c)
                    poly_expr += c * self.vars[0]**i * self.vars[1]**j
            polys.append(poly_expr)
        return Derivation(polys, self.vars), all_coeffs

    @staticmethod
    def check_proportionality(d1: Derivation, d2: Derivation) -> bool:
        """Checks if D1 = f * D2 for some scalar function f."""
        # Cross product of 2D vector fields: P1*Q2 - P2*Q1 == 0
        cross_prod = d1.polynomials[0] * d2.polynomials[1] - d1.polynomials[1] * d2.polynomials[0]
        return simplify(cross_prod).equals(0)