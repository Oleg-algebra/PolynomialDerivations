from sympy import symbols, diff, expand, Poly, Matrix, nsimplify, linear_eq_to_matrix


class Derivation:
    """Клас для представлення диференціювання D = P1*d/dx + P2*d/dy."""

    def __init__(self, polynomials, variables):
        self.polynomials = polynomials  # Список символьних виразів
        self.variables = variables

    def apply(self, expression):
        """Обчислює дію диференціювання на вираз: D(f)."""
        return expand(sum(self.polynomials[i] * diff(expression, self.variables[i])
                          for i in range(len(self.variables))))


class ConstantSearch:
    """Пошук поліноміальних перших інтегралів."""

    def __init__(self, derivation: Derivation, powers: list, k_extra: int = 1, strategy: str = "special"):
        self.der = derivation
        self.powers = powers  # [k, n, l, m] для мономів alpha*x^k*y^n, beta*x^l*y^m
        self.k_extra = k_extra
        self.strategy = strategy

    def _get_search_degrees(self):
        """Визначає максимальні степені N, M для пошуку шуканого полінома."""
        p = self.powers
        # Перевірка на нульові компоненти диференціювання
        f1 = not self.der.polynomials[0].is_zero
        f2 = not self.der.polynomials[1].is_zero

        if self.strategy == "special":
            n_deg = abs(p[0] * f1 - p[2] * f2) + self.k_extra
            m_deg = abs(p[1] * f1 - p[3] * f2) + self.k_extra
        else:
            n_deg = max(p[0] * f1, p[2] * f2) + self.k_extra
            m_deg = max(p[1] * f1, p[3] * f2) + self.k_extra
        return int(n_deg), int(m_deg)

    def _generate_unknown_poly(self, n_deg, m_deg):
        """Створює поліном з невідомими коефіцієнтами."""
        coeffs = []
        poly_expr = 0
        x, y = self.der.variables

        for i in range(n_deg + 1):
            for j in range(m_deg + 1):
                c = symbols(f'a_{i}_{j}')
                coeffs.append(c)
                poly_expr += c * x ** i * y ** j
        return poly_expr, coeffs

    def find_first_integral(self):
        """Основний цикл пошуку константи диференціювання."""
        n_deg, m_deg = self._get_search_degrees()

        while True:
            unknown_poly, coeffs = self._generate_unknown_poly(n_deg, m_deg)

            # Рівняння D(f) = 0
            derivative = self.der.apply(unknown_poly)
            poly_eq = Poly(derivative, self.der.variables)
            equations = poly_eq.coeffs()

            if not equations:
                return symbols('C'), True  # Тривіальна константа

            # Перетворення у матрицю A*x = 0
            matrix_a, _ = linear_eq_to_matrix(equations, coeffs)
            solutions_basis = matrix_a.nullspace()

            if not solutions_basis:
                # Якщо розв'язків немає, збільшуємо степінь пошуку
                n_deg += 1
                m_deg += 1
                continue

            # Беремо перший нетривіальний вектор з базису (ядра матриці)
            final_vector = solutions_basis[0]
            sol_dict = dict(zip(coeffs, final_vector))

            result_poly = nsimplify(unknown_poly.subs(sol_dict), rational=True)

            # Перевірка, чи не є результат нулем
            if result_poly.is_zero:
                n_deg += 1
                continue

            return result_poly, True