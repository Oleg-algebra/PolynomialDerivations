import time

from typing import List
from dataclasses import dataclass
from giacpy import giac,  syst2mat
from giacpy.giacpy import Pygen


@dataclass
class Derivation:
    """Represents a derivation D = P1*d/dx1 + P2*d/dx2."""
    polynomials: List[Pygen]  # SymPy Expressions
    variables: List[Pygen]

    def apply(self, expression) -> Pygen:
        """
        poly: об'єкт giac (поліном)
        components: список [f_x, f_y]
        vars: список змінних [x, y]
        """
        # res:Pygen = 0
        # for f, v in zip(self.polynomials, self.variables):
        #     res += f * expression.diff(v)
        # return res.normal()

        res = sum(f * expression.diff(v) for f, v in zip(self.polynomials, self.variables))
        return res.normal()

    def is_zero(self) -> bool:
        return all(p.normal() == 0 for p in self.polynomials)

    def bracket(self, other):
        """
        Обчислює дужку Лі [self, other].
        Формула для компонент: c_k = self(other.polynomials[k]) - other(self.polynomials[k])
        """
        if self.variables != other.variables:
            raise ValueError("Змінні деривацій повинні збігатися!")

        new_polynomials = [
            (self.apply(other.polynomials[k]) - other.apply(self.polynomials[k])).normal()
            for k in range(len(self.polynomials))
        ]

        return Derivation(polynomials=new_polynomials, variables=self.variables)

    def __matmul__(self, other):
        """Дозволяє запис D3 = D1 @ D2 для комутатора."""
        return self.bracket(other)




class FastCommutatorFinder:
    def __init__(self, derivation: Derivation, max_k: int = 5):
        self.der: Derivation = derivation
        self.vars: List[Pygen] = derivation.variables
        self.max_k: int = max_k

    def create_poly_simple(self,degree,symbol_coeffs:str):
        # 1. Оголошуємо основні змінні

        poly: Pygen = 0
        coeffs = []


        # 2. Вкладені цикли для формування всіх комбінацій x^i * y^j
        # Умова i + j <= degree гарантує, що загальний ступінь моному не перевищить d
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                # Створюємо назву коефіцієнта (c0, c1, c2...)
                c_name = f"{symbol_coeffs}{i}_{j}"
                c_val = giac(c_name)

                # Формуємо доданок: c_k * x^i * y^j
                term = c_val * (self.vars[0] ** i) * (self.vars[1] ** j)

                # Додаємо до загального полінома
                poly += term

                # Зберігаємо коефіцієнт у список для подальшого розв'язання системи
                coeffs.append(c_val)

        # .rat() перетворює результат у внутрішню раціональну форму для швидкості
        return poly, coeffs

    def get_sparsity_info(self,M):
        # Загальна кількість елементів
        rows = M.nrows()
        cols = M.ncols()
        total_cells = rows * cols

        if total_cells == 0:
            return 0, 0

        # Конвертуємо в розріджену та рахуємо ненульові елементи
        # В Giac розріджена матриця (sparse) зберігається як таблиця ненульових значень
        S = M.sparse()
        non_zeros = len(S)

        # Щільність (density) = (кількість ненульових) / (загальна кількість)
        density = (non_zeros / total_cells) * 100
        return density, non_zeros

    def hash_polynomialPygen(self,derivation: Derivation) -> int:
        key = "--".join([str(p) for p in derivation.polynomials])
        return hash(key)

    def find_commutator(self) -> dict:
        """Швидкий пошук через nullspace матриці."""
        current_k = 1

        best_solution = None
        is_proportional = False
        all_solutions = {}
        while current_k <= self.max_k:


            # 1. Генеруємо невідомі коефіцієнти
            unknown_der, coeffs = self._generate_unknown_derivation(current_k)
            # print(unknown_der.polynomials)
            # 2. Формуємо рівняння [D, Du] = 0
            bracket_lie: Derivation = self.der @ unknown_der
            equations = []
            for i in range(len(bracket_lie.polynomials)):

                equations.extend(bracket_lie.polynomials[i].normal().coeffs(self.vars))
            # print(equations)
            if not equations:
                current_k += 1
                continue
            # print(len(coeffs),coeffs)
            # print(equations)
            M = syst2mat(equations,coeffs)
            print("Matrix dim: ",M.dim())
            solution = M.ker()
            # print("Solution:\n",solution)
            print(f"solution dim: ",solution.dim())
            # density, nnz = self.get_sparsity_info(M)
            # print(f"Матриця {M.nrows()}x{M.ncols()}")
            # print(f"Ненульових елементів: {nnz}")
            # print(f"Щільність: {density:.2f}%")

            for vector in solution:
                new_polynomials = []
                for i in range(len(self.vars)):
                    m = len(coeffs) // len(self.vars)
                    old = coeffs[m*i:m*(i + 1)]
                    new = vector[m*i:m*(i + 1)]
                    new_polynomials.append(unknown_der.polynomials[i].subst(old,new).simplify())
                    # print(new_polynomials)


                potential_solution = Derivation(new_polynomials,self.vars)

                if potential_solution.is_zero():
                    continue

                if not self.is_solution_valid(potential_solution):
                    print("Not valid solution")
                    continue

                hash_der = self.hash_polynomialPygen(potential_solution)
                if hash_der not in all_solutions:
                    all_solutions[hash_der] = {
                        "derivation_solution" : potential_solution,
                        "is_proportional" : self.check_proportionality(self.der, potential_solution),
                        "is_valid" : self.is_solution_valid(potential_solution)
                    }


            current_k += 1

        return all_solutions

    def _generate_unknown_derivation(self, degree):
        """Створює Du з символьними коефіцієнтами."""
        all_coeffs = []
        polys = []
        for i in range(len(self.vars)):
            poly, coeffs = self.create_poly_simple(degree,chr(ord('a') + i))
            polys.append(poly)
            all_coeffs += coeffs
        return Derivation(polys, self.vars), all_coeffs

    def is_solution_valid(self, solution: Derivation):
        lie_bracket = self.der @ solution
        return lie_bracket.is_zero()


    def check_proportionality(self,d1: Derivation, d2: Derivation) -> bool:
        """Checks if D1 = f * D2 for some scalar function f."""
        # Cross product of 2D vector fields: P1*Q2 - P2*Q1 == 0
        for i in range(len(self.vars)):
            for j in range(i + 1, len(self.vars)):
                if i == j:
                    continue
                cross_prod: Pygen = d1.polynomials[i] * d2.polynomials[j] - d1.polynomials[j] * d2.polynomials[i]
                if cross_prod.normal() == 0:
                    return True
        return False


if __name__ == "__main__":
    x, y = giac('x, y')
    f_x:Pygen = y**2
    f_y = x**2
    print(f_x.str())

    K = 10
    der = Derivation([f_x, f_y], [x, y])

    dct = {"--".join([str(p) for p in der.polynomials]) : der}
    print(dct)
    print(f"Given derivative: {der}")
    commut_search = FastCommutatorFinder(der,K)
    # commuting_derivative, is_proportional = commut_search.find_commutator()
    # print(f"Commuting derivative: {commuting_derivative}")
    # print(f"Is proportional: {is_proportional}")
    start = time.time()
    all_solutions = commut_search.find_commutator()
    end = time.time()


    for hash, solution in all_solutions.items():
        print(solution)

    print(f"execution time: {end - start}")

