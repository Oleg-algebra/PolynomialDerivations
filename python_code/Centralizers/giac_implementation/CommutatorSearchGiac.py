import time
from sympy import sympify, Poly, symbols
from typing import List, Any
from dataclasses import dataclass
from giacpy import giac,  syst2mat, solve, matrix
from giacpy.giacpy import Pygen
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify
import os
import hashlib


@dataclass
class Derivation:
    """Represents a derivation D = P1*d/dx1 + P2*d/dx2."""
    polynomials: List[Pygen | Poly] # SymPy Expressions
    variables: List[Pygen | Any]

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

    def find_critical_points(self) -> List[Pygen]:
        """
        Знаходить критичні точки деривації, розв'язуючи систему P1=0, P2=0...
        Використовує базиси Грьобнера всередині Giac для пошуку всіх коренів.
        """

        # 1. Формуємо список рівнянь (коефіцієнти прирівнюємо до 0)
        # В Giac запис solve([eq1, eq2], [var1, var2]) шукає спільні корені
        try:
            # Використовуємо .rat().normal() для спрощення перед розв'язанням
            equations = [p.normal() for p in self.polynomials]

            # 2. Викликаємо солвер Giac
            # solve повертає список розв'язків у форматі [[x1, y1], [x2, y2]...]
            points: Pygen = solve(equations, self.variables)

            return list(points)
        except Exception as e:
            return f"Помилка при пошуку критичних точок: {e}"

    def get_jacobian(self) -> Pygen:
        """
        Будує матрицю Якобі вручну: J[i][j] = d(Pi) / d(xj)
        """

        # Створюємо список списків (матрицю)
        # Зовнішній цикл — по поліномах (рядки)
        # Внутрішній — по змінних (стовпці)
        matrix_data = []
        for p in self.polynomials:
            row = [p.diff(v).normal() for v in self.variables]
            matrix_data.append(row)

        # Конвертуємо список Python у матрицю Giac
        return matrix(matrix_data)

    def classify_critical_points(self) -> List[dict]:
        """
        Класифікує критичні точки (ізольовані та лінії) за власними значеннями.
        """
        points = self.find_critical_points()
        # print(points)
        if isinstance(points, str):
            return []

        jac = self.get_jacobian()
        classification = []

        for pt in points:
            # 1. Визначаємо, чи є точка ізольованою (числовою) чи частиною континууму
            is_symbolic = any(any(pt[i].has(v) == 1 for v in self.variables) for i in range(len(pt)))

            # Підставляємо точку в Якобіан
            current_jac = jac.subs(self.variables, pt)
            evs = current_jac.eigenvalues()
            eigen_vectors = current_jac.eigenvectors()
            if eigen_vectors.type() != "vecteur":
                print(eigen_vectors)
            entry = {
                "point": [str(c) for c in pt],
                "eigenvalues": [str(e) for e in evs],
                "is_continuum": is_symbolic,
                "jacobian_at_point" : [[str(cell) for cell in row] for row in current_jac],
                "eigen_vectors": [[str(cell) for cell in vector] for vector in eigen_vectors]
                if eigen_vectors.type() == "vecteur" else []
            }

            if not is_symbolic:
                # --- ЛОГІКА ДЛЯ ІЗОЛЬОВАНИХ ТОЧОК ---
                try:
                    reals = [float(ev.re()) for ev in evs]
                    imags = [float(ev.im()) for ev in evs]

                    if all(r < -1e-9 for r in reals):
                        pt_type = "Stable (Sink)"
                    elif all(r > 1e-9 for r in reals):
                        pt_type = "Unstable (Source)"
                    elif any(r > 1e-9 for r in reals) and any(r < -1e-9 for r in reals):
                        pt_type = "Saddle"
                    elif all(abs(r) < 1e-9 for r in reals) and any(abs(i) > 1e-9 for i in imags):
                        pt_type = "Center (Linearized)"
                    else:
                        pt_type = "Degenerate/Other"
                except (TypeError, ValueError):
                    pt_type = "Non-numeric (Parameters present)"
            else:
                # --- ЛОГІКА ДЛЯ КОНТИНУУМУ (ЛІНІЙ ТОЧОК) ---
                # На лінії рівноваги принаймні одне власне значення завжди 0
                non_zero_evs = [ev for ev in evs if str(ev.normal()) != "0"]

                if not non_zero_evs:
                    pt_type = "Line of Critical Points (Fully Degenerate)"
                else:
                    # Беремо дійсні частини ненульових (трансверсальних) власних значень
                    try:
                        # Спробуємо оцінити знаки, якщо вираз дозволяє
                        t_reals = [float(ev.re()) for ev in non_zero_evs if ev.re().is_number()]

                        if not t_reals:
                            pt_type = "Line: Parametric Stability"
                        elif all(r < -1e-9 for r in t_reals):
                            pt_type = "Stable Line (Transversal Sink)"
                        elif all(r > 1e-9 for r in t_reals):
                            pt_type = "Unstable Line (Transversal Source)"
                        else:
                            pt_type = "Line: Mixed/Saddle Stability"
                    except:
                        pt_type = "Line: Complex Symbolic Stability"

            entry["type"] = pt_type
            classification.append(entry)

        return classification

    def count_critical_points(self):
        """
        Повертає кількість точок або float('inf'), якщо розв'язків нескінченно.
        """
        points = self.find_critical_points()
        print(points)
        if isinstance(points, str):  # Помилка
            return 0

        # Перевіряємо кожен розв'язок на наявність символів
        for pt in points:
            for coord in pt:
                # Якщо координата містить будь-яку зі змінних системи,
                # значить розв'язок залежить від параметра (нескінченна множина)
                # В Giac перевірка наявності змінної через .has()
                for var in self.variables:
                    flag = coord.has(var)
                    if flag == 1:
                        return float('inf')

                # Додаткова перевірка: якщо координата не є числом
                # (наприклад, буквенний параметр alpha, який не був заданий числівником)
                # if not coord.is_number():
                #     return float('inf')

        return len(points)

    def __matmul__(self, other):
        """Дозволяє запис D3 = D1 @ D2 для комутатора."""
        return self.bracket(other)

    def to_sympy(self) -> 'Derivation':
        """
        Конвертує поточну деривацію з Giac у SymPy.Poly для передачі через MPI/Pool.
        """


        # Створюємо символи SymPy
        s_vars = symbols([str(v) for v in self.variables])

        s_polynomials = []
        for p in self.polynomials:
            # 1. str(p) у Giac використовує ^ для степеня.
            # sympify(..., convert_xor=True) коректно перетворить ^ у **.
            p_str = str(p.normal())
            s_expr = sympify(p_str)

            # 2. Створюємо SymPy Poly. Це важливо для збереження методів .LT(), .coeffs() тощо.
            s_poly = Poly(s_expr, *s_vars)
            s_polynomials.append(s_poly)

        return Derivation(polynomials=s_polynomials, variables=s_vars)

    def from_sympy(self) -> 'Derivation':
        """
        Конвертує деривацію з SymPy назад у Giac для виконання швидких обчислень.
        """
        from giacpy import giac

        # 1. Відновлюємо змінні Giac
        # Створюємо список об'єктів giac(gen) на основі імен символів SymPy
        g_vars = [giac(str(v)) for v in self.variables]

        g_polynomials = []
        for p in self.polynomials:
            # 2. Перетворюємо SymPy Poly назад у вираз, а потім у рядок
            # p.as_expr() видаляє специфічну обгортку Poly, залишаючи чистий поліном
            p_str = str(p.as_expr())

            # 3. Створюємо об'єкт Giac.
            # Метод .normal() гарантує, що Giac правильно розпарсить і спростить вираз
            g_poly = giac(p_str).normal()
            g_polynomials.append(g_poly)

        # Повертаємо новий екземпляр класу з Giac-об'єктами
        return Derivation(polynomials=g_polynomials, variables=g_vars)

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
                term = c_val * (self.variables[0] ** i) * (self.variables[1] ** j)

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

    def hash_polynomialPygen(self) -> int:
        # 1. Формуємо стабільний рядок (використовуємо .normal() для Giac-об'єктів)
        # Важливо, щоб порядок поліномів у списку був завжди однаковим
        key = "--".join([str(p.normal()) for p in self.polynomials])

        # 2. Використовуємо hashlib для отримання детермінованого хешу
        # sha256 повертає 64-символьний хеш, який завжди однаковий для однакового рядка
        hash_object = hashlib.sha256(key.encode('utf-8'))
        hex_dig = hash_object.hexdigest()

        # 3. Конвертуємо частину hex-рядка в int (наприклад, 16 символів для 64-бітного int)
        # або весь рядок, якщо вам потрібне дуже велике число
        return int(hex_dig[:16], 16)

    def find_commutator(self, max_k) -> tuple:
        """Швидкий пошук через nullspace матриці."""
        current_k = 0

        all_solutions = {}
        while current_k <= max_k:

            # 1. Генеруємо невідомі коефіцієнти
            unknown_der, coeffs = self._generate_unknown_derivation(current_k)

            # 2. Формуємо рівняння [D, Du] = 0
            bracket_lie: Derivation = self @ unknown_der
            equations = []
            for i in range(len(bracket_lie.polynomials)):
                equations.extend(
                    bracket_lie.polynomials[i]
                    .normal()
                    .coeffs(self.variables)
                )
            # print(equations)
            if not equations:
                current_k += 1
                continue

            M = syst2mat(equations,coeffs)
            solution = M.ker()


            for vector in solution:
                new_polynomials = []
                for i in range(len(self.variables)):
                    m = len(coeffs) // len(self.variables)
                    old = coeffs[m*i:m*(i + 1)]
                    new = vector[m*i:m*(i + 1)]
                    new_polynomials.append(unknown_der.polynomials[i].subst(old,new).simplify())


                potential_solution = Derivation(new_polynomials,self.variables)

                if potential_solution.is_zero():
                    continue

                if not self.is_solution_valid(potential_solution):
                    print("Not valid solution")
                    continue

                hash_der = potential_solution.hash_polynomialPygen()
                if hash_der not in all_solutions:
                    if not self.check_proportionality(self,potential_solution):
                        return {hash_der : {
                        "derivation_solution" : potential_solution,
                        "is_proportional" : self.check_proportionality(self, potential_solution),
                        "is_valid" : self.is_solution_valid(potential_solution)
                    }
                        }, False

                    all_solutions[hash_der] = {
                        "derivation_solution" : potential_solution,
                        "is_proportional" : self.check_proportionality(self, potential_solution),
                        "is_valid" : self.is_solution_valid(potential_solution)
                    }


            current_k += 1


        return all_solutions, True


    def _generate_unknown_derivation(self, degree):
        """Створює Du з символьними коефіцієнтами."""
        all_coeffs = []
        polys = []
        for i in range(len(self.variables)):
            poly, coeffs = self.create_poly_simple(degree,chr(ord('a') + i))
            polys.append(poly)
            all_coeffs += coeffs
        return Derivation(polys, self.variables), all_coeffs

    def is_solution_valid(self, solution: 'Derivation'):
        lie_bracket = self @ solution
        return lie_bracket.is_zero()


    def check_proportionality(self,d1: 'Derivation', d2: 'Derivation') -> bool:
        """Checks if D1 = f * D2 for some scalar function f."""
        # Cross product of 2D vector fields: P1*Q2 - P2*Q1 == 0
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                if i == j:
                    continue
                cross_prod: Pygen = d1.polynomials[i] * d2.polynomials[j] - d1.polynomials[j] * d2.polynomials[i]
                if cross_prod.normal() == 0:
                    return True
        return False



    def draw_phase_portrait(self, x_range=(-5, 5),
                            y_range=(-5, 5),
                            density=1.5,
                            directory: str = "phase_portraits/"):
        """
        Малює фазовий портрет системи (P1, P2) та зберігає його як PNG.
        """
        # 1. Отримуємо SymPy-версію деривації для швидких числових обчислень
        s_der = self.to_sympy()
        vars_sympy = s_der.variables
        poly_sympy = [p.as_expr() for p in s_der.polynomials]

        # 2. Створюємо lambdify-функції для компонент вектора (u, v)
        # Це перетворює символи у швидкі операції з масивами numpy
        f_u = lambdify(vars_sympy, poly_sympy[0], 'numpy')
        f_v = lambdify(vars_sympy, poly_sympy[1], 'numpy')

        # 3. Готуємо сітку координат
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)

        # 4. Обчислюємо векторне поле на сітці
        U = f_u(X, Y)
        V = f_v(X, Y)

        # 5. Візуалізація
        plt.figure(figsize=(10, 8))

        # Малюємо лінії току
        # color: швидкість потоку (magnitude) для кращого сприйняття
        speed = np.sqrt(U ** 2 + V ** 2)
        strm = plt.streamplot(X, Y, U, V, color=speed, linewidth=1, cmap='autumn',
                              density=density, arrowstyle='->', arrowsize=1.5)

        plt.colorbar(strm.lines, label='Швидкість потоку')

        # 6. Додаємо критичні точки, якщо вони вже обчислені
        try:
            points = self.find_critical_points()
            for pt in points:
                if not any(not coord.is_number() for coord in pt):
                    px, py = float(pt[0]), float(pt[1])
                    plt.plot(px, py, 'go', markersize=8, label='Критична точка' if px == pt[0] else "")
        except:
            pass  # Якщо точок немає або вони символьні — ігноруємо

        hash = self.hash_polynomialPygen()
        # Оформлення
        plt.title(f"Phase Portrait (Hash: {hash})")
        plt.xlabel(str(self.variables[0]))
        plt.ylabel(str(self.variables[1]))
        plt.grid(alpha=0.3)
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)





        if os.path.exists(directory) and os.path.isdir(directory):
            print("Директорія на місці.")
        else:
            os.makedirs(directory, exist_ok=True)

        # 7. Збереження
        filename = f"{directory+str(hash)}.png"
        plt.savefig(filename, dpi=150)
        plt.close()  # Закриваємо фігуру, щоб не переповнювати RAM
        return filename














if __name__ == "__main__":
    x, y = giac('x, y')
    f_x:Pygen = y**2
    f_y = x**2

    K = 10
    der = Derivation([f_x, f_y], [x, y])

    dct = {"--".join([str(p) for p in der.polynomials]) : der}
    print(dct)
    print(f"Given derivative: {der}")

    # commuting_derivative, is_proportional = commut_search.find_commutator()
    # print(f"Commuting derivative: {commuting_derivative}")
    # print(f"Is proportional: {is_proportional}")
    start = time.time()
    all_solutions,is_proportional = der.find_commutator(12)
    end = time.time()


    for hash, solution in all_solutions.items():
        print(hash)
        print(solution)

    print(f"execution time: {end - start}")

