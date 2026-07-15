import time
from math import inf
import gc
from dataclasses import dataclass
from giacpy import  syst2mat, solve, matrix
import matplotlib.pyplot as plt
from sympy import lambdify
import os
from poly_tools import *


# noinspection PyTypeChecker
@dataclass
class Derivation:
    """Represents a derivation D = P1*d/dx1 + P2*d/dx2."""
    polynomials: List[Pygen | Poly] # SymPy Expressions
    variables: List[Pygen | Any]

    def __post_init__(self):

        if len(self.polynomials) != len(self.variables):
            raise ValueError(
                f"List's lengthens are not equal "
                f"polynomials number is {len(self.polynomials)}, but variables number is {len(self.variables)}."
            )


    def apply(self, expression) -> Pygen:
        """
        poly: об'єкт giac (поліном)
        components: список [f_x, f_y]
        vars: список змінних [x, y]
        """

        res = sum(f * expression.diff(v) for f, v in zip(self.polynomials, self.variables))
        return res.normal()

    def is_zero(self) -> bool:
        return all(p.normal() == 0 for p in self.polynomials)

    def bracket(self, other) -> 'Derivation':
        """
        Обчислює дужку Лі [self, other].
        Формула для компонент: c_k = self(other.polynomials[k]) - other(self.polynomials[k])
        """
        if self.variables != other.variables:
            raise ValueError("Змінні диференціювань повинні збігатися!")

        new_polynomials = [
            (self.apply(other.polynomials[k]) - other.apply(self.polynomials[k])).normal()
            for k in range(len(self.polynomials))
        ]

        return Derivation(polynomials=new_polynomials, variables=self.variables)

    def find_critical_points(self) -> List[List[Pygen]]:
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

    def count_critical_points(self) -> int:
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

    def to_sympy(self) -> 'Derivation':
        """
        Конвертує поточну деривацію з Giac у SymPy.Poly для передачі через MPI/Pool.
        """
        s_vars = None
        s_polynomials = []

        for p in self.polynomials:
            s_poly, sympy_vars = polynomial_to_sympy(p,self.variables)
            if s_vars == None:
                s_vars = list(sympy_vars)
            s_polynomials.append(s_poly)

        # Створюємо новий чистий об'єкт контейнера
        sympy_derivation = Derivation(polynomials=s_polynomials, variables=s_vars)

        # Руйнуємо локальні вказівники на важкі SymPy об'єкти в поточному стеку
        s_vars = None
        s_polynomials = None
        s_poly = None
        sympy_vars = None

        return sympy_derivation

    def from_sympy(self) -> 'Derivation':
        """
        Конвертує деривацію з SymPy назад у Giac для виконання швидких обчислень.
        """

        # 1. Відновлюємо змінні Giac
        # Створюємо список об'єктів giac(gen) на основі імен символів SymPy
        g_vars = [giac(str(v)) for v in self.variables]

        g_polynomials = []
        for p in self.polynomials:
            g_polynomials.append(polynomial_from_sympy(p))

        giac_derivation = Derivation(polynomials=g_polynomials, variables=g_vars)

        # Чистимо локальні посилання
        g_vars = None
        g_polynomials = None

        return giac_derivation

    def get_sparsity_info(self,M) -> tuple:
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

    def find_commutator(self, max_k = None) -> Tuple:
        """Швидкий пошук через nullspace матриці."""
        current_k = 0
        all_solutions = {}

        if max_k is None:
            max_deg_var = []
            for poly in self.polynomials:
                max_deg_var.append(
                    get_polynomial_degree(poly,self.variables)
                )
            max_k = max(max_deg_var) + 2

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
                # Чистимо невикористані змінні
                unknown_der = None
                coeffs = None
                bracket_lie = None
                continue

            M = syst2mat(equations,coeffs)
            # print(f"[SYSTEM DIMENSION]: {M.dim()}")
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
                    potential_solution = None
                    # print(f"[ZERO SOLUTION]: Derivation: {self} --- k = {current_k} --- max_K = {max_k}")
                    continue

                if not self.is_solution_valid(potential_solution):
                    print("Not valid solution")
                    raise RuntimeError("[!!!][INVALID SOLUTION]")
                    continue

                hash_der = hash_polynomialPygen(potential_solution.polynomials)
                log_result = {
                        "derivation_solution" : potential_solution,
                        "is_proportional" : self.check_proportionality(self, potential_solution),
                        "system_dim" : [int(d) for d in M.dim()]
                    }
                if hash_der not in all_solutions:
                    if not self.check_proportionality(self, potential_solution):
                        # Перед швидким виходом чистимо пам'ять
                        unknown_der = None
                        coeffs = None
                        bracket_lie = None
                        M = None
                        gc.collect()
                        return {hash_der: log_result}, False

                    all_solutions[hash_der] = log_result

            # Очищення наприкінці кожної ітерації ступеня
            unknown_der = None
            coeffs = None
            bracket_lie = None
            M = None
            giac('purge()')
            gc.collect()

            current_k += 1

        if all_solutions == {}:
            print(f"--> empty {self}")
            # raise RuntimeError("[ONLY ZERO SOLUTIONS!!!]")
            all_solutions[hash_polynomialPygen(self.polynomials)] = {
                "derivation_solution": Derivation(list(self.polynomials), list(self.variables)),
                "is_proportional": True,
                "system_dim": []
            }

        # ПЕРЕД ТИМ ЯК ПОВЕРНУТИ РЕЗУЛЬТАТ, ОБНУЛЯЄМО ТИМЧАСОВІ ОБ'ЄКТИ:
        potential_solution = None
        vector = None
        M = None
        equations = None
        bracket_lie = None
        unknown_der = None

        # Викликаємо збір сміття

        gc.collect()

        return all_solutions, True


    def _generate_unknown_derivation(self, degree) -> tuple['Derivation',list[Pygen]]:
        """Створює Du з символьними коефіцієнтами."""
        all_coeffs = []
        polys = []
        for i in range(len(self.variables)):
            poly, coeffs = create_multivariate_poly(degree,
                                                    chr(ord('a') + i),
                                                    self.variables)
            polys.append(poly)
            all_coeffs += coeffs
        return Derivation(polys, self.variables), all_coeffs #TODO: check

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

    def find_first_integral(self,
                            min_degree: int = 0,
                            max_degree: int | None | float = inf,
                            is_truncated_search = False) -> dict:
        """
        Основний метод пошуку перших інтегралів (констант) диференціювання.
        Перебирає загальний степінь полінома від min_degree до max_degree.
        Повертає словник зі списком знайдених нетривіальних інтегралів.
        """
        current_deg = min_degree

        # Якщо max_degree не вказано, беремо максимум зі степенів компонент деривації
        if  max_degree is None :
            max_deg_var = [get_polynomial_degree(poly, self.variables) for poly in self.polynomials]
            max_degree = max(max_deg_var) + 1


        is_trivial_found = False
        is_non_trivial_found = False
        results = {}
        while current_deg <= max_degree:
            print(f"[Poly DEGREE]: {current_deg}")
            # 1. Генеруємо невідомий поліном f та його символьні коефіцієнти
            unknown_poly, coeffs = create_multivariate_poly(current_deg,
                                                            "c",
                                                            self.variables)

            # 2. Обчислюємо дію деривації: D(f)
            df_expr = self.apply(unknown_poly)

            # 3. Виділяємо рівняння (коефіцієнти прирівнюємо до 0)
            equations = df_expr.normal().coeffs(self.variables)

            if not equations:
                current_deg += 1
                # Чистимо посилання
                unknown_poly = None
                coeffs = None
                df_expr = None
                continue

            # 4. Формуємо матрицю системи та знаходимо її ядро (nullspace) через Giac
            M = syst2mat(equations, coeffs)
            solution_basis = M.ker()

            # 5. Аналізуємо знайдені вектори-розв'язки

            for vector in solution_basis:
                # Підставляємо знайдені числові значення замість символьних коефіцієнтів
                found_poly = unknown_poly.subst(coeffs, vector[:-1]).simplify()

                if not self._validate_first_integral(found_poly):
                    print(f"[INVALID FIRST INTEGRAL]: {self} ---> {found_poly}")
                    raise RuntimeError(f"[INVALID FIRST INTEGRAL]: Derivation: {self}")

                # Перевіряємо, чи є знайдений поліном нетривіальним
                # self._is_nontrivial_integral(found_poly) and

                if not  self._is_nontrivial_integral(found_poly):
                    if not is_trivial_found:
                        # print(f"[FOUND TRIVIAL FIRST INTEGRAL]: {found_poly}")
                        results[hash_polynomialPygen([found_poly])] = found_poly
                        is_trivial_found = True
                        # print(f"ADDED TRIVIAL --> {self} --> Results: {results}")
                else:
                    hash_poly = hash_polynomialPygen([found_poly])
                    if  hash_poly not in results:
                        # print(f"[FOUND NON TRIVIAL FIRST INTEGRAL]: {self}: {found_poly}")
                        results[hash_poly] = found_poly
                        is_non_trivial_found = True

            # АГРЕСИВНЕ ОЧИЩЕННЯ ПАМ'ЯТІ всередині циклу пошуку інтегралів
            unknown_poly = None
            coeffs = None
            df_expr = None
            equations = None
            M = None
            solution_basis = None

            giac('purge()')  # Скидаємо символьний кеш Giac
            gc.collect()

            # Якщо знайшли нетривіальні інтеграли на поточному кроці степеня — зупиняємо пошук
            if is_truncated_search and is_non_trivial_found:
                print("[TRUNCATING SEARCH]")
                break

            current_deg += 1

        else:
            print("WENT THROUGH ALL DEGREES")

        # if results == {}:
        #     print(f"---->NO FIRST INTEGRALS: {self}")
        # else:
        #     print(f"---->FOUND SOMETHING: {self}")

        return {"first_integrals": results}



    def _validate_first_integral(self,poly: Pygen):
        res = self.apply(poly)
        return is_poly_zero(res)

    def _is_nontrivial_integral(self, poly: Pygen) -> bool:
        """
        Перевіряє, чи є знайдений інтеграл нетривіальним.
        Поліном тривіальний, якщо він тотожно рівний 0 або є чистою константою.
        """
        if is_poly_zero(poly):
            return False
        # Якщо степінь відносно всіх змінних системи дорівнює 0 — це константа C
        if get_polynomial_degree(poly, self.variables) == 0:
            return False

        return True

    def draw_phase_portrait(self, x_range=(-5, 5),
                            y_range=(-5, 5),
                            density=1.5,
                            directory: str = "phase_portraits/") -> str:
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
        U_raw = f_u(X, Y)
        V_raw = f_v(X, Y)


        # Перевіряємо, чи є результат масивом.
        # Якщо повертається скаляр (для константних поліномів), розгортаємо його в масив форми X
        U = U_raw if isinstance(U_raw, np.ndarray) else np.full_like(X, U_raw)
        V = V_raw if isinstance(V_raw, np.ndarray) else np.full_like(Y, V_raw)

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

        hash = hash_polynomialPygen(self.polynomials)
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

        # Чистимо важкі об'єкти графіків
        s_der = None
        U = None
        V = None
        X = None
        Y = None
        gc.collect()
        return filename


    def __matmul__(self, other):
        """Дозволяє запис D3 = D1 @ D2 для комутатора."""
        return self.bracket(other)











if __name__ == "__main__":
    # l, k, n, m, alpha, beta = (8, 9, 5, 8, 1, 1)
    # l, k, n, m, alpha, beta = (2, 1, 2, 0, 6, -8)
    # l, k, n, m, alpha, beta = (2, 0, 2, 0, 1, 1)
    # l, k, n, m, alpha, beta = [5, 6, 4, 0, -9, 5]
    l, k, n, m, alpha, beta = [9,0,0,7,-14,0]

    x, y = giac('x, y')
    f_x:Pygen = alpha*x**k*y**n
    f_y = beta*x**l*y**m
    K = 10
    der = Derivation([f_x, f_y], [x, y])

    dct = {"--".join([str(p) for p in der.polynomials]) : der}
    print(dct)
    print(f"Given derivative: {der}")


    start = time.time()
    all_solutions,is_proportional = der.find_commutator(max_k= None)
    end = time.time()





    print(f"execution time: {end - start}")

