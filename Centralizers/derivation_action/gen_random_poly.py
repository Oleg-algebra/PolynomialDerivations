import random
from sympy import symbols, Poly, nsimplify


def generate_random_polynomials(count, max_n, max_m, variables):
    """
    Генерує список випадкових многочленів.
    :param count: Кількість многочленів.
    :param max_n: Максимальний степінь по x.
    :param max_m: Максимальний степінь по y.
    :param variables: Список змінних [x, y].
    """
    x, y = variables
    polynom_list = []

    for _ in range(count):
        expr = 0
        for i in range(max_n + 1):
            for j in range(max_m + 1):
                # Випадковий коефіцієнт від -5 до 5
                coeff = random.randint(-5, 5)
                if coeff != 0:
                    expr += coeff * x ** i * y ** j

        # Якщо випадково вийшов 0, додаємо просту константу
        if expr == 0:
            expr = nsimplify(random.randint(1, 5))

        polynom_list.append(expr)

    return polynom_list