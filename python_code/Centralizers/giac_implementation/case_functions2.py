import gc
from typing import List, Tuple

import numpy as np
from giacpy.giacpy import Pygen, giac
from poly_tools import generate_sparse_random_poly, generate_sparse_random_poly_sympy
import random
from sympy import symbols, Poly


def idenctical_polynomials_sympy(min_power: int,
                                 max_power: int,
                                 min_coeff: int,
                                 max_coeff: int,
                                 zero_percentage: float = 0.60,
                                 vars_names: list[str] = None) -> List[Poly]:
    if vars_names is None:
        vars_names = ["x", "y"]
    degree = int(np.random.randint(0, max_power + 1))

    poly = generate_sparse_random_poly_sympy(
        variables_names=vars_names,
        degree=degree,
        zero_percentage=zero_percentage,
        value_range=(min_coeff, max_coeff)
    )
    return [poly, poly]

def identical_polynomials(
        min_power: int,
        max_power: int,
        min_coeff: int,
        max_coeff: int,
        zero_percentage: float = 0.60,
        vars: list[Pygen] = None
) -> List[Pygen]:
    if vars is None:
        raise ValueError("Polynomial variables is not defined. Defined variables.")

    degree = np.random.randint(0, max_power + 1)
    polynomial = generate_sparse_random_poly(
        variables=vars,
        degree=degree,
        zero_percentage=zero_percentage,
        value_range=(min_coeff, max_coeff)
    )

    # Повертаємо чисту копію списку
    res_list = [polynomial, polynomial]

    # Занулюємо локальну змінну для швидшої утилізації
    polynomial = None

    return res_list


def get_monomials(case: int,
                  min_power: int,
                  max_power: int,
                  min_coeff: int,
                  max_coeff: int,
                  vars: List[Pygen] = None
                  ) -> List[Pygen]:
    cases = {
        111: arbitrary,
        101: alpha_beta_zero,
        777: nonPropCase,
        888: propCase,
        1: case1,
        2: case2,
        3: case3,
        4: case4,
        5: case5,
        6: case6,
        7: case7,
        8: case8,
        9: case9
    }
    if vars is None:
        raise ValueError("Polynomial variables is not defined. Defined variables.")

    if case not in cases:
        raise KeyError(f"Key {case} is not in cases list")

    # 1. Отримуємо числові параметри степенів та коефіцієнтів
    l, k, n, m, alpha, beta = cases[case](min_power, max_power, min_coeff, max_coeff)

    # 2. Будуємо мономи ОДНИМ швидким рядковим викликом Giac
    # Це значно швидше за операції Python: giac(alpha) * x**k * y**n
    # Оскільки парсер Giac на C++ миттєво будує моном у пам'яті C++ купи
    m1_str = f"({alpha})*x^{k}*y^{n}"
    m2_str = f"({beta})*x^{l}*y^{m}"

    m1 = giac(m1_str).normal()
    m2 = giac(m2_str).normal()

    # Чистимо локальні рядкові посилання
    m1_str = None
    m2_str = None

    return [m1, m2]


def arbitrary(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(0, max_power + 1))
    k = int(np.random.randint(0, max_power + 1))
    n = int(np.random.randint(0, max_power + 1))
    m = int(np.random.randint(0, max_power + 1))

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def nonPropCase(min_power, max_power, min_coeff, max_coeff):
    while True:
        l = int(np.random.randint(0, max_power))
        k = l + 1
        n = int(np.random.randint(0, max_power))
        m = n + 1

        a = int(np.random.randint(min_coeff, max_coeff))
        alpha = -a * m
        beta = a * k

        if alpha == -beta:
            continue
        else:
            return l, k, n, m, alpha, beta


def propCase(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(1, max_power))
    k = l + 1
    n = l
    m = n + 1

    a = int(np.random.randint(min_coeff, max_coeff))
    alpha = -a * m
    beta = a * k

    return l, k, n, m, alpha, beta


def alpha_beta_zero(min_power, max_power, min_coeff, max_coeff):
    l = 0
    k = int(np.random.randint(1, max_power + 1))
    n = int(np.random.randint(1, max_power + 1))
    m = int(np.random.randint(1, max_power + 1))

    alpha = 0
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case1(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(1, max_power + 1))
    k = l
    n = int(np.random.randint(1, max_power + 1))
    m = n

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case2(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(2, max_power + 1))
    k = l
    m = int(np.random.randint(2, max_power + 1))
    n = 0

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case3(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(1, max_power + 1))
    k = l
    n = int(np.random.randint(1, max_power + 1))
    m = int(np.random.randint(0, n))

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case4(min_power, max_power, min_coeff, max_coeff):
    k = 2
    l = 0
    m = k - 1
    n = m

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case5(min_power, max_power, min_coeff, max_coeff):
    k = int(np.random.randint(1, max_power + 1))
    l = int(np.random.randint(0, k))

    m = int(np.random.randint(1, max_power + 1))
    n = int(np.random.randint(0, m))

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case6(min_power, max_power, min_coeff, max_coeff):
    k = int(np.random.randint(2, max_power + 1))
    l = 0
    n = int(np.random.randint(1, max_power + 1))
    m = int(np.random.randint(0, n))

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case7(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(1, max_power + 1))
    k = int(np.random.randint(0, l))
    m = int(np.random.randint(0, max_power + 1))
    n = m

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case8(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(1, max_power + 1))
    k = int(np.random.randint(0, l))
    m = int(np.random.randint(1, max_power + 1))
    n = int(np.random.randint(0, m))

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta


def case9(min_power, max_power, min_coeff, max_coeff):
    l = int(np.random.randint(1, max_power + 1))
    k = int(np.random.randint(0, l))
    n = int(np.random.randint(1, max_power + 1))
    m = int(np.random.randint(0, n))

    alpha = int(np.random.randint(min_coeff, max_coeff))
    beta = int(np.random.randint(min_coeff, max_coeff))

    return l, k, n, m, alpha, beta