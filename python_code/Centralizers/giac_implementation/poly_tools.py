import hashlib
import itertools

import numpy as np
from typing import List, Tuple, Any
from giacpy import giac
from giacpy.giacpy import Pygen
from sympy import symbols, sympify, Poly, Symbol


# noinspection PyTypeChecker
def get_polynomial_degree(polynomial: Pygen | Poly,
                          variables: list[Pygen]) -> int:


    return int(polynomial.total_degree(variables))

def hash_polynomialPygen(polynomials: list[Pygen | Poly]) -> int:
    # 1. Формуємо стабільний рядок (використовуємо .normal() для Giac-об'єктів)
    # Важливо, щоб порядок поліномів у списку був завжди однаковим
    key = "--".join([str(p.normal()) for p in polynomials])

    # 2. Використовуємо hashlib для отримання детермінованого хешу
    # sha256 повертає 64-символьний хеш, який завжди однаковий для однакового рядка
    hash_object = hashlib.sha256(key.encode('utf-8'))
    hex_dig = hash_object.hexdigest()

    # 3. Конвертуємо частину hex-рядка в int (наприклад, 16 символів для 64-бітного int)
    # або весь рядок, якщо вам потрібне дуже велике число
    return int(hex_dig[:16], 16)

def create_multivariate_poly(degree,
                             symbol_prefix:str,
                             variables: List[Pygen] | None = None) -> tuple[Pygen,List[Pygen]]:
    """
        Генерує повний поліном заданого степеня від довільної кількості змінних
        із символьними коефіцієнтами.

        :param variables: Список об'єктів Pygen (змінні системи, наприклад [x, y])
        :param degree: Максимальний загальний степінь полінома (total degree)
        :param symbol_prefix: Префікс для позначення невідомих коефіцієнтів (наприклад, "c")
        :return: Кортеж (поліном Pygen, список створених коефіцієнтів)
        """
    if variables == None:
        raise ValueError("variables parameter is not defined")
    poly: Pygen = giac("0")
    coeffs = []
    num_vars = len(variables)

    # itertools.product генерує всі можливі набори степенів від 0 до degree
    for exponents in itertools.product(range(degree + 1), repeat=num_vars):
        # Відсікаємо комбінації, де сума степенів не більша за заданий total degree
        if sum(exponents) <= degree:
            # Створюємо індекс для назви коефіцієнта, наприклад: c_2_1 для моному x^2 * y^1
            suffix = "_".join(map(str, exponents))
            c_name = f"{symbol_prefix}_{suffix}"
            c_val = giac(c_name)

            # Будуємо моном: c * (x^exponents[0]) * (y^exponents[1]) * ...
            term = c_val
            for var, exp in zip(variables, exponents):
                term *= (var ** exp)

            poly += term
            coeffs.append(c_val)

    # .normal() переводить поліном у раціональне представлення для швидших обчислень в Giac
    return poly.normal(), coeffs


def polynomial_from_sympy(polynomial: Poly) -> Pygen:

    # 2. Перетворюємо SymPy Poly назад у вираз, а потім у рядок
    # p.as_expr() видаляє специфічну обгортку Poly, залишаючи чистий поліном
    p_str = str(polynomial.as_expr())

    # 3. Створюємо об'єкт Giac.
    # Метод .normal() гарантує, що Giac правильно розпарсить і спростить вираз
    return giac(p_str).normal()

def polynomial_to_sympy(polynomial: Pygen, variables: List[Pygen | Any]) -> Tuple[Poly,Symbol]:
    s_vars = symbols([str(v) for v in variables])
    p_str = str(polynomial.normal())
    s_expr = sympify(p_str)

    # 2. Створюємо SymPy Poly. Це важливо для збереження методів .LT(), .coeffs() тощо.
    return Poly(s_expr, *s_vars), s_vars

def is_poly_zero(poly: Pygen):
    return poly.normal() == 0

def generate_sparse_random_poly(variables: List[Pygen],
                                degree: int,
                                zero_percentage: float = 0.60,
                                value_range: Tuple[int, int] = (-10, 10)) -> Pygen:
    """
    Генерує випадковий поліном заданого степеня, де вказаний відсоток
    коефіцієнтів гарантовано замінюється на нуль.

    :param variables: Список змінних Giac, наприклад [x, y]
    :param degree: Максимальний загальний степінь полінома
    :param zero_percentage: Коефіцієнт занулення від 0.0 (без нулів) до 1.0 (повне занулення)
    :param value_range: Діапазон для генерації випадкових ненульових коефіцієнтів (мінімум, максимум)
    :return: Числовий поліном Pygen
    """
    # 1. Використовуємо нашу попередню універсальну функцію для створення структури
    # (Припустимо, що create_multivariate_poly визначена вище)
    abstract_poly, coeffs = create_multivariate_poly(variables=variables,
                                                     degree=degree,
                                                     symbol_prefix="c")

    num_coeffs = len(coeffs)
    if num_coeffs == 0:
        return giac(0)

    # 2. Обчислюємо точну кількість нулів, яку нам потрібно отримати
    num_zeros = int(np.round(num_coeffs * zero_percentage))
    num_values = num_coeffs - num_zeros

    # 3. Генеруємо масив випадкових цілих чисел (за винятком нуля, щоб не спотворювати відсоток)
    # Якщо випадково згенерується 0, ми замінимо його на 1
    random_values = np.random.randint(value_range[0], value_range[1] + 1, size=num_values)
    random_values[random_values == 0] = 1

    # 4. Створюємо фінальний масив: додаємо потрібну кількість нулів
    final_array = np.concatenate([random_values, np.zeros(num_zeros, dtype=int)])

    # 5. Перемешуємо масив, щоб нулі розподілилися випадково по мономах
    np.random.shuffle(final_array)

    # 6. Конвертуємо масив Python/NumPy у формат, який підтримує Giac для підстановки
    # Оскільки subst приймає списки Giac, перетворюємо елементи у типи Giac
    giac_values = [giac(int(val)) for val in final_array]

    # 7. Замінюємо символьні коефіцієнти на згенеровані числові значення
    # subst(вираз, список_того_що_міняємо, список_на_що_міняємо)
    sparse_poly = abstract_poly.subst(coeffs, giac_values)

    return sparse_poly.normal()