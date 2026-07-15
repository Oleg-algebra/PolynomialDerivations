import hashlib
import itertools
import gc
import numpy as np
from typing import List, Tuple, Any
from giacpy import giac
from giacpy.giacpy import Pygen
from sympy import symbols, sympify, Poly, Symbol


# Глобальний кеш символів SymPy для уникнення витоків при багаторазовому створенні symbols()
_SYMPY_VARS_CACHE = {}

def get_sympy_symbols(var_names: List[str]) -> tuple[Symbol]:
    """Повертає символи SymPy з глобального кешу, запобігаючи витоку ОЗУ."""
    cache_key = tuple(var_names)
    if cache_key not in _SYMPY_VARS_CACHE:
        _SYMPY_VARS_CACHE[cache_key] = symbols(var_names)
    return _SYMPY_VARS_CACHE[cache_key]

# noinspection PyTypeChecker
def get_polynomial_degree(polynomial: Pygen | Poly,
                          variables: list[Pygen]) -> int:


    return int(polynomial.total_degree(variables))

def hash_polynomialPygen(polynomials: list[Pygen | Poly]) -> int:
    """Швидке детерміноване хешування без створення зайвих Giac C++ об'єктів."""
    # Замість p.normal() беремо str(p). Якщо p вже спрощений, це працює миттєво і без витоків
    key_parts = []
    for p in polynomials:
        if hasattr(p, 'as_expr'):  # Якщо це SymPy Poly
            key_parts.append(str(p.as_expr()))
        else:  # Якщо це Giac Pygen
            key_parts.append(str(p))

    key = "--".join(key_parts)

    hash_object = hashlib.sha256(key.encode('utf-8'))
    hex_dig = hash_object.hexdigest()

    # Руйнуємо локальні посилання на рядки
    key = None
    key_parts = None

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

    # Повертаємо спрощений поліном
    res_poly = poly.normal()

    # Примусово очищуємо проміжні накопичення
    poly = None

    return res_poly, coeffs


def polynomial_from_sympy(polynomial: Poly) -> Pygen:

    # 2. Перетворюємо SymPy Poly назад у вираз, а потім у рядок
    # p.as_expr() видаляє специфічну обгортку Poly, залишаючи чистий поліном
    p_str = str(polynomial.as_expr())

    res = giac(p_str).normal()
    p_str = None
    return res

def polynomial_to_sympy(polynomial: Pygen, variables: List[Pygen | Any]) -> Tuple[Poly,Symbol]:
    # Використовуємо наш безпечний кешований генератор символів SymPy
    var_names = [str(v) for v in variables]
    s_vars = get_sympy_symbols(var_names)

    # Отримуємо чистий нормалізований рядок з Giac без створення нового об'єкта
    p_str = str(polynomial)
    s_expr = sympify(p_str)

    res_poly = Poly(s_expr, *s_vars)

    # Звільняємо пам'ять у локальному стеку
    p_str = None
    s_expr = None
    var_names = None

    return res_poly, s_vars

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

    res_poly = sparse_poly.normal()

    # !!! ВАЖЛИВА ДЕСТРУКТУРИЗАЦІЯ ДЛЯ ЗВІЛЬНЕННЯ C++ КУПИ !!!
    abstract_poly = None
    coeffs = None
    giac_values = None
    sparse_poly = None
    final_array = None
    random_values = None

    # Викликаємо збирач сміття для видалення С++ дескрипторів
    gc.collect()

    return res_poly


def generate_sparse_random_poly_sympy(variables_names: List[str],
                                      degree: int,
                                      zero_percentage: float = 0.60,
                                      value_range: Tuple[int, int] = (-10, 10)) -> Poly:
    """Генерує випадковий розріджений поліном суто через SymPy (без залучення Giac)."""
    s_vars = symbols(variables_names)
    num_vars = len(s_vars)

    # 1. Генеруємо всі можливі комбінації степенів (мономи)
    all_exponents = []
    for exponents in itertools.product(range(degree + 1), repeat=num_vars):
        if sum(exponents) <= degree:
            all_exponents.append(exponents)

    num_coeffs = len(all_exponents)
    if num_coeffs == 0:
        return Poly(0, *s_vars)

    # 2. Визначаємо кількість нульових коефіцієнтів
    num_zeros = int(np.round(num_coeffs * zero_percentage))
    num_values = num_coeffs - num_zeros

    # 3. Генеруємо випадкові значення
    random_values = np.random.randint(value_range[0], value_range[1] + 1, size=num_values)
    random_values[random_values == 0] = 1

    final_coeffs = np.concatenate([random_values, np.zeros(num_zeros, dtype=int)])
    np.random.shuffle(final_coeffs)

    # 4. Будуємо поліном SymPy
    poly_expr = 0
    for coeffs_val, exponents in zip(final_coeffs, all_exponents):
        if coeffs_val == 0:
            continue
        term = coeffs_val
        for var, exp in zip(s_vars, exponents):
            term *= (var ** exp)
        poly_expr += term

    return Poly(poly_expr, *s_vars)