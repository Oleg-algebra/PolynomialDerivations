import numpy as np
import random
from sympy import symbols, Poly, simplify
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import Derivation


def get_matrix_and_basis(derivation, max_degree):
    """Будує базис та матрицю M один раз для оператора."""
    x, y = derivation.variables
    basis = [x ** (d - i) * y ** i for d in range(max_degree + 1) for i in range(d + 1)]

    dim = len(basis)
    matrix = np.zeros((dim, dim))
    basis_indices = {expr: idx for idx, expr in enumerate(basis)}

    for idx, mono in enumerate(basis):
        result = derivation.apply(mono)
        # Явно вказуємо змінні x, y як генератори
        p_res = Poly(result, *derivation.variables)

        for deg, coeff in p_res.terms():
            target_mono = x ** deg[0] * y ** deg[1]
            if target_mono in basis_indices:
                # Використовуємо nsimplify для числових дробів або float для чисел
                matrix[basis_indices[target_mono], idx] = float(coeff.evalf())


    return matrix, basis, basis_indices


def verify_single_poly(derivation, matrix, basis, basis_indices, test_poly_expr):
    """Порівнює символьне диференціювання із матричним множенням."""
    x, y = derivation.variables
    dim = len(basis)

    # 1. Символьний результат
    symbolic_result = derivation.apply(test_poly_expr)

    # 2. Матричний результат
    input_vector = np.zeros(dim)
    p_input = Poly(test_poly_expr, x, y)
    for deg, coeff in p_input.terms():
        mono = x ** deg[0] * y ** deg[1]
        if mono in basis_indices:
            input_vector[basis_indices[mono]] = float(coeff)

    output_vector = matrix @ input_vector
    matrix_result_expr = sum(coeff * basis[i] for i, coeff in enumerate(output_vector) if abs(coeff) > 1e-9)

    # 3. Порівняння
    return simplify(symbolic_result - matrix_result_expr).equals(0)


def run_test_suite(derivation, max_degree, num_tests=20):
    """Генерує випадкові поліноми та підраховує статистику."""
    x, y = derivation.variables
    matrix, basis, basis_indices = get_matrix_and_basis(derivation, max_degree)

    correct = 0

    for _ in range(num_tests):
        # Генеруємо випадковий поліном степеня <= max_degree - 1
        deg = random.randint(1, max_degree - 1)
        poly = sum(random.randint(-10, 10) * x ** i * y ** j
                   for i in range(deg + 1) for j in range(deg + 1 - i))

        if verify_single_poly(derivation, matrix, basis, basis_indices, poly):
            correct += 1

    print(f"Результати для D=({derivation.polynomials[0]}, {derivation.polynomials[1]}):")
    print(f"  Всього тестів: {num_tests}")
    print(f"  Вірно: {correct}")
    print(f"  Помилок: {num_tests - correct}")
    print(f"  Точність: {(correct / num_tests) * 100:.1f}%")
    return correct, num_tests


def generate_random_derivations(count, max_deg_op, variables):
    """
    Генерує список випадкових диференціальних операторів.
    :param count: Кількість операторів для генерації.
    :param max_deg_op: Максимальний степінь многочленів у компонентах оператора.
    :param variables: Список символьних змінних [x, y].
    """
    x, y = variables
    random_derivations = []

    for _ in range(count):
        components = []
        for _ in range(2):  # Створюємо P1 та P2 для оператора D = P1*d/dx + P2*d/dy
            # Випадковий степінь для цієї компоненти
            deg = random.randint(0, max_deg_op)
            poly = sum(random.randint(-3, 3) * x ** i * y ** j
                       for i in range(deg + 1) for j in range(deg + 1 - i))

            # Гарантуємо, що компонента не буде нульовою для цікавіших тестів
            if poly == 0:
                poly = symbols("1") if random.random() > 0.5 else x
            components.append(poly)

        random_derivations.append(Derivation(components, variables))

    return random_derivations


if __name__ == "__main__":
    x, y = symbols("x y")
    vars = [x, y]

    MAX_BASIS_DEG = 10  # Степінь базису для матриці M
    NUM_OPERATORS = 10  # Скільки випадкових операторів створити
    TESTS_PER_OP = 100  # Скільки поліномів перевірити для кожного оператора

    # 1. Генеруємо випадкові оператори
    # Обмежуємо степінь оператора 1, щоб він не так швидко виводив за межі базису
    random_ops = generate_random_derivations(NUM_OPERATORS, 1, vars)

    total_stats = {"correct": 0, "total": 0}

    for i, der in enumerate(random_ops):
        print(f"\n--- Тестування випадкового оператора №{i + 1} ---")
        correct, total = run_test_suite(der, MAX_BASIS_DEG, num_tests=TESTS_PER_OP)

        total_stats["correct"] += correct
        total_stats["total"] += total

    # Загальний звіт
    print("\n" + "=" * 50)
    print(f"ПІДСУМКОВА СТАТИСТИКА ПО ВСІХ ОПЕРАТОРАХ:")
    print(f"Всього успішних тестів: {total_stats['correct']} з {total_stats['total']}")
    print(f"Загальна точність: {(total_stats['correct'] / total_stats['total']) * 100:.1f}%")
    print("=" * 50)