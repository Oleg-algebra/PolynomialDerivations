import numpy as np
from sympy import symbols, simplify, expand
# Використовуємо наші нові швидкі класи
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import Derivation, CommutatorFinder
from Centralizers.basicClasses.constantSearchSymbolic import ConstantSearch



# --- Допоміжні функції генерації ---

def get_non_proportional_coeffs(alpha, beta):
    """Генерує пару (a, b), яка не пропорційна (alpha, beta)."""
    while True:
        a = np.random.randint(-abs(alpha) - 1, abs(alpha) + 1)
        b = np.random.randint(-abs(beta) - 1, abs(beta) + 1)
        if a * beta != b * alpha:
            return a, b


def lie_bracket(der1: Derivation, der2: Derivation) -> Derivation:
    """Обчислює дужку Лі двох диференціювань [D1, D2]."""
    bracket_polys = []
    for i in range(len(der1.variables)):
        # Формула: [D1, D2]_i = D1(P2_i) - D2(P1_i)
        comp = der1.apply(der2.polynomials[i]) - der2.apply(der1.polynomials[i])
        bracket_polys.append(expand(comp))
    return Derivation(bracket_polys, der1.variables)


# --- Основна функція тестування ---

def run_example(case_id: int = 777, max_k_search: int = 5):
    """
    Проводить повний цикл: генерація D -> пошук комутатора -> пошук константи.
    """
    from Centralizers.mpiTests.case_functions2 import get_parameters

    # 1. Отримання параметрів та налаштування
    l, k, n, m, alpha, beta = get_parameters(case_id, 0, 7, -10, 10)
    x, y = symbols("x y")
    vars_list = [x, y]

    # Створюємо вихідне диференціювання
    monomials = [alpha * x ** k * y ** n, beta * x ** l * y ** m]
    der = Derivation(monomials, vars_list)

    print(f"--- Case {case_id} ---")
    print(f"Given D = ({monomials[0]})d/dx + ({monomials[1]})d/dy\n")

    # 2. Пошук комутатора (використовуємо FastCommutatorFinder)
    # Метод nullspace автоматично знайде точний розв'язок
    comm_finder = CommutatorFinder(der, max_k=5)
    commutator, is_prop = comm_finder.find_commutator()

    # 3. Пошук першого інтеграла (константи)
    # Шукаємо константу, поступово збільшуючи K, поки не знайдемо нетривіальну
    final_constant = None
    found_is_constant = False

    for k_val in range(1, max_k_search):
        const_search = ConstantSearch(der, [k, n, l, m], k_extra=k_val, strategy="general")
        constant, is_const = const_search.find_first_integral()

        # Перевіряємо, чи є результат нетривіальним
        if is_const and not constant.is_Number:
            final_constant = constant
            found_is_constant = True
            print(f"Константу знайдено при K = {k_val}")
            break

    # 4. Вивід результатів
    print("=" * 50)
    print("КОМУТАТОР:")
    if commutator:
        print(f"Пропорційний: {is_prop}")
        print(f"P_comm: {simplify(commutator.polynomials[0])}")
        print(f"Q_comm: {simplify(commutator.polynomials[1])}")
    else:
        print("Комутатор не знайдено.")

    print("-" * 50)
    print("ПЕРШИЙ ІНТЕГРАЛ (КОНСТАНТА):")
    print(f"Знайдено: {found_is_constant}")
    print(f"f(x, y) = {final_constant}")
    print("=" * 50)

    return der, commutator, final_constant

