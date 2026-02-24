import time
from tqdm import tqdm
from sympy import symbols
# Імпортуємо оновлені класи з попереднього кроку
from Centralizers.basicClasses.constantSearchSymbolic import Derivation, ConstantSearch

# --- Налаштування ---
N = 1
MAX_K = 15
variables = symbols("x y")
x, y = variables

# Визначення компонент диференціювання
# Приклад: D = 0*d/dx + (x^3 * y)*d/dy
monomials = [
    x * 0,  # P1
    x ** 3 * y  # P2
]

# Параметри степенів для стратегії пошуку [k, n, l, m]
# Використовуються для визначення початкового степеня шуканого інтеграла
powers = [0, 0, 3, 1]

der = Derivation(monomials, variables)

# --- Основний цикл тестування ---
start_time = time.time()

report_path = "report.txt"
with open(report_path, "w", encoding="utf-8") as file:
    file.write("Результати тестування пошуку перших інтегралів\n")
    file.write("=" * 80 + "\n")
    file.write(f"Вихідне диференціювання: D = ({monomials[0]})d/dx + ({monomials[1]})d/dy\n\n")

    # Використовуємо tqdm як контекстний менеджер
    with tqdm(range(MAX_K + 1), desc=f"Case N={N}") as pbar:
        for k_val in pbar:
            # Ініціалізація пошуку з поточним зміщенням степеня k_val
            search_engine = ConstantSearch(der, powers, k_extra=k_val)

            # nullspace метод забезпечує швидке знаходження точного розв'язку
            constant, is_constant = search_engine.find_first_integral()

            # Запис у звіт
            file.write(f"K = {k_val:2} | IsConstant: {str(is_constant):5} | Constant: {constant}\n")

            # Останній знайдений результат зберігаємо для виводу в консоль
            final_res = (constant, is_constant)

# --- Фінальний звіт у консоль ---
total_time = time.time() - start_time
final_constant, final_is_constant = final_res

print(f"\n" + "=" * 30)
print(f"Тестування для N = {N} завершено")
print(f"Загальний час: {total_time:.4f} сек")
print(f"Знайдена константа: {final_constant}")
print(f"Валідність: {final_is_constant}")
print("=" * 30)