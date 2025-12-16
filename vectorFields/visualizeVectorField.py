import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.gridspec as gridspec
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from matplotlib.colors import Normalize


def generate_phase_portrait():
    """
    Головна функція для генерації фазового портрету.
    Забезпечує взаємодію з користувачем, парсинг, обчислення та візуалізацію.
    """

    # --- БЛОК 1: Ініціалізація символьного середовища ---
    # Визначаємо символи як дійсні числа (real=True), що спрощує роботу
    # з комплексними числами при обчисленні коренів або логарифмів.
    x, y = sp.symbols('x y', real=True)

    # Словник дозволених локальних змінних. Це механізм безпеки ("білий список"),
    # що запобігає виконанню довільного коду через eval.
    # Також це забезпечує зручність: користувач може писати 'sin' замість 'sp.sin'.
    local_dict = {
        'x': x,
        'y': y,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': sp.log,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'e': sp.E,
        'abs': sp.Abs
    }

    # Налаштування трансформацій парсера.
    # implicit_multiplication_application дозволяє писати "2x" замість "2*x"
    # та "sin(x)y" замість "sin(x)*y".
    transformations = (standard_transformations + (implicit_multiplication_application,))

    print("=== Генератор Фазових Портретів Векторних Полів (v1.0) ===")
    print("Введіть координатні функції векторного поля P(x, y) та Q(x, y).")
    print("Область візуалізації: [-10, 10] x [-10, 10]")
    print("Приклад вводу:")
    print("  P(x, y): y - x**3")
    print("  Q(x, y): -x - y")
    print("-" * 60)

    try:
        # --- БЛОК 2: Ввід та Парсинг ---
        p_str = input("Введіть P(x, y) [dx/dt]: ").strip()
        q_str = input("Введіть Q(x, y) [dy/dt]: ").strip()

        if not p_str or not q_str:
            raise ValueError("Помилка: Вирази не можуть бути порожніми.")

        # Парсинг рядків у символьні вирази SymPy
        # Використовуємо parse_expr замість sympify для кращого контролю
        p_expr = parse_expr(p_str, local_dict=local_dict, transformations=transformations)
        q_expr = parse_expr(q_str, local_dict=local_dict, transformations=transformations)

        print(f"\n[INFO] Система розпізнана як:")
        print(f"  dx/dt = {p_expr}")
        print(f"  dy/dt = {q_expr}")

        # --- БЛОК 3: Компіляція (Lambdification) ---
        # Перетворення символьних виразів у швидкі функції Python, що працюють з масивами NumPy.
        # modules='numpy' гарантує використання векторизованих ufuncs (np.sin, np.exp).
        p_func = sp.lambdify((x, y), p_expr, modules='numpy')
        q_func = sp.lambdify((x, y), q_expr, modules='numpy')

        # --- БЛОК 4: Генерація сітки та обчислення поля ---
        # Створення сітки 100x100 точок.
        # Використання 100j у зрізі забезпечує включення граничних значень.
        w = 1
        Y, X = np.mgrid[-w:w:100j, -w:w:100j]

        # Обчислення компонент вектора швидкості у кожній точці сітки
        U = p_func(X, Y)
        V = q_func(X, Y)

        # Корекція: якщо вираз є константою (наприклад, "1"), lambdify поверне скаляр.
        # Потрібно розмножити цей скаляр до розміру сітки.
        if not isinstance(U, np.ndarray):
            U = np.full_like(X, float(U))
        if not isinstance(V, np.ndarray):
            V = np.full_like(Y, float(V))

        # Обчислення магнітуди (швидкості) для кольорування
        speed = np.sqrt(U ** 2 + V ** 2)

        # Обробка сингулярностей: Inf -> NaN
        # Це критично для коректної роботи colormap
        U[np.isinf(U)] = np.nan
        V[np.isinf(V)] = np.nan
        speed[np.isinf(speed)] = np.nan

        # --- БЛОК 5: Візуалізація ---
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax = fig.add_subplot(1,1,1)

        # Динамічне налаштування товщини ліній
        # Лінії стають товстішими там, де швидкість вища.
        # Використовуємо np.nanmax для ігнорування NaN значень.
        max_speed = np.nanmax(speed)
        if max_speed == 0 or np.isnan(max_speed):
            lw = 1.0  # Якщо поле нульове, товщина стандартна
        else:
            # Нормалізація товщини від 0.5 до 2.5 пунктів
            lw = 0.5 + 2 * speed / max_speed

        # Побудова streamplot
        # density=1.4: підвищена щільність ліній для деталізації
        # arrowsize=1.5: збільшені стрілки для кращої видимості напрямку

        strm = ax.streamplot(X, Y, U, V,
                             color=speed,  # Колір залежить від швидкості
                             linewidth=lw,  # Масив товщин
                             cmap='inferno',  # Тепла кольорова гама
                             density=1.4,
                             arrowsize=1.5)

        # Додавання кольорової шкали
        cbar = fig.colorbar(strm.lines, ax=ax)
        cbar.set_label('Магнітуда фазової швидкості |v|', rotation=270, labelpad=20)

        # --- Оформлення: Класичні осі координат ---
        ax.set_title(f'Фазовий портрет системи\n$P={sp.latex(p_expr)}$\n$Q={sp.latex(q_expr)}$')

        # Приховуємо рамку
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Центруємо осі
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))

        # Додаємо стрілочки
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        # Підписи осей біля стрілочок
        ax.set_xlabel('x', loc='right', fontsize=12)
        ax.set_ylabel('y', loc='top', rotation=0, fontsize=12)

        ax.grid(True, linestyle=':', alpha=0.6)  # Сітка стає менш помітною
        ax.set_aspect('equal')

        # Важливо: aspect='equal' гарантує, що кола виглядатимуть як кола,
        # а перпендикулярність векторів не буде спотворена.
        ax.set_aspect('equal')

        print(" Графік згенеровано успішно.")
        plt.tight_layout()
        plt.savefig(f"{p_str}dx+{q_str}dy.png", dpi=300)
        plt.show()
        # plt.savefig(f"{p_str}dx+{q_str}dy.png", dpi=300)

    except sp.SympifyError as e:
        print(f"\n Помилка синтаксису математичного виразу: {e}")
        print("Перевірте розстановку дужок та правильність написання функцій.")
    except Exception as e:
        print(f"\n Виникла непередбачувана помилка: {e}")


if __name__ == "__main__":
    generate_phase_portrait()