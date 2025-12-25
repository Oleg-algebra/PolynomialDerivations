import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.gridspec as gridspec
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


def get_system_input(system_name, local_dict, transformations):
    """
    Допоміжна функція для вводу однієї системи рівнянь.
    """
    print(f"\n--- {system_name} ---")
    p_str = input(f"Введіть P(x, y) [dx/dt] для {system_name}: ").strip()
    q_str = input(f"Введіть Q(x, y) [dy/dt] для {system_name}: ").strip()

    if not p_str or not q_str:
        raise ValueError("Помилка: Вирази не можуть бути порожніми.")

    p_expr = parse_expr(p_str, local_dict=local_dict, transformations=transformations)
    q_expr = parse_expr(q_str, local_dict=local_dict, transformations=transformations)

    return p_expr, q_expr


def compute_field_data(p_func, q_func, X, Y):
    """
    Обчислює компоненти U, V та швидкість для заданої сітки.
    """
    U = p_func(X, Y)
    V = q_func(X, Y)

    # Корекція скалярів
    if not isinstance(U, np.ndarray): U = np.full_like(X, float(U))
    if not isinstance(V, np.ndarray): V = np.full_like(Y, float(V))

    # Обчислення швидкості
    speed = np.sqrt(U ** 2 + V ** 2)

    # Обробка сингулярностей
    U[np.isinf(U)] = np.nan
    V[np.isinf(V)] = np.nan
    speed[np.isinf(speed)] = np.nan

    return U, V, speed


def generate_dual_phase_portrait():
    # --- БЛОК 1: Ініціалізація ---
    x, y = sp.symbols('x y', real=True)
    local_dict = {
        'x': x, 'y': y, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E
    }
    transformations = (standard_transformations + (implicit_multiplication_application,))

    print("=== Генератор Двох Фазових Портретів (v2.0) ===")
    print("Область візуалізації: [-5, 5] x [-5, 5]")  # Changed to fit standard view better

    try:
        # --- БЛОК 2: Ввід двох систем ---
        # Система 1 (Синя)
        p1_expr, q1_expr = get_system_input("Системи #1 (Blue)", local_dict, transformations)
        # Система 2 (Червона)
        p2_expr, q2_expr = get_system_input("Системи #2 (Red)", local_dict, transformations)

        # --- БЛОК 3: Компіляція ---
        p1_func = sp.lambdify((x, y), p1_expr, modules='numpy')
        q1_func = sp.lambdify((x, y), q1_expr, modules='numpy')

        p2_func = sp.lambdify((x, y), p2_expr, modules='numpy')
        q2_func = sp.lambdify((x, y), q2_expr, modules='numpy')

        # --- БЛОК 4: Генерація сітки ---
        w = 1  # Range limit
        Y, X = np.mgrid[-w:w:100j, -w:w:100j]

        # Обчислення даних для обох полів
        U1, V1, speed1 = compute_field_data(p1_func, q1_func, X, Y)
        U2, V2, speed2 = compute_field_data(p2_func, q2_func, X, Y)

        # --- БЛОК 5: Візуалізація ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)

        # --- Побудова Системи 1 (Холодні кольори) ---
        max_speed1 = np.nanmax(speed1)
        lw1 = 0.5 + 2 * speed1 / max_speed1 if max_speed1 > 0 else 1.0

        strm1 = ax.streamplot(X, Y, U1, V1,
                              # color=speed1,
                              color = "blue",
                              linewidth=lw1,
                              cmap='Blues',  # Синя гама
                              density=1.2,
                              arrowsize=1.5)

        # --- Побудова Системи 2 (Теплі кольори) ---
        max_speed2 = np.nanmax(speed2)
        lw2 = 0.5 + 2 * speed2 / max_speed2 if max_speed2 > 0 else 1.0

        strm2 = ax.streamplot(X, Y, U2, V2,
                              # color=speed2,
                              color = "red",
                              linewidth=lw2,
                              cmap='Reds',  # Червона гама
                              density=1.2,
                              arrowsize=1.5)

        # --- Оформлення ---
        # Додаємо легенду через "proxy artists", бо streamplot не підтримує легенди напряму
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label=f'Sys 1: dx={sp.latex(p1_expr)}, dy={sp.latex(q1_expr)}')
        red_patch = mpatches.Patch(color='red', label=f'Sys 2: dx={sp.latex(p2_expr)}, dy={sp.latex(q2_expr)}')
        ax.legend(handles=[blue_patch, red_patch], loc='upper left', fontsize=10, framealpha=0.9)

        # Налаштування осей (Centering)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        ax.set_title('Суперпозиція двох векторних полів')
        ax.grid(True, linestyle=':', alpha=1)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{p1_expr}dx+{q1_expr}dy - {p2_expr}dx+{q2_expr}dy.png", dpi=300)
        plt.show()

    except Exception as e:
        print(f"\n Виникла помилка: {e}")


if __name__ == "__main__":
    generate_dual_phase_portrait()
