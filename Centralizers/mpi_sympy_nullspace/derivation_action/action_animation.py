from sympy import Poly, symbols
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import Derivation
from time import time

import matplotlib.pyplot as plt
import cv2
import os


def create_multi_trajectory_animation(derivation, start_polys, steps=20, fps=2, filename="multi_evolution.mp4"):
    # Копіюємо список, щоб не змінювати вхідні дані
    current_polys = list(start_polys)
    num_polys = len(current_polys)

    # Історія координат для кожного многочлена: список списків
    hist_x = [[] for _ in range(num_polys)]
    hist_y = [[] for _ in range(num_polys)]

    # Кольори для різних траєкторій
    colors = plt.cm.get_cmap('viridis', num_polys)

    if not os.path.exists("temp_frames"):
        os.makedirs("temp_frames")

    frames = []

    for step in range(steps):
        print(f"creating frame {step}...")
        plt.figure(figsize=(10, 8))

        for k in range(num_polys):
            p_expr = current_polys[k]
            p_obj = Poly(p_expr, derivation.variables)
            terms = p_obj.terms()

            if p_obj.expr.equals(0):
                print("zero polynomial")
                avg_i = 0
                avg_j = 0
            else:
                # Обчислення центру мас за вашою формулою
                # \bar{x} = \frac{\sum |a_{i,j}| \cdot i}{\sum |a_{i,j}|}
                total_w = sum(abs(float(c)) for _, c in terms)
                avg_i = sum(abs(float(c)) * d[0] for d, c in terms) / total_w
                avg_j = sum(abs(float(c)) * d[1] for d, c in terms) / total_w

            hist_x[k].append(avg_i)
            hist_y[k].append(avg_j)

            # Малюємо траєкторію k-го многочлена
            plt.plot(hist_x[k], hist_y[k], color=colors(k), alpha=0.5, linewidth=1)
            # Поточне положення
            plt.scatter(avg_i, avg_j, color=colors(k), s=40, edgecolors='black', zorder=3)

            # Застосовуємо диференціювання для наступного кроку
            current_polys[k] = derivation.apply(p_expr)

        plt.title(f"Еволюція {num_polys} многочленів (Крок {step})")
        plt.xlabel("Степінь x")
        plt.ylabel("Степінь y")
        plt.grid(True, linestyle='--', alpha=0.7)

        # Межі осей (можна налаштувати динамічно)
        plt.xlim(-0.5, max([max(h, default=2) for h in hist_x]) + 1)
        plt.ylim(-0.5, max([max(h, default=2) for h in hist_y]) + 1)

        frame_path = f"temp_frames/frame_{step:03d}.png"
        plt.savefig(frame_path)
        frames.append(frame_path)
        plt.close()

    # Склеювання відео
    if frames:
        first_frame = cv2.imread(frames[0])
        h, w, _ = first_frame.shape
        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            video.write(cv2.imread(f))
            os.remove(f)
        video.release()
        os.rmdir("temp_frames")
        print(f"Анімацію збережено у файл: {filename}")


if __name__ == "__main__":
    # 1. Setup variables and input polynomials
    x, y = symbols("x y")
    variables = [x, y]

    # You can define your polynomials directly as SymPy expressions
    # Example: The rotation derivation (-y*d/dx + x*d/dy)
    poly1 = y
    poly2 = x

    startTime = time()

    # 2. Initialize Derivation and Finder
    # We pass the list of polynomials directly
    der = Derivation([poly1, poly2], variables)

    # Після того, як ви знайшли комутатор 'res_der'
    create_multi_trajectory_animation(
        derivation=der,
        start_polys=[x ** 2 + y ** 2 + 2*x - 3*y],  # Початковий многочлен для тесту
        steps=50,
        fps=3,
        filename="comm_evolution.mp4"
    )

