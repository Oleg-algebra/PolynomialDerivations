import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sympy import Poly, symbols, simplify, expand
import cv2
import os
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import Derivation


def calculate_single_trajectory_symbolic(p_expr, derivation, steps):
    """Обчислює траєкторію одного полінома символьно."""
    x, y = derivation.variables
    path = []
    curr_p = p_expr
    last_valid_pos = np.array([0.0, 0.0])

    for t in range(steps):
        # Якщо поліном став 0 через equals(0), фіксуємо останню позицію
        if curr_p.equals(0):
            path.append(last_valid_pos)
            continue

        p_obj = Poly(curr_p, x, y)
        terms = p_obj.terms()

        # Обчислення баріцентричного представлення (центр мас)
        total_w = sum(abs(float(c)) for _, c in terms)
        if total_w > 1e-11:
            avg_i = sum(abs(float(c)) * deg[0] for deg, c in terms) / total_w
            avg_j = sum(abs(float(c)) * deg[1] for deg, c in terms) / total_w
            last_valid_pos = np.array([float(avg_i), float(avg_j)])
            path.append(last_valid_pos)
        else:
            path.append(last_valid_pos)

        # Символьне застосування диференціювання (без обмеження степеня)
        curr_p = derivation.apply(curr_p)

    return np.array(path)


def create_heatmap_animation(all_trajectories, steps, filename="symbolic_dynamics_heatmap.mp4"):
    """Створює відео з тепловою картою."""
    if not os.path.exists("temp"): os.makedirs("temp")
    frames = []

    # Визначаємо динамічні межі для графіку на основі всіх траєкторій
    max_x = np.max(all_trajectories[:, :, 0]) + 1
    max_y = np.max(all_trajectories[:, :, 1]) + 1

    for t in range(steps):
        points = all_trajectories[:, t, :]

        plt.figure(figsize=(8, 6))
        # Малюємо 2D гістограму щільності
        plt.hexbin(points[:, 0], points[:, 1], gridsize=30, cmap='inferno', mincnt=1)
        plt.colorbar(label='Кількість поліномів')

        plt.title(f"Символьна еволюція (Крок {t})")
        plt.xlabel("Степінь X")
        plt.ylabel("Степінь Y")
        plt.xlim(-0.5, max_x)
        plt.ylim(-0.5, max_y)
        plt.grid(True, linestyle='--', alpha=0.5)

        path = f"temp/f_{t:03d}.png"
        plt.savefig(path)
        frames.append(path)
        plt.close()

    # Склеювання відео
    img = cv2.imread(frames[0])
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 7, (img.shape[1], img.shape[0]))
    for f in frames:
        video.write(cv2.imread(f))
        os.remove(f)
    video.release()
    os.rmdir("temp")


if __name__ == "__main__":
    x, y = symbols("x y")
    vars = [x, y]

    # Приклад диференціювання: D = x*d/dx + y*d/dy (Ейлерівське, ростить степінь)
    given_der = Derivation([y-1,x+2], vars)

    # Генерація початкових поліномів (символьно)
    num_polys = 100  # Для символьного методу краще брати менше число, ніж для матричного
    steps = 30

    max_degree = 3
    # Випадкові поліноми степеня до 3
    initial_polys = []
    for _ in range(num_polys):
        p = sum(np.random.randint(-3, 4) * x ** i * y ** j
                for i in range(max_degree) for j in range(max_degree - i))
        initial_polys.append(p if not p.equals(0) else x)

    # Паралельне обчислення (символьне)
    # n_jobs=12 використовує всі ядра вашого процесора
    results = Parallel(n_jobs=12)(
        delayed(calculate_single_trajectory_symbolic)(p, given_der, steps)
        for p in initial_polys
    )

    all_trajectories = np.array(results)

    # Генерація відео
    create_heatmap_animation(all_trajectories, steps, "symbolic_heatmap.mp4")