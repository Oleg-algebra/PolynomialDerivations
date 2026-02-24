from sympy import Poly, symbols, diff, Add
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import Derivation
from time import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import cv2
import os
import shutil

np.set_printoptions(threshold=sys.maxsize)

def build_derivation_matrix(derivation, max_degree):
    """Будує матрицю оператора в базисі мономів x^i * y^j де i+j <= max_degree."""
    x, y = derivation.variables
    basis = []
    for d in range(max_degree + 1):
        for i in range(d + 1):
            basis.append(x ** (d - i) * y ** i)
    print(basis)
    dim = len(basis)
    matrix = np.zeros((dim, dim))
    basis_indices = {expr: idx for idx, expr in enumerate(basis)}
    print(basis_indices)

    for idx, mono in enumerate(basis):
        # Обчислюємо D(mono)
        result = derivation.apply(mono)
        p_res = Poly(result, x, y)
        for deg, coeff in p_res.terms():
            target_mono = x ** deg[0] * y ** deg[1]
            if target_mono in basis_indices:
                matrix[basis_indices[target_mono], idx] = float(coeff)
    return matrix, basis





def log_iteration_to_str(t, curr_V, basis):
    """Формує рядок для логування поточної ітерації."""
    log_lines = [f"--- Ітерація {t} ---\n"]
    num_polys = curr_V.shape[1]

    for k in range(num_polys):
        coeffs = curr_V[:, k]
        # Відновлюємо поліном тільки з ненульових коефіцієнтів
        poly_expr = Add(*[coeffs[i] * basis[i] for i in range(len(basis)) if abs(coeffs[i]) > 1e-11])
        log_lines.append(f"P_{k}: {poly_expr}\n")
    log_lines.append("\n")
    return "".join(log_lines)


def calculate_trajectories_ultra_fast(coeffs_vectors, matrix, steps, max_degree, basis,is_logging = False):
    """
    Векторизоване обчислення траєкторій з поверненням історії для логування.
    """
    M = np.array(matrix, dtype=np.float64)
    V = np.array(coeffs_vectors, dtype=np.float64).T

    num_polys = V.shape[1]
    dim = V.shape[0]

    basis_degrees = []
    for d in range(max_degree + 1):
        for i in range(d + 1):
            basis_degrees.append([d - i, i])
    B = np.array(basis_degrees, dtype=np.float64)

    all_paths = np.zeros((num_polys, steps, 2), dtype=np.float64)
    last_valid_positions = np.zeros((num_polys, 2), dtype=np.float64)

    # Список для зберігання текстового логу
    history_log = []
    curr_V = V

    for t in range(steps):
        # Логування: перетворюємо поточний стан матриці коефіцієнтів у текст
        if is_logging:
            history_log.append(log_iteration_to_str(t, curr_V, basis))

        abs_V = np.abs(curr_V)
        total_weights = np.sum(abs_V, axis=0)
        weighted_sums = B.T @ abs_V

        alive_mask = total_weights > 1e-11
        current_coords = np.zeros((num_polys, 2))

        if np.any(alive_mask):
            current_coords[alive_mask] = (weighted_sums[:, alive_mask] / total_weights[alive_mask]).T
            last_valid_positions[alive_mask] = current_coords[alive_mask]

        current_coords[~alive_mask] = last_valid_positions[~alive_mask]
        all_paths[:, t, :] = current_coords

        # Крок диференціювання
        curr_V = M @ curr_V

    return all_paths, "".join(history_log)

def calculate_trajectories_batch(coeffs_vectors, matrix, steps, max_degree):
    """Обчислює траєкторії з обробкою занулення многочленів."""
    basis_degrees = []
    for d in range(max_degree + 1):
        for i in range(d + 1):
            basis_degrees.append([d - i, i])
    basis_degrees = np.array(basis_degrees)

    num_polys = coeffs_vectors.shape[0]
    # Одразу створюємо масив потрібного розміру, заповнений нулями
    all_paths = np.zeros((num_polys, steps, 2))

    for k in range(num_polys):
        curr_v = coeffs_vectors[k]
        last_valid_pos = np.array([0.0, 0.0])  # Дефолт для нульового многочлена

        for t in range(steps):
            abs_v = np.abs(curr_v)
            total_w = np.sum(abs_v)

            if total_w > 1e-11:
                # Обчислюємо баріцентричне представлення (центр мас)
                avg_coords = np.sum(abs_v[:, None] * basis_degrees, axis=0) / total_w
                last_valid_pos = avg_coords
                all_paths[k, t, :] = avg_coords
            else:
                # Якщо total_w == 0, многочлен зник.
                # Залишаємо його в останній відомій позиції або в (0,0)
                all_paths[k, t, :] = last_valid_pos

            # Ітерація через матрицю оператора
            curr_v = matrix @ curr_v

    return all_paths




def render_frame(t, points, max_val, temp_dir):
    """Функція для малювання одного кадру (виконується в окремому процесі)."""
    plt.figure(figsize=(8, 6))
    # Використовуємо hexbin для heatmap
    plt.hexbin(points[:, 0], points[:, 1], gridsize=30, cmap='inferno', mincnt=1)
    plt.colorbar(label='Кількість поліномів')

    plt.title(f"Розподіл центрів мас (Крок {t})")
    plt.xlabel("Степінь X")
    plt.ylabel("Степінь Y")
    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    plt.grid(True, linestyle='--', alpha=0.6)

    frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
    plt.savefig(frame_path)
    plt.close()
    return frame_path


def create_heatmap_animation_parallel(all_trajectories, steps, filename="density_evolution.mp4", n_jobs=12):
    """Створює анімацію, розпаралелюючи рендеринг кадрів."""
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Визначаємо межі графіку один раз для всіх кадрів
    max_val = np.max(all_trajectories) + 1

    print(f"Початок рендерингу {steps} кадрів на {n_jobs} ядрах...")

    # Розпаралелюємо цикл по часу (t)
    Parallel(n_jobs=n_jobs)(
        delayed(render_frame)(t, all_trajectories[:, t, :], max_val, temp_dir)
        for t in range(steps)
    )

    # Збираємо список файлів у правильному порядку
    frame_files = [os.path.join(temp_dir, f"frame_{t:04d}.png") for t in range(steps)]

    print("Склеювання відео...")
    # Зчитуємо перший кадр для отримання розмірів
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape

    # Налаштування відео (7 кадрів на секунду, кодек mp4v)
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 7, (width, height))

    for f in frame_files:
        img = cv2.imread(f)
        video.write(img)

    video.release()

    # Очищення тимчасових файлів
    shutil.rmtree(temp_dir)
    print(f"Відео збережено як {filename}")


# Виклик функції:
# create_heatmap_animation_parallel(all_trajectories, steps, "lie_dynamics_heatmap.mp4", n_jobs=12)


if __name__ == "__main__":
    # Налаштування
    x, y = symbols("x y")
    vars = [x, y]

    # Диференціювання (приклад)
    given_der = Derivation([y**2 ,-x**2], vars)
    # 1. Створюємо матрицю оператора (наприклад, для макс. степеня 10)
    start_matrix = time()
    max_deg = 20
    M_matrix, basis = build_derivation_matrix(given_der, max_deg)
    end_matrix = time()

    print(f"matrix generation time: {end_matrix-start_matrix}")

    # 2. Генеруємо початкові вектори коефіцієнтів (1000 поліномів)
    start_poly_gen = time()
    num_polys = 1000
    coeff_limit = 10
    dim = len(basis)
    initial_coeffs = np.random.uniform(-coeff_limit,coeff_limit, (num_polys, dim))
    end_poly_gen = time()
    print(f"polynomial generation time: {end_poly_gen-start_poly_gen}")

    steps = 100

    # 3. Паралельне обчислення траєкторій (Joblib)
    # Розбиваємо 1000 поліномів на 12 пачок
    start_trajectory_gen = time()
    batches = np.array_split(initial_coeffs, 12)

    is_logging = False
    # Тепер отримуємо і траєкторії, і лог-рядки
    results_with_logs = Parallel(n_jobs=12)(
        delayed(calculate_trajectories_ultra_fast)(batch, M_matrix, steps, max_deg, basis,is_logging)
        for batch in batches
    )

    # Розпаковуємо результати
    all_trajectories_list = [r[0] for r in results_with_logs]
    all_logs_list = [r[1] for r in results_with_logs]

    all_trajectories = np.vstack(all_trajectories_list)

    if is_logging:
    # Записуємо лог у файл
        with open("polynomial_evolution.txt", "w", encoding="utf-8") as f:
            # Оскільки ми хочемо порядок: Ітерація 0 для всіх, потім Ітерація 1...
            # а процеси повернули лог "Ітерація 0-100 для поліномів 0-1000",
            # ми можемо просто об'єднати їх, або пересортувати за ітераціями.
            # Найпростіше — вивести їх послідовно по пачках поліномів:
            for batch_log in all_logs_list:
                f.write(batch_log)

    end_trajectory_gen = time()
    print(f'trajectory generation and logging time: {end_trajectory_gen - start_trajectory_gen}')
    # 4. Генерація Heatmap-відео

    start_heat_map_gen = time()
    create_heatmap_animation_parallel(all_trajectories, steps, "lie_dynamics_heatmap.mp4")
    end_heat_map_gen = time()
    print(f'heat map generation time: {end_heat_map_gen-start_heat_map_gen}')

    # Альтернатива склеюванню через cv2:
    # os.system(f"ffmpeg -y -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {filename}")