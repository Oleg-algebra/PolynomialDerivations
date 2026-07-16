import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


class LogAnalyzer:
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> Path:
        """
        Перевіряє, чи існує директорія. Якщо ні, то створює її
        разом з усіма вкладеними піддиректоріями (повне дерево).
        Повертає об'єкт Path для зручної подальшої роботи.
        """
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def analyze_logs(self, log_file_path: str, output_dir: str, case_name: str) -> None:
        """
        Зчитує файл із логами, рахує кількість рангів, будує діаграму
        та записує параметри (params) у відповідні файли rank1.txt та rank2.txt.
        """
        # 1. Перевірка та створення директорії за допомогою окремого методу
        out_path = self.ensure_directory_exists(output_dir)

        rank1_params = []
        rank2_params = []

        # 2. Зчитування файлу з логами (підтримка JSONL формату)
        if not os.path.exists(log_file_path):
            print(f"[-] Файлу логів не знайдено за шляхом: {log_file_path}")
            return

        with open(log_file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    data = entry.get("data", {})
                    rank = data.get("RANK")

                    # Витягуємо список параметрів (поліномів)
                    params = data.get("params", [])

                    if rank == 1:
                        rank1_params.append(params)
                    elif rank == 2:
                        rank2_params.append(params)
                except json.JSONDecodeError as e:
                    print(f"[!] Помилка декодування JSON у рядку {line_idx}: {e}")
                    continue

        count_rank1 = len(rank1_params)
        count_rank2 = len(rank2_params)

        print(f"[+] Оброблено логів. Знайдено Rank 1: {count_rank1}, Rank 2: {count_rank2}")

        # 3. Запис у файли rank1.txt та rank2.txt
        self._write_params_to_file(out_path / "rank1.txt", rank1_params)
        self._write_params_to_file(out_path / "rank2.txt", rank2_params)

        # 4. Побудова та збереження діаграми
        self._build_and_save_chart(count_rank1, count_rank2, out_path, case_name)

    @staticmethod
    def _write_params_to_file(file_path: Path, params_list: list) -> None:
        """Внутрішній метод для запису поліномів у текстові файли."""
        with open(file_path, "w", encoding="utf-8") as f:
            for params in params_list:
                for poly in params:
                    f.write(f"{poly}\n")
                f.write("=" * 100 + "\n")
        print(f"[+] Дані успішно збережено у {file_path}")

    @staticmethod
    def _build_and_save_chart(count_r1: int, count_r2: int, output_path: Path, case_name: str) -> None:
        """Внутрішній метод для генерації та збереження діаграми."""
        categories = ['Rank 1', 'Rank 2']
        counts = [count_r1, count_r2]
        colors = ['#3498db', '#e74c3c']  # Синій та червоний кольори

        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, counts, color=colors, edgecolor='black', width=0.6)

        # Додаємо числові значення над стовпчиками
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{int(yval)}",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.title(f"Distribution of Ranks for Case: {case_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Rank Category", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Збереження діаграми
        chart_filename = output_path / f"{case_name}_ranks_distribution.png"
        plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Діаграму збережено як {chart_filename}")

    def analyze_polynomial_degrees(self, log_file_path: str, output_dir: str, case_name: str) -> None:
        """
        Зчитує текстовий файл із поліномами, знаходить степінь кожного унікального полінома
        та кількість змінних у його записі за допомогою Giac, після чого будує
        діаграму з накопиченням (степені vs кількість змінних).
        """
        import gc
        from giacpy import giac

        # 1. Перевірка та створення директорії
        out_path = self.ensure_directory_exists(output_dir)

        if not os.path.exists(log_file_path):
            print(f"[-] Файл із поліномами не знайдено за шляхом: {log_file_path}")
            return

        # Структура для двовимірного аналізу: {категорія_степеня: {кількість_змінних: count}}
        degree_categories = {
            "0": {},
            "1": {},
            "2": {},
            ">= 3": {}
        }

        # 2. Зчитування та парсинг файлу
        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.split("=" * 100)
        parsed_count = 0

        for block in blocks:
            lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            if not lines:
                continue

            # Беремо першого представника блоку
            poly_str = lines[0]

            try:
                # 3. Обчислення степеня та кількості змінних через Giac
                poly_giac = giac(poly_str)
                variables = poly_giac.lvar()

                num_vars = len(variables) if variables else 0

                if num_vars == 0:
                    degree = 0
                    print(poly_giac)
                else:
                    degree = int(poly_giac.total_degree(variables))

                # Визначення категорії степеня
                if degree == 0:
                    deg_cat = "0"
                elif degree == 1:
                    deg_cat = "1"
                elif degree == 2:
                    deg_cat = "2"
                else:
                    deg_cat = ">= 3"

                # Фіксація двовимірних даних
                degree_categories[deg_cat][num_vars] = degree_categories[deg_cat].get(num_vars, 0) + 1
                parsed_count += 1

                # Очищення об'єктів Giac
                poly_giac = None
                variables = None

            except Exception as e:
                print(f"[!] Помилка обробки полінома '{poly_str}': {e}")
                continue

        print(f"[+] Успішно проаналізовано {parsed_count} унікальних поліномів.")

        # 4. Побудова та збереження покращеного графіка
        self._build_and_save_stacked_chart(degree_categories, out_path, case_name)

        # Очищення пам'яті
        gc.collect()

    @staticmethod
    def _build_and_save_stacked_chart(categories_data: dict, output_path: Path, case_name: str) -> None:
        """
        Внутрішній метод для генерації та збереження стовпчастої діаграми з накопиченням (stacked bar chart).
        """
        import numpy as np
        import matplotlib.patheffects as path_effects

        x_categories = ["0", "1", "2", ">= 3"]

        # Збираємо всі унікальні кількості змінних, які зустрілися (наприклад, 0, 1, 2)
        unique_vars = set()
        for cat in x_categories:
            unique_vars.update(categories_data[cat].keys())
        unique_vars = sorted(list(unique_vars))

        # Підготовка даних для кожного шару (Series) кількості змінних
        series_data = {v: [] for v in unique_vars}
        for v in unique_vars:
            for cat in x_categories:
                series_data[v].append(categories_data[cat].get(v, 0))

        plt.figure(figsize=(9, 7))
        bottoms = np.zeros(len(x_categories))

        # Налаштування кольорової гами для шарів
        color_map = {
            0: '#95a5a6',  # Сірий (для констант)
            1: '#3498db',  # Синій (для 1 змінної)
            2: '#2ecc71',  # Зелений (для 2 змінних)
        }
        default_colors = ['#e67e22', '#9b59b6', '#f1c40f']
        color_idx = 0

        # Будуємо шари стовпчиків
        for v in unique_vars:
            color = color_map.get(v)
            if color is None:
                color = default_colors[color_idx % len(default_colors)]
                color_idx += 1

            label = f"{v} змінна(і)" if v > 0 else "0 змінних (константа)"
            values = np.array(series_data[v])

            # Малюємо поточний сегмент стовпчиків
            bars = plt.bar(x_categories, values, bottom=bottoms, color=color,
                           edgecolor='black', width=0.55, label=label)

            # Додаємо числові підписи всередині кожного сегмента стовпчика
            for i, bar in enumerate(bars):
                val = values[i]
                if val > 0:
                    # Позиціонуємо підпис по центру поточного кольорового сегмента
                    y_pos = bottoms[i] + val / 2.0
                    txt = plt.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{int(val)}",
                                   ha='center', va='center', color='white',
                                   fontsize=10, fontweight='bold')
                    # Додаємо легку тінь до тексту для кращої читабельності на світлих/темних кольорах
                    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

            # Збільшуємо основу (bottom) для наступного колірного шару
            bottoms += values

        # Додаємо загальну суму (Всього) над кожним стовпчиком
        for i, total_val in enumerate(bottoms):
            if total_val > 0:
                plt.text(i, total_val + 0.15, f"Всього: {int(total_val)}",
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')

        plt.title(f"Polynomial Degrees & Variables Distribution: {case_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Polynomial Degree Category", fontsize=12)
        plt.ylabel("Count of Polynomials", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Налаштування легенди
        plt.legend(title="Кількість змінних у записі", fontsize=10, title_fontsize=11, loc="upper right")

        # Робимо динамічний запас зверху по осі Y під підписи загальної кількості
        if len(bottoms) > 0 and max(bottoms) > 0:
            plt.ylim(0, max(bottoms) * 1.15)

        # Збереження
        chart_filename = output_path / f"{case_name}_degrees_and_vars_distribution.png"
        plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Діаграму з накопиченням успішно збережено: {chart_filename}")