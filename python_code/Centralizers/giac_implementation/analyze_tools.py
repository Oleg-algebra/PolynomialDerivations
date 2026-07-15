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