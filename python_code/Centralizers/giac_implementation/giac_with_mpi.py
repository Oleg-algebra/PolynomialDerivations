import time
import os
import argparse
import sys
from mpi4py import MPI
from contextlib import contextmanager

import json
import datetime
import uuid
from typing import Any


def get_existing_hashes(filename="results_log.jsonl"):
    existing_hashes = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Дістаємо хеш із вкладеної структури data
                h = entry.get('data', {}).get('hash')
                if h:
                    existing_hashes.add(h)
    except FileNotFoundError:
        pass # Файлу ще немає, це нормально
    return existing_hashes

def serialize_research_data(obj: Any) -> Any:
    """
    Рекурсивно конвертує об'єкти дослідження в JSON-сумісні типи.
    """
    # 1. Обробка вашого класу Derivation
    if hasattr(obj, 'polynomials') and hasattr(obj, 'variables'):
        return {
            "type": "Derivation",
            "poly": [str(p.as_expr()) for p in obj.polynomials],
            "vars": [str(v) for v in obj.variables]
        }

    # 2. Обробка SymPy Poly
    if hasattr(obj, 'as_expr'):
        return str(obj.as_expr())

    # 3. Обробка стандартних структур
    if isinstance(obj, dict):
        return {k: serialize_research_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_research_data(i) for i in obj]

    # 4. Базові типи
    return obj


def append_to_research_log(result: dict, filename: str = "results_log.jsonl",directory: str = "logs/"):
    """
    Додає один результат обчислення в файл.
    """
    # Збагачуємо дані метаінформацією
    payload = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.datetime.now().isoformat(),
        "data": serialize_research_data(result)
    }

    if os.path.exists(directory) and os.path.isdir(directory):
        print("Директорія на місці.")
    else:
        os.makedirs(directory, exist_ok=True)

    with open(directory+filename, "a", encoding="utf-8") as f:
        # ensure_ascii=False дозволяє бачити кирилицю та змінні як є
        line = json.dumps(payload, ensure_ascii=False)
        f.write(line + "\n")

# 1. КОНТЕКСТ ТИШІ
@contextmanager
def silence_giac():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


# --- MPI НАЛАШТУВАННЯ ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_JOB = 1
TAG_RESULT = 2
TAG_STOP = 3



def worker():
    # Late import всередині воркера
    with silence_giac():
        from giacpy import giac
        from CommutatorSearchGiac import Derivation
        # Ініціалізація ядра
        _ = giac('x')
        giac('nthreads:=1')
        giac('print_time:=0')
        giac('timeout:=40')
        # У воркері після giac('x')
        giac('debug_infolevel:=0')
        giac('threads_allowed:=0')  # Жорстке вимкнення будь-яких потоків на рівні C++

    while True:
        status = MPI.Status()
        # Чекаємо на завдання від Master
        params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP:
            break

        try:
            l, k, n, m, alpha, beta = params
            x, y = giac('x, y')

            # Обчислення
            m1 = giac(alpha) * x ** k * y ** n
            m2 = giac(beta) * x ** l * y ** m

            start_t = time.time()
            given_der = Derivation([m1, m2], [x, y])
            all_solutions, is_proportional = given_der.find_commutator(12)

            # Формуємо JSON-сумісний словник
            found_dict = {}
            for s_id, sol in all_solutions.items():
                der_obj = sol["derivation_solution"]
                found_dict[str(s_id)] = {
                    "is_valid": bool(sol["is_valid"]),
                    "is_proportional": bool(sol["is_proportional"]),
                    "commuting_derivative": der_obj.to_sympy()
                }

            result_payload = {
                "status": "success",
                "params": params,
                "hash" : given_der.hash_polynomialPygen(),
                "GIVEN": given_der.to_sympy(),
                "RANK": 1 if is_proportional else 2,
                "FOUND": found_dict,
                "critical_points_types" : given_der.classify_critical_points(),
                "jacobian": [[str(cell) for cell in row] for row in given_der.get_jacobian()],
                "time": time.time() - start_t
            }
            given_der.draw_phase_portrait()
            print(f"Phase portrait for derivation {given_der.hash_polynomialPygen()} is saved.")
        except Exception as e:
            result_payload = {"status": "error", "message": str(e), "params": params}
            print(result_payload)
            # raise e

        # Відправляємо JSON-рядок (це найбезпечніше)
        comm.send(result_payload, dest=0, tag=TAG_RESULT)

        # Примусове очищення Giac після кожного завдання
        with silence_giac():
            giac('restart')


def master(total_it, case_id):
    from case_functions2 import get_parameters

    with silence_giac():
        from giacpy import giac
        from CommutatorSearchGiac import Derivation
        # Ініціалізація ядра
        _ = giac('x')
        giac('nthreads:=1')
        giac('print_time:=0')
        giac('timeout:=40')
        # У воркері після giac('x')
        giac('debug_infolevel:=0')
        giac('threads_allowed:=0')

    def is_already_computed(params, existing_hashes):
        """
        Перевіряє, чи була деривація з такими параметрами вже обчислена.

        params: кортеж (l, k, n, m, alpha, beta)
        existing_hashes: set() із хешами, зчитаними з вашого JSONL файлу
        """
        l, k, n, m, alpha, beta = params
        x, y = giac('x, y')

        # 1. Створюємо мономи точно за вашою формулою
        # Використовуємо .normal() для стабілізації представлення в Giac
        m1 = (giac(alpha) * x ** k * y ** n).normal()
        m2 = (giac(beta) * x ** l * y ** m).normal()

        # 2. Створюємо об'єкт деривації
        # Примітка: Об'єкт Derivation повинен мати метод hash_polynomialPygen
        # який використовує hashlib.sha256
        temp_der = Derivation([m1, m2], [x, y])

        # 3. Отримуємо стабільний хеш
        current_hash = temp_der.hash_polynomialPygen()

        # 4. Перевірка наявності
        if current_hash in existing_hashes:
            return True, current_hash

        return False, current_hash

    limit_cfg = {"min_power": 0, "max_power": 10, "min_coeff": -20, "max_coeff": 20}

    final_results = []
    tests_sent = 0
    tests_received = 0

    processed_hashes = get_existing_hashes()
    print(f"[*] Завантажено {len(processed_hashes)} існуючих результатів.")

    # Роздаємо перші завдання
    for worker_id in range(1, size):
        if tests_sent < total_it:
            params = get_parameters(case_id, **limit_cfg)
            comm.send(params, dest=worker_id, tag=TAG_JOB)
            tests_sent += 1

    # Збираємо результати
    while tests_received < total_it:
        print(tests_received)
        status = MPI.Status()

        raw_json = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        # res_data = json.loads(raw_json)
        res_data = raw_json
        worker_id = status.Get_source()
        tests_received += 1
        current_hash = res_data.get('hash')

        if res_data["status"] == "success":
            final_results.append(res_data)

            if current_hash in processed_hashes:
                print(f"[!] Дублікат пропущено: {current_hash}")
            else:
                append_to_research_log(res_data)
                processed_hashes.add(current_hash)
                print(f"[+] Новий результат збережено: {current_hash}")

            print(f"[{tests_received}/{total_it}] Success from Worker {worker_id}")
        else:
            with open("logs/errors_log.jsonl", "a", encoding="utf-8") as f:
                # ensure_ascii=False дозволяє бачити кирилицю та змінні як є
                line = json.dumps(res_data, ensure_ascii=False)
                f.write(line + "\n")
            print(res_data)
            print(f"[{tests_received}/{total_it}] Failed. Resending...")
            # Якщо треба — можна тут зменшити tests_received і переслати завдання

        # Відправляємо нове завдання або стоп-сигнал
        if tests_sent < total_it:
            while True:
                params = get_parameters(case_id, **limit_cfg)
                is_already_exists, h = is_already_computed(params, processed_hashes)

                if not is_already_exists:
                    print("FOUND A NEW SET OF PARAMETERS")
                    break
                print("ALREADY EXISTS")
            comm.send(params, dest=worker_id, tag=TAG_JOB)
            tests_sent += 1
        else:
            comm.send(None, dest=worker_id, tag=TAG_STOP)

    return final_results



if __name__ == "__main__":
    if rank == 0:
        start_research_time = time.time()

        parser = argparse.ArgumentParser(description="MPI Commutator Search Test")
        parser.add_argument("--case", type=int, required=True, help="case number")
        parser.add_argument("--it", type=int, default=1, help="iteration number")
        args = parser.parse_args()
        case, total_tests = args.case, args.it
        print(f"Case: {case}, Total Iterations: {total_tests}")
        # Вкажіть параметри тут прямо
        results = master(total_it=total_tests, case_id=case)
        print(f"Done! Collected {len(results)} tests.")
        # for i in range(len(results)):
        #     print()
        #     print(f"count {i}")
        #     res = results[i]
        #     # print(res.keys())
        #     # print(res)
        #     for k ,v in res.items():
        #         print(k," : ",v)
        print("=======ALL DONE=======")
        exit(0)
    else:
        worker()