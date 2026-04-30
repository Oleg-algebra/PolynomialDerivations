import time
import json
import os
import argparse
import sys
from mpi4py import MPI
from contextlib import contextmanager


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


def get_complexity(giac_obj):
    """Повертає кількість вузлів у дереві Giac-об'єкта"""
    from giacpy import giac
    # Ми використовуємо int(), бо giac() повертає об'єкт типу gen
    return int(giac(f"size({giac_obj})"))

def worker():
    # Late import всередині воркера
    with silence_giac():
        from giacpy import giac
        from CommutatorSearchGiac import Derivation, FastCommutatorFinder
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
            finder = FastCommutatorFinder(given_der, max_k=12)
            all_solutions, is_proportional = finder.find_commutator()

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
                "GIVEN": given_der.to_sympy(),
                "RANK": 1 if is_proportional else 2,
                "FOUND": found_dict,
                "time": time.time() - start_t
            }
        except Exception as e:
            result_payload = {"status": "error", "message": str(e), "params": params}

        # Відправляємо JSON-рядок (це найбезпечніше)
        comm.send(result_payload, dest=0, tag=TAG_RESULT)

        # Примусове очищення Giac після кожного завдання
        with silence_giac():
            giac('restart')


def master(total_it, case_id):
    from case_functions2 import get_parameters
    limit_cfg = {"min_power": 0, "max_power": 10, "min_coeff": -20, "max_coeff": 20}

    final_results = []
    tests_sent = 0
    tests_received = 0

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

        if res_data["status"] == "success":
            final_results.append(res_data)
            print(f"[{tests_received}/{total_it}] Success from Worker {worker_id}")
        else:
            print(f"[{tests_received}/{total_it}] Failed. Resending...")
            # Якщо треба — можна тут зменшити tests_received і переслати завдання

        # Відправляємо нове завдання або стоп-сигнал
        if tests_sent < total_it:
            params = get_parameters(case_id, **limit_cfg)
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
        for res in results:
            print()
            print(f"{res["GIVEN"]}, --> rank(C_W) = {res["RANK"]}")
            for der in res["FOUND"].values():
                print(der['commuting_derivative'])

    else:
        worker()