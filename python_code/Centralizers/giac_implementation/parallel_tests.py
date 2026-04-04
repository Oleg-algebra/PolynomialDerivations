import time
import argparse
import multiprocessing as mp
from case_functions2 import get_parameters

import os
import sys
from contextlib import contextmanager

@contextmanager
def silence_giac():
    # Відкриваємо /dev/null
    with open(os.devnull, 'w') as devnull:
        # Зберігаємо копії оригінальних дескрипторів stdout/stderr
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            # Підміняємо системні дескриптори на /dev/null
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            # Повертаємо все як було
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

# ПРИМІТКА: Імпорт giacpy ТУТ НЕ РОБИМО, щоб уникнути конфліктів при fork()

def worker_task(task_packet):
    """
    Функція-воркер, яка виконує один тест.
    Параметр maxtasksperchild=1 у Pool гарантує, що цей процес
    буде вбитий та перезапущений після виконання цієї функції.
    """


    # 1. Тимчасово "осліплюємо" систему перед імпортом та ініціалізацією
    with silence_giac():
        from giacpy import giac
        import os

        # 1. Створюємо фіктивну змінну, щоб ядро Giac ініціалізувалося
        _ = giac('x')

        # 2. Вимикаємо час та вербальність напряму через рядкові команди
        # Це замінює libgiac.eval()
        giac('print_time:=0')
        giac('verblevel:=0')
        giac('nthreads:=1')  # КРИТИЧНО для лікування malloc()
        giac('timeout:=30')

        from CommutatorSearchGiac import Derivation, FastCommutatorFinder

    params, variables_str, max_k = task_packet
    l, k, n, m, alpha, beta = params

    try:
        # 2. Ініціалізація Giac-змінних
        # Важливо: використовуємо giac() для коефіцієнтів, щоб уникнути конфліктів з SymPy
        x, y = giac('x, y')
        vars_giac = [x, y]

        # Створюємо мономи виключно в Giac
        m1 = giac(alpha) * x ** k * y ** n
        m2 = giac(beta) * x ** l * y ** m
        monomials_giac = [m1, m2]

        start_time = time.time()

        # 3. Обчислення
        given_der = Derivation(monomials_giac, vars_giac)
        finder = FastCommutatorFinder(given_der, max_k=max_k)
        all_solutions = finder.find_commutator()

        # 4. СЕРІАЛІЗАЦІЯ: Перетворюємо об'єкти Giac на рядки для передачі в Master
        serialized_solutions = {}
        for sol_id, sol in all_solutions.items():
            # Глибока копія словника результатів
            clean_sol = {k: v for k, v in sol.items() if k != "derivation_solution"}

            # Конвертуємо деривацію-результат у рядки
            der_obj = sol["derivation_solution"]
            clean_sol["derivation_solution_str"] = [str(p) for p in der_obj.polynomials]

            serialized_solutions[sol_id] = clean_sol

        return {
            "status": "success",
            "params": params,
            "GIVEN": [str(m) for m in monomials_giac],
            "FOUND": serialized_solutions,
            "time": time.time() - start_time
        }

    except Exception as e:
        return {
            "status": "error",
            "params": params,
            "message": str(e)
        }


def run_research():

    start_research = time.time()
    # --- Конфігурація ---
    limit_cfg = {"min_power": 0, "max_power": 10, "min_coeff": -20, "max_coeff": 20}

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--it", type=int, default=100)
    parser.add_argument("--cores", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    print(f"Starting research for Case {args.case} with {args.it} iterations on {args.cores} cores.")

    # 1. Генеруємо всі завдання наперед (якщо вони не занадто важкі)
    max_k = 10
    tasks = [(get_parameters(args.case, **limit_cfg), ['x', 'y'], max_k) for _ in range(args.it)]

    final_results = []
    with mp.Pool(processes=args.cores, maxtasksperchild=1) as pool:
        # 1. Запускаємо всі задачі асинхронно
        async_results = [pool.apply_async(worker_task, (t,)) for t in tasks]

        for i, r in enumerate(async_results):
            try:
                # 2. Чекаємо на результат не більше 60 секунд (налаштуйте під себе)
                waiting_time = 5
                res = r.get(timeout=waiting_time)

                final_results.append(res)
                print(f"[{i}] Success")
            except mp.TimeoutError:
                # 3. Якщо зависло — йдемо далі, воркер буде вбитий через maxtasksperchild
                print(f"[{i}] Skipping: Test timed out after {waiting_time}s")
                final_results.append({"status": "timeout", "params": tasks[i][0],"time" : waiting_time})
            except Exception as e:
                print(f"[{i}] Failed: {e}")

    # 3. Аналіз результатів (Master)
    metrics = {
        "correct": 0, "incorrect": 0, "proportional": 0,
        "unproportional": 0, "zero": 0, "total_time": 0.0, "errors": 0, "timed_out": 0
    }

    for res_data in final_results:
        # print("=" * 100)
        # print(res_data)

        if res_data["status"] == "error":
            print("--> error")
            metrics["errors"] += 1
            continue
        elif res_data["status"] == 'timeout':
            print("--> time")
            metrics["timed_out"] += 1
            metrics["total_time"] += res_data["time"]
        elif res_data["status"] == 'success':
            print("--> OK")
            # print("="*100)
            # print(res_data)
            metrics["total_time"] += res_data["time"]

            # Аналіз FOUND

            for sol in res_data['FOUND'].values():
                if not sol.get("is_valid", False):
                    metrics["incorrect"] += 1
                else:
                    metrics["correct"] += 1

                if sol.get("is_proportional", True):
                    metrics["proportional"] += 1
                else:
                    metrics["unproportional"] += 1
    end_research = time.time()
    # --- Print Summary (як у вас) ---
    print("\n" + "=" * 50)
    print(f"FINAL SUMMARY FOR CASE {args.case}")
    print(f"Total Tests: {len(final_results)} (Errors: {metrics['errors']}, Time_out: {metrics["timed_out"]})")
    print(f"Correct/Incorrect: {metrics['correct']}/{metrics['incorrect']}")
    print(f"Proportional/Unprop: {metrics['proportional']}/{metrics['unproportional']}")
    if len(final_results) > 0:
        print(f"Avg Time per Test: {metrics['total_time'] / len(final_results):.4f}s")
    print(f"Total time: {end_research-start_research}")
    print("=" * 50)


if __name__ == "__main__":
    run_research()