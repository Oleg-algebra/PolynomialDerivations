import time
import argparse
import sys
from mpi4py import MPI
from sympy import symbols, expand, nsimplify
from case_functions2 import get_parameters

# Використовуємо новий швидкий клас
# (Впевніться, що перейменували або оновили файл)
from commutatorSearchFast import Derivation, FastCommutatorFinder

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Теги для повідомлень MPI
TAG_JOB = 1
TAG_RESULT = 2
TAG_STOP = 3


def master(total_tests, case, limit_cfg):
    results = {}
    tests_sent = 0
    tests_received = 0

    # 1. Роздаємо перші завдання всім вільним воркерам
    for worker_id in range(1, size):
        if tests_sent < total_tests:
            params = get_parameters(case, **limit_cfg)
            comm.send(params, dest=worker_id, tag=TAG_JOB)
            tests_sent += 1

    # 2. Отримуємо результати та відправляємо нові завдання
    while tests_received < total_tests:
        print(tests_received)
        status = MPI.Status()
        res_data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        worker_id = status.Get_source()
        tests_received += 1

        # Зберігаємо результат за параметрами як за ключем
        results[res_data['params']] = res_data

        # Якщо ще є робота — відправляємо цьому ж воркеру
        if tests_sent < total_tests:
            params = get_parameters(case, **limit_cfg)
            comm.send(params, dest=worker_id, tag=TAG_JOB)
            tests_sent += 1
        else:
            # Робота закінчилася, кажемо воркеру зупинитися
            comm.send(None, dest=worker_id, tag=TAG_STOP)

    return results


def worker(variables):
    while True:
        status = MPI.Status()
        params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP:
            break

        start_time = time.time()
        l, k, n, m, alpha, beta = params

        monomials = [alpha * x ** k * y ** n, beta * x ** l * y ** m]

        # Обчислення
        given_der = Derivation(monomials, variables)
        finder = FastCommutatorFinder(given_der, max_k=5)
        found_der, is_prop = finder.find_commutator()

        # Швидка валідація без heavy simplify
        is_correct = True
        if found_der and not found_der.is_zero():
            for i in range(len(variables)):
                bracket = given_der.apply(found_der.polynomials[i]) - \
                          found_der.apply(given_der.polynomials[i])
                if not nsimplify(expand(bracket)).equals(0):
                    is_correct = False
                    break

        # Відправка результату назад
        result_payload = {
            "params": params,
            "GIVEN": monomials,
            "FOUND": found_der.polynomials,
            "isCorrect": is_correct,
            "isProp": is_prop,
            "isZero": found_der.is_zero() if found_der else True,
            "time": time.time() - start_time
        }
        comm.send(result_payload, dest=0, tag=TAG_RESULT)


# --- Точка входу ---
x, y = symbols("x y")
limit_cfg = {"min_power": 0, "max_power": 10, "min_coeff": -20, "max_coeff": 20}

if rank == 0:
    # Парсинг аргументів
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--it", type=int, default=100)
    args = parser.parse_args()
    case = args.case

    final_results = master(args.it, args.case, limit_cfg)
    # Тут вивід статистики (як у вашому оригінальному коді)

    metrics = {
        "correct": 0, "incorrect": 0, "proportional": 0,
        "unproportional": 0, "zero": 0, "total_time": 0.0
    }
    for res in final_results.values():
        metrics["total_time"] += res["time"]
        if not res["isCorrect"]:
            metrics["incorrect"] += 1
        else:
            metrics["correct"] += 1

        if res["isZero"]:
            metrics["zero"] += 1
        elif res["isProp"]:
            metrics["proportional"] += 1
        else:
            metrics["unproportional"] += 1

    # --- Print Summary ---
    print("\n" + "=" * 50)
    print(f"RESULTS FOR CASE {case}")
    print(f"Total Unique Cases: {len(final_results)}")
    print(f"Correct/Incorrect: {metrics['correct']}/{metrics['incorrect']}")
    print(f"Proportional/Unprop/Zero: {metrics['proportional']}/{metrics['unproportional']}/{metrics['zero']}")
    print(f"Avg Time per Test: {metrics['total_time'] / len(final_results):.4f}s")
    print("=" * 50)

    # --- Save Log ---
    with open(f"case_{case}_log.txt", "w") as f:
        f.write(f"Commutator Search Report - Case {case}\n")
        f.write("-" * 30 + "\n")
        for key, val in metrics.items():
            f.write(f"{key}: {val}\n")

        f.write("\nDetailed Unproportional Solutions:\n")
        for params, data in final_results.items():
            if not data["isProp"] and not data["isZero"]:
                f.write(f"Params {params} -> Found: {data['FOUND']}\n")

        f.write("\nDetailed Proportional Solutions:\n")
        for params, data in final_results.items():
            if data["isProp"] and not data["isZero"]:
                f.write(f"Params {params} -> Found: {data['FOUND']}\n")
else:
    worker([x, y])