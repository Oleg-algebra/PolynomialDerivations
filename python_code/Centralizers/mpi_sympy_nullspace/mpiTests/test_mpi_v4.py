import time
import argparse


from mpi4py import MPI
from case_functions2 import get_parameters


# Використовуємо новий швидкий клас
# (Впевніться, що перейменували або оновили файл)


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


def worker():
    from CommutatorSearchGiac import Derivation, FastCommutatorFinder
    from giacpy import giac
    x, y = giac('x, y')
    variables = [x,y]
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
        finder = FastCommutatorFinder(given_der, max_k=10)
        all_solutions = finder.find_commutator()

        for sol in all_solutions.values():
            giac_obj: Derivation = sol["derivation_solution"]
            polynomials = [str(p) for p in giac_obj.polynomials]
            variables_str = [str(v) for v in giac_obj.variables]
            sol["derivation_solution"] = polynomials

        # Відправка результату назад
        result_payload = {
            "params": params,
            "GIVEN": [str(monomial) for monomial in monomials],
            "FOUND": all_solutions,
            "time": time.time() - start_time
        }
        comm.send(result_payload, dest=0, tag=TAG_RESULT)


# --- Точка входу ---
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
    for sol in final_results.values():
        metrics["total_time"] += sol["time"]
        for res in sol["FOUND"].values():
            if not res["is_valid"]:
                metrics["incorrect"] += 1
            else:
                metrics["correct"] += 1

            if res["is_proportional"]:
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
            for res in data["FOUND"].values():
                if not data["is_proportional"]:
                    f.write(f"Params {params} -> Found: {data['FOUND']}\n")

        f.write("\nDetailed Proportional Solutions:\n")
        for params, data in final_results.items():
            for res in data["FOUND"].values():
                if data["is_proportional"]:
                    f.write(f"Params {params} -> Found: {data['FOUND']}\n")
else:
    worker()