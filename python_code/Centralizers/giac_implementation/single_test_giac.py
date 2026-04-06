import time
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from case_functions2 import get_parameters


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


def worker_task(task_packet):
    with silence_giac():
        from giacpy import giac
        _ = giac('x')
        giac('print_time:=0')
        giac('verblevel:=0')
        giac('nthreads:=1')
        giac('timeout:=60')  # Збільшено для одного тесту

        from CommutatorSearchGiac import Derivation, FastCommutatorFinder

    params, variables_str, max_k = task_packet
    params = (9, 10, 0, 1, 15, -150)
    l, k, n, m, alpha, beta = params

    print(f"Executing task with params: {params}...")
    try:
        x, y = giac('x, y')
        vars_giac = [x, y]

        m1 = giac(alpha) * x ** k * y ** n
        m2 = giac(beta) * x ** l * y ** m
        monomials_giac = [m1, m2]

        start_time = time.time()

        given_der = Derivation(monomials_giac, vars_giac)

        # Вивід у воркері зазвичай пригнічений silence_giac,
        # тому ми покладаємося на те, що повернемо в Master
        finder = FastCommutatorFinder(given_der, max_k=max_k)
        all_solutions = finder.find_commutator()

        serialized_solutions = {}
        for sol_id, sol in all_solutions.items():
            clean_sol = {key: v for key, v in sol.items() if key != "derivation_solution"}
            der_obj: Derivation = sol["derivation_solution"]
            clean_sol["derivation_solution_sympy"] = der_obj.to_sympy()
            serialized_solutions[sol_id] = clean_sol

        return {
            "status": "success",
            "params": params,
            "GIVEN": given_der.to_sympy(),
            "FOUND": serialized_solutions,
            "time": time.time() - start_time
        }
    except Exception as e:
        return {
            "status": "error",
            "params": params,
            "message": str(e)
        }


def run_single_test():
    # --- Фіксовані параметри (замість argparse) ---
    case_number = 1  # Вкажіть потрібний номер кейсу
    max_k_value = 12  # Максимальна степінь
    waiting_time = 60  # Час очікування в секундах
    limit_cfg = {"min_power": 0, "max_power": 10, "min_coeff": -20, "max_coeff": 20}

    print(f"--- Starting Single Test for Case {case_number} ---")
    start_research = time.time()

    # Генеруємо параметри для одного тесту
    params = get_parameters(case_number, **limit_cfg)
    task_packet = (params, ['x', 'y'], max_k_value)

    # Використовуємо Pool з одним процесом для ізоляції Giac
    with mp.Pool(processes=1, maxtasksperchild=1) as pool:
        handle = pool.apply_async(worker_task, (task_packet,))

        try:

            res_data = handle.get(timeout=waiting_time)

            if res_data["status"] == "success":
                print("\n" + "=" * 50)
                print("SUCCESSFUL RESULT")
                print(f"Parameters: {res_data['params']}")
                print(f"Given Derivation: {res_data['GIVEN']}")
                print(f"Time taken: {res_data['time']:.4f}s")
                print(f"Found {len(res_data['FOUND'])} solutions.")

                # Детальний вивід знайдених розв'язків
                for i, (sol_id, sol) in enumerate(res_data['FOUND'].items()):
                    print(f"\nSolution {i + 1} (ID: {sol_id}):")
                    print(f"  Valid: {sol.get('is_valid')}")
                    print(f"  Proportional: {sol.get('is_proportional')}")
                    print(f"  Derivation: {sol['derivation_solution_sympy']}")
                print("=" * 50)
            else:
                print(f"Worker Error: {res_data.get('message')}")

        except mp.TimeoutError:
            print(f"CRITICAL: Test timed out after {waiting_time}s")
        except Exception as e:
            print(f"CRITICAL: System crash: {e}")

    end_research = time.time()
    print(f"\nTotal execution time: {end_research - start_research:.2f}s")


if __name__ == "__main__":
    run_single_test()