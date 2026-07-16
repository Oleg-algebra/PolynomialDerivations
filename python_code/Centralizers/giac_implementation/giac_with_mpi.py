import gc
import time
import os
import argparse
import sys
import multiprocessing
from giacpy.giacpy import Pygen
from mpi4py import MPI
from contextlib import contextmanager

import json
import datetime
import uuid
from typing import Any, Tuple, List

from poly_tools import hash_polynomialPygen
from sympy.core.cache import clear_cache

from case_functions2 import identical_polynomials, get_monomials

os.environ["SYMPY_USE_CACHE"] = "no"

polynomial_cases = {
    "identical_polynomials" : identical_polynomials
}

def get_polynomials_list(
        variables = None,
        is_monomial_case = False,
        case_id: str = "111",
) -> List[Pygen]:

    if variables == None:
        raise ValueError("Polynomial variables is not defined. Defined variables.")

    coeff_majorant = 20
    limit_cfg = {"min_power": 0,
                 "max_power": 5,
                 "min_coeff": -coeff_majorant,
                 "max_coeff": coeff_majorant}

    if is_monomial_case:
        listPygen = get_monomials(int(case_id),
                                  **limit_cfg,
                                  vars=variables)
    else:
        if not case_id in polynomial_cases:
            raise KeyError(f"Key {case_id} is not in polynomial_cases")
        listPygen = polynomial_cases[case_id](zero_percentage=0.0,
                                       vars=variables,
                                       **limit_cfg)

    return listPygen


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


def append_to_research_log(result: dict,
                           filename: str = "results_log.jsonl",
                           directory: str = "logs/"):
    """
    Додає один результат обчислення в файл.
    """
    # Збагачуємо дані метаінформацією
    payload = {
        # "id": str(uuid.uuid4())[:8],
        # "timestamp": datetime.datetime.now().isoformat(),
        "data": serialize_research_data(result),
        # "data": result
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


def run_commutator_isolated(given_der_sympy) -> dict:
    """
    Ця функція виконується в окремому ізольованому процесі.
    Вона завантажує giacpy, робить обчислення і повністю завершується,
    гарантуючи 100% вивільнення C++ оперативної пам'яті.
    """
    with silence_giac():
        from giacpy import giac
        # Ініціалізація локального ядра
        _ = giac('x')
        giac('nthreads:=1')
        giac('print_time:=0')
        giac('threads_allowed:=0')

        # Виконуємо обчислення
        given_der = given_der_sympy.from_sympy()
        all_solutions, is_proportional = given_der.find_commutator()

        found_dict = {}
        for s_id, sol in all_solutions.items():
            der_obj = sol["derivation_solution"]
            found_dict[str(s_id)] = {
                "is_proportional": bool(sol["is_proportional"]),
                "commuting_derivative": der_obj.to_sympy(),
                "system_dim": sol["system_dim"]
            }

        result = {
            "status": "success",
            "RANK": 1 if is_proportional else 2,
            "CENTRALIZER": found_dict
        }

        # Очищення перед виходом з процесу
        giac('restart')
        return result

def worker():

    while True:
        status = MPI.Status()
        # Чекаємо на завдання від Master
        given_der_sympy: 'Derivation' = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        # Формуємо JSON-сумісний словник
        found_dict = {}

        if status.Get_tag() == TAG_STOP:
            break

        start_t = time.time()
        try:
            # Запускаємо обчислення в ізольованому пулі з 1 процесу
            # max_tasks_per_child=1 гарантує, що процес буде вбито після виконання
            with multiprocessing.Pool(processes=1, maxtasksperchild=1) as pool:
                async_res = pool.apply_async(run_commutator_isolated, (given_der_sympy,))
                # Встановлюємо таймаут на випадок зависання Giac
                calc_res = async_res.get(timeout=50)

            result_payload = {
                "status": "success",
                "params": given_der_sympy.polynomials,
                "hash": hash_polynomialPygen(given_der_sympy.polynomials),
                "GIVEN": given_der_sympy,
                "RANK": calc_res["RANK"],
                "CENTRALIZER": calc_res["CENTRALIZER"],
                "time": time.time() - start_t
            }
        except Exception as e:
            print(f"[FOUND ERROR] {e}")
            result_payload = {"status": "error", "message": str(e), "params": given_der_sympy.polynomials}
            print(result_payload)

        # Відправляємо JSON-рядок (це найбезпечніше)
        comm.send(result_payload, dest=0, tag=TAG_RESULT)


        # !!! ПОВНЕ ЗВІЛЬНЕННЯ ВСІХ ЛОКАЛЬНИХ ПОСИЛАНЬ В РОБОЧОМУ ЦИКЛІ ВОРКЕРА !!!
        given_der = None
        given_der_sympy = None
        all_solutions = None
        found_dict = None
        result_payload = None

        gc.collect()
        clear_cache()


def master(total_it = 100,
           case_id:  str = "111",
           is_monomial_case: bool = False
           ):

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

    def is_already_computed(poly_list: List[Pygen], existing_hashes):
        current_hash = hash_polynomialPygen(poly_list)
        return current_hash in existing_hashes, current_hash

    success_count = 0
    tests_sent = 0
    tests_received = 0

    processed_hashes = get_existing_hashes()
    print(f"[*] Завантажено {len(processed_hashes)} існуючих результатів.")

    x, y = giac('x, y')
    variables = [x, y]

    # Роздаємо перші завдання
    for worker_id in range(1, size):
        if tests_sent < total_it:

            # listPygen = get_monomials(case_id, **limit_cfg,vars=variables)
            listPygen = get_polynomials_list(variables = variables,
                                             case_id= case_id,
                                             is_monomial_case = is_monomial_case)
            # Конвертуємо у SymPy одразу, щоб очистити Giac-версію
            der_giac = Derivation(listPygen, variables)
            params = der_giac.to_sympy()

            comm.send(params, dest=worker_id, tag=TAG_JOB)
            tests_sent += 1

            # !!! ОЧИЩЕННЯ ПІСЛЯ ПЕРШОЇ ВІДПРАВКИ !!!
            der_giac = None
            listPygen = None
            params = None

            gc.collect()

    # Збираємо результати
    while tests_received < total_it:
        print(f"Tests received: {tests_received}/{total_it}")
        status = MPI.Status()

        res_data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)

        worker_id = status.Get_source()
        tests_received += 1
        current_hash = res_data.get('hash')

        if res_data["status"] == "success":
            success_count += 1

            if current_hash in processed_hashes:
                print(f"[!] Дублікат пропущено: {current_hash}")
                # raise RuntimeError("[DUPLICATE!!!]")
            else:
                append_to_research_log(res_data)
                processed_hashes.add(current_hash)
                print(f"[+] Новий результат збережено: {current_hash}")

            print(f"[{tests_received}/{total_it}] Success from Worker {worker_id}")
        else:
            append_to_research_log(res_data,"errors_log.jsonl")

            print(res_data)
            print(f"[{tests_received}/{total_it}] Failed. Resending...")
            # Якщо треба — можна тут зменшити tests_received і переслати завдання

        # Відправляємо нове завдання або стоп-сигнал
        if tests_sent < total_it:
            while True:

                # listPygen = get_monomials(case_id, **limit_cfg, vars=variables)
                listPygen = get_polynomials_list(
                                            variables = variables,
                                            case_id= case_id,
                                            is_monomial_case = is_monomial_case)
                # Перевіряємо за чистим списком поліномів
                is_already_exists, h = is_already_computed(listPygen, processed_hashes)

                if not is_already_exists:
                    der_giac = Derivation(listPygen, variables)
                    params = der_giac.to_sympy()
                    print("FOUND A NEW SET OF PARAMETERS")
                    break
                print("ALREADY EXISTS")

                # Запобігаємо витоку при частих дублікатах в Master-процесі
                with silence_giac():
                    giac('restart')
                    giac('purge()')
                    x, y = giac('x, y')
                    variables = [x, y]
                gc.collect()

            comm.send(params, dest=worker_id, tag=TAG_JOB)
            tests_sent += 1

            # !!! КРИТИЧНЕ ОЧИЩЕННЯ ПІСЛЯ КОЖНОЇ УСПІШНОЇ ВІДПРАВКИ В MASTER !!!
            der_giac = None
            params = None
            listPygen = None
            res_data = None

            with silence_giac():
                giac('restart')
                giac('purge()')
                x, y = giac('x, y')
                variables = [x, y]

            gc.collect()
            clear_cache()
        else:
            # Якщо завдань більше немає, надсилаємо сигнал зупинки і занулюємо результат
            comm.send(None, dest=worker_id, tag=TAG_STOP)
            res_data = None
            gc.collect()

    return success_count

def str2bool(v):
    """Конвертує текстове значення аргументу в логічний тип bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False).')

if __name__ == "__main__":
    if rank == 0:
        start_research_time = time.time()

        parser = argparse.ArgumentParser(description="MPI Commutator Search Test")
        parser.add_argument("--case", type=str, required=True, help="case_id")
        parser.add_argument("--it", type=int, default=1, help="iteration_number")

        # Використовуємо наш кастомний тип str2bool
        parser.add_argument("--is-monomial", type=str2bool, default=False,
                            help="Use monomials (True/False)")

        args = parser.parse_args()
        case, total_tests, is_monomial_case = args.case, args.it, args.is_monomial
        print(f"Case: {case}, Total Iterations: {total_tests}")
        # Тепер args.is_monomial — це чистий Python bool (True або False)
        print(f"Is Monomial mode active: {is_monomial_case}")
        # Вкажіть параметри тут прямо
        start = time.time()
        success_runs = master(total_it=total_tests,
                              case_id=case,
                              is_monomial_case = is_monomial_case)
        end = time.time()
        print(f"Done! Collected {success_runs} tests.")
        print(f"Total time: {end - start}")
        print("=======ALL DONE=======")
        exit(0)
    else:
        worker()
