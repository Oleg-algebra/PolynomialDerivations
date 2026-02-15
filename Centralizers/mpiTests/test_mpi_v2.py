import os
import time
import argparse
import sys
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from sympy import symbols, simplify

# Assuming your refactored class is in CommutatorSearchSymbolic.py
from commutatorSearchSymbolicV2 import Derivation, CommutatorFinder

# Performance tuning
sys.setrecursionlimit(10 ** 6)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# --- Argument Parsing (Root only) ---
if rank == 0:
    parser = argparse.ArgumentParser(description="MPI Commutator Search Test")
    parser.add_argument("--case", type=int, required=True, help="case number")
    parser.add_argument("--it", type=int, default=1, help="iteration number")
    args = parser.parse_args()
    case, total_tests = args.case, args.it
    print(f"Case: {case}, Total Iterations: {total_tests}")
else:
    case, total_tests = None, None

case = comm.bcast(case, root=0)
total_tests = comm.bcast(total_tests, root=0)

# Workload distribution
tests_per_rank = total_tests // size + (1 if rank < total_tests % size else 0)

# --- Configuration ---
from case_functions2 import get_parameters  # Import helper

limit_cfg = {"min_c": -15, "max_c": 15, "min_p": 0, "max_p": 15}
MAX_K_SEARCH = 5

# --- Metrics ---
results = {}
metrics = {
    "correct": 0, "incorrect": 0, "proportional": 0,
    "unproportional": 0, "zero": 0, "total_time": 0.0
}

x, y = symbols("x y")
variables = [x, y]

# --- Main Test Loop ---
with tqdm(total=tests_per_rank, desc=f"Rank {rank}", position=rank, leave=False) as pbar:
    counter = 0
    while counter < tests_per_rank:
        start_time = time.time()

        # 1. Parameter Generation
        l, k, n, m, alpha, beta = get_parameters(
            case, limit_cfg["min_p"], limit_cfg["max_p"],
            limit_cfg["min_c"], limit_cfg["max_c"]
        )

        if alpha ** 2 + beta ** 2 == 0 or (k, n, l, m, alpha, beta) in results:
            continue

        # 2. Setup Derivation
        monomials = [alpha * x ** k * y ** n, beta * x ** l * y ** m]
        given_der = Derivation(monomials, variables)

        # 3. Search for Commutator
        finder = CommutatorFinder(given_der, max_k=MAX_K_SEARCH)
        found_der, is_prop = finder.find_commutator()

        # 4. Validation
        # Check if [D_given, D_found] == 0
        is_correct = True
        for i in range(len(variables)):
            bracket = given_der.apply(found_der.polynomials[i]) - \
                      found_der.apply(given_der.polynomials[i])
            if not simplify(bracket).equals(0):
                is_correct = False
                break

        is_zero = found_der.is_zero()

        # 5. Store Result
        results[(k, n, l, m, alpha, beta)] = {
            "GIVEN": monomials,
            "FOUND": found_der.polynomials,
            "isCorrect": is_correct,
            "isProp": is_prop,
            "isZero": is_zero,
            "K_reached": finder.k,
            "time": time.time() - start_time
        }

        counter += 1
        pbar.update(1)

# --- MPI Result Gathering ---
comm.Barrier()
all_gathered = comm.gather(results, root=0)

if rank == 0:
    final_report = {}
    for d in all_gathered:
        final_report.update(d)

    # Process Stats
    for res in final_report.values():
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
    print(f"Total Unique Cases: {len(final_report)}")
    print(f"Correct/Incorrect: {metrics['correct']}/{metrics['incorrect']}")
    print(f"Proportional/Unprop/Zero: {metrics['proportional']}/{metrics['unproportional']}/{metrics['zero']}")
    print(f"Avg Time per Test: {metrics['total_time'] / len(final_report):.4f}s")
    print("=" * 50)

    # --- Save Log ---
    with open(f"case_{case}_log.txt", "w") as f:
        f.write(f"Commutator Search Report - Case {case}\n")
        f.write("-" * 30 + "\n")
        for key, val in metrics.items():
            f.write(f"{key}: {val}\n")

        f.write("\nDetailed Unproportional Solutions:\n")
        for params, data in final_report.items():
            if not data["isProp"] and not data["isZero"]:
                f.write(f"Params {params} -> Found: {data['FOUND']}\n")

        f.write("\nDetailed Proportional Solutions:\n")
        for params, data in final_report.items():
            if  data["isProp"] and not data["isZero"]:
                f.write(f"Params {params} -> Found: {data['FOUND']}\n")