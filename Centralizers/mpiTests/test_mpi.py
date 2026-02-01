
import os
import time
from mpi4py import MPI
from tqdm import tqdm
import sys
# from cases_functions import get_parameters
from case_functions2 import get_parameters
import os
import argparse

from CommutatorSearchSymbolic import *



sys.setrecursionlimit(10**6)





comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

comm.Barrier()
if rank == 0:
    parser = argparse.ArgumentParser(description="A simple script with command-line arguments.")
    parser.add_argument("--case", help="case number")
    parser.add_argument("--it", type=int, default=1, help="iteration number")

    args = parser.parse_args()
    print(f"case: {args.case}, iterations: {args.it}")
    case = int(args.case)
    total_tests_number = int(args.it)
    # case = int(input("Enter the case number: "))
    # total_tests_number = int(input("Enter the total number of tests: "))
else:
    total_tests_number = None
    case = None
case = comm.bcast(case, root=0)
total_tests_number = comm.bcast(total_tests_number, root=0)
comm.Barrier()

tests_number = total_tests_number // size + 2

coeff_limit = 15
min_coeff = -coeff_limit
max_coeff = coeff_limit

max_power = 15
min_power = 0

K = 2
max_K = 5

variables_number = 2
proportionalCounter = 0
unproportionalCounter = 0
zeroDerivationCounter = 0
correct_answers_counter = 0
incorrect_answers_counter = 0

results = {}
givenDerivationKEY = "GIVEN_DERIVATION"
commutatorKEY = "COMMUTATOR"
isProportionalKEY = "isProportional"
isZeroDerivationKEY = "IsZeroDerivation"
matrixDimension  = "matrixDimension"
isSolutionCorrectKey = "isSolutionCorrect"

correctAnswersNumberKEY = "correctAnswersNumber"
falseAnswersNumberKey = "falseAnswersNumber"
proportionalKEY = "proportionalCounter"
unproportionalKEY = "unproportionalCounter"
zeroDerivaionsKEY = "zeroDerivaionsCounter"
time_exec_KEY = "time_elapsed"

keysForReport = [givenDerivationKEY,isSolutionCorrectKey,matrixDimension,commutatorKEY]
# keysForReport = [givenDerivationKEY,commutatorKEY,isProportionalKEY,isSolutionCorrectKey]

isShortReport = True
isSearchNonZero = True

s = 0

l = 0
k = 0
n = 0
m = 0

alpha = 0
beta = 0

counter = 0
strategy = "special"
# print(f'rank: {rank} started testing')
with tqdm(total=tests_number,desc=f"Rank: {rank}",position=rank,leave=False,disable=False) as pbar:
    while counter < tests_number:

        start = time.time()

        # print(f"rank {rank} ---> Testing {str(counter+1) + "/" + str(tests_number)}" )
        result = {}

        #   Determine parameters
        #########################################################################
        if K == 2:


            l,k,n,m,alpha,beta = get_parameters(case, min_power, max_power, min_coeff, max_coeff)
            # n = 0
            # l = 0
            # k = np.random.randint(1,15)
            # m = np.random.randint(1,15)
            # alpha = beta = 1

            # if alpha*beta == 0:
            #     continue
            #
            # if m < 2 :
            #     continue

            # if (k + n) != (l + m):
            #     continue

            if alpha**2 + beta**2 == 0:
                continue
        else:
            # print("--> increased K by 1")
            pass
        ############################################################################

        powers1 = [k,n]
        powers2 = [l, m]


        if (k,n,l,m,alpha,beta) in results.keys():
            continue

        x, y = symbols("x"), symbols("y")
        variables = [x, y]

        monomial1 = alpha * x ** k * y ** n
        monomial2 = beta * x ** l * y ** m

        polynomial1 = Polynomial(poly_symbols=monomial1, variables=variables)
        polynomial2 = Polynomial(poly_symbols=monomial2, variables=variables)

        result[givenDerivationKEY] = [polynomial1.polynomial_symbolic, polynomial2.polynomial_symbolic]

        der = Derivation([polynomial1,polynomial2],variables)

        commutator = Commutator(der,K,degreeStrategy=strategy)

        result[isZeroDerivationKEY] = False

        res, isProportional = commutator.get_commutator()
        result[matrixDimension] = commutator.unknown_derivation.polynomials[0].coefficients.shape
        commutatorPolynomials = []

        # if isProportional and K < max_K:
        #     K+=1
        #     continue

        if commutator.isSolution(der,res):
            result[isSolutionCorrectKey] = True
        else:
            result[isSolutionCorrectKey] = False

        result[isProportionalKEY] = isProportional

        zeroCounter = 0
        for i in range(len(res.polynomials)):
            polynom = res.polynomials[i].polynomial_symbolic
            if isProportional:
                polynom = simplify(res.polynomials[i].polynomial_symbolic)
            commutatorPolynomials.append(polynom)

            if res.polynomials[i].polynomial_symbolic.equals(0):
                zeroCounter += 1
        if zeroCounter == variables_number:
            if isSearchNonZero:
                if K < max_K:
                    # print("Increase K")
                    K += 1
                    continue
                else:
                    # print("Max K reached")
                    if strategy == "special":
                        # print("change strategy")
                        strategy = "general"
                        K = 2
                        continue
            result[isZeroDerivationKEY] = True

        else:

            result[isZeroDerivationKEY] = False

        result["K"] = K
        result[commutatorKEY] = commutatorPolynomials[:]

        end = time.time()
        time_elapsed = end - start
        result[time_exec_KEY] = time_elapsed

        s+=time_elapsed
        results[(k, n, l, m, alpha, beta)] = result

        strategy = "special"
        K = 2
        counter += 1
        pbar.update(1)


comm.Barrier()
results_container = comm.gather(results, root=0)
comm.Barrier()
if rank == 0:

    average_K = 0
    max_K = 2

    total_time = 0
    all_results = {}
    for dct in results_container:
        for key in dct.keys():
            all_results[key] = dct[key]

    for res in all_results.values():

        average_K += res["K"]
        if res["K"] > max_K:
            max_K = res["K"]

        if res[isZeroDerivationKEY]:
            zeroDerivationCounter += 1
        if res[isProportionalKEY] and not res[isZeroDerivationKEY]:
            proportionalCounter += 1
        if not res[isProportionalKEY]:
            unproportionalCounter += 1
        if res[isSolutionCorrectKey]:
            correct_answers_counter += 1
        else:
            incorrect_answers_counter += 1
        total_time += res[time_exec_KEY]

    average_K = average_K / len(all_results.keys())
    average_time_per_process = total_time / size

    os.system("clear")

    print("\n"+"="*100)
    print(f"Total number of different cases: {len(all_results.keys())}")
    print(f"rank: {rank} gathering results")
    # print(results_container)

    print(f'proportional: {proportionalCounter}')
    print(f'unproportional: {unproportionalCounter}')
    print(f'zeroDerivations: {zeroDerivationCounter}')
    print(f"correct answers number: {correct_answers_counter}")
    print(f"incorrect answers number: {incorrect_answers_counter}")
    print(f"Average K: {average_K}")
    print(f"Max K: {max_K}")
    print("Average time per process: ", average_time_per_process)
    print("average time per test: ", total_time / (tests_number*size))
    print("="*100)

    fileName = f"case_{case}"
    file = open(fileName+"_log.txt", "w")

    file.write("Report of testing\n")
    file.write("======================General information===================\n")
    file.write(f"Total number of different cases: {len(all_results.keys())}\n")
    file.write(f"proportional: {proportionalCounter}\n")
    file.write(f"unproportional: {unproportionalCounter}\n")
    file.write(f"zeroDerivations: {zeroDerivationCounter}\n")
    file.write(f"correct answers number: {correct_answers_counter}\n")
    file.write(f"incorrect answers number: {incorrect_answers_counter}\n")
    file.write(f"Average K: {average_K}\n")
    file.write(f"Max K: {max_K}\n")
    file.write(f"Average time per process: {average_time_per_process}\n")
    file.write(f"average time per test: {total_time / (tests_number*size)}\n")
    file.write("======================Special cases=========================\n")

    file.write("=======================Incorrect answers=========================\n")
    count = 1
    for param, res in all_results.items():
        if not res[isSolutionCorrectKey]:
            if isShortReport:
                file.write(f"{count}):  {param}: ")
                for key in keysForReport:
                    file.write(f" {res[key]} |")
                file.write("\n")
            else:
                file.write(f"{count}):  {param}: {res}\n")
            count += 1

    file.write("=======================Zero derivations=========================\n")
    count = 1
    for param, res in all_results.items():
        if res[isZeroDerivationKEY]:
            if isShortReport:
                file.write(f"{count}):  {param}: ")
                for key in keysForReport:
                    file.write(f" {key} : {res[key]} |-----|")
                file.write("\n")
            else:
                file.write(f"{count}):  {param}: {res}\n")
            count += 1

    file.write("=======================Unproportional derivations=========================\n")
    count = 1
    for param, res in all_results.items():
        if not res[isProportionalKEY]:
            if isShortReport:
                file.write(f"{count}):  {param}: ")
                for key in keysForReport:
                    file.write(f" {key} : {res[key]} |-----|")
                file.write("\n")
            else:
                file.write(f"{count}):  {param}: {res}\n")
            count += 1

    file.write("=====================Proportional derivations==================\n")
    count = 1
    for param, res in all_results.items():

        if res[isProportionalKEY] and not res[isZeroDerivationKEY]:
            if isShortReport:
                file.write(f"{count}):  {param}: ")
                for key in keysForReport:
                    file.write(f" {key} : {res[key]} |-----|")
                file.write("\n")
            else:
                file.write(f"{count}):  {param}: {res}\n")
            count += 1


    file.write("=======================Correct answers=========================\n")
    count = 1
    for param, res in all_results.items():
        if res[isSolutionCorrectKey]:
            if isShortReport:
                file.write(f"{count}):  {param}: ")
                for key in keysForReport:
                    file.write(f" {key} : {res[key]} |-----|")
                file.write("\n")
            else:
                file.write(f"{count}):  {param}: {res}\n")
            count += 1




    file.write("==================END of REPORT=======================\n")
    file.close()



