from basicClasses.constantSearchSymbolic import *
from time import time
from tqdm import tqdm

variables_number = 2
isSearchNonZero = True
isZeroDerivation = False

startTime = time()
#
# parser = argparse.ArgumentParser(description="A simple script with command-line arguments.")
# parser.add_argument("--N", help="power")
# parser.add_argument("--K", help="max bias power")
#
# args = parser.parse_args()
# N = int(args.N)
# max_K = int(args.K)
N = 1

K = 0
max_K = 15
k = 0
n= N
l = N
m = 0


alpha = 1
beta = 1

powers1 = [k, n]
powers2 = [l, m]

x,y = symbols("x"),symbols("y")
variables = [x,y]

monomial1 = alpha*x**k*y**n
monomial2 = beta*x**l*y**m

monomial1 = -x*0
monomial2 = x**3*y

polynomial1 = Polynomial(poly_symbols=monomial1, variables=variables)
polynomial2 = Polynomial(poly_symbols=monomial2, variables=variables)

der = Derivation([polynomial1,polynomial2], polynomial1.variables_polynom)



file = open(f"report.txt","w")
file.write("Test results\n")
file.write("="*100 + "\n")
file.write(f"Given derivation: {[polynomial1.polynomial_symbolic,polynomial2.polynomial_symbolic]}\n")

with tqdm(total=max_K,desc=f"Case N = {N}",position = 0,leave=False,disable=False) as pbar:
    pbar.update(2)
    while K <= max_K:
        # print(f"K = {K}")
        commutatorPolynomials = []
        constanSearch = ConstantSearchSymbolic(der,[*powers1,*powers2],K)
        # print(f"Matrices size: {commutator.unknown_derivation.polynomials[0].coefficients.shape}")

        constant, isConstant = constanSearch.get_constant()

        file.write(f"K = {K} --- isConstant: {isConstant} --- constant: {constant}\n")

        K += 1
        pbar.update(1)

endTime = time()
totalTime = endTime - startTime
file.close()

print(f"Tests for N = {N} finished")
print(f"Total time: {totalTime}")
print(f"Variables: {polynomial1.variables_polynom}")
print("========Given derivation=======")
for i in range(len(der.polynomials)):
    print(f'poly {i}: {der.polynomials[i].polynomial_symbolic}')

print("==========Unknown derivation=======")

print(f"isConstant: {isConstant}")
print(f"FINAL CONSTANT SEARCH RESULT: {constant}")

