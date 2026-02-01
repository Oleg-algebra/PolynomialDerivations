from time import time

from basicClasses.CommutatorSearchSymbolic import *

variables_number = 2
isSearchNonZero = True
isZeroDerivation = False

startTime = time()

# parser = argparse.ArgumentParser(description="A simple script with command-line arguments.")
# parser.add_argument("--N", help="power")
# parser.add_argument("--K", help="max bias power")
#
# args = parser.parse_args()
# N = int(args.N)
# max_K = int(args.K)
N = 1
max_K = 3
k = 0
n= N
l = N
m = 0

#powers of monomial by d/dx
k = 0
n = 1

#powers of monomial by d/dy
l = 1
m = 0

alpha = 1
beta = -1

powers1 = [k, n]
powers2 = [l, m]

x,y = symbols("x"),symbols("y")
variables = [x,y]

monomial1 = alpha*x**k*y**n
monomial2 = beta*x**l*y**m

polynomial1 = Polynomial(poly_symbols=monomial1, variables=variables)
polynomial2 = Polynomial(poly_symbols=monomial2, variables=variables)


# polynomial1 = 2*x**2+3*x*y -4*y**2
# polynomial2 = polynomial1 + 3*x

polynomial1 = 2*x*y
polynomial2 = 2*y**2

# polynomial1 = -2*x*y
# polynomial2 = y**2

polynomial1 = 0*x
polynomial2 = x**2*y**1

# polynomial1 = -x
# polynomial2 = y*0
#
# n = 1
# monom = (x*y)**n
# polynomial1 *= monom
# polynomial2 *= monom

polynomial1 = -y
polynomial2 = x

polynomial1 = Polynomial(poly_symbols=polynomial1, variables=variables)
polynomial2 = Polynomial(poly_symbols=polynomial2, variables=variables)

der = Derivation([polynomial1,polynomial2], polynomial1.variables_polynom)
K = 2
commutator = Commutator(der,K=0,max_K= 10,degreeStrategy="general")

commutatorPolynomials = []
# print(f"Matrices size: {commutator.unknown_derivation.polynomials[0].coefficients.shape}")

res, isProportional = commutator.get_commutator(solver = "general")

zeroCounter = 0
for i in range(len(res.polynomials)):
    commutatorPolynomials.append(simplify(res.polynomials[i].polynomial_symbolic))
    if res.polynomials[i].polynomial_symbolic.equals(0):
        zeroCounter += 1

if zeroCounter == 2:
    isZeroDerivation = True
else:
    isZeroDerivation = False



endTime = time()
totalTime = endTime - startTime

print(f"Total time: {totalTime}")
print(f"Variables: {polynomial1.variables_polynom}")
print("========Given derivation=======")
for i in range(len(der.polynomials)):
    print(f'poly {i}: {der.polynomials[i].polynomial_symbolic}')

print("==========Unknown derivation=======")

print(f"proportional: {isProportional}")
print(f"is Solution correct: {commutator.isSolution(derivation1=der,derivation2=res)}")
print("COMMUTATOR")
for i in range(len(res.polynomials)):
    print(f'poly {i}: {simplify(res.polynomials[i].polynomial_symbolic)}' )
print("="*100)

