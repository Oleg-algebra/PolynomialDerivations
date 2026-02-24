from time import time
from sympy import symbols, simplify
# Ensure your refactored classes are in CommutatorSearchSymbolic.py
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import  Derivation, CommutatorFinder

# 1. Setup variables and input polynomials
x, y = symbols("x y")
variables = [x, y]

# You can define your polynomials directly as SymPy expressions
# Example: The rotation derivation (-y*d/dx + x*d/dy)
poly1 = y
poly2 = x

startTime = time()

# 2. Initialize Derivation and Finder
# We pass the list of polynomials directly
der = Derivation([poly1, poly2], variables)

# Use the new Finder class which handles the degree strategies internally
finder = CommutatorFinder(der, max_k=10)

# 3. Execute Search
# The find_commutator method returns the resulting derivation and proportionality flag
res_der, is_proportional = finder.find_commutator()

endTime = time()

# 4. Output Results
print(f"Total time: {endTime - startTime:.4f}s")
print(f"Variables: {variables}")

print("\n======== Given Derivation =========")
for i, p in enumerate(der.polynomials):
    print(f"Component {i}: {p}")

print("\n======== Found Commutator =========")
print(f"Proportional: {is_proportional}")

# Use the built-in logic to check if the Lie Bracket [D1, D2] is zero
is_correct = True
for i in range(len(variables)):
    bracket = der.apply(res_der.polynomials[i]) - res_der.apply(der.polynomials[i])
    if not simplify(bracket).equals(0):
        is_correct = False
        break

print(f"Is Solution Correct: {is_correct}")
print(f"Is Zero Derivation: {res_der.is_zero()}")

print("\nCOMMUTATOR POLYNOMIALS:")
for i, p in enumerate(res_der.polynomials):
    print(f"Poly {i}: {p}")
print("="*60)