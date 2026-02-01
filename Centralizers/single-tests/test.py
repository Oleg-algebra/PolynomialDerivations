from time import time

from basicClasses.CommutatorSearchSymbolic import *

x,y = symbols("x"),symbols("y")
variables = [x,y]
polynomial = 2*x**2+3*x*y -4*y**2
polynomial2 = Polynomial(poly_symbols=polynomial, variables=variables)

print(polynomial2.coefficients)