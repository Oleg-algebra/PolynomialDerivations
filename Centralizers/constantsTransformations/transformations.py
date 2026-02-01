from searchExample import *
from basicClasses.CommutatorSearchSymbolic import *
from basicClasses.constantSearchSymbolic import *
from sympy import Symbol

derivation,commutator,constant = getExample()

# x,y = Symbol('x'), Symbol('y')
# vars = [x,y]
# poly1 = Polynomial(poly_symbols=-56*x**3*y**7,vars =vars )
# poly2 = Polynomial(poly_symbols=21*x**2*y**8,vars = vars)
# derivation = Derivation([poly1,poly2],vars)
#
# constant = 9*x**3*y**8
#
# poly_comm1 = Polynomial(poly_symbols=-7*x,vars = vars)
# poly_comm2 = Polynomial(poly_symbols=2*y,vars = vars)
# commutator = Derivation([poly_comm1,poly_comm2],vars)

print(derivation.take_derivative(constant))
lie_bracket_result = lie_bracket(derivation1=derivation,derivation2=commutator)
for poly in lie_bracket_result.polynomials:
    print(poly.polynomial_symbolic)

counter = 0
print(f"{counter} -- {constant}")
while True and counter < 20:

    new_constant = commutator.take_derivative(constant)

    if new_constant.equals(0):
        print(f"{counter+1} -- {new_constant}")
        break
    print(f"{counter} -- {new_constant}")
    constant = new_constant
    counter+=1
