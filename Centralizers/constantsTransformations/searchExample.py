from basicClasses.CommutatorSearchSymbolic import *
from basicClasses.constantSearchSymbolic import *
from basicClasses.cases_functions import get_parameters


coeff_limit = 10
min_coeff = -coeff_limit
max_coeff = coeff_limit

max_power = 7
min_power = 0

def nonPropCase(min_power, max_power, min_coeff, max_coeff):
    while True:
        l = np.random.randint(0, max_power)
        k = l + 1
        n = np.random.randint(0, max_power)
        m = n + 1

        a = np.random.randint(min_coeff, max_coeff)
        alpha = -a * m
        beta = a * k

        if alpha == -beta:
            continue
        else:
            return l, k, n, m, alpha, beta

def lie_bracket(derivation1: Derivation, derivation2: Derivation):

    derivatives = []
    for i in range(len(derivation1.variables)):
        der1 = derivation1.take_derivative(derivation2.polynomials[i].polynomial_symbolic)
        der2 = derivation2.take_derivative(derivation1.polynomials[i].polynomial_symbolic)
        poly = Polynomial(poly_symbols=der1 - der2, variables= derivation1.variables)
        derivatives.append(poly)

    return Derivation(derivatives,derivation1.variables)

def get_nonProportional(alpha, beta):
    while True:
        a = np.random.randint(-alpha, alpha)
        b = np.random.randint(-beta, beta)
        if a*beta == b*alpha:
            continue
        return a,b

def getExample(case = 222, max_K = 20):


    l,k,n,m,alpha,beta = get_parameters(case,min_power,max_power,min_coeff,max_coeff)


    powers1 = [k, n]
    powers2 = [l, m]

    x, y = symbols("x"), symbols("y")
    variables = [x, y]

    monomial1 = alpha * x ** k * y ** n
    monomial2 = beta * x ** l * y ** m

    polynomial1 = Polynomial(poly_symbols=monomial1, variables=variables)
    polynomial2 = Polynomial(poly_symbols=monomial2, variables=variables)


    der = Derivation([polynomial1,polynomial2], polynomial1.variables_polynom)
    K = n*2+1
    commutatorSearch = Commutator(der, K)

    commutator, isProportional = commutatorSearch.get_commutator()
    constant = None
    isConstant = False

    K = 1
    while True and K < max_K:
        constantSearch = ConstantSearchSymbolic(der, [*powers1, *powers2], K,strategy="general")
        constant, isConstant = constantSearch.get_constant()
        if not constant.is_constant():
            break
        print("K = ", K)
        K += 1

    print("Given derivation: ")
    print(f"P(x,y): {der.polynomials[0].polynomial_symbolic}")
    print(f"Q(x,y): {der.polynomials[1].polynomial_symbolic}")
    print("=" * 100)
    print("Commutator:")
    print(f"isProportional: {isProportional}")
    print(f"isCommute: {Commutator.isSolution(der, commutator)}")
    print(f"P(x,y): {simplify(commutator.polynomials[0].polynomial_symbolic)}")
    print(f"Q(x,y): {simplify(commutator.polynomials[1].polynomial_symbolic)}")
    print("=" * 100)

    print("Constant of given derivation:")
    print(f"isConstant: {isConstant}")
    print(f"const: {constant}")

    return der,commutator,constant





