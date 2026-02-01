from sympy import Matrix, solve_linear_system,symbols, diff, simplify, expand, Expr, solve, Poly, N, nsimplify
import numpy as np





class Polynomial:
    def __init__(self, coefficients: np.ndarray = None, n_var: int = None, poly_symbols = None, variables = None):

        self.polynomial_symbolic = 0
        self.coefficients = coefficients
        if poly_symbols is not None:
            self.polynomial_symbolic = poly_symbols
        self.variables_polynom = []
        if variables is not None:
            self.variables_polynom = variables
        else:
            self.variables_polynom = [symbols(f"x_{i}") for i in range(n_var)]
        if coefficients is not None:
            for i in range(coefficients.shape[0]):
                for j in range(coefficients.shape[1]):
                    self.polynomial_symbolic += coefficients[i,j]*self.variables_polynom[0]**i*self.variables_polynom[1]**j
        else:
            row_max = 0
            column_max = 0
            poly = Poly(poly_symbols, variables)
            for term in poly.terms():
                degree = term[0]
                if degree[0]> row_max:
                    row_max = degree[0]
                if degree[1]> column_max:
                    column_max = degree[1]
            self.coefficients = np.zeros((row_max+1,column_max+1))
            for term in poly.terms():
                degree = term[0]
                self.coefficients[degree[0]][degree[1]] = term[1]




class Derivation:
    def __init__(self, polynomials: list[Polynomial], variables):
        self.polynomials = polynomials
        self.variables = variables

    def take_derivative(self,expression: Expr ) -> Expr:
        res = 0
        for i in range(len(self.variables)):
            res += self.polynomials[i].polynomial_symbolic * diff(expression, self.variables[i])
        res = expand(res)
        return res


class Commutator:

    def __init__(self, derivation: Derivation, K=0, degreeStrategy: str = "special",matrixStrategy: str = "degreeStrategy",max_K = 10):
        self.derivation = derivation
        self.powers = self.powers = [*self.derivation.polynomials[0].coefficients.shape,
                       *self.derivation.polynomials[1].coefficients.shape]
        self.unknown_coeffients = []
        self.K = K
        self.max_K = max_K
        self.degreeStrategy = degreeStrategy
        self.unknown_derivation = None
        self.searchCommutator = {
            "general" : self.generalSolver,
            "linear" : self.linearSolver
        }


    def specialStrategy(self):
        flag1 = self.derivation.polynomials[0].polynomial_symbolic.equals(0)
        flag2 = self.derivation.polynomials[1].polynomial_symbolic.equals(0)
        N = abs(self.powers[0]*(not flag1)-self.powers[2]*(not flag2)) + self.K
        M = abs(self.powers[1]*(not flag1)-self.powers[3]*(not flag2)) + self.K
        return N,M

    def generalStrategy(self):
        flag1 = self.derivation.polynomials[0].polynomial_symbolic.equals(0)
        flag2 = self.derivation.polynomials[1].polynomial_symbolic.equals(0)
        N = max(self.powers[0]*(not flag1),self.powers[2]*(not flag2)) + self.K
        M = max(self.powers[1]*(not flag1),self.powers[3]*(not flag2)) + self.K
        return N,M

    def strategy1(self):
        return max(self.generalStrategy())


    def getDegree(self,strategy = "special"):
        strategies = {
            "special" : self.specialStrategy,
            "general" : self.generalStrategy,
            "degreeStrategy" : self.strategy1
        }
        return strategies[strategy]()


    def getMatrix(self,strategy = "degreeStrategy"):
        strategies = {
            "degreeStrategy" : self.getCoefficientsMatrixFixedDegree,
            "full" : self.getFullCoefficientsMatrix
        }

        return strategies[strategy]()

    """
    Generate full coefficients matrix
    """
    def getFullCoefficientsMatrix(self) -> Derivation:
        self.unknown_coeffients = []
        variables = self.derivation.variables
        N,M = self.getDegree(strategy=self.degreeStrategy)

        Matrices = []
        symb = ["a","b"]

        for k in range(len(variables)):
            sym = symb[k]
            matrix = []
            for i in range(N):
                row = []
                for j in range(M):
                    coef = symbols(f'{sym}{i}_{j}')
                    row.append(coef)
                    self.unknown_coeffients.append(coef)
                matrix.append(row)
            Matrices.append(Matrix(matrix))
        polynomials = []
        for m in Matrices:
            polynomials.append(Polynomial(m, len(variables), variables= self.derivation.variables))

        der_unknown = Derivation(polynomials,variables)

        return der_unknown

    """
        Generate coefficients matrix for polynomial of given degree
    """
    def getCoefficientsMatrixFixedDegree(self) -> Derivation:
        self.unknown_coeffients = []
        variables = self.derivation.variables
        N = self.getDegree(strategy="degreeStrategy")
        Matrices = []
        symb = ["a", "b"]

        for k in range(len(variables)):
            sym = symb[k]
            matrix = np.zeros((N,N),dtype=object)
            for i in range(N):
                # row = []
                for j in range(N-i):
                    coef = symbols(f'{sym}{i}_{j}')
                    # row.append(coef)
                    matrix[i][j] = coef
                    self.unknown_coeffients.append(coef)
                # matrix.append(row)
            Matrices.append(Matrix(matrix))
        polynomials = []
        for m in Matrices:
            polynomials.append(Polynomial(m, len(variables),variables= variables))

        der_unknown = Derivation(polynomials, variables)

        return der_unknown


    def generalSolver(self):

        # for poly in self.unknown_derivation.polynomials:
        #     print(f'unknown polynomial: {poly.polynomial_symbolic}')

        # print(self.unknown_coeffients)
        self.unknown_derivation = self.getCoefficientsMatrixFixedDegree()
        # self.unknown_derivation = self.getFullCoefficientsMatrix()
        derivatives1 = []
        for poly in self.unknown_derivation.polynomials:
            derivatives1.append(self.derivation.take_derivative(poly.polynomial_symbolic))

        derivatives2 = []
        for poly in self.derivation.polynomials:
            derivatives2.append(self.unknown_derivation.take_derivative(poly.polynomial_symbolic))

        polys = []
        for i in range(len(derivatives1)):
            polys.append(derivatives1[i]-derivatives2[i])



        equations = []
        variables = self.unknown_derivation.variables
        for poly in polys:
            p = Poly(poly,variables)
            for term in p.terms():
                equations.append(term[1])
        # print(f'equations: {equations}')
        res = solve(equations,self.unknown_coeffients)

        return res

    def linearSolver(self):
        # TODO: works well only with monomial derivations. For geneal case needs modification

        self.unknown_derivation = self.getCoefficientsMatrixFixedDegree()
        derivatives1 = []
        for poly in self.unknown_derivation.polynomials:
            derivatives1.append(self.derivation.take_derivative(poly.polynomial_symbolic))

        derivatives2 = []
        for poly in self.derivation.polynomials:
            derivatives2.append(self.unknown_derivation.take_derivative(poly.polynomial_symbolic))

        polys = []
        for i in range(len(derivatives1)):
            polys.append(derivatives1[i] - derivatives2[i])

        equations = []
        variables = self.unknown_derivation.variables

        terms = []
        for poly in polys:
            p = Poly(poly, variables)
            for coeff in p.terms():
                p1  = Poly(coeff[1],self.unknown_coeffients)
                terms.append(p1)

        for term in terms:
            # print(term)
            row = np.zeros((1,len(self.unknown_coeffients)))
            for t in term.terms():
                coeffs = np.array(t[0])
                scalar = float(N(t[1], chop=True))
                row = row + coeffs * scalar
            equations.append(row[0])

        # print(f'equations: {equations}')
        matrix = np.array(equations)
        # print(matrix)
        matrix = np.concatenate((matrix,np.zeros((matrix.shape[0],1))),axis=1)
        matrix = Matrix(matrix)
        res = solve_linear_system(matrix, *self.unknown_coeffients)

        return res

    def get_commutator(self,solver = "general"):

        while True:
            coefficients = self.searchCommutator[solver]()

            arbitrary_coefficients = []
            for coeff in self.unknown_coeffients:
                if coeff not in coefficients.keys():
                    arbitrary_coefficients.append(coeff)


            for poly in self.unknown_derivation.polynomials:
                for coeff in coefficients.keys():
                    new_symbolic_expr = poly.polynomial_symbolic.subs(coeff,coefficients[coeff])
                    poly.polynomial_symbolic = new_symbolic_expr

            for coeff in arbitrary_coefficients:
                number = np.random.randint(1, 10)
                for poly in self.unknown_derivation.polynomials:
                    new_symbolic_expr = poly.polynomial_symbolic.subs(coeff, number)
                    poly.polynomial_symbolic = nsimplify(new_symbolic_expr, rational=True)
                    # poly.polynomial_symbolic = new_symbolic_expr

            # is_proportional = self.is_proportional()

            if not self.is_zero_derivation() or self.K > self.max_K - 1:
                break
            self.K += 1

        is_proportional = self.is_proportional2()
        return self.unknown_derivation,is_proportional


    def is_zero_derivation(self):
        for poly in self.unknown_derivation.polynomials:
            if not poly.polynomial_symbolic.equals(0):
                return False
        return True

    def is_proportional(self):
        poly_unknown = self.unknown_derivation.polynomials
        poly_given = self.derivation.polynomials

        fractions = []



        for i in range(len(poly_unknown)):
            fraction = poly_unknown[i].polynomial_symbolic / poly_given[i].polynomial_symbolic
            fraction = simplify(fraction)
            fractions.append(fraction)
        # print(fractions)
        const = simplify(fractions[0]/fractions[0])
        for i in range(1,len(fractions)):
            check  = fractions[0].equals(fractions[i])
            if not check:
                return False

        return True

    def is_proportional2(self):
        #TODO: Needs to be rewritten, because of the case where both
        # coordinate functions are zero (for example functions
        # are zeros near d/dx in both derivations)

        poly_unknown = self.unknown_derivation.polynomials
        poly_given = self.derivation.polynomials

        prop = poly_unknown[0].polynomial_symbolic*poly_given[1].polynomial_symbolic-poly_unknown[1].polynomial_symbolic*poly_given[0].polynomial_symbolic
        prop = simplify(prop)
        # print(f"proportion: {prop}")
        return prop.equals(0)

    @staticmethod
    def isSolution(derivation1: Derivation,derivation2: Derivation) -> bool:
        polyDerivatives1 = []
        polyDerivatives2 = []

        for poly in derivation1.polynomials:
            der = derivation2.take_derivative(poly.polynomial_symbolic)
            polyDerivatives1.append(der)

        for poly in derivation2.polynomials:
            der = derivation1.take_derivative(poly.polynomial_symbolic)
            polyDerivatives2.append(der)

        for i in range(len(polyDerivatives1)):
            difference = polyDerivatives1[i] - polyDerivatives2[i]
            if not difference.equals(0):
                return False
        return True








