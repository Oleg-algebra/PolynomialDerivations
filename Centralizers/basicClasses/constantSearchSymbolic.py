
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


class ConstantSearchSymbolic:

    def __init__(self, derivation: Derivation,powers: list,K = 1, strategy: str = "special"):
        self.derivation = derivation
        self.powers = powers
        self.unknown_coeffients = []
        self.K = K
        self.strategy = strategy
        self.unknown_constant = None
        self.searchCommutator = {
            "general" : self.generalSolver
        }


    def specialStrategy(self):
        # N = max(abs(self.powers[0]-self.powers[2]),abs(self.powers[1]-self.powers[3])) + self.K
        flag1 = self.derivation.polynomials[0].polynomial_symbolic.equals(0)
        flag2 = self.derivation.polynomials[1].polynomial_symbolic.equals(0)
        N = abs(self.powers[0]*(not flag1)-self.powers[2]*(not flag2)) + self.K
        M = abs(self.powers[1]*(not flag1)-self.powers[3]*(not flag2)) + self.K
        return N,M

    def generalStrategy(self):
        # N = max(self.powers[0],self.powers[1],self.powers[2],self.powers[3]) + self.K
        flag1 = self.derivation.polynomials[0].polynomial_symbolic.equals(0)
        flag2 = self.derivation.polynomials[1].polynomial_symbolic.equals(0)
        N = max(self.powers[0]*(not flag1),self.powers[2]*(not flag2)) + self.K
        M = max(self.powers[1]*(not flag1),self.powers[3]*(not flag2)) + self.K
        return N,M

    def getDegree(self,strategy = "special"):
        strategies = {
            "special" : self.specialStrategy,
            "general" : self.generalStrategy
        }
        return strategies[strategy]()

    def generatePolynomial(self) -> Polynomial:
        self.unknown_coeffients = []
        variables = self.derivation.variables
        N,M = self.getDegree(strategy=self.strategy)

        sym = "a"
        matrix = []
        for i in range(N+1):
            row = []
            for j in range(M+1):
                coef = symbols(f'{sym}{i}_{j}')
                row.append(coef)
                self.unknown_coeffients.append(coef)
            matrix.append(row)
        matrix = Matrix(matrix)
        return Polynomial(matrix,len(variables),variables=variables)

    def generalSolver(self):


        derivative = self.derivation.take_derivative(self.unknown_constant.polynomial_symbolic)


        equations = []
        variables = self.unknown_constant.variables_polynom

        poly = Poly(derivative,variables)
        for term in poly.terms():
            equations.append(term[1])
        # print(f'equations: {equations}')
        res = solve(equations,self.unknown_coeffients)

        return res



    def get_constant(self, solver ="general"):


        while True:
            self.unknown_constant = self.generatePolynomial()
            coefficients = self.searchCommutator[solver]()

            if coefficients == []:
                print(self.K)
                self.K += 1
                continue

            arbitrary_coefficients = []
            for coeff in self.unknown_coeffients:
                if coeff not in coefficients.keys():
                    arbitrary_coefficients.append(coeff)



            for coeff in coefficients.keys():
                new_symbolic_expr = self.unknown_constant.polynomial_symbolic.subs(coeff,coefficients[coeff])
                self.unknown_constant.polynomial_symbolic = new_symbolic_expr



            for coeff in arbitrary_coefficients:
                number = np.random.randint(1,10)
                # number = 1
                new_symbolic_expr = self.unknown_constant.polynomial_symbolic.subs(coeff,number)
                self.unknown_constant.polynomial_symbolic = nsimplify(new_symbolic_expr,rational=True)
                # poly.polynomial_symbolic = new_symbolic_expr

            # is_proportional = self.is_proportional()
            is_constant = self.is_constant()
            return self.unknown_constant.polynomial_symbolic,is_constant




    def is_constant(self):
        derivative = self.derivation.take_derivative(self.unknown_constant.polynomial_symbolic)

        return derivative.equals(0)











