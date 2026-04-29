from giacpy import giac
from giacpy.giacpy import Pygen

from CommutatorSearchGiac import Derivation

def get_complexity(giac_obj):
    """Повертає кількість вузлів у дереві Giac-об'єкта"""
    from giacpy import giac
    # Ми використовуємо int(), бо giac() повертає об'єкт типу gen
    return int(giac(f"size({giac_obj})"))

x, y = giac('x, y')
print(get_complexity(x**2 - 1))

# Приклад: D = (x^2 - 1) d/dx + (y^2 - 4) d/dy
D = Derivation([x**2 - y,   x** 2 - y ], [x, y])
f_x: Pygen = x**2*y**4 +  x**3*y**2 - 1 - y - 4
deg = f_x.degree([x,y])
print(deg)


print(D)
points = D.find_critical_points()
print(f"Критичні точки: {points}")
print(f"Number of critical points: {D.count_critical_points()}")
print(f"Classification: {D.classify_critical_points()}")

print(points[0][0].has(x) )

from giacpy import giac

x, y, a = giac('x, y, a')
expr = x**2 + a
print()
print(expr.has(x))  # True
print(expr.has(y))  # False
print(expr.has(a))  # True (знайшов параметр)
res = giac('x').has(x) == 1
print(res)

# # Виведе: [[1,2], [1,-2], [-1,2], [-1,-2]]
#
# # Обчислимо матрицю Якобі
# J = D.get_jacobian()
# print(f"Матриця Якобі:\n{J}")
#
# # Перевірка стійкості в першій точці [1, 2]
# for point in points:
#
#     J_at_point = J.subst([x,y], point)
#     print(f"J at point {point} : {J_at_point}")
#     eigenvalues = J_at_point.eigenvalues()
#     print(f"Власні значення в точці {point}: {eigenvalues}")
#     print(f"Eigen vectors: {J_at_point.eigenvectors()}")
#     print("="*100)
