from typing import List

import giacpy
from giacpy import giac, linsolve,scalar_product
from giacpy.giacpy import Pygen
from CommutatorSearchGiac import Derivation, FastCommutatorFinder

x, y = giac('x, y')
print(type(x))

def create_poly_simple(degree):
    # 1. Оголошуємо основні змінні


    poly = 0
    coeffs = []
    k = 0  # Лічильник для імен коефіцієнтів

    # 2. Вкладені цикли для формування всіх комбінацій x^i * y^j
    # Умова i + j <= degree гарантує, що загальний ступінь моному не перевищить d
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            # Створюємо назву коефіцієнта (c0, c1, c2...)
            c_name = f"c{k}"
            c_val = giac(c_name)

            # Формуємо доданок: c_k * x^i * y^j
            term = c_val * (x ** i) * (y ** j)

            # Додаємо до загального полінома
            poly += term

            # Зберігаємо коефіцієнт у список для подальшого розв'язання системи
            coeffs.append(c_val)
            k += 1

    # .rat() перетворює результат у внутрішню раціональну форму для швидкості
    return poly, coeffs


# ПРИКЛАД ВИКОРИСТАННЯ:
d = 2
P: Pygen
all_c: List[Pygen]
P, all_c = create_poly_simple(d)

print(f"Створено поліном ступеня {d}:")
print(P)
print(f"\nСписок коефіцієнтів: {all_c}")



# Визначаємо компоненти деривації D = f_x * d/dx + f_y * d/dy
# Наприклад: D = (x+y)*d/dx + (x*y)*d/dy
f_x = y
f_y = x

der = Derivation([f_x,f_y],[x,y])
res = der.apply(P)
eqs = res.coeffs([x,y])
print(eqs)
solution = linsolve(eqs,all_c)
print(solution)
P_new = P.subst(all_c,solution)

print(P_new)
print(P_new.normal() == 0)


