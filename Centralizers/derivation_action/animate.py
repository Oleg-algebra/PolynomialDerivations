from sympy import symbols
from Centralizers.basicClasses.commutatorSearchSymbolicV2 import Derivation
from action_animation import create_multi_trajectory_animation
from gen_random_poly import generate_random_polynomials
from time import time

# Налаштування
x, y = symbols("x y")
vars = [x, y]

# Диференціювання (приклад)
D = Derivation([y, x], vars)

start_gen = time()
# 1. Генеруємо 10 випадкових многочленів
random_polys = generate_random_polynomials(count=10, max_n=3, max_m=3, variables=vars)
end_gen = time()
print(f"polynomial generation time: {end_gen - start_gen}")

start_anime = time()
# 2. Запускаємо анімацію еволюції
create_multi_trajectory_animation(
    derivation=D,
    start_polys=random_polys,
    steps=30,
    fps=5,
    filename="trajectories_cloud.mp4"
)
end_anime = time()
print(f"animation time: {end_anime - start_anime}")