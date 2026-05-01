#!/bin/bash

## 1. Знімаємо обмеження на стек (запобігання Segfault 11 у giac::depth)
ulimit -s unlimited

## 2. Обмежуємо кількість арен пам'яті (запобігання malloc errors та віртуального роздуття)
#export MALLOC_ARENA_MAX=1
#
## 3. ВИМИКАЄМО ВНУТРІШНІЙ ПАРАЛЕЛІЗМ (Найважливіше для економії RAM)
## Це не дозволить кожному з 5 процесів MPI створювати власні під-потоки
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#
## 4. Додатковий контроль для Giac (якщо бібліотека їх зчитує)
export GIAC_NTHREADS=1

echo "Запуск MPI з обмеженням ресурсів для Case 777..."

# 5. Запуск із використанням jemalloc
# Використовуємо \ для розбиття довгого рядка для читабельності
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
mpirun --mca btl_openlib_warn_no_device_params_found 0 \
       --mca btl ^openib \
       --bind-to core --map-by core \
       -n 10 \
       python3 giac_with_mpi.py \
       --case 777 --it 1000

# Звуковий сигнал про завершення ( PhD-бонус )
# play -n synth 0.5 sin 880
#