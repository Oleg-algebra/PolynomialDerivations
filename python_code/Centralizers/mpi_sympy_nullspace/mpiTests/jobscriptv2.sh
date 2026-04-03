#!/bin/bash

mpirun --mca btl_openlib_warn_no_device_params_found 0 --mca btl ^openib -n 12 python3 test_mpi_v2.py --case 777 --it 50
#play -n synth 1 sin 440
