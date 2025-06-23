#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
#  CUDA_test.py
#  
#  Copyright 2019 Zachary Dwight



#### Compare CPU vs GPU computation time w/ easy example (you can swap in your functions)
#### Helped me compare performance when legacy systems were required, different systems

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from timeit import default_timer as timer
from numba import vectorize


import os

os.environ['NUMBAPRO_NVVM'] = r'/home/user/anaconda2/lib/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = r'/home/user/anaconda2/lib/'


print 'CUDA Test'
# This should be a substantially high value. On my test machine, this took
# 33 seconds to run via the CPU and just over 3 seconds on older GPU.
NUM_ELEMENTS = 1000000
 
# This is the CPU version.
def vector_add_cpu(a, b):
  c = np.zeros(NUM_ELEMENTS, dtype=np.float32)
  for i in range(NUM_ELEMENTS):
    c[i] = a[i] * b[i] / 8 + b[i] - 2
  return c
 
# This is the GPU version. Note the @vectorize decorator. This tells
# numba to turn this into a GPU vectorized function.
@vectorize(["float32(float32, float32)"], target='cuda')
def vector_add_gpu(a, b):
  return a + b;
 
def main():
  a_source = np.ones(NUM_ELEMENTS, dtype=np.float32)
  b_source = np.ones(NUM_ELEMENTS, dtype=np.float32)
 
  # Time the CPU function
  start = timer()
  vector_add_cpu(a_source, b_source)
  vector_add_cpu_time = timer() - start
 
  # Time the GPU function
  start = timer()
  vector_add_gpu(a_source, b_source)
  vector_add_gpu_time = timer() - start
 
  # Report times
  print("CPU function took %f seconds." % vector_add_cpu_time)
  print("GPU function took %f seconds." % vector_add_gpu_time)
 
  return 0
 
if __name__ == "__main__":
  main()
