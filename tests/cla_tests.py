from __future__ import print_function

import numpy as np
import pyopencl as cl
from pyopencl import elementwise

from tests.timeit import timeit_repeat

done = False
reps = 256
n = 2 ** 20
a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)


@timeit_repeat(reps)
def test_np():
    a = (np.exp(a_np) + np.exp(b_np))


test_np()
for platform in cl.get_platforms()[1:]:
    print("Platform:", platform)
    for device in platform.get_devices():
        print("Device:", device)

        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        a_g = cl.array.to_device(queue, a_np)
        b_g = cl.array.to_device(queue, b_np)

        results = cl.array.empty_like(a_g)
        exp_add = elementwise.ElementwiseKernel(context,
                                                "float* arr_a, float* arr_b, float* arr_out",
                                                "arr_out[i] = exp(arr_a[i]) / exp(arr_b[i])",
                                                "exp_add")


        @timeit_repeat(reps)
        def test_cl():
            exp_add(a_g, b_g, results)
            results.get()
            # print results


        test_cl()
