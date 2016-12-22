import pyopencl as cl
import pyopencl.bitonic_sort
import pyopencl.array
import numpy as np
import gc

gc.disable()
ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
queue = cl.CommandQueue(ctx)
sorter = cl.bitonic_sort.BitonicSort(ctx)

vals = np.random.randint(0, 1024, 4096)


def np_sort(values):
    np.argsort(values)


def cl_sort(values):
    # cl_vals = cl.array.Array(queue, data=vals)
    # sorter(cl_vals)
    pass


for i in xrange(10000):
    a = np_sort(vals)
    b = cl_sort(vals)
