import pyopencl as cl
import pyopencl.cltypes
import numpy as np
import functools
from datetime import datetime

device = cl.get_platforms()[1].get_devices()[1]
ctx = cl.Context([device])
print(device)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

search = cl.cltypes.uint(1024)
vals = np.random.randint(0, 4096, 10000, dtype=cl.cltypes.uint)

prog = cl.Program(ctx, """
__kernel void where(__constant uint* vals, __global char* idxs, const uint needle) {
    const int n = get_global_id(0);
    idxs[n] = vals[n] == needle;
}
""").build()


def np_where(values):
    return values == search


def cl_where(values):
    cl_values = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
    idxs = np.empty(values.size, dtype=cl.cltypes.char)
    cl_out = cl.Buffer(ctx, mf.WRITE_ONLY, idxs.nbytes)
    prog.where(queue, (values.size,), None, cl_values, cl_out, search)
    cl.enqueue_copy(queue, idxs, cl_out).wait()
    return idxs


lim = 1000


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = datetime.now()
        func(*args, **kwargs)
        elapsedTime = datetime.now() - startTime
        print('function [{}] finished in {}'.format(
            func.__name__, elapsedTime))

    return newfunc


@timeit
def test_np():
    for _ in xrange(lim):
        # a = np_where(vals)
        a = np.zeros(10000)


@timeit
def test_cl():
    for _ in xrange(lim):
        # a = cl_where(vals)
        a = np.empty(10000)


print("testing cl")
test_cl()
print("testing np")
test_np()
