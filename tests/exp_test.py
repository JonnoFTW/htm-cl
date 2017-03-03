import pyopencl as cl
from src.utils import cltypes
from pyopencl.elementwise import ElementwiseKernel
import numpy as np
from tests.timeit import timeit_repeat

device = cl.get_platforms()[1].get_devices()[0]
print(device)
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
reps = 500
num = 16000

mf = cl.mem_flags

boostFactors = np.ones(num, dtype=cltypes.float)
cl_boostFactorsRW = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=boostFactors)
cl_boostFactorsR = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boostFactors)
cl_boostFactorsW = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=boostFactors)
new_boostFactors = np.ones(num, dtype=cltypes.float)

src = """
__kernel void exp_overwrite(__global float* boost, const float boostStrength) {
    const int gid = get_global_id(0);
    boost[gid] = exp(boost[gid]) * boostStrength;
}
__kernel void exp_copy(__constant float* boost, __global float* boost_out, const float boostStrength) {
    const int gid = get_global_id(0);
    boost_out[gid] = exp(boost[gid]) * boostStrength;
}
"""
exp_el = ElementwiseKernel(ctx,
                           "const __global float* boost, __global float* boost_out, const float boostStrength",
                           "boost_out[i] = exp(boost[i]) * boostStrength",
                           "exp_el")
boostStrength = cltypes.float(1.001)
prog = cl.Program(ctx, src).build()


@timeit_repeat(reps)
def test_copy():
    prog.exp_copy(queue, (num,), None, cl_boostFactorsR, cl_boostFactorsW, boostStrength).wait()


@timeit_repeat(reps)
def test_overwrite():
    prog.exp_overwrite(queue, (num,), None, cl_boostFactorsRW, boostStrength).wait()


@timeit_repeat(reps)
def test_el():
    exp_el(cl_boostFactorsR, cl_boostFactorsW, boostStrength, range=True).wait()


@timeit_repeat(reps)
def test_numpy():
    return np.exp(boostFactors) * boostStrength


if __name__ == "__main__":
    test_copy()
    test_overwrite()
    test_el()
    test_numpy()
