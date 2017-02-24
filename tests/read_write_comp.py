from tests.timeit import timeit_repeat
import pyopencl as cl
import numpy as np

device = cl.get_platforms()[1].get_devices()[0]
print(device)
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
reps = 1000

num_els = 2000000
a_buffer = np.zeros(num_els, dtype=np.uint32)
r_buffer = np.random.randint(0, num_els, num_els/2, dtype=np.uint32)
mf = cl.mem_flags
cl_rw_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_buffer)

cl_read_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_els, dtype=np.uint32))
out_buffer = np.zeros(num_els, dtype=np.uint32)
cl_write_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=out_buffer)

cl_r_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r_buffer)

src = """
__kernel void inc1(
    __global uint* vals,
    const __global uint* indexes
) {
    vals[indexes[get_global_id(0)]] += 1;
}
__kernel void inc2(
    const __global uint* vals,
    __global uint* out,
    const __global uint* indexes
) {
    const int gid = get_global_id(0);
    out[indexes[gid]] = vals[indexes[gid]] + 1;
}
"""

prog = cl.Program(ctx, src).build()

"""
Use a read write buffer
"""
@timeit_repeat(reps)
def test_inc1():
    prog.inc1(queue, (r_buffer.size,), None, cl_rw_buffer, cl_r_buffer).wait()

"""
Use separate read and write buffers
"""
@timeit_repeat(reps)
def test_inc2():
    prog.inc2(queue, (r_buffer.size,), None, cl_read_buffer, cl_write_buffer, cl_r_buffer).wait()


if __name__ == "__main__":
    test_inc1()
    test_inc2()

