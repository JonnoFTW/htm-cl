import pyopencl as cl
import numpy as np
import pyopencl.array
from tests.timeit import  timeit_repeat

device = cl.get_platforms()[1].get_devices()[0]
print(device)
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
reps = 1000

zero_buffer =  np.zeros(200048, dtype=cl.array.vec.uint2)
empty_buffer = np.empty(200048, dtype=cl.array.vec.uint2)
empty_buffer.fill(16)
mf = cl.mem_flags
cl_zero_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zero_buffer)
cl_empty_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=empty_buffer)
zero = np.zeros(2, np.uint32)
# make sure both buffers are initialised
cl.enqueue_copy(queue, empty_buffer, cl_empty_buffer)
print(empty_buffer)
cl.enqueue_copy_buffer(queue, cl_zero_buffer, cl_empty_buffer, zero_buffer.nbytes).wait()

cl.enqueue_copy(queue, empty_buffer, cl_empty_buffer)
print(empty_buffer)


@timeit_repeat(reps)
def test_overwrite_ecb():
    cl.enqueue_copy_buffer(queue, cl_zero_buffer, cl_empty_buffer, zero_buffer.nbytes).wait()


@timeit_repeat(reps)
def test_overwrite_efb():
    cl.enqueue_fill_buffer(queue, cl_empty_buffer, zero, 0, zero_buffer.nbytes).wait()


if __name__ == "__main__":
    test_overwrite_ecb()
    test_overwrite_efb()
