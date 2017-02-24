import pyopencl as cl
from pyopencl import tools
import numpy as np

device = cl.get_platforms()[0].get_devices()[0]
print (device)
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

my_struct = np.dtype([('myFloat', np.float32), ('myUint', np.uint32)])
my_struct, my_struct_c_decl = cl.tools.match_dtype_to_c_struct(device, "my_struct", my_struct)
my_struct = cl.tools.get_or_register_dtype('my_struct', my_struct)

src = my_struct_c_decl + """
__kernel void foo(__global my_struct* s) {
    const int gid = get_global_id(0);
    s[gid].myFloat = gid * 0.5f;
    s[gid].myUint = gid;
}
__kernel void bar(__global my_struct* s) {
    const int gid = get_global_id(0);
    s[gid].myUint = 2* gid;
}
"""
print(src)
structs = np.empty((64,), dtype=my_struct)
prog = cl.Program(ctx, src).build()
cl_structs = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, structs.nbytes)

e1 = prog.foo(queue, (structs.size,), None, cl_structs)
e2 = cl.enqueue_fill_buffer(queue, cl_structs, np.array([0]), 0, structs.nbytes,[e1])
prog.bar(queue, (structs.size,), None, cl_structs, wait_for=[e2]).wait()

cl.enqueue_copy(queue, structs, cl_structs).wait()
print(structs)
