import pyopencl as cl

device = cl.get_platforms()[0].get_devices()[0]
print device
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

src = """
__kernel void test() {
    printf((__constant char*)"global id %d/%d work group %d/%d local id %d/%d\\n",
    get_global_id(0), get_global_size(0)-1, get_group_id(0), get_num_groups(0), get_local_id(0), get_local_size(0)-1);
}
"""
prog = cl.Program(ctx, src).build()
prog.test(queue, (16*4,1), (4,1))
queue.finish()
