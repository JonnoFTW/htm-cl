from __future__ import absolute_import, print_function
import numpy as np
import os
out = os.popen('locate libOpenCL.so').readlines()
def follow_link(fname):
    print (fname, end=" ")
    if os.path.islink(fname):
        print("\n\t->", end="")
        next_l = os.readlink(fname)
        d = '/'.join(fname.split('/')[:-1])
        if os.path.isabs(next_l):
            follow_link(next_l)
        else:
            follow_link(d+'/'+next_l)
    else:
        print()
# for i in out:
#     i = i.strip()
#     follow_link(i)
# print(os.environ['PYTHONPATH'].split(os.pathsep))
# print(os.environ['LD_LIBRARY_PATH'])
from tests.timeit import timeit, timeit_repeat
import pyopencl as cl
from src.algorithms import SpatialPooler

for platform in cl.get_platforms():
    print (platform)
    for device in platform.get_devices():
        print ('\t', device)

        print("\t\tOPENCL VERSION ", device.get_info(cl.device_info.OPENCL_C_VERSION ))
        print("\t\tGLOBAL MEM SIZE {} MB".format(device.get_info(cl.device_info.GLOBAL_MEM_SIZE)/1024/1024))
        print("\t\tLOCAL MEM SIZE {} KB".format(device.get_info(cl.device_info.LOCAL_MEM_SIZE)/1024))
        print("\t\tDEVICE ADDRESS BITS {}".format(device.get_info(cl.device_info.ADDRESS_BITS)))
        try:
            print("\t\tMEM SIZE PER COMPUTE {} KB".format(device.get_info(cl.device_info.LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD)/1024))
        except:
            pass
        print("\t\tMAX CONST BUFFER MEM SIZE {} KB".format(device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE)/1024))
device = cl.get_platforms()[1].get_devices()[0]
@timeit
def test_nupic(nupic_sp, encoder, lim):
    out = {'bottomUpOut': np.zeros(64, dtype=np.float), 'anomalyScore': np.empty(1)}
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])

        print(nupic_sp._sfdr._calculateOverlap(enc))
        print(nupic_sp._sfdr.getBoostedOverlaps())


@timeit
def test_cl(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score(enc))


@timeit
def test_cl_idx(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_bitidx(enc))


@timeit
def test_numpy_idx(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_numpy_bitidx(enc)[:32])


@timeit
def test_cl_loop_bin(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_loop_bin(enc))

@timeit
def test_cl_overlap_all_synapse(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_by_synapse(enc))


@timeit
def test_cl_loop_all(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_column_loop_all(enc))

@timeit
def test_input_inverse(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        # print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_by_input_connections(enc)[:32])

reps = 10000

@timeit_repeat(reps)
def test_overwrite_ecb(cl_sp):
    cl_sp._reset_overlap_ecb()

def compare_overlap():
    print("Using device: ", device)

    from nupic.encoders import ScalarEncoder
    from nupic.regions import SPRegion
    cols = 2048
    se = ScalarEncoder(n=1024, w=33, minval=0, maxval=20, forced=True, clipInput=True, name='testInput')
    queue = cl.CommandQueue(cl.Context([device]))
    potentialPct = 0.25
    sp_nupic = SPRegion.SPRegion(columnCount=cols, inputWidth=se.n, spatialImp='py', spVerbosity=1,
                                 potentialPct=potentialPct)
    sp_cl = SpatialPooler(queue, columnCount=cols, inputWidth=se.n, spVerbosity=1, potentialPct=potentialPct)
    sp_nupic.initialize(None, None)
    lim = 1
    print("testing nupic")
    test_nupic(sp_nupic, se, lim)
    print("testing cl loop all")
    test_cl_loop_all(sp_cl, se, lim)

    # print("testing cl bit idx")
    # test_cl_idx(sp_cl, se, lim)
    #
    # sp_cl.dump_kernel_info()
    print("testing cl column ")
    test_cl_overlap_all_synapse(sp_cl, se, lim)

    print("Testing numpy")
    test_numpy_idx(sp_cl, se, lim)
    print("testing inverse")
    test_input_inverse(sp_cl, se, lim)

    print("testing cl for loop bin search")
    test_cl_loop_bin(sp_cl, se, lim)


def test_sp():
    from nupic.encoders import ScalarEncoder
    from nupic.regions import SPRegion
    columns = 128
    se = ScalarEncoder(n=21+50, w=3+9, minval=0, maxval=100, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    sp = SpatialPooler(queue, columnCount=columns, inputWidth=se.n, spVerbosity=1)
    sp_nupic = SPRegion.SPRegion(columnCount=columns, inputWidth=se.n)

    val = 5
    # return
    for _ in range(0, 2):
        for i in range(0, 10):
            encoding = se.encode(val)
            bucketIdx = se.getBucketIndices(val)[0]
            print("Actual Value: {} , Active Bits: {}, BucketIdx: {}".format(val, np.where(encoding == 1), bucketIdx))
            sp.compute(encoding, True, method=2)
            val += 0.5
            print("-" * 10)


if __name__ == "__main__":
    # test_sp()
    compare_overlap()
