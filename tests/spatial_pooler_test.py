from __future__ import absolute_import, print_function
import pyopencl as cl
import numpy as np
import os
print(os.environ['PYTHONPATH'].split(os.pathsep))
from tests.timeit import timeit
from src.algorithms import SpatialPooler

device = cl.get_platforms()[0].get_devices()[0]


@timeit
def test_nupic(nupic_sp, encoder, lim):
    out = {'bottomUpOut': np.zeros(64, dtype=np.float), 'anomalyScore': np.empty(1)}
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])

        print(nupic_sp._sfdr._calculateOverlap(enc))


@timeit
def test_cl(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score(enc))


@timeit
def test_cl_idx(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_bitidx(enc))


@timeit
def test_numpy_idx(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_numpy_bitidx(enc))


@timeit
def test_cl_loop_bin(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_loop_bin(enc))


@timeit
def test_cl_loop_all(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_column_loop_all(enc))

@timeit
def test_input_inverse(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_by_input_connections(enc))


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
    # sp_cl._test(32)
    # sp_cl._test(2048)
    # # return
    print("testing nupic")
    test_nupic(sp_nupic, se, lim)
    # print("testing cl loop all")
    # test_cl_loop_all(sp_cl, se, lim)

    # print("testing cl bit idx")
    # test_cl_idx(sp_cl, se, lim)
    #
    print("Testing numpy")
    test_numpy_idx(sp_cl, se, lim)
    print("testing inverse")
    test_input_inverse(sp_cl, se, lim)

    # print("testing cl for loop bin search")
    # test_cl_loop_bin(sp_cl, se, lim)


def test_sp():
    from nupic.encoders import ScalarEncoder
    from nupic.regions import SPRegion
    columns = 128
    se = ScalarEncoder(n=128, w=3, minval=0, maxval=20, forced=True)
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
