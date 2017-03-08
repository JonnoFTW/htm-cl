from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from src.algorithms import TemporalMemory

device = cl.get_platforms()[1].get_devices()[0]


def test_tm():
    from nupic.encoders import ScalarEncoder
    from nupic.regions import TPRegion
    columns = 128
    se = ScalarEncoder(n=21 + 50, w=3 + 9, minval=0, maxval=100, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    tm = TemporalMemory(queue, columnCount=columns, inputWidth=se.n, verbosity=1, inputActive=se.w)
    tm_nupic = TPRegion.TPRegion(columnCount=columns, inputWidth=se.n)

    val = 5

    def make_output_dict():
        return {
            'topDownOut': np.zeros(64),
            'bottomUpOut': np.zeros(columns, dtype=np.float),
            'lrnActiveStateT': np.zeros(columns),
            'anomalyScore': np.empty(1),
            'activeCells': np.zeros(64),
            'predictiveActiveCells': np.zeros(64)
        }

    cl_out = make_output_dict()
    nupic_out = make_output_dict()
    for _ in range(0, 2):
        for i in range(0, 10):
            encoding = se.encode(val)
            bucketIdx = se.getBucketIndices(val)[0]
            print("Actual Value: {} , Active Bits: {}, BucketIdx: {}".format(val, np.where(encoding == 1), bucketIdx))
            tm.compute(encoding, cl_out)
            tm_nupic.compute(encoding, nupic_out)
            val += 0.5
            print("-" * 10)


if __name__ == "__main__":
    test_tm()
