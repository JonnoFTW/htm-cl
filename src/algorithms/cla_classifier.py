from __future__ import print_function
import pyopencl as cl
from pyopencl import array
from pyopencl import dtypes
import numpy as np

kernel_src = """
__kernel void infer_compute(
    __constant uint* activeBitIdx, // work size is the number of activations, array of active bits indices in the input
    __constant float2 *table, // x=histogram, y=moving average
    __global float2 *new_table, // x=histogram, y=moving average
    float const alpha, // moving average alpha
    float const actualValue, // actual input value
    uint const bucketIdx, // bucket that actualValue falls into
    uint const bucketCount,
    char const learn,
    char const infer,
    __global float2* predictions, // if infer is off, this should be null
    ulong long const bigNum
){
    const int gid = get_global_id(0);
    const int n = activeBitIdx[get_global_id(0)]*bucketCount; // each job updates the table for a single active bit of the input


    if(learn){ // add values to the table

        const int nbI = n+bucketIdx;
        // increment the active count for this bit's bucket
        new_table[nbI] = (float2)(table[nbI].x +1, ((1-alpha)*table[nbI].y) + alpha * actualValue);
        printf("Kernel Id %d activeBit=%d n=%d new_table[%d]=%2.2v2hlf\\n", gid,activeBitIdx[gid], n, nbI,new_table[nbI]);
    }


    if(infer) { // make predictions
        for(int b = 0; b < bucketCount; b++) {

            const int nb = n+b;
            printf((__constant char *)"GID %d bit=%d bkt=%d x=%0.2f  y=%0.2f\\n", gid, activeBitIdx[gid], b,table[nb].x,table[nb].y);
            predictions[b] += table[nb];

        }
    }
}
"""


class CLAClassifier(object):
    def __init__(self, queue, numbuckets, steps=[1], bits=2048, alpha=0.001, actValueAlpha=0.3, verbose=False):
        self._prg = cl.Program(queue.context, kernel_src).build()
        self._learn_iteration = 0
        self.bit_activations = np.zeros(bits, dtype=np.uint32)
        self.bucket_activations = np.zeros(numbuckets, dtype=np.uint32)
        self.steps = {i: np.zeros((bits * numbuckets), dtype=cl.array.vec.float2) for i in steps}
        self.alpha = alpha
        self.actValueAlpha = actValueAlpha
        self.bits = bits  # number of bits in the input
        self._queue = queue  # the opencl queue
        self._ctx = queue.context  # the opencl context
        self._numBuckets = numbuckets
        self._verbose = verbose

    def _show_table(self, table):
        """
        Show the internal table
        Bits are rows, buckets are columns,
        Each value has x=number of activations of a bit with give bucket
                       y=moving average of values falling into a bucket when a bit was active
        :param table:
        :return:
        """
        for idx in xrange(self.bits):
            print(idx, table[idx * self._numBuckets:idx * self._numBuckets + self._numBuckets])

    def compute(self, recordNum, pattern, bucketIdx, actValue, learn, infer):
        """
        Computes 1 step
        :param recordNum:
        :param pattern: indices of active columns in the TM layer
        :param classification: dict of bucketIdx and actualValue
        :param learn:
        :param infer:
        :return:
        """
        pattern = np.array(pattern).astype(np.uint32)
        self.bit_activations[pattern] += 1  # number of times each bit was active

        multiStepPredictions = {}
        mf = cl.mem_flags
        if learn or infer:
            cl_activeBitIdx = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pattern)
            for step, table in self.steps.iteritems():
                # print("old table")
                # self._show_table(table)
                """
                 int* activeBitIdx
                 float2 *table, // x=histogram, y=moving average
                float const alpha, // moving average alpha
                float const actualValue, // actual input value
                int const bucketIdx, // bucket that actualValue falls into
                int const bucketCount,
                bool const learn,
                bool const infer,
                __global float *predictions
                """
                new_table = table.copy()
                cl_new_table = cl.Buffer(self._ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=new_table)
                cl_table = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=table)
                if learn:
                    cl_table = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=table)
                if infer:
                    predictions = np.zeros(self._numBuckets, dtype=cl.array.vec.float2)
                    cl_predictions = cl.Buffer(self._ctx, mf.READ_WRITE, predictions.nbytes)
                else:
                    cl_predictions = cl.Buffer(self._ctx, mf.WRITE_ONLY, 1)
                self._prg.infer_compute(self._queue, (pattern.shape[0],), None, cl_activeBitIdx, cl_table, cl_new_table,
                                        cl.dtypes.float(self.actValueAlpha), cl.dtypes.float(actValue), cl.dtypes.uint(bucketIdx),
                                        cl.dtypes.uint(self._numBuckets),
                                        cl.dtypes.char(learn), cl.dtypes.char(infer), cl_predictions, cl.dtypes.ulonglong(240))

                if learn:
                    cl.enqueue_copy(self._queue, new_table, cl_new_table).wait()
                    self.steps[step] = new_table
                    if self._verbose:
                        print("new table")
                        self._show_table(new_table)
                        pass
                if infer:
                    cl.enqueue_copy(self._queue, predictions, cl_predictions).wait()
                print("Activations", self.bucket_activations)
                multiStepPredictions['actualValues'] = predictions['x'] / len(pattern)
                multiStepPredictions[step] = predictions['y'] / len(pattern)  # the probability for each bucket
                print("Actual Values", multiStepPredictions['actualValues'])
                print("Probability", multiStepPredictions[step])
        self.bucket_activations[bucketIdx] += 1

        return multiStepPredictions


def test_cla_se():
    from nupic.encoders import ScalarEncoder
    from nupic.algorithms import CLAClassifier as npCLAClassifier

    se = ScalarEncoder(n=10, w=3, minval=0, maxval=20, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    classifier = CLAClassifier(queue, numbuckets=len(se.getBucketValues()), bits=se.n, verbose=True)
    np_cla = npCLAClassifier.CLAClassifier(verbosity=1)
    print("Buckets", se.getBucketValues())
    val = 5
    for _ in range(0, 2):
        for i in range(0, 10):
            encoding = np.where(se.encode(val) == 1)[0]
            bucketIdx = se.getBucketIndices(val)[0]
            print("Actual Value: {} , Active Bits: {}, BucketIdx: {}".format(val, encoding, bucketIdx))
            cl_preds = classifier.compute(i, encoding, bucketIdx, val, True, True)
            nupic_preds = np_cla.compute(i, encoding, {'bucketIdx': bucketIdx, 'actValue': val}, True, True)
            print("cl", cl_preds)
            print("nup", np_cla._actualValues)
            print("nup", nupic_preds)
            # assert cl_preds == nupic_preds
            val += 0.5
            print("-" * 32)


if __name__ == "__main__":
    test_cla_se()
