import pyopencl as cl
from pyopencl import array
import numpy as np

kernel_src = """
__kernel void infer_compute(
    __constant int* activeBitIdx, // work size is the number of activations, array of active bits indices in the input
    __global float2 *table, // x=histogram, y=moving average
    float const alpha, // moving average alpha
    float const actualValue, // actual input value
    int const bucketIdx, // bucket that actualValue falls into
    int const bucketCount,
    char const learn,
    char const infer,
    __global float *predictions // if infer is off, this should be null
){
    const int n = activeBitIdx[get_global_id(0)]*bucketCount; // each job updates the table for a single active bit of the input
    if(learn){ // add values to the table
        const int nbI = n+bucketIdx;
        table[nbI].x++; // increment the active count for this bit's bucket
        table[nbI].y = ((1-alpha)*table[nbI].y) + alpha * actualValue;
    }
    if(infer) { // make predictions
        for(int b = 0; b < bucketCount; b++) {
            const int nb = n+b;
            predictions[b] += table[nb].x * table[nb].y;
        }
    }
}
"""


class CLAClassifier(object):
    def __init__(self, queue, numbuckets, steps=[1], bits=2048, alpha=0.001):
        self._prg = cl.Program(queue.context, kernel_src).build()
        self.iteration = 0
        self.activations = np.zeros(bits)
        self.steps = {i: np.zeros((bits * numbuckets), dtype=cl.array.vec.float2) for i in steps}
        self.alpha = alpha
        self.bits = bits  # number of bits in the input
        self._queue = queue  # the opencl queue
        self._ctx = queue.context  # the opencl context
        self._numBuckets = numbuckets

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
        pattern = np.array(pattern)
        self.activations[pattern] += 1

        multiStepPredictions = {}
        mf = cl.mem_flags
        if learn or infer:
            print pattern
            cl_activeBitIdx = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pattern)
            for step, table in self.steps.iteritems():

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
                cl_table = cl.Buffer(self._ctx, mf.COPY_HOST_PTR, hostbuf=table)
                if learn:
                    cl_table = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=table)
                if infer:
                    predictions = np.zeros(self._numBuckets, dtype=np.float)
                    cl_predictions = cl.Buffer(self._ctx, mf.WRITE_ONLY, predictions.nbytes)
                else:
                    cl_predictions = cl.Buffer(self._ctx, mf.WRITE_ONLY, 1)
                self._prg.infer_compute(self._queue, (pattern.shape[0],), None, cl_activeBitIdx, cl_table,
                                  np.float32(self.alpha), np.float32(actValue), np.int32(bucketIdx),
                                  np.int32(self._numBuckets),
                                  np.uint8(learn), np.uint8(infer), cl_predictions)

                if learn:
                    cl.enqueue_copy(self._queue, table, cl_table).wait()
                if infer:
                    cl.enqueue_copy(self._queue, predictions, cl_predictions).wait()

                multiStepPredictions[step] = predictions
        return multiStepPredictions


def test_cla_se():
    from nupic.encoders import ScalarEncoder
    from nupic.algorithms import CLAClassifier as npCLAClassifier

    se = ScalarEncoder(n=22, w=3, minval=0, maxval=20, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    classifier = CLAClassifier(queue, len(se.getBucketValues()))
    np_cla = npCLAClassifier.CLAClassifier()
    val = 5
    for _ in range(0,2):
        for i in range(0, 10):
            encoding = np.where(se.encode(val) == 1)[0]
            bucketIdx = se.getBucketIndices(val)[0]
            cl_preds = classifier.compute(i, encoding, bucketIdx, val, True, True)
            nupic_preds = np_cla.compute(i, encoding, {'bucketIdx': bucketIdx, 'actValue':val}, True, True)
            print "Actual Value: {} , Active Bits: {}, BucketIdx: {}".format(val, encoding, bucketIdx)
            print "cl", cl_preds
            print "nup", nupic_preds
            # assert cl_preds == nupic_preds
            val += 0.5
            print "-"*10
if __name__ == "__main__":

    test_cla_se()