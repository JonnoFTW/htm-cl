from __future__ import print_function
import pyopencl as cl
from pyopencl import array

try:
    from pyopencl import cltypes
except ImportError:
    from ..utils import cltypes
import numpy as np

print(cl.get_platforms())
kernel_src = """
/**
 * Updates the table for every bit in every step
 *
**/
__kernel void learn(
    __global const uint* activeBitIdx, // work size is the number of activations, array of active bits indices in the input
    __global float* averages,
    __global uint* count,
    float const alpha, // moving average alpha
    float const actualValue, // actual input value from the PF
    uint  const bucketIdx, // bucket that actualValue falls into
    uint  const bucketCount // number of buckets
) {
    const int gid = get_global_id(0);
    const int n = activeBitIdx[gid]; // each job updates the table for a single active bit of the input

    const int nbI = n*bucketCount + bucketIdx;
    // increment the active count for this bit's bucket
    averages[nbI] = ((1-alpha)*averages[nbI]) + alpha * actualValue;
    count[nbI] += 1;
}

/**
 * Make a prediction given a particular input
 * Each kernel does a separate bucket, and iterates over the
 * active bits in the input pattern
**/

__kernel void infer(
    const __global float* averages,
    const __global uint* counts,
    const __global uint* activeBitIdx,
    __global float2* predictions, // the array of predictions
    const __global uint* bitActivations, // the number of times each bit has been active
    const uint activeBits
) {
    const int gid = get_global_id(0);
    const uint bucketCount = get_global_size(0);
    float frequencySum = 0;
    float avgSum = 0;
    for(int i = 0; i < activeBits; i++) {
        const int bitIdx = activeBitIdx[i];
        const int tblIdx = bitIdx * bucketCount + gid;
        frequencySum += counts[tblIdx] / bitActivations[bitIdx];
        avgSum       += averages[tblIdx];
    }
    predictions[gid] = ((float2)(frequencySum, avgSum))/activeBits;
}

__kernel void update_bit_activations(
    __global uint* bitActivations,
    const __global uint* inputPattern
) {
    const int gid = get_global_id(0);
    bitActivations[inputPattern[gid]] += 1;
}
"""
mf = cl.mem_flags


class CLAClassifier(object):
    def __init__(self, queue, numbuckets, steps=[1], bits=2048, alpha=0.001, actValueAlpha=0.3, verbosity=False):
        self._prg = cl.Program(queue.context, kernel_src).build()
        self._learn_iteration = 0
        self.bit_activations = np.zeros(bits, dtype=cltypes.uint)
        self.bucket_activations = np.zeros(numbuckets, dtype=cltypes.uint)
        self.steps = steps
        self.step_count = len(steps)
        self.alpha = cltypes.float(alpha)
        self.actValueAlpha = cltypes.float(actValueAlpha)
        self.bits = bits  # number of bits in the input
        self._queue = queue  # the opencl queue
        self._ctx = queue.context  # the opencl context
        self._numBuckets = cltypes.uint(numbuckets)
        self._verbose = verbosity

        self._init_buffers = False

    def _setup_buffers(self, pattern):
        # buffer for ALL the tables
        # That is, for each timestep there is a
        # table of count,buffer for every single input bit
        #
        # bit(4)        buckets(6)
        #         counts       averages
        # 0  -> [0,0,0,0,0,0], [0,0,0,0,0,0]
        # 1
        # 0
        # 0

        self._table_average = np.zeros(self.bits * self._numBuckets, dtype=cltypes.float)
        self._table_counts = np.zeros(self.bits * self._numBuckets, dtype=cltypes.uint)

        self.cl_table_average = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._table_average)
        self.cl_table_counts = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._table_counts)

        self.cl_activeBitIdx = cl.Buffer(self._ctx, mf.READ_ONLY, size=pattern.nbytes)

        self._predictions = np.zeros(self.step_count * self._numBuckets, dtype=cltypes.float2)
        self.cl_predictions = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._predictions)

        self.cl_bit_activations = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.bit_activations)

        self._init_buffers = True

    def _get_bit_activations(self):
        cl.enqueue_copy(self._queue, self.bit_activations, self.cl_bit_activations).wait()
        return self.bit_activations

    def _get_table(self):
        cl.enqueue_copy(self._queue, self._table_average, self.cl_table_average)
        cl.enqueue_copy(self._queue, self._table_counts, self.cl_table_counts).wait()
        return np.dstack((self._table_counts, self._table_average))[0].reshape(self.bits, self._numBuckets * 2)

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
        pattern = np.array(pattern, dtype=cltypes.uint)
        if not self._init_buffers:
            self._setup_buffers(pattern)

        ev_copy_pattern = cl.enqueue_write_buffer(self._queue, self.cl_activeBitIdx, pattern)
        # update bit activations on device side
        ev_update_bit = self._prg.update_bit_activations(self._queue, (pattern.size,), None,
                                                         self.cl_bit_activations, self.cl_activeBitIdx,
                                                         wait_for=[ev_copy_pattern])

        multiStepPredictions = {}
        ev_learn = None
        if learn:
            ev_learn = [self._prg.learn(self._queue, (self.step_count * pattern.size,), None,
                                        self.cl_activeBitIdx, self.cl_table_average, self.cl_table_counts, self.alpha,
                                        self.actValueAlpha,
                                        cltypes.uint(bucketIdx), self._numBuckets,
                                        wait_for=[ev_update_bit])]
        if infer:
            """
                const __global float* averages,
                const __global uint* counts,
                const __global uint* activeBitIdx,
                __global float2* predictions, // the array of predictions
                __global const  uint* bitActivations, // the number of times each bit has been active
                uint const activeBits
            """
            # kernel for every active bit in each step
            ev_infer = self._prg.infer(self._queue, (self._numBuckets,), None,
                                       self.cl_table_average, self.cl_table_counts, self.cl_activeBitIdx,
                                       self.cl_predictions, self.cl_bit_activations,
                                       cltypes.uint(pattern.size), wait_for=ev_learn)

            cl.enqueue_copy(self._queue, self._predictions, self.cl_predictions, wait_for=[ev_infer]).wait()
        # print("Activations", self.bucket_activations)
        # multiStepPredictions['actualValues'] = predictions['x'] / len(pattern)
        # multiStepPredictions[step] = predictions['y'] / len(pattern)  # the probability for each bucket
        # print("Actual Values", multiStepPredictions['actualValues'])
        multiStepPredictions[1] = self._predictions.copy()
        # print("Probability", multiStepPredictions[1])
        self.bucket_activations[bucketIdx] += 1

        return multiStepPredictions


def test_cla_se():
    from nupic.encoders import ScalarEncoder
    from nupic.algorithms.CLAClassifier import CLAClassifier as npCLAClassifier

    se = ScalarEncoder(n=10, w=3, minval=0, maxval=20, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    classifier = CLAClassifier(queue, numbuckets=len(se.getBucketValues()), bits=se.n, verbosity=True)
    np_cla = npCLAClassifier(verbosity=1)
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
