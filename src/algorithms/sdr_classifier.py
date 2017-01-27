from __future__ import print_function
import pyopencl as cl
from pyopencl import array
from pyopencl import cltypes
import numpy as np
from collections import deque

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
    __global float2* predictions // if infer is off, this should be null
){
    const int gid = get_global_id(0);
    const int n = activeBitIdx[get_global_id(0)]*bucketCount; // each job updates the table for a single active bit of the input

    if(infer) { // make predictions
        for(int b = 0; b < bucketCount; b++) {

            const int nb = n+b;
            printf((__constant char *)"GID %d bit=%d bkt=%d x=%0.2f  y=%0.2f\\n", gid, activeBitIdx[gid], b,table[nb].x,table[nb].y);
            predictions[b] += table[nb];

        }
    }
    if(learn){ // add values to the table

        const int nbI = n+bucketIdx;
        // increment the active count for this bit's bucket
        new_table[nbI] = (float2)(table[nbI].x +1, ((1-alpha)*table[nbI].y) + alpha * actualValue);
        printf("Kernel Id %d activeBit=%d n=%d new_table[%d]=%2.2v2hlf\\n", gid,activeBitIdx[gid], n, nbI,new_table[nbI]);
    }



}
__kernel void inferSingleStep(
        __constant char* pattern,
        __constant float* weights,
        __global float* predictions,
        __global float* sums,
        uint const numBuckets,
        uint const maxInput) {
    // get column sum for matrix for active bit
    const int gid = get_global_id(0);
    const int num_preds = get_global_size(0);
    float sum = 0;
    const int bitIdx = pattern[gid];
    // maxInput rows, numBucket columns
    for(int i=0; i < numBuckets; i++) {
        sum += weights[maxInput* i + bitIdx];
    }
    sums[bitIdx] = exp(sum);

    // barrier
    barrier(CLK_GLOBAL_MEM_FENCE);
    sum = 0; // find sum of all other sums
    for(int i=0; i < num_preds; i++) {
        sum += sums[i];
    }
    predictions[gid] = sums[bitIdx] / sum;
}

"""

mf = cl.mem_flags


class SDRClassifier(object):
    def __init__(self, queue, maxinput, numbuckets, steps=[1], alpha=0.001, actValueAlpha=0.3,
                 verbosity=False):
        """Constructor for the SDR classifier.
          Parameters:
          ---------------------------------------------------------------------
          @param queue (CLCommandQueue) PyOpenCL command queue
          @param maxinput the largets possible input
          @param numbuckets (int) number of buckets or classes the encoder uses
          @param steps (list) Sequence of the different steps of multi-step
              predictions to learn
          @param alpha (float) The alpha used to adapt the weight matrix during
              learning. A larger alpha results in faster adaptation to the data.
          @param actValueAlpha (float) Used to track the actual value within each
              bucket. A lower actValueAlpha results in longer term memory
          @param verbosity (int) verbosity level, can be 0, 1, or 2
          """
        if numbuckets <= 0:
            raise ValueError("numbuckets must be positive")
        if len(steps) == 0:
            raise TypeError("steps cannot be empty")
        if not all(isinstance(item, int) for item in steps):
            raise TypeError("steps must be a list of ints")
        if any(item < 0 for item in steps):
            raise ValueError("steps must be a list of non-negative ints")

        if alpha < 0:
            raise ValueError("alpha (learning rate) must be a positive number")
        if actValueAlpha < 0 or actValueAlpha >= 1:
            raise ValueError("actValueAlpha be a number between 0 and 1")
        self.steps = steps
        self._maxSteps = max(self.steps) + 1
        self._patternNZHistory = deque(maxlen=self._maxSteps)
        self._maxinput = maxinput
        self._weights = {step: np.zeros((maxinput, numbuckets), dtype=cltypes.float) for step in steps}
        self._actualValues = [None]
        self._prg = cl.Program(queue.context, kernel_src).build()
        self._learn_iteration = 0
        self.alpha = alpha
        self.actValueAlpha = actValueAlpha
        self._queue = queue  # the opencl queue
        self._ctx = queue.context  # the opencl context
        self._numBuckets = numbuckets
        self.verbosity = verbosity

    def _show_table(self, table):
        for idx in xrange(self.bits):
            print(idx, table[idx * self._numBuckets:idx * self._numBuckets + self._numBuckets])

    def inferSingleStep(self, pattern, weights):
        expOutputActivation = np.exp(weights[pattern].sum(axis=0))
        predictDist = expOutputActivation / np.sum(expOutputActivation)
        return predictDist

    def inferSingleStepCL(self, pattern, weights):
        """
        __constant char* pattern,
        __constant float* weights,
        __global float* predictions,
        __global float* sums,
        uint const numBuckets

        :param pattern:
        :param param:
        :return:
        """
        cl_pattern = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pattern)
        cl_weights = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
        predictions = np.empty(len(pattern), dtype=cltypes.float)
        cl_predictions = cl.Buffer(self._ctx, mf.WRITE_ONLY, predictions.nbytes)
        cl_sums = cl.Buffer(self._ctx, mf.READ_WRITE, 32 * len(pattern))
        self._prg.inferSingleStep(self._queue, (pattern.shape[0],), None, cl_pattern, cl_weights, cl_predictions,
                                  cl_sums, cltypes.uint(self._numBuckets), cltypes.uint(self._maxinput))
        cl.enqueue_copy(self._queue, predictions, cl_predictions).wait()
        return predictions

    def infer(self, pattern, classification):
        if self.steps[0] == 0 or classification is None:
            defaultValue = 0
        else:
            defaultValue = classification["actValue"]
        actValues = [x if x is not None else defaultValue
                     for x in self._actualValues]
        retval = {"actualValues": actValues}
        for step in self.steps:
            retval[step] = self.inferSingleStep(pattern, self._weights[step])
            retval[str(step) + 'cl'] = self.inferSingleStepCL(pattern, self._weights[step])
        return retval

    def compute(self, recordNum, pattern, classification, learn, infer):
        """
        Computes 1 step
        :param recordNum:
        :param pattern: indices of active columns in the TM layer
        :param classification: dict of bucketIdx and actualValue
        :param learn:
        :param infer:
        :return:
        """
        if self.verbosity:
            print("  recordNum:", recordNum)
            print("  patternNZ (%d):" % len(pattern), pattern)
            print("  classificationIn:", classification)

        bucketIdx, actValue = classification['bucketIdx'], classification['actValue']
        pattern = np.array(pattern).astype(cltypes.uint)
        self._patternNZHistory.append((recordNum, pattern))

        retval = None
        if infer:
            retval = self.infer(pattern, classification)
        return retval

        if learn and bucketIdx is not None:
            cl_activeBitIdx = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pattern)
            for learnRecordNum, learnPattern in self._patternNZHistory:
                error = dict()
                targetDist = np.zeros(self._numBuckets + 1, dtype=cltypes.float)
                targetDist[bucketIdx] = 1.0

            for step, table in self._weights.iteritems():
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
                                        cltypes.float(self.actValueAlpha), cltypes.float(actValue),
                                        cltypes.uint(bucketIdx),
                                        cltypes.uint(self._numBuckets),
                                        cltypes.char(learn), cltypes.char(infer), cl_predictions)

                if learn:
                    cl.enqueue_copy(self._queue, self.steps[step], cl_new_table).wait()
                if infer:
                    cl.enqueue_copy(self._queue, predictions, cl_predictions).wait()
                print("Activations", self.bucket_activations)

                multiStepPredictions[step] = predictions['y'] / len(pattern)  # the probability for each bucket
                print("Actual Values", multiStepPredictions['actualValues'])
                print("Probability", multiStepPredictions[step])
        self.bucket_activations[bucketIdx] += 1

        return multiStepPredictions


def test_cla_se():
    from nupic.encoders import ScalarEncoder
    from nupic.algorithms.sdr_classifier import SDRClassifier as npSDRClassifier

    se = ScalarEncoder(n=10, w=3, minval=0, maxval=20, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    classifier = SDRClassifier(queue, 30, len(se.getBucketValues()))
    np_cla = npSDRClassifier(verbosity=1)
    print("Buckets", se.getBucketValues())
    val = 5
    for _ in range(0, 2):
        for i in range(0, 10):
            encoding = np.where(se.encode(val) == 1)[0]
            bucketIdx = se.getBucketIndices(val)[0]
            print("Actual Value: {} , Active Bits: {}, BucketIdx: {}".format(val, encoding, bucketIdx))
            classification = {'bucketIdx': bucketIdx, 'actValue': val}
            cl_preds = classifier.compute(i, encoding, classification, True, True)
            nupic_preds = np_cla.compute(i, encoding, classification, True, True)
            print("cl", cl_preds)
            print("nup", np_cla._actualValues)
            print("nup", nupic_preds)
            # assert cl_preds == nupic_preds
            val += 0.5
            print("-" * 32)


if __name__ == "__main__":
    test_cla_se()
