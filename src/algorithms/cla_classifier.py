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
    bool const learn,
    bool const infer,
    __global float *predictions // if infer is off, this should be null
){
    const int n = activeBitIdx[get_global_id(0)]*bucketCount; // each job updates the table for a single active bit of the input
    const int nbI = n+bucketIdx;
    if(learn){ // add values to the table
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

            cl_activeBitIdx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pattern)
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

                self._prg.compute(self._queue, (pattern.shape[0]), None, cl_activeBitIdx, cl_table,
                                  np.float32(self.alpha), np.float32(actValue), np.int32(bucketIdx), np.int32())

                if learn:
                    cl.enqueue_copy(self._queue, table, cl_table).wait()
                if infer:
                    cl.enqueue_copy(self._queue, predictions, cl_predictions)

                multiStepPredictions[step] = predictions
        return multiStepPredictions


if __name__ == "__main__":
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    classifier = CLAClassifier(queue, 20)
