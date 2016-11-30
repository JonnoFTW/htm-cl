import pyopencl as cl
from pyopencl import array
import numpy as np

kernel_src = """
__kernel void compute(
    __constant int* activeBitIdx, // work size is the number of activations
    __global float2 *table, // x=histogram, y=moving average
    float const alpha, // moving average alpha
    float const actualValue, // actual input value
    int const bucketIdx, // bucket that actualValue falls into
    int const bucketCount
){
    const int n = activeBitIdx[get_global_id(0)]*bucketCount + bucketIdx; // each job updates the table for a single active bit

    table[n].x++; // increment the active count for this bit's bucket
    table[n].y = ((1-alpha)*table[n].y) + alpha * actualValue;
}
__kernel void infer(
    __constant int* activeBitIdx,
    __constant float2 *table,
    __global float *predictions,
    int const bucketCount)
){
    const int n = activeBitIdx[get_global_id(0)] * bucketCount;
    for(int b = 0; b < bucketCount; b++) {
        const int nb = n+b;
        predictions[nb] += table[nb].x * table[nb].y;
    }
}
"""


class CLAClassifier(object):
    def __init__(self, ctx, queue, numbuckets, steps=[1], bits=2048, alpha=0.001):
        self._prg = cl.Program(ctx, kernel_src).build()
        self.iteration = 0
        self.activations = np.zeros(bits)
        self.steps = {i: np.zeros((bits * numbuckets,2), dtype=np.float32) for i in steps}
        self.alpha = alpha
        self._maxBucketIdx = 0
        self.bits = bits  # number of bits in the input
        self._queue = queue # the opencl queue
        self._ctx = ctx # the opencl context


    def compute(self, recordNum, pattern, classification, learn, infer):
        """
        Computes 1 step
        :param recordNum:
        :param pattern: indices of active columns
        :param classification: dict of bucketIdx and actualValue
        :param learn:
        :param infer:
        :return:
        """
        pattern = np.array(pattern)
        self.activations[pattern] += 1

        retval = None
        if infer:
            retval = self.infer(pattern, classification)

        if learn and classification['bucketIdx'] is not None:
            # current input
            bucketIdx = classification['bucketIdx']
            actualValue = classification['actValue']

            for step, table in self.steps.iteritems():
                mf = cl.mem_flags
                img_opencl = cl.Buffer(self.ctx, mf.WRITE_ONLY, img.nbytes)
                lut_opencl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lut)
                points_opencl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.points)
                self._prg.compute(self._queue, (pattern.shape[0]), None, )

                cl.enqueue_copy(self.queue, img, img_opencl).wait()
                cl.enqueue_copy(self.queue, img, img_opencl).wait()
        return  retval

    def infer(self, pattern, classification):
        """
        Make a prediction from the current activations
        :param pattern:
        :param classification:
        :return:
        """

        multiStepPredictions = {}
        for step, table in self.steps.iteritems():
            multiStepPredictions[step] = self._prg.infer(self._queue, ())
        return multiStepPredictions

if __name__ == "__main__":
    ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
    queue = cl.CommandQueue(ctx)
    classifier = CLAClassifier()

