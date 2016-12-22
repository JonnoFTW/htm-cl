from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import pyopencl.cltypes

kernel_src = """
int clip(int a, int min, int max) {
    if (a>max) return max;
    if (a<min) return min;
    return a;
};
__kernel void overlap2(
    __constant uchar* encoding, // the encoded input as a binary array
    __constant synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    __constant float* boostFactors, // boost values for columns
    const float synPermConnected
){
    const int n = get_global_id(0); // job for the nth column
    const int synIdx = get_local_id(0);
    const int numColumns = get_global_size(0);
    const int synapsesPerColumn = get_local_size(0);
   // printf("GID(0) %d GID(1) %d\\n", n, synIdx);
    synapse_struct synapse = synapses[n * synapsesPerColumn + synIdx];

    if (encoding[synapse.bitIdx] && synapse.permanence > synPermConnected) {
    //    printf("Column %d Synapse %d overlaps (synapse.bitIdx=%d, permanence=%0.2f)\\n", n, synIdx, synapse.bitIdx, synapse.permanence);
        overlaps[n].x += 1; // regular overlap score
        overlaps[n].y += 1 * boostFactors[n]; //boosted overlap score
    }

   // printf("Column %d has overlap: %d boosted %d \\n",n,overlaps[n].x, overlaps[n].y);
}
__kernel void overlap(
    __constant uchar* encoding, // the encoded input as a binary array
    __constant synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    __constant float* boostFactors, // boost values for columns
    const float synPermConnected,
    const int synapsesPerColumn
){
    const int n = get_global_id(0); // job for the nth column
  //  const int y = get_global_id(1);
    const int numColumns = get_global_size(0);
    // kernel runs for every column
    uint overlap = 0;
    const uint columnsStartSynapseIdx = n * synapsesPerColumn;
    for(int i=0; i < synapsesPerColumn; i++) {
        // the ith synapse belonging to the nth column
        synapse_struct synapse = synapses[columnsStartSynapseIdx+i];
        overlap += encoding[synapse.bitIdx] && synapse.permanence > synPermConnected;
       // if (encoding[synapse.bitIdx] && synapse.permanence > synPermConnected)
       //     printf("Column %d Synapse %d overlaps (synapse.bitIdx=%d, permanence=%0.2f)\\n", n, i, synapse.bitIdx, synapse.permanence);
    }
    overlaps[n].x = overlap; // regular overlap score
    overlaps[n].y = overlap * boostFactors[n] ; //boosted overlap score

   // printf("Column %d has overlap: %d boosted %d \\n",n,overlaps[n].x, overlaps[n].y);
}
/*
__kernel void update_synapses(
    __constant uchar* encoding, // encoded input
    __constant synapse_struct* synapses, //  synapses
    __constant uint* active_columns, // array of active column indices
    __constant uint* activeDutyCycles,
    const char updateRound,
    const int synapsesPerColumn,
    const float synPermInactiveDec,
    const float synPermActiveInc
) {
    const int n = get_global_id(0); // job for the nth column
    const int numColumns = get_global_size(0);
    const uint columnsStartSynapseIdx = n * synapsesPerColumn;

    // apply inhibition, uses the boosted column score


    // update stuff

    // adapt synapses for this column

    float perm =
    //if() {

   // }

    // update duty cycles

    // bump up weak columns

    // update boost factors global

    //for(int i=0; i < synapsesPerColumn; i++) {
   //     synapses[columnsStartSynapseIdx+i].boostFactor = exp((targetDensity - activeDutyCycles) * boostStrength);
    //}


    // update round
     if (updateRound) {
         // update inhibition radius

         // update min duty cycles
     }
}*/
"""
mf = cl.mem_flags


class SpatialPooler(object):
    def __init__(self,
                 queue,
                 columnCount=2048,
                 globalInhibition=1,
                 inputWidth=500,
                 boostStrength=0.0,
                 numActiveColumnsPerInhArea=40,
                 potentialPct=0.3,
                 seed=1956,
                 spVerbosity=0,
                 spatialImp='cl',
                 synPermActiveInc=0.05,
                 synPermConnected=0.1,
                 synPermInactiveDec=0.05015):
        if spatialImp != 'cl':
            raise ValueError('This implementation only supports OpenCL Temporal Memory')
        self.columnCount = columnCount
        self.globalInhibition = globalInhibition
        self.inputWidth = inputWidth
        self.boostStrength = boostStrength
        self.numActiveColumnPerInhArea = cl.cltypes.uint(numActiveColumnsPerInhArea)
        self.potentialPct = cl.cltypes.float(potentialPct)
        np.random.seed(seed)
        self.verbosity = spVerbosity
        self.synPermActiveInc = cl.cltypes.float(synPermActiveInc)
        self.synPermConnected = cl.cltypes.float(synPermConnected)
        self.synPermInactiveDec = cl.cltypes.float(synPermInactiveDec)
        # store the TM as an array of int, either on or off
        self.columns = np.zeros(columnCount, dtype=cl.cltypes.uint)
        self.synapsesPerColumn = cl.cltypes.uint(columnCount * potentialPct)
        self._stimulusThreshold = 0

        self._activeDutyCycles = np.zeros(self.columnCount, cl.cltypes.uint)
        self._overlapDutyCycles = np.zeros(self.columnCount, cl.cltypes.uint)
        self._minOverlapDutyCycles = np.zeros(self.columnCount, cl.cltypes.uint)
        self._boostFactors = np.ones(self.columnCount, dtype=cl.cltypes.float)

        self._queue = queue
        self._ctx = queue.context
        self._updatePeriod = 50

        synapse_struct = np.dtype([('permanence', cl.cltypes.float), ('bitIdx', cl.cltypes.uint)])
        synapse_struct, synapse_struct_c_decl = cl.tools.match_dtype_to_c_struct(self._ctx.devices[0], "synapse_struct",
                                                                                 synapse_struct)
        synapse_struct = cl.tools.get_or_register_dtype('synapse_struct', synapse_struct)
        self.synapses = np.zeros((columnCount * self.synapsesPerColumn),
                                 dtype=synapse_struct)  # x is permanence value, y is input bit idx
        if spVerbosity:
            # print("Synapse Struct", synapse_struct_c_decl)
            print("Synapses", self.synapses.size)
            print("Columns", self.columnCount)
            print("Input Width", self.inputWidth)
            print("Synapses Per Column", self.synapsesPerColumn)
            print("Synapse Connection Threshold", self.synPermConnected)

        self.synPermMin_ = 0.0
        self.synPermMax_ = 1.0

        self.synapses['permanence'] = np.clip(
            np.random.normal(synPermConnected, (self.synPermMax_ - self.synPermMin_) / 10,
                             size=self.synapses.shape[0]).astype(np.float32), 0, 1)
        for column in range(self.columnCount):
            idx = column * self.synapsesPerColumn
            self.synapses['bitIdx'][idx:idx + self.synapsesPerColumn] = np.random.choice(range(0, inputWidth),
                                                                                         self.synapsesPerColumn, False)
        # each column connects to exactly columnCount*potentialPct inputs

        self.prog = cl.Program(self._ctx, synapse_struct_c_decl + kernel_src).build()
        self._iterationNum = 0
        self._iterationLearnNum = 0
        self._inhibitionRadius = self.columnCount
        if spVerbosity > 1:
            self._show_synapses()

    def _show_synapses(self):
        reshaped = self.synapses.reshape((self.columnCount, self.synapsesPerColumn))
        for idx, i in enumerate(reshaped):

            print("Column:", idx)
            for sIdx, synapse in enumerate(i):
                print(" Syn", sIdx, synapse)

    def _get_overlap_score(self, encoding, method):
        """
        Returns an array with boosted and non-boosted scores as a vector
        :param encoding: the encoded data
        :return:
        """
        cl_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        cl_boostFactors = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._boostFactors)
        cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
        overlap = np.zeros(self.columnCount, dtype=cl.cltypes.uint2)  # array of overlap and boosted overlap scores
        cl_overlap = cl.Buffer(self._ctx, mf.READ_WRITE, overlap.nbytes)
        # print("Copying synapses: {} bytes, boostFactors: {} bytes, encoding: {}".format(self.synapses.nbytes, self._boostFactors.nbytes, encoding.nbytes, overlap.nbytes))
        if method == 2:
            self.prog.overlap2(self._queue, (self.columnCount,), (self.synapsesPerColumn,),
                               cl_encoding, cl_synapses, cl_overlap, cl_boostFactors, self.synPermConnected,
                               )
        else:
            self.prog.overlap(self._queue, (self.columnCount,), None, cl_encoding, cl_synapses, cl_overlap,
                              cl_boostFactors, self.synPermConnected, self.synapsesPerColumn)
        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        return overlap

    def _inhibit_columns(self, overlaps):

        inhibitionArea = min(self.columnCount, (2 * self._inhibitionRadius + 1) ** self.columnCount)
        targetDensity = min(0.5, float(self.numActiveColumnPerInhArea) / inhibitionArea)
        numActive = int(targetDensity * self.columnCount)

        winnerIndices = np.argsort(overlaps, kind='mergesort')
        start = winnerIndices.size - numActive
        while start < winnerIndices.size:
            i = winnerIndices[start]
            if overlaps[i]['y'] >= self._stimulusThreshold:
                break
            else:
                start += 1
        return winnerIndices[start:][::-1]

    def compute(self, encoding, learn, method=1):
        """
        Updates the spatial pooler. Returns the the indices of the  on-bits
        :param encoding:
        :param learn:
        :return:
        """
        if not isinstance(encoding, np.ndarray):
            raise TypeError("Input vector must be a numpy array, not %s" %
                            str(type(encoding)))

        if encoding.size != self.inputWidth:
            raise ValueError(
                "Input vector dimensions don't match. Expecting %s but got %s" % (
                    encoding.size, self.inputWidth))
        encoding = encoding.astype(cl.cltypes.uchar)
        self._iterationNum += 1
        if learn:
            self._iterationLearnNum += 1

        encoding.reshape(-1)

        overlaps = self._get_overlap_score(encoding, method=method)
        return
        if self.verbosity:
            print("Overlaps:", overlaps)
        # Apply inhibition to determine the winning columns
        active_columns = self._inhibit_columns(overlaps)
        if self.verbosity:
            print("active columns", active_columns)

        if learn:
            self.update_synapse_boost(encoding, active_columns, overlaps)

        return active_columns

    def update_synapse_boost(self, encoding, active_columns, overlaps):
        updateRound = cl.cltypes.char((self._iterationNum % self._updatePeriod) == 0)
        self._adaptSynapses(encoding, active_columns)
        self._updateDutyCycles(overlaps, active_columns, updateRound)
        self._bumpUpWeakColumns()
        self._updateBoostFactors()
        if updateRound:
            self._updateInhibitionRadius()
            self._updateMinDutyCycles()

    def _adaptSynapses(self, encoding, active_columns, updateRound):
        """
            __constant uchar* encoding, // encoded input
            __constant synapse_struct* synapses, //  synapses
            __constant uint* active_columns, // array of active column indices
            __const uint* activeDutyCycles,
            const char updateRound,
            const int synapsesPerColumn,
            const float synPermInactiveDec,
            const float synPermActiveInc
        :param encoding:
        :param active_columns:
        :return:
        """
        inputIndices = np.where(encoding == 1)[0]
        permChanges = np.full(self.columnCount, -1 * self.synPermInactiveDec)
        permChanges[inputIndices] = self.synPermActiveInc
        # for each column, adjust the permanence

        cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
        cl_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        cl_active_columns = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=active_columns)
        cl_active_duty_cycles = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COYP_HOST_PTR, hostbuf=self._activeDutyCycles)
        self.prog.update_synapses(self._queue, (self.columnCount,), None, cl_encoding, cl_synapses, cl_active_columns,
                                  cl_active_duty_cycles, updateRound, self.synapsesPerColumn, self.synPermInactiveDec,
                                  self.synPermActiveInc)
        cl.enqueue_copy(self._queue, ).wait()


import functools
from datetime import datetime


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = datetime.now()
        func(*args, **kwargs)
        elapsedTime = datetime.now() - startTime
        print('function [{}] finished in {}'.format(
            func.__name__, elapsedTime))

    return newfunc


@timeit
def test_np(nupic_sp, encoder, lim):
    out = {'bottomUpOut': np.zeros(64, dtype=np.float), 'anomalyScore': np.empty(1)}
    for i in xrange(lim):
        enc = encoder.encode(i)

        nupic_sp.compute({'bottomUpIn': enc}, out)  # eventually need to compare this SP output to my own


@timeit
def test_cl(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)

        cl_sp.compute(enc, True)


@timeit
def test_cl2(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)

        cl_sp.compute(enc, True, method=2)

device = cl.get_platforms()[1].get_devices()[0]
def compare_overlap():
    from nupic.encoders import ScalarEncoder
    from nupic.regions import SPRegion
    cols = 128
    se = ScalarEncoder(n=256, w=3, minval=0, maxval=20, forced=True, clipInput=True, name='testInput')
    queue = cl.CommandQueue(cl.Context([device]))
    sp_cl = SpatialPooler(queue, columnCount=cols, inputWidth=se.n, spVerbosity=1)
    sp_nupic = SPRegion.SPRegion(columnCount=cols, inputWidth=se.n, spatialImp='py')
    sp_nupic.initialize(None, None)
    lim = 1000
    print("testing cl")
    test_cl(sp_cl, se, lim)
    print("testing cl2")
    test_cl2(sp_cl, se, lim)
    # print("testing np")
    # test_np(sp_nupic, se, lim)


def test_sp():
    from nupic.encoders import ScalarEncoder
    from nupic.regions import SPRegion

    se = ScalarEncoder(n=30, w=3, minval=0, maxval=20, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    sp = SpatialPooler(queue, columnCount=2048, inputWidth=se.n, spVerbosity=1)
    sp_nupic = SPRegion.SPRegion(columnCount=2048, inputWidth=se.n)

    val = 5
    # return
    for _ in range(0, 2):
        for i in range(0, 10):
            encoding = se.encode(val)
            bucketIdx = se.getBucketIndices(val)[0]
            print("Actual Value: {} , Active Bits: {}, BucketIdx: {}".format(val, encoding, bucketIdx))
            sp.compute(encoding, True)
            val += 0.5
            print("-" * 10)


if __name__ == "__main__":
    # test_sp()
    compare_overlap()
