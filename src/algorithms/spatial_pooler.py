from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import pyopencl.dtypes

kernel_src = """
__kernel void compute(
    __constant uchar* encoding, // the encoded input as a binary array
    __constant synapse_struct* synapses, // all the synapses
    __global uint2* columns, // columns to store overlap scores
    const uint synapsesPerColumn,
    const float boostStrength
){
    const int n = get_global_id(0); // job for the nth column
    const int numColumns = get_global_size(0);
    // kernel runs for every column
    uint overlap = 0;
    const uint columnsStartSynapseIdx = n * synapsesPerColumn;
    for(int i=0; i < synapsesPerColumn; i++) {
        // the ith synapse belonging to the nth column
        synapse_struct synapse = synapses[columnsStartSynapseIdx+i];
        overlap += encoding[synapse.bitIdx] == 1;
       // if (encoding[synapse.bitIdx]==1)
         //   printf("Column %d Synapse %d overlaps (synapse.bitIdx=%d)\\n", n, i, synapse.bitIdx);
    }
    columns[n].x = overlap; // regular overlap score
    columns[n].y = synapse.boostFactor * overlap; //boosted overlap score

    printf("Column %d has overlap: %d boosted %d\\n",n,columns[n].x, columns[n].y);
    // apply inhibition, uses the boosted column score
    const float inhibitionArea = min(numColumns, pow(2*inhibitionRadius+1, ))


    // update stuff
    if(learn) {
        // adapt synapses

        // update duty cycles

        // bumpup weak columns

        // update boost factors
        for(int i=0; i < synapsesPerColumnl i++) {
            synapse.boostFactor = exp((targetDensity - activeDutyCycles)(boostStrength);
        }


        // update round
        if (updateRound) {
            // update inhibition radius

            // update min duty cycles
        }
    }

}
"""
mf = cl.mem_flags


class SpatialPooler(object):
    def __init__(self,
                 queue,
                 columnCount=2048,
                 globalInhibition=1,
                 inputWidth=500,
                 maxBoost=2.0,
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
        self.maxBoost = maxBoost
        self.numActiveColumnPerInhArea = numActiveColumnsPerInhArea
        self.potentialPct = potentialPct
        np.random.seed(seed)
        self.verbosity = spVerbosity
        self.synPermActiveInc = synPermActiveInc
        self.synPermConnected = synPermConnected
        self.synPermInactiveDec = synPermInactiveDec
        # store the TM as an array of int, either on or off
        self.columns = np.zeros(columnCount, dtype=cl.dtypes.uint)
        self.synapsesPerColumn = np.int32(columnCount * potentialPct)

        self._queue = queue
        self._ctx = queue.context

        synapse_struct = np.dtype([('permanence', cl.dtypes.float), ('bitIdx', cl.dtypes.uint), ('boostFactor', cl.dtypes.float)])
        synapse_struct, synapse_struct_c_decl = cl.tools.match_dtype_to_c_struct(self._ctx.devices[0], "synapse_struct",
                                                                                 synapse_struct)
        synapse_struct = cl.tools.get_or_register_dtype('synapse_struct', synapse_struct)
        self.synapses = np.zeros((columnCount * self.synapsesPerColumn),
                                 dtype=synapse_struct)  # x is permanence value, y is input bit idx
        if spVerbosity:
            print("Synapse Struct", synapse_struct_c_decl)
            print("Synapses", self.synapses.size)
            print("Columns", self.columnCount)
            print("Input Width", self.inputWidth)
            print("Synapses Per Column", self.synapsesPerColumn)
            print("Overlap threshold", self.synPermConnected)

        self.synPermMin_ = 0.0
        self.synPermMax_ = 1.0
        self.synapses['boostFactor'][:] = 1
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
        if spVerbosity:
            self._show_synapses()

    def _show_synapses(self):
        reshaped = self.synapses.reshape((self.columnCount, self.synapsesPerColumn))
        for idx, i in enumerate(reshaped):

            print("Column:", idx)
            for sIdx, synapse in enumerate(i):
                print(" Syn", sIdx, synapse)

    def _get_overlap_score(self, encoding):
        """
        __constant uchar* encoding, // the encoded input as a binary array
        __constant synapse_struct* synapses, // all the synapses
        __global uint* columns, // columns to store overlap scores
        const uint synapsesPerColumn

        :param encoding:
        :return:
        """
        cl_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
        overlap = np.zeros(self.columnCount, dtype=cl.arrays.vec.uint2)
        cl_overlap = cl.Buffer(self._ctx, mf.WRITE_ONLY, overlap.nbytes)
        self.prog.overlap(self._queue, (self.columnCount,), None, cl_encoding, cl_synapses, cl_overlap,
                          self.synapsesPerColumn)
        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        print(overlap)
        return overlap

    def compute(self, encoding, learn):
        """
        Updates the spatial pooler. Returns the the indices of the  on-bits
        :param encoding:
        :return:
        """
        if not isinstance(encoding, np.ndarray):
            raise TypeError("Input vector must be a numpy array, not %s" %
                            str(type(encoding)))

        if encoding.size != self.inputWidth:
            raise ValueError(
                "Input vector dimensions don't match. Expecting %s but got %s" % (
                    encoding.size, self.inputWidth))
        encoding = encoding.astype(cl.dtypes.uchar)
        self._iterationNum += 1
        if learn:
            self._iterationLearnNum += 1

        encoding.reshape(-1)
        self._overlaps = self._get_overlap_score(encoding)

        # Apply boosting when learning is on
        if learn:
            self._boostedOverlaps = self._boostFactors * self._overlaps
        else:
            self._boostedOverlaps = self._overlaps

        # Apply inhibition to determine the winning columns
        activeColumns = self._inhibitColumns(self._boostedOverlaps)

        if learn:
            self._adaptSynapses(encoding, activeColumns)
            self._updateDutyCycles(self._overlaps, activeColumns)
            self._bumpUpWeakColumns()
            self._updateBoostFactors()
            if self._isUpdateRound():
                self._updateInhibitionRadius()
                self._updateMinDutyCycles()

        return activeColumns


def test_sp():
    from nupic.encoders import ScalarEncoder
    from nupic.regions import SPRegion

    se = ScalarEncoder(n=30, w=3, minval=0, maxval=20, forced=True)
    queue = cl.CommandQueue(cl.Context([cl.get_platforms()[0].get_devices()[0]]))
    sp = SpatialPooler(queue, columnCount=32, inputWidth=se.n, spVerbosity=1)
    sp_nupic = SPRegion.SPRegion(columnCount=64, inputWidth=se.n)
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
    test_sp()
cl.array.vec.types
