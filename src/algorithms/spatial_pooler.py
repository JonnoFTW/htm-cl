from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.array
from pyopencl import cltypes

kernel_src = """
__kernel void overlap_by_active_input(
    __constant uint* activeBit, // all the active input bit indexes (use np.where() or loop through all bits?
    __constant synapse_struct* synapses, //all synapses
    __global overlap_struct* overlaps, // overlap scores
    __constant float* boostFactors, // boost factors
    const float synPermConnected, // synapse connection threshold
    const int synapsesPerColumn
) {

    const int bitIdx = get_global_id(0); // index of the active bit
    const int synapseCount = get_global_size(0);
    const int columnCount = synapseCount / synapsesPerColumn;
    // loop through each synapse and check if this input bit matches
    for(int i=0;i< synapseCount; i++) {
        const synapse_struct syn = synapses[i];
        if(syn.bitIdx == bitIdx && syn.permanence > synPermConnected) {

            const int cellIdx = i/synapsesPerColumn;
         //   barrier(CLK_GLOBAL_MEM_FENCE);
         //   overlaps[cellIdx].x += 1;
         //   barrier(CLK_GLOBAL_MEM_FENCE);
         //   overlaps[cellIdx].y = overlaps[cellIdx].x * boostFactors[cellIdx];
             const int overlap = 1+ atomic_inc(&overlaps[cellIdx].overlap);
            //  atomic_xchg(&(overlaps[cellIdx].boosted), overlap * boostFactors[cellIdx]);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if(bitIdx==0) {
    //    printf((__constant char *)"%d synapseCount", synapseCount);
        for(int i=0; i < columnCount; i++) {
            // overlaps[i].overlap =  0;
            overlaps[i].boosted = 0;//overlaps[i].overlap * boostFactors[i];
        }
    }
}
/**
    Calculates the overlap score for a given input
**/
__kernel void overlap(
    __constant uchar* encoding, // the encoded input as a binary array
    __constant synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    __constant float* boostFactors, // boost values for columns
    const float synPermConnected,
    const uint synapsesPerColumn,
    const uint columnCount
){
    const int global_id = get_global_id(0);
    const int global_size = get_global_size(0);
    const int synapseCount = synapsesPerColumn * columnCount;
    // 256 workers * synapsesPerColumn/global_size = synapseCount
    // each worker processes

    for(int i=0; i < synapsesPerColumn/global_size; i++) {
        const int synIdx =  global_size * i + global_id;

        // each batch of columns
         // printf("column id %d synapse id %d/%d \\n", n, synIdx, synapsesPerColumn);
/*
        synapse_struct synapse = synapses[n * synapsesPerColumn + synIdx];
        __local int overlap;
        if(synIdx==0) {
            overlap = 0;
        }
        if (encoding[synapse.bitIdx] && synapse.permanence > synPermConnected) {
         //   printf((__constant char *)"Column %d Synapse %d overlaps (synapse.bitIdx=%d, permanence=%0.2f)\\n", n, synIdx, synapse.bitIdx, synapse.permanence);
            atomic_inc(&overlap); // overlap score for column
        }
        barrier (CLK_LOCAL_MEM_FENCE);
        if (synIdx==0) {
            overlaps[n].x = overlap;
            overlaps[n].y = overlap * boostFactors[n];
            // printf("Column %d has overlap: %d boosted %d \\n",n,overlaps[n].x, overlaps[n].y);
        }*/
    }
}
bool bin_search(__constant long* a, const int n, const uint key) {
    int low = 0;
    int high = n - 1;

    while (low <= high) {
        int mid = low + ((high - low) / 2);
        int midVal = a[mid];
        if (midVal < key)
            low = mid + 1;
        else if (midVal > key)
            high = mid - 1;
        else
            return 1; // key found
    }
    return 0;  // key not found
}
__kernel void overlap_loop(
    __constant long* activeBits, // active bits in sorted order
    __constant synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    __constant float* boostFactors, // boost values for columns
    const float synPermConnected,
    const int synapsesPerColumn,
    const uint numActiveBits
) {
    // for each synapse, check if permanence exceeds threshold and bitIdx in active bits, add to overlaps
    const int columnIdx   = get_global_id(0);
    //const int columnCount = get_global_size(0);
    int overlap = 0;
  /*  if(columnIdx==0) {
        printf("SynapsesPerColumn: %d\\nActiveBits: ", synapsesPerColumn);
        for(int i=0;i < numActiveBits; i++) {
            printf("%d ", activeBits[i]);
        }
        printf("\\n");
    }
    barrier(CLK_GLOBAL_MEM_FENCE);*/
    for(int i =0; i< synapsesPerColumn; i++) {
        const int synapseIdx = columnIdx*synapsesPerColumn +i;
        const synapse_struct synapse = synapses[synapseIdx];
      //  printf("Column %d Synapse %d perm: %.2f bitIdx=%d on=%d\\n", columnIdx, i, synapse.permanence, synapse.bitIdx,synapse.permanence > synPermConnected && bin_search(activeBits, numActiveBits,synapse.bitIdx));
        if (synapse.permanence > synPermConnected && bin_search(activeBits, numActiveBits,synapse.bitIdx)) {
            overlap += 1;
        }
    }
    overlaps[columnIdx].x = overlap;
    overlaps[columnIdx].y = overlap * boostFactors[columnIdx];
}
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
    //const uint columnsStartSynapseIdx = n * synapsesPerColumn;

    // apply inhibition, uses the boosted column score


    // update stuff

    // adapt synapses for this column

   // float perm =
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
         ;
         // update min duty cycles
     }
};
__kernel void test(__global uint* nums) {
    nums[get_global_id(0)] = get_global_id(0);
}
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
                 potentialPct=.5,
                 seed=1956,
                 spVerbosity=0,
                 spatialImp='cl',
                 synPermActiveInc=0.05,
                 synPermConnected=0.10,
                 synPermInactiveDec=0.008):
        if spatialImp != 'cl':
            raise ValueError('This implementation only supports OpenCL Temporal Memory')
        if globalInhibition != 1:
            raise ValueError('This implementation does not support local inhibition')
        self.columnCount = cltypes.uint(columnCount)
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
        self.synapsesPerColumn = cl.cltypes.uint(inputWidth * potentialPct)
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

        overlap_struct = np.dtype([('overlap', cl.cltypes.uint), ('boosted', cl.cltypes.uint)])
        overlap_struct, overlap_struct_c_decl = cl.tools.match_dtype_to_c_struct(self._ctx.devices[0], "overlap_struct",
                                                                                 overlap_struct)
        self.overlap_struct = cl.tools.get_or_register_dtype('overlap_struct', overlap_struct)

        self.synapses = np.zeros((columnCount * self.synapsesPerColumn),
                                 dtype=synapse_struct)  # x is permanence value, y is input bit idx
        if spVerbosity >= 1:
            print('------------CL  SpatialPooler Parameters ------------------')
            # print("Synapse Struct", synapse_struct_c_decl)
            print("Synapses\t", self.synapses.size)
            print("Columns\t", self.columnCount)
            print("Input Width\t", self.inputWidth)
            print("Synapses Per Column\t", self.synapsesPerColumn)
            print("Synapse Connection Threshold\t", self.synPermConnected)

        self.synPermMin_ = 0.0
        self.synPermMax_ = 1.0

        self.synapses['permanence'] = np.clip(
            np.random.normal(synPermConnected, (self.synPermMax_ - self.synPermMin_) / 10,
                             size=self.synapses.shape[0]).astype(np.float32), 0, 1)
        input_synapses = np.arange(0, inputWidth)
        for column in range(self.columnCount):
            idx = column * self.synapsesPerColumn
            self.synapses['bitIdx'][idx:idx + self.synapsesPerColumn] = np.random.choice(input_synapses,
                                                                                         self.synapsesPerColumn, False)
        print("Connected synapses: ", np.where(self.synapses['permanence'] > synPermConnected)[0].size / float(
            self.synapses['permanence'].size))
        # each column connects to exactly columnCount*potentialPct inputs
        src = ''.join([synapse_struct_c_decl, overlap_struct_c_decl, kernel_src])
        self.prog = cl.Program(self._ctx, src).build()
        # print (map(lambda x: x.get_info(pyopencl.kernel_info.FUNCTION_NAME), self.prog.all_kernels()))
        self._iterationNum = 0
        self._iterationLearnNum = 0
        self._inhibitionRadius = self.columnCount
        self.synapseCount = self.synapsesPerColumn * self.columnCount
        if spVerbosity >= 1:
            # self._show_synapses()
            pass

    def _show_synapses(self):
        reshaped = self.synapses.reshape((self.columnCount, self.synapsesPerColumn))
        for idx, i in enumerate(reshaped):

            print("Column:", idx)
            for sIdx, synapse in enumerate(i):
                print(" Syn", sIdx, synapse)
    def _test(self, elems=32):

        data = np.empty(elems, dtype=cltypes.uint)
        cl_data = cl.Buffer(self._ctx, mf.WRITE_ONLY, data.nbytes)
        self.prog.test(self._queue, (elems,), None, cl_data).wait()
        cl.enqueue_copy(self._queue, data, cl_data).wait()
        print(data)
    def _get_overlap_score(self, encoding):
        """
        Returns an array with boosted and non-boosted scores as a vector
        :param encoding: the encoded data
        :return:
        """


        cl_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        cl_boostFactors = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._boostFactors)
        cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
        overlap = np.zeros(self.columnCount, dtype=cl.array.vec.uint2)  # array of overlap and boosted overlap scores
        cl_overlap = cl.Buffer(self._ctx, mf.WRITE_ONLY, overlap.nbytes)
        max_wg_size = self._ctx.devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        print("Max work group size= {}".format(max_wg_size))
        print("Copying:\t"+self._determine_bytes(encoding, self.synapses, overlap, self._boostFactors))
        self.prog.overlap(self._queue,
                          (8,), None,
                          cl_encoding, cl_synapses, cl_overlap, cl_boostFactors, self.synPermConnected, self.synapsesPerColumn, self.columnCount).wait()

        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        return overlap

    def _get_cl_synapses_buffer(self):
        return cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)

    def _get_cl_boost_factor_buffer(self):
        return cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._boostFactors)

    def _get_overlap_score_loop(self, encoding):
        """
        Returns an array with boosted and non-boosted scores as a vector
        :param encoding: the encoded data
        :return:

        overlap_loop(
            __constant int64* activeBits, // active bits in sorted order
            __constant synapse_struct* synapses, // all the synapses
            __global uint2* overlaps, // columns to store overlap scores
            __constant float* boostFactors, // boost values for columns
            const float synPermConnected,
            const int synapsesPerColumn,
            const uint numActiveBits
        )
        """
        active_bits = np.where(encoding == 1)[0]
        cl_active_bits = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=active_bits)
        cl_synapses = self._get_cl_synapses_buffer()
        cl_boostFactors = self._get_cl_boost_factor_buffer()
        overlap = np.zeros(self.columnCount, dtype=cl.array.vec.uint2)  # array of overlap and boosted overlap scores
        cl_overlap = cl.Buffer(self._ctx, mf.WRITE_ONLY, overlap.nbytes)
        self.prog.overlap_loop(self._queue,
                               (self.columnCount,), None,
                               cl_active_bits, cl_synapses, cl_overlap, cl_boostFactors, self.synPermConnected,
                               self.synapsesPerColumn, cltypes.uint(active_bits.size)).wait()

        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        return overlap

    def _get_overlap_score_numpy_bitidx(self, encoding):
        activeBits = set(np.where(encoding == 1)[0])
        overlaps = np.zeros(self.columnCount, dtype=[('x', np.uint32), ('y', np.uint32)])
        for synIdx, synapse in enumerate(self.synapses):
            cellIdx = synIdx / self.synapsesPerColumn
            if synapse['permanence'] > self.synPermConnected and synapse['bitIdx'] in activeBits:
                overlaps[cellIdx][0] += 1
                overlaps[cellIdx][1] = overlaps[cellIdx][0] * self._boostFactors[cellIdx]

        return overlaps

    def _get_overlap_score_bitidx(self, encoding):
        """
        overlap_by_active_input(
            __constant uint* activeBit, // all the active input bit indexes (use np.where() or loop through all bits?
            __constant synapse_struct synapses, //all synapses
             __global uint2* overlaps, // overlap scores
             __constant float* boostFactors, // boost factors
             const float synPermConnected, // synapse connection threshold
             const int synapsesPerColumn
        )
        :param encoding:
        :return:
        """
        cl_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        cl_boostFactors = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._boostFactors)
        active_bits = np.where(encoding == 1)[0].astype(np.uint32)
        cl_active_bits = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=active_bits)
        overlap = np.zeros(self.columnCount, dtype=self.overlap_struct)  # array of overlap and boosted overlap scores
        cl_overlap2 = cl.Buffer(self._ctx, mf.READ_WRITE, overlap.nbytes)

        self.prog.overlap_by_active_input(self._queue,
                                          (self.synapseCount,), None,
                                          cl_active_bits, cl_synapses, cl_overlap2, cl_boostFactors,
                                          self.synPermConnected, self.synapsesPerColumn).wait()
        cl.enqueue_copy(self._queue, overlap, cl_overlap2).wait()
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
        self._adaptSynapses(encoding, active_columns, updateRound)
        self._updateDutyCycles(overlaps, active_columns, updateRound)
        self._bumpUpWeakColumns()
        self._updateBoostFactors()
        if updateRound:
            self._updateInhibitionRadius()
            self._updateMinDutyCycles()
    @staticmethod
    def _determine_bytes(*args):
        return "{} bytes".format(sum(map(lambda x: x.nbytes, args)))
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
        cl_active_duty_cycles = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._activeDutyCycles)
        self.prog.update_synapses(self._queue, (self.columnCount,), None, cl_encoding, cl_synapses, cl_active_columns,
                                  cl_active_duty_cycles, updateRound, self.synapsesPerColumn, self.synPermInactiveDec,
                                  self.synPermActiveInc).wait()
        cl.enqueue_copy(self._queue, self.synapses, cl_synapses).wait()
        pass


from tests.timeit import timeit


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
def test_cl_loop(cl_sp, encoder, lim):
    for i in xrange(lim):
        enc = encoder.encode(i)
        print("Active bits:", np.where(enc == 1)[0])
        print(cl_sp._get_overlap_score_loop(enc))


device = cl.get_platforms()[1].get_devices()[0]


def compare_overlap():
    print("Using device: ", device)
    from nupic.encoders import ScalarEncoder
    from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
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
    # return
    print("testing nupic")
    test_nupic(sp_nupic, se, lim)
    # print("testing cl")
    # test_cl(sp_cl, se, lim)

    # print("testing cl bit idx")
    # test_cl_idx(sp_cl, se, lim)

    print("testing numpy bit idx")
    test_numpy_idx(sp_cl, se, lim)

    print("testing cl for loop")
    test_cl_loop(sp_cl, se, lim)


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
