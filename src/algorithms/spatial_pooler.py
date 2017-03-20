from __future__ import print_function

import numpy as np
import pyopencl as cl

try:
    from pyopencl import cltypes
except ImportError:
    from ..utils import cltypes

# __package__ = "spatial_pooler"
kernel_src = """
__kernel void overlap_by_active_input(
    __constant uint* activeBit, // all the active input bit indexes (use np.where() or loop through all bits?
    const __global synapse_struct* synapses, //all synapses
    __global overlap_struct* overlaps, // overlap scores
    const __global float* boostFactors, // boost factors
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
    const __global synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    const __global float* boostFactors, // boost values for columns
    const float synPermConnected,
    const uint synapsesPerColumn,
    const uint columnCount
){
    const int global_id = get_global_id(0);
    const int global_size = get_global_size(0);

   // const int synapseCount = synapsesPerColumn * columnCount;
    // max wg size workers * synapsesPerColumn/global_size = synapseCount
    // each worker processes as many columns as it can:
    //

    for(int i=0; i < synapsesPerColumn/global_size; i++) {
   //     const int synIdx =  global_size * i + global_id;

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
__kernel void overlap_column_loop_all(
    __constant uchar* inputBits, // active bits in sorted order
    const __global synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    const __global float* boostFactors, // boost values for columns
    const float synPermConnected, // if a permenance exceeds this for an active input bit, the synapse is connected
    const int synapsesPerColumn
) {
    // for each synapse, check if permanence exceeds threshold and bitIdx in active bits, add to overlaps
    const int columnIdx = get_global_id(0);
    overlaps[columnIdx] = 0;
    int overlap = 0;
    for(int i =0; i< synapsesPerColumn; i++) {
        const int synapseIdx = columnIdx*synapsesPerColumn +i;
        const synapse_struct synapse = synapses[synapseIdx];
        if (inputBits[synapse.bitIdx] && synapse.permanence > synPermConnected) {
            overlap += 1;
        }
    }
    overlaps[columnIdx].x = overlap;
    overlaps[columnIdx].y = overlap * boostFactors[columnIdx];

}
bool bin_search(__constant uint* a, const int n, const uint key) {
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
int interpolation_search(__constant uint* arr, const int n, const uint key) {
    int low = 0;
    int high = n -1;
    int mid;
    while (arr[high] != arr[low] && key >= arr[low] && key <= arr[high]) {
        mid = low + ((key - arr[low]) * (high - low) / (arr[high] - arr[low]));

        if (arr[mid] < key)
            low = mid + 1;
        else if (key < arr[mid])
            high = mid - 1;
        else
            return mid;
    }
    if (key == arr[low])
        return low ;
    else
        return -1;
}
__kernel void overlap_by_synapse(
    __constant uchar* inputPattern, // the input bits
    const __global synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    const __global float* boostFactors, // boost values for columns
    const float synPermConnected,
    const int synapsesPerColumn
) {
    const uint gid = get_global_id(0); // one thread per column
    uint overlap = 0;
    for(int i = 0; i < synapsesPerColumn; i++) {
        const int synapseIdx = gid * synapsesPerColumn + i;
        const synapse_struct synapse = synapses[synapseIdx];
        if(synapse.permanence > synPermConnected && inputPattern[synapse.bitIdx])
            overlap++;
    }
    overlaps[gid].x = overlap;
    overlaps[gid].y = overlap * boostFactors[gid];
}


__kernel void overlap_loop_bin(
    __constant uint* activeBits, // active bits in sorted order
    const __global synapse_struct* synapses, // all the synapses
    __global uint2* overlaps, // columns to store overlap scores
    const __global float* boostFactors, // boost values for columns
    const float synPermConnected,
    const int synapsesPerColumn,
    const uint numActiveBits
) {
    // for each synapse, check if permanence exceeds threshold and bitIdx in active bits, add to overlaps
    const int columnIdx  = get_global_id(0);
    int overlap = 0;
    for(int i =0; i< synapsesPerColumn; i++) {
        const int synapseIdx = columnIdx*synapsesPerColumn +i;
        const synapse_struct synapse = synapses[synapseIdx];
      //  printf("Column %d Synapse %d perm: %.2f bitIdx=%d on=%d\\n", columnIdx, i, synapse.permanence, synapse.bitIdx,synapse.permanence > synPermConnected && bin_search(activeBits, numActiveBits,synapse.bitIdx));
        if (synapse.permanence > synPermConnected && bin_search(activeBits, numActiveBits,synapse.bitIdx)) {
            overlap++;
        }
    }
    overlaps[columnIdx].x = overlap;
    overlaps[columnIdx].y = overlap * boostFactors[columnIdx];
}
/**
 * Calculate overlap by having each work-item
 * run a single input bit, so there will only
 * be `w` items running the loop
**/
__kernel void overlap_by_input(
    __constant uchar* inputBits,
    const __global float* synapses,
    const __global int* inputSynapses, // synapse indexes for each input bit
 //   __constant float* boostFactors,
    __global uint2* overlaps,
    const float synPermConnected,
    const int synapsesPerColumn,
    const int max_count
) {
    const int gid = get_global_id(0);
    // process the ith bit
    if(inputBits[gid]){
        for(int i=0;i<max_count;i++) {
            const int synapseIdx = inputSynapses[gid * max_count + i];
            if(synapseIdx==-1)
                break;
            if(synapses[synapseIdx] > synPermConnected) {
                const int out = atomic_inc(&overlaps[synapseIdx / synapsesPerColumn].x);
                atomic_mul(&overlaps[column].y, overlaps[column].x * boostFactors[column]);
            }
        }
    } else {
        return;
    }
}
__kernel void update_synapses(
    __constant uchar* encoding, // encoded input
    const __global synapse_struct* synapses, //  synapses
    const __global uint* active_columns, // array of active column indices
    const __global uint* activeDutyCycles,
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

// From this point I'll just




/**
 * For each synapse in each active column,
 * Increment the permanence if synPermThreshold met
 * Otherwise, decrement
**/
__kernel void learnActiveColumns(
    __global uint* activeColumnIdxs,
    __global uchar* inputBits,
    __global synapse_struct* synapses,
    const float synPermActiveInc,
    const float synPermInactiveDec,
    const float synPermConnected
) {

    const uint gid = get_global_id(0);
    synapse_struct s = synapses[gid];

    if(inputBits[s.bitIdx] && s.permanence > synPermActive)
        s.permanence = max(1.0f, s.permanence + synPermActiveInc);
    else  {
        s.permanence = min(0.0f, s.permanence + synPermInactiveDec);
    }
}

/**
 * Update book keeping stuff, after learning
 *
**/

__kernel void bookKeeping(

    __global float* boostFactors,
    const uchar learning

) {

    const uint c = get_global_id(0); // the column
    minDutyCycle[c] = 0.01 * maxDutyCycle[neighbours(c)];
    activeDutyCycle[c] = updateActiveDutyCycle();

    boostFactors[c] = exp(boostFactors[col]);
    overlapDutyCycle[c] = updateOverlapDutyCycle(c);
    if(overlapDutyCycle < minDutyCycle[c]) {
        // increase permanences for this column
        for (int i =0; i < synapsesPerColumn; i++){
            synapses[c*synapsesPerColumn].permanence = 00.1 * connectedPerm;
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
                 inputActive=33,
                 boostStrength=0.0,
                 numActiveColumnsPerInhArea=40,
                 potentialPct=.5,
                 stimulusThreshold=0,
                 seed=1956,
                 dutyCyclePeriod=1000,
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
        self.numActiveColumnPerInhArea = cltypes.uint(numActiveColumnsPerInhArea)
        self.potentialPct = cltypes.float(potentialPct)
        np.random.seed(seed)
        self.verbosity = spVerbosity
        self.synPermActiveInc = cltypes.float(synPermActiveInc)
        self.synPermConnected = cltypes.float(synPermConnected)
        self.synPermInactiveDec = cltypes.float(synPermInactiveDec)
        # store the TM as an array of int, either on or off
        self.columns = np.zeros(columnCount, dtype=cltypes.uint)
        self.synapsesPerColumn = cltypes.uint(inputWidth * potentialPct)
        self._stimulusThreshold = cltypes.uint(stimulusThreshold)
        self._dutyCyclePeriod = cltypes.uint(dutyCyclePeriod)

        self._activeDutyCycles = np.zeros(self.columnCount, cltypes.uint)
        self._overlapDutyCycles = np.zeros(self.columnCount, cltypes.uint)
        self._minOverlapDutyCycles = np.zeros(self.columnCount, cltypes.uint)
        self._boostFactors = np.ones(self.columnCount, dtype=cltypes.float)

        self._queue = queue
        self._ctx = queue.context
        self._updatePeriod = 50

        synapse_struct = np.dtype([('permanence', cltypes.float), ('bitIdx', cltypes.uint)])
        synapse_struct, synapse_struct_c_decl = cl.tools.match_dtype_to_c_struct(self._ctx.devices[0], "synapse_struct",
                                                                                 synapse_struct)
        synapse_struct = cl.tools.get_or_register_dtype('synapse_struct', synapse_struct)

        overlap_struct = np.dtype([('overlap', cltypes.uint), ('boosted', cltypes.uint)])
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
        self.synapses_no_bit = self.synapses['permanence']
        input_synapses = np.arange(0, inputWidth)
        for column in range(self.columnCount):
            idx = column * self.synapsesPerColumn
            self.synapses['bitIdx'][idx:idx + self.synapsesPerColumn] = np.random.choice(input_synapses,
                                                                                         self.synapsesPerColumn, False)
        bits, counts = np.unique(self.synapses['bitIdx'], return_counts=True)
        # array mapping each input bit to it's synapses indexes
        max_count = np.max(counts)
        self.max_input_to_synapse = cltypes.int(max_count)
        self.input_bitIdx = np.full((len(counts) * max_count), -1, dtype=cltypes.int)

        for inputBitIdx in xrange(inputWidth):
            idx = inputBitIdx * max_count
            synapseIndexes = np.where(self.synapses['bitIdx'] == inputBitIdx)[0]
            self.input_bitIdx[idx: idx + synapseIndexes.size] = synapseIndexes

        # print("Connected synapses: ", np.where(self.synapses['permanence'] > synPermConnected)[0].size / float(
        #     self.synapses['permanence'].size))
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

        # initialise host buffers for commonly used things
        # we only copy stuff between host and device when we need to

        self.overlap = np.zeros(self.columnCount, dtype=cltypes.uint2)  # array of overlap and boosted overlap scores
        self.cl_boost_factors = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._boostFactors)
        self.cl_overlap = cl.Buffer(self._ctx, mf.READ_WRITE, self.overlap.nbytes)

        encoding_temp = np.empty(inputWidth, dtype=cltypes.uchar)  # output is a np.uint8 == cltypes.uchar

        self.cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY, size=encoding_temp.nbytes)
        self.active_bits = np.zeros(inputActive, dtype=cltypes.long)
        self.inputActive = inputActive
        self.cl_active_bits = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.active_bits)
        self.cl_synapses = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        self.cl_input_bitIdx = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input_bitIdx)

        self.cl_synapses_no_bit = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses_no_bit)

    def _show_synapses(self):
        reshaped = self.synapses.reshape((self.columnCount, self.synapsesPerColumn))
        for idx, i in enumerate(reshaped):

            print("Column:", idx)
            for sIdx, synapse in enumerate(i):
                print(" Syn", sIdx, synapse)

    def _reset_overlap(self):
        cl.enqueue_fill_buffer(self._queue, self.cl_overlap, np.uint32(0), 0, self.overlap.nbytes).wait()

    def _get_overlap_by_synapse(self, encoding):
        """
            __constant uint* inputPattern, // the input bits
            const __global synapse_struct* synapses, // all the synapses
            __global uint2* overlaps, // columns to store overlap scores
            const __global float* boostFactors, // boost values for columns
            const float synPermConnected,
            const int synapsesPerColumn
        """
        # copy encoding into the buffer
        ev1 = cl.enqueue_write_buffer(self._queue, self.cl_encoding, encoding)
        # run the kernel
        ev2 = self.prog.overlap_by_synapse(self._queue, (self.columnCount,), None,
                                           self.cl_encoding, self.cl_synapses, self.cl_overlap, self.cl_boost_factors,
                                           self.synPermConnected, self.synapsesPerColumn,
                                           wait_for=[ev1])
        cl.enqueue_copy(self._queue, self.overlap, self.cl_overlap, wait_for=[ev2])
        return self.overlap

    def _get_overlap_by_input_connections(self, encoding):
        """
            __constant uchar* inputBits,
            __constant synapse_struct* synapses,
            __constant int* inputSynapses, // synapse indexes for each input bit
            __constant float* boostFactors,
            __global uint2* overlaps,
            const float synPermConnected,
            const int synapsesPerColumn,
            const int max_count
        :param encoding:
        :return:
        """
        cl.enqueue_copy(self._queue, self.cl_encoding, encoding).wait()
        # cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)

        # cl.enqueue_fill_buffer(self._queue, cl_overlap, np.array([0]),0,overlap.nbytes)
        cl_input_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input_bitIdx)
        cl_synapses = self._get_cl_synapses_buffer_no_bit_idx()
        # print("synapse_idx kb :", self.input_bitIdx.nbytes/1024)
        # print("synapses    kb:", self.synapses_no_bit.nbytes/1024)
        # print("encoding    kb:", encoding.nbytes/1024)
        # cl_boostFactors = self._get_cl_boost_factor_buffer()
        self._reset_overlap()
        self.prog.overlap_by_input(self._queue, (self.inputWidth,), None,
                                   self.cl_encoding, cl_synapses, cl_input_synapses,
                                   # cl_boostFactors,
                                   self.cl_overlap,
                                   self.synPermConnected, self.synapsesPerColumn, self.max_input_to_synapse).wait()
        cl.enqueue_copy(self._queue, self.overlap, self.cl_overlap).wait()
        return self.overlap

    def dump_kernel_info(self):
        for knl in self.prog.all_kernels():
            print(knl.function_name)
            print("  Work Group Info")
            for attr in [getattr(cl.kernel_work_group_info, i) for i in dir(cl.kernel_work_group_info) if
                         not i.startswith('_')]:
                try:
                    res = knl.get_work_group_info(attr, knl.context.devices[0])
                except:
                    res = ""

                print("\t" + cl.kernel_work_group_info.to_string(attr), res)

    def _get_overlap_column_loop_all(self, encoding):
        """
        Get overlaps with a single worker per column that loops over each of its synapses
        :param encoding:
        :return:
           __constant uchar* inputBits, // active bits in sorted order
            __constant synapse_struct* synapses, // all the synapses
            __global uint2* overlaps, // columns to store overlap scores
            __constant float* boostFactors, // boost values for columns
            const float synPermConnected, // if a permenance exceeds this for an active input bit, the synapse is connected
            const int synapsesPerColumn,
        """

        cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
        overlap = np.empty(self.columnCount, dtype=cl.array.vec.uint2)  # array of overlap and boosted overlap scores
        cl_overlap = cl.Buffer(self._ctx, mf.WRITE_ONLY, overlap.nbytes)
        cl_synapses = self._get_cl_synapses_buffer()
        cl_boostFactors = self._get_cl_boost_factor_buffer()
        self.prog.overlap_column_loop_all(self._queue, (self.columnCount,), None,
                                          cl_encoding, cl_synapses, cl_overlap, cl_boostFactors,
                                          self.synPermConnected, self.synapsesPerColumn).wait()
        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        return overlap

    def _get_overlap_score(self, encoding):
        """
        Returns an array with boosted and non-boosted scores as a vector
        :param encoding: the encoded data
        :return:
        """
        cl_synapses = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.synapses)
        cl_boostFactors = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._boostFactors)
        cl_encoding = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
        overlap = np.empty(self.columnCount, dtype=cl.array.vec.uint2)  # array of overlap and boosted overlap scores
        cl_overlap = cl.Buffer(self._ctx, mf.WRITE_ONLY, overlap.nbytes)
        max_wg_size = self._ctx.devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        print("Max work group size= {}".format(max_wg_size))
        print("Copying:\t" + self._determine_bytes(encoding, self.synapses, overlap, self._boostFactors))
        self.prog.overlap(self._queue,
                          (max_wg_size,), None,
                          cl_encoding, cl_synapses, cl_overlap, cl_boostFactors, self.synPermConnected,
                          self.synapsesPerColumn, self.columnCount).wait()

        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        return overlap

    def _get_cl_synapses_buffer(self):
        return self.cl_synapses

    def _get_cl_synapses_buffer_no_bit_idx(self):
        return self.cl_synapses_no_bit

    def _get_cl_boost_factor_buffer(self):
        return self.cl_boost_factors

    def _get_overlap_score_loop_bin(self, encoding):
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
        cl.enqueue_write_buffer(self._queue, self.cl_active_bits, active_bits)
        cl_synapses = self._get_cl_synapses_buffer()
        cl_boostFactors = self._get_cl_boost_factor_buffer()
        overlap = np.zeros(self.columnCount, dtype=cl.array.vec.uint2)  # array of overlap and boosted overlap scores
        cl_overlap = cl.Buffer(self._ctx, mf.WRITE_ONLY, overlap.nbytes)
        self.prog.overlap_loop_bin(self._queue,
                                   (self.columnCount,), None,
                                   self.cl_active_bits, cl_synapses, cl_overlap, cl_boostFactors, self.synPermConnected,
                                   self.synapsesPerColumn, cltypes.uint(active_bits.size)).wait()

        cl.enqueue_copy(self._queue, overlap, cl_overlap).wait()
        return overlap

    def _get_overlap_score_numpy_bitidx(self, encoding):
        sa = self.synapses
        # activeBits = np.where(encoding == 1)[0]
        return np.sum(
            np.split((encoding[sa['bitIdx']] == 1) & (sa['permanence'] > self.synPermConnected), self.columnCount),
            axis=1)

    def _copy_active_bits(self, encoding):
        active_bits = np.where(encoding == 1)[0]
        return cl.enqueue_write_buffer(self._queue, self.cl_active_bits, active_bits)

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
        cl_synapses = self.cl_synapses
        cl_boostFactors = self.cl_boost_factors


        cl_active_bits = self.cl_active_bits
        overlap = np.zeros(self.columnCount, dtype=self.overlap_struct)  # array of overlap and boosted overlap scores

        cl_overlap = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=overlap)

        self.prog.overlap_by_active_input(self._queue,
                                          (self.synapseCount,), None,
                                          cl_active_bits, cl_synapses, cl_overlap, cl_boostFactors,
                                          self.synPermConnected, self.synapsesPerColumn).wait()
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
        encoding = encoding.astype(cltypes.uchar)
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
        updateRound = cltypes.char((self._iterationNum % self._updatePeriod) == 0)
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
