import pyopencl as cl
from pyopencl import tools
import numpy as np

src = """
__kernel void overlap_by_input(
    __constant uchar* inputBits,
    __constant synapse_struct* synapses,
    __constant int* inputSynapses, // synapse indexes for each input bit
    __global uint* overlaps,
    const float synPermConnected,
    const int synapsesPerColumn,
    const int max_count
) {
    const int gid = get_global_id(0);
    // process the ith bit
    if(inputBits[gid]==1){
        for(int i=0;i<max_count;i++) {
            const int synapseIdx = inputSynapses[gid * max_count + i];
            if(synapseIdx==-1)
                break;
            if(synapses[synapseIdx].permanence > synPermConnected) {
                atomic_inc(&overlaps[synapseIdx / synapsesPerColumn]);
            }
        }
    }
}
"""

device = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

synapse_struct = np.dtype([('permanence', np.float32), ('bitIdx', np.uint32)])
synapse_struct, synapse_struct_c_decl = cl.tools.match_dtype_to_c_struct(ctx.devices[0], "synapse_struct",
                                                                         synapse_struct)
synapse_struct = cl.tools.get_or_register_dtype('synapse_struct', synapse_struct)
synPermMin_ = 0.0
synPermMax_ = 1.0
synPermConnected = np.float32(0.1)
synapsesPerColumn = np.int32(256)
columnCount = 2048
inputWidth = 4096

synapses = synapses = np.zeros((columnCount * synapsesPerColumn), dtype=synapse_struct)

synapses['permanence'] = np.clip(
    np.random.normal(synPermConnected, (synPermMax_ - synPermMin_) / 10,
                     size=synapses.shape[0]).astype(np.float32), 0, 1)
input_synapses = np.arange(0, inputWidth)
for column in range(columnCount):
    idx = column * synapsesPerColumn
    synapses['bitIdx'][idx:idx + synapsesPerColumn] = np.random.choice(input_synapses, synapsesPerColumn, False)
bits, counts = np.unique(synapses['bitIdx'], return_counts=True)
# array mapping each input bit to it's synapses indexes
max_count = np.max(counts)
max_input_to_synapse = max_count
input_bitIdx = np.full((inputWidth * max_count), -1, dtype=np.int32)

for inputBitIdx in xrange(inputWidth):
    idx = inputBitIdx * max_count
    synapseIndexes = np.where(synapses['bitIdx'] == inputBitIdx)[0]
    input_bitIdx[idx: idx + synapseIndexes.size] = synapseIndexes


def expected_overlap(encoding):
    return np.sum(
        np.split((encoding[synapses['bitIdx']] == 1) & (synapses['permanence'] > synPermConnected), columnCount),
        axis=1)


prog = cl.Program(ctx, synapse_struct_c_decl + src).build()
mf = cl.mem_flags


def overlap(encoding):
    cl_encoding = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=encoding)
    overlap = np.zeros(columnCount, dtype=np.uint32)  # array of overlap and boosted overlap scores
    cl_overlap = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=overlap)
    cl_input_synapses = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_bitIdx)
    cl_synapses = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=synapses)

    prog.overlap_by_input(queue, (inputWidth,), None,
                          cl_encoding, cl_synapses, cl_input_synapses, cl_overlap,
                          synPermConnected, synapsesPerColumn, np.int32(max_input_to_synapse)).wait()
    cl.enqueue_copy(queue, overlap, cl_overlap).wait()
    return overlap


encoded = np.zeros(inputWidth, dtype=np.uint8)
# bits = np.random.randint(0, inputWidth, 32)
encoded[:64] = 1

expected = expected_overlap(encoded)
cl_result = overlap(encoded)
print(expected)
print(cl_result)
print("Equals? ", np.array_equal(expected, cl_result))
