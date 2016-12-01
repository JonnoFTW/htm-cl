import pyopencl as cl
from pyopencl import array
import numpy as np

kernel_src = """
__kernel void learn(
    __constant int* encoding, // the encoded inputs
    __global int2* synapses,
    __global int* columns
    const int synapsesPerColumn)
){
    const int n = get_global_id(0);
    // a column's synapses start at columnIdx * potentialPct

}

"""


class SpatialPooler(object):
    def __init__(self,
                 columnCount=2048,
                 globalInhibition=1,
                 inputWidth=500,
                 maxBoost=2.0,
                 numActiveColumnsPerInhArea=40,
                 potentialPct=0.8,
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
        self.columns = np.zeros(columnCount, dtype=np.uint32)
        self.synapsesPerColumn = int(columnCount * potentialPct)
        self.synapses = np.zeros((columnCount * self.synapsesPerColumn),
                                 dtype=cl.array.vec.float2)  # x is permanence value, y is input bit idx

        self.synapses[:, 1] = np.random.randint(0, inputWidth, self.synapses.shape[0])
        # each column maps to columnCount*potentialPct inputs

    def compute(self, encoding):
        """
        Updates the spatial pooler. Returns the the indices of the  on-bits
        :param encoding:
        :return:
        """
        return self.columns[:, 0] == 1
