import pyopencl as cl
import numpy as np


class TemporalMemory(object):
    def __init__(self,
                 activationThreshold=14,
                 cellsPerColumn=32,
                 columnCount=2048,
                 globalDecay=0.0,
                 initialPerm=0.21,
                 inputWidth=2048,
                 maxAge=0,
                 maxSegmentsPerCell=128,
                 maxSynapsesPerSegment=32,
                 minThreshold=11,
                 newSynapseCount=20,
                 outputType='normal',
                 pamLength=3,
                 permanenceDec=0.1,
                 permanenceInc=0.1,
                 seed=1960,
                 temporalImp='cl',
                 verbosity=0):
        self.columnCount = 2048
        if temporalImp != 'cl':
            raise ValueError('This implementation only supports OpenCL')
        self.columnCount = columnCount
        self.inputWidth = inputWidth
        self.columns = np.zeros((columnCount, cellsPerColumn), dtype=np.uint8)
        np.random.seed(seed)

    def compute(self, spActiveIndices):
        return
