import pyopencl as cl
import numpy as np
try:
    from pyopencl import cltypes
except ImportError:
    from ..utils import cltypes
src = """



"""
mf = cl.mem_flags
class TemporalMemory(object):
    def __init__(self,
                 queue,
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
        if temporalImp != 'cl':
            raise ValueError('This implementation only supports OpenCL')
        self.activationThreshold = cltypes.uint(activationThreshold)
        self.columnCount = cltypes.uint(columnCount)
        self.cellsPerColumn = cltypes.uint(cellsPerColumn)
        self.globalDecay = cltypes.float(globalDecay)
        self.initialPerm = cltypes.float(initialPerm)
        self.maxAge = cltypes.uint(maxAge)
        self.maxSegmentsPerCell = cltypes.uint(maxSegmentsPerCell)
        self.maxSynapsesPerSegment = cltypes.uint(maxSynapsesPerSegment)
        self.minThreshold = cltypes.uint(minThreshold)
        self.newSynapseCount = cltypes.uint(newSynapseCount)
        self.outputType = outputType
        self.pamLength = cltypes.uint(pamLength)
        self.permanenceDec = cltypes.float(permanenceDec)
        self.permanenceInc = cltypes.float(permanenceInc)
        np.random.seed(seed)

        self.verbosity = verbosity
        self.columnCount = columnCount
        self.inputWidth = inputWidth

        self._queue = queue
        self._ctx = queue.ctx

        np.random.seed(seed)
        self._setup_cl_buffers()

    def _setup_cl_buffers(self):
        """
        There temporal memory has many columns.
        Each column is connected to some number of feed forward inputs from the spatial pooler
        Each column has many cells, each cell can be connected to some other cells in the TM.

        Predictive cells will become active in the next timestep
        :return:
        """
        self.columns = np.zeros((self.columnCount * self.cellsPerColumn), dtype=cltypes.char)
        self.cl_columns = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.columns)

        # a mapping from input bits to each column
        self.feedForward = np.full(self.columnCount, self.initialPerm)
        self.cl_feedForward = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.feedForward)

        self.contextInputs = np.empty(self.columnCount * self.maxSegmentsPerCell, dtype=cltypes.float)
        for c in range(self.columnCount):
            idx = c*self.maxSegmentsPerCell
            self.contextInputs[idx:idx+self.maxSegmentsPerCell] = np.random.normal(0.1, 0.01, size=self.maxSegmentsPerCell)
        self.cl_contextInputs = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.contextInputs)

    def infer(self, pattern):
        pass

    def compute(self, spActiveIndices):

        return
