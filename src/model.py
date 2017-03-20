from algorithms import CLAClassifier, SpatialPooler, TemporalMemory
import numpy as np
import nupic
import nupic.encoders
import pyopencl as cl


class Model(object):
    def __init__(self, params, ctx=None):
        """

        :param params: A dict of modelParams in the format
         {'clParams':{'alpha':float,'steps':'1,2,3'},
          'sensorParams':{'encoders':{}
        """
        if ctx is None:
            self.ctx = cl.create_some_context()
        else:
            import pyopencl.cffi_cl
            if type(ctx) is not cl.cffi_cl.Context:
                raise ValueError("ctx must be a valid PyOpenCL Context")

        self.queue = cl.CommandQueue(self.ctx)

        modelParams = params['modelParams']
        for i in ['spParams', 'tpParams', 'clParams']:
            modelParams[i]['queue'] = self.queue

        modelParams['spParams']['spatialImp'] = 'cl'
        modelParams['tpParams']['temporalImp'] = 'cl'
        self.encoders = {
            field: getattr(nupic.encoders, args['type'])(
                **dict((arg, val) for arg, val in args.items() if arg not in ['type', 'fieldname']))
            for field, args in
            modelParams['sensorParams']['encoders'].items() if args is not None
            }

        self.predicted_field = modelParams['predictedField']
        modelParams['spParams']['inputWidth'] = sum(map(lambda x: x.getWidth(), self.encoders.values()))
        self.sp = SpatialPooler(**modelParams['spParams'])
        self.tm = TemporalMemory(**modelParams['tpParams'])

        modelParams['clParams']['numBuckets'] = len(self.encoders[self.predicted_field].getBucketValues())
        modelParams['clParams']['bits'] = modelParams['tpParams']['columnCount']
        self.classifier = CLAClassifier(**modelParams['clParams'])

        self.recordNum = 0

    def encode(self, inputs):
        """

        :param inputs: dict of input names to their values
        inputs
        :return:
        """
        return np.concatenate([encoder.encode(inputs[name]) for name, encoder in self.encoders.iteritems()])

    def run(self, inputs):
        """
        Runs a single timestep
        :param inputs: a dict mapping input names to their values
        :return:  a dict of predictions
        """
        self.recordNum += 1
        encodings = self.encode(inputs)
        predictedValue = inputs[self.predicted_field]
        bucketIdx = self.encoders[self.predicted_field].getBucketIndices(predictedValue)[0]
        self.recordNum += 1

        spActiveColumnIdxs = self.sp.compute(encodings)
        tmActiveColumnIdxs = self.tm.compute(spActiveColumnIdxs)

        return self.classifier.compute(self.recordNum, tmActiveColumnIdxs, bucketIdx, predictedValue, True, True)
