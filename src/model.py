from algorithms import CLAClassifier, SpatialPooler, TemporalMemory
import numpy as np
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
        for i in ['spParams', 'tpParams', 'claParams']:
            modelParams[i]['queue'] = self.queue


        self.encoders = {
            field: getattr(nupic.encoders, args['type'])(**dict((arg, val) for arg, val in args.items() if arg not in ['type', 'fieldname']))
            for field, args in
            modelParams['sensorParams']['encoders'].items() if args is not None
         }

        self.predicted_field = modelParams['predictedField']
        params['spParams']['inputWidth'] = sum(map(lambda x: x.n, self.encoders))
        self.sp = SpatialPooler(**modelParams['spParams'])
        self.tm = TemporalMemory(**modelParams['tpParams'])

        modelParams['claParams']['numBuckets'] = len(self.encoders[self.predicted_field].getBucketValues())
        modelParams['claParams']['bits'] = modelParams['tpParams']['columnCount']
        self.classifier = CLAClassifier(**modelParams['claParams'])

        self.recordNum = 0

    def encode(self, inputs):
        """

        :param inputs: dict of input names to their values
        inputs
        :return:
        """
        return np.concatenate((self.encoders[name].encode(val) for name, val in sorted(inputs.items())))

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
