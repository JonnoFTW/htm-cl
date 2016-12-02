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
            self.ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
        self.queue = cl.CommandQueue(self.ctx)

        self.encoders = {'encoder': getattr(nupic.encoders, field)(**args) for field, args in
                         params['sensorParams']['encoders'].iteritems()}
        params['spParams']['inputWidth'] = sum(map(lambda x: x.n, self.encoders))
        self.sp = SpatialPooler(**params['spParams'])
        self.tm = TemporalMemory(**params['tpParams'])
        self.classifier = CLAClassifier(**params['claParams'])
        self.predicted_field = params['predictedField']
        self.recordNum = 0

    def encode(self, inputs):
        """

        :param inputs: dict of input names to their values
        :return:
        """
        return np.concatenate((self.encoders[name].encode(val) for name, val in sorted(inputs.iteritems())))

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
