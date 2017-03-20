import numpy as np
import nupic.encoders
from nupic.regions import SPRegion, TPRegion
from nupic.algorithms.CLAClassifier import CLAClassifier

SpatialPooler = SPRegion.SPRegion
TemporalMemory = TPRegion.TPRegion


class Model(object):
    def __init__(self, params):
        """

        :param params: A dict of modelParams in the format
         {'clParams':{'alpha':float,'steps':'1,2,3'},
          'sensorParams':{'encoders':{}
        """

        modelParams = params['modelParams']
        self._encoders = {
            field: getattr(nupic.encoders, args['type'])(
                **dict((arg, val) for arg, val in args.items() if arg not in ['type', 'fieldname']))
            for field, args in
            modelParams['sensorParams']['encoders'].items() if args is not None
            }

        self.predicted_field = modelParams['predictedField']
        modelParams['spParams']['inputWidth'] = sum(map(lambda x: x.getWidth(), self._encoders.values()))
        self.sp = SpatialPooler(**modelParams['spParams'])
        self.sp.initialize(None, None)
        self.tm = TemporalMemory(**modelParams['tpParams'])
        self.tm.initialize(None, None)
        self.classifier = CLAClassifier(**modelParams['clParams'])

        self.spOutputs = {'bottomUpOut': np.zeros(modelParams['spParams']['columnCount'], dtype=np.float32),
                          'anomalyScore': np.zeros(modelParams['spParams']['columnCount'], dtype=np.float32)}
        self.tmOutputs = {
            'bottomUpOut': np.zeros(modelParams['tpParams']['columnCount'] * modelParams['tpParams']['cellsPerColumn'],
                                    dtype=np.float32)}

        self.recordNum = 0

    def encode(self, inputs):
        """

        :param inputs: dict of input names to their values
        inputs
        :return: encoded inputs concatenated
        """
        return np.concatenate([encoder.encode(inputs[name]) for name, encoder in self._encoders.iteritems()])

    def run(self, inputs):
        """
        Runs a single timestep
        :param inputs: a dict mapping input names to their values
        :return:  a dict of predictions-++
        """
        self.recordNum += 1
        encodings = self.encode(inputs)
        predictedValue = inputs[self.predicted_field]
        bucketIdx = self._encoders[self.predicted_field].getBucketIndices(predictedValue)[0]
        self.recordNum += 1

        self.sp.compute({'bottomUpIn': encodings}, self.spOutputs)
        self.tm.compute({'bottomUpIn': self.spOutputs['bottomUpOut']}, self.tmOutputs)

        return self.classifier.compute(self.recordNum, self.tmOutputs['bottomUpOut'],
                                       {'bucketIdx': bucketIdx, 'actValue': predictedValue}, True, True)
