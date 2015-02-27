from layers.input          import InputLayer
from layers.dropout        import DropoutLayer
from layers.nonlinearities import ReluLayer, SigmoidLayer, MaxoutLayer, TanhLayer
from layers.loss           import SoftmaxLayer, RegressionLayer, SVMLayer
from layers.normalization  import LocalResponseNormalizationLayer
from layers.pooling        import PoolLayer
from layers.dotproducts    import ConvLayer, FullyConnectedLayer
from layers.similarity     import SimilarityLayer, MexLayer
from layers.operations     import AddLayer

class Net(object):

    """
    Net manages a set of layers.
    For now: Simple linear order of layers, first layer input and last layer a cost layer
    """

    def __init__(self, layers=None):
        self.layers = []
        if layers and type(layers) == list:
            self.makeLayers(layers)

    def __str__(self):
        return '\n'.join(
            '{} {} {} {}'.format(
                layer.layer_type,
                layer.out_sx,
                layer.out_sy,
                layer.out_depth
            ) for layer in self.layers
        )

    def makeLayers(self, layers):
        # Takes a list of layer definitions and creates the network layer objects

        # Checks
        if len(layers) < 2:
            print 'Error: Net must have at least one input and one softmax layer.'
        if layers[0]['type'] != 'input':
            print 'Error: First layer should be input.'

        # Add activations and dropouts
        def addExtraLayers():
            newLayers = []
            for layer in layers:
                layerType = layer['type']
                layerKeys = layer.keys()

                if layerType == 'softmax' or layerType == 'svm':
                    # add an fc layer
                    newLayers.append({
                        'type': 'fc', 
                        'num_neurons': layer['num_classes']
                    })

                if layerType == 'regression':
                    # add an fc layer
                    newLayers.append({
                        'type': 'fc',
                        'num_neurons': layer['num_neurons']
                    })

                if ((layerType == 'fc' or layerType == 'conv') 
                    and ('bias_pref' not in layerKeys)):
                    layer['bias_pref'] = 0.0
                    if 'activation' in layerKeys and layer['activation'] == 'relu':
                        layer['bias_pref'] = 0.1 # prevent dead relu by chance

                if layerType != 'capsule':
                    newLayers.append(layer)

                if 'activation' in layerKeys:
                    layerAct = layer['activation']
                    if layerAct in ['relu', 'sigmoid', 'tanh', 'mex']:
                        newLayers.append({'type': layerAct})
                    elif layerAct == 'maxout':
                        newLayers.append({
                            'type': 'maxout',
                            'group_size': layer['group_size'] if group_size in layerKeys else 2
                        })
                    else:
                        print 'Error: Unsupported activation'

                if 'drop_prob' in layerKeys and layerType != 'dropout':
                    newLayers.append({
                        'type': 'dropout',
                        'drop_prob': layer['drop_prob']
                    })

                if layerType == 'capsule':
                    fc_recog = {'type': 'fc', 'num_neurons': layer['num_recog']}
                    pose = {'type': 'add', 'delta': [layer['dx'], layer['dy']], 
                            'skip': 1, 'num_neurons': layer['num_pose']}
                    fc_gen = {'type': 'fc', 'num_neurons': layer['num_gen']}

                    newLayers.append(fc_recog)
                    newLayers.append(pose)
                    newLayers.append(fc_gen)

            return newLayers

        all_layers = addExtraLayers()

        # Create the layers
        for i in xrange(len(all_layers)):
            layer = all_layers[i]
            if i > 0:
                prev = self.layers[i - 1]
                layer['in_sx'] = prev.out_sx
                layer['in_sy'] = prev.out_sy
                layer['in_depth'] = prev.out_depth

            layerType = layer['type']
            obj = None
        
            if   layerType == 'fc':         obj = FullyConnectedLayer(layer)
            elif layerType == 'lrn':        obj = LocalResponseNormalizationLayer(layer)
            elif layerType == 'dropout':    obj = DropoutLayer(layer)
            elif layerType == 'input':      obj = InputLayer(layer)
            elif layerType == 'softmax':    obj = SoftmaxLayer(layer)
            elif layerType == 'regression': obj = RegressionLayer(layer)
            elif layerType == 'conv':       obj = ConvLayer(layer)
            elif layerType == 'pool':       obj = PoolLayer(layer)
            elif layerType == 'relu':       obj = ReluLayer(layer)
            elif layerType == 'sigmoid':    obj = SigmoidLayer(layer) 
            elif layerType == 'tanh':       obj = TanhLayer(layer)
            elif layerType == 'maxout':     obj = MaxoutLayer(layer)
            elif layerType == 'svm':        obj = SVMLayer(layer)
            elif layerType == 'sim':        obj = SimilarityLayer(layer)
            elif layerType == 'mex':        obj = MexLayer(layer)
            elif layerType == 'add':        obj = AddLayer(layer)
            elif layerType == 'capsule':    pass
            else: print 'Unrecognized layer type'

            if obj: self.layers.append(obj)

    def forward(self, V, is_training=False):
        # Forward propogate through the network. 
        # Trainer will pass is_training=True
        activation = self.layers[0].forward(V, is_training)
        for i in xrange(1, len(self.layers)):
            activation = self.layers[i].forward(activation, is_training)
        return activation

    def getCostLoss(self, V, y):
        self.forward(V, False)
        loss = self.layers[-1].backward(y)
        return loss

    def backward(self, y):
        # Backprop: compute gradients wrt all parameters
        loss = self.layers[-1].backward(y) #last layer assumed loss layer
        for i in xrange(len(self.layers) - 2, 0, -1): # first layer assumed input
            self.layers[i].backward()
        return loss

    def getParamsAndGrads(self):
        # Accumulate parameters and gradients for the entire network
        return [ pg
                for layer in self.layers
                for pg in layer.getParamsAndGrads() ]

    def getPrediction(self):
        softmax = self.layers[-1]
        p = softmax.out_act.w
        return p.index(max(p))

    def toJSON(self):
        return { 'layers': [ layer.toJSON() for layer in self.layers ] }

    def fromJSON(self, json):
        self.layers = []
        for layer in json['layers']:
            layerType = layer['type']
            obj = None

            if   layerType == 'fc':         obj = FullyConnectedLayer(layer)
            elif layerType == 'lrn':        obj = LocalResponseNormalizationLayer(layer)
            elif layerType == 'dropout':    obj = DropoutLayer(layer)
            elif layerType == 'input':      obj = InputLayer(layer)
            elif layerType == 'softmax':    obj = SoftmaxLayer(layer)
            elif layerType == 'regression': obj = RegressionLayer(layer)
            elif layerType == 'conv':       obj = ConvLayer(layer)
            elif layerType == 'pool':       obj = PoolLayer(layer)
            elif layerType == 'relu':       obj = ReluLayer(layer)
            elif layerType == 'sigmoid':    obj = SigmoidLayer(layer) 
            elif layerType == 'tanh':       obj = TanhLayer(layer)
            elif layerType == 'maxout':     obj = MaxoutLayer(layer)
            elif layerType == 'svm':        obj = SVMLayer(layer)
            elif layerType == 'sim':        obj = SimilarityLayer(layer)
            elif layerType == 'mex':        obj = MexLayer(layer)

            obj.fromJSON(layer)
            self.layers.append(obj)