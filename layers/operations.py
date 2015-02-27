from util import getopt
from vol import Vol

"""
Layers that perform an operation of inputs.
"""

class AddLayer(object):

    def __init__(self, opt={}):
        self.out_depth = opt['in_depth']
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.num_inputs = opt['in_sx'] * opt['in_sy'] * opt['in_depth']
        self.layer_type = 'add'

        self.skip = getopt(opt, 'skip', 0) # skip n activations in input
        self.delta = getopt(opt, 'delta', [0] * self.num_inputs)
        self.num_neurons = getopt(opt, 'num_neurons', self.num_inputs)

    def forward(self, V, is_training):
        self.in_act = V

        A = Vol(1, 1, self.num_inputs, 0.0)
        applied = 0
        for n in xrange(self.num_inputs):
            if n < self.skip:
                A.w[n] = V.w[n]
            else:
                A.w[n] = V.w[n] + self.delta[n - self.skip]
                applied += 1
            if applied == self.num_neurons:
                break

        self.out_act = A
        return self.out_act

    def backward(self):
        pass

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']