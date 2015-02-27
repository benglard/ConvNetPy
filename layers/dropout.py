from util import getopt, zeros
from random import random

class DropoutLayer(object):

    """
    Randomly omits half the feature detectors on each training case.
    Each neuron tends to learn something 'useful'.
    http://arxiv.org/pdf/1207.0580.pdf
    """

    def __init__(self, opt={}):
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth']
        self.layer_type = 'dropout'
        self.drop_prob = getopt(opt, 'drop_prob', 0.5)
        self.dropped = zeros(self.out_sx * self.out_sy * self.out_depth)

    def forward(self, V, is_training=False):
        self.in_act = V
        V2 = V.clone()
        N = len(V.w)

        if is_training: 
            # do dropout
            for i in xrange(N):
                if random() < self.drop_prob: # drop
                    V2.w[i] = 0.0 
                    self.dropped[i] = True
                else:
                    self.dropped[i] = False
        else: 
            # scale the activations during prediction
            for i in xrange(N):
                V2.w[i] *= self.drop_prob

        self.out_act = V2
        return self.out_act

    def backward(self):
        V = self.in_act
        chain_grad = self.out_act
        N = len(V.w)
        V.dw = zeros(N) # zero out gradient wrt data
        for i in xrange(N):
            if not self.dropped[i]:
                V.dw[i] = chain_grad.dw[i] # copy over the gradient

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type,
            'drop_prob' : self.drop_prob
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']
        self.drop_prob  = json['drop_prob']