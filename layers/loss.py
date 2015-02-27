from util import getopt, zeros
from vol import Vol
from math import exp, log

"""
Layers that implement a loss. Currently these are the layers that 
can initiate a backward() pass. In future we probably want a more 
flexible system that can accomodate multiple losses to do multi-task
learning, and stuff like that. But for now, one of the layers in this
file must be the final layer in a Net.
"""

class SoftmaxLayer(object):

    """
    This is a classifier, with N discrete classes from 0 to N-1
    it gets a stream of N incoming numbers and computes the softmax
    function (exponentiate and normalize to sum to 1 as probabilities should)
    """

    def __init__(self, opt={}):
        self.num_inputs = opt['in_sx'] * opt['in_sy'] * opt['in_depth']
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = 'softmax'
        self.old = None

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(1, 1, self.out_depth, 0.0)

        # max activation
        max_act = max(V.w) 

        # compute exponentials (carefully to not blow up)
        # normalize
        exps = [ exp(w - max_act) for w in V.w ]
        exps_sum = float(sum(exps))
        exps_norm = [ elem / exps_sum for elem in exps ]

        self.es = exps_norm
        A.w = exps_norm

        self.out_act = A
        return self.out_act

    def backward(self, y):
        # compute and accumulate gradient wrt weights and bias of this layer
        x = self.in_act
        x.dw = zeros(len(x.w))

        for i in xrange(self.out_depth):
            indicator = float(i == y)
            mul = - (indicator - self.es[i])
            x.dw[i] = mul

        # loss is the class negative log likelihood
        try:
            return -log(self.es[y])
        except ValueError:
            return -log(0.001)

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type,
            'num_inputs': self.num_inputs
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']
        self.num_inputs = json['num_inputs']

class RegressionLayer(object):

    """
    Implements an L2 regression cost layer,
    so penalizes \sum_i(||x_i - y_i||^2), where x is its input
    and y is the user-provided array of "correct" values.
    """

    def __init__(self, opt={}):
        self.num_inputs = opt['in_sx'] * opt['in_sy'] * opt['in_depth']
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = 'regression'

    def forward(self, V, is_training):
        self.in_act = V
        self.out_act = V
        return V

    def backward(self, y):
        # y is a list here of size num_inputs
        # compute and accumulate gradient wrt weights and bias of this layer
        x = self.in_act
        x.dw = zeros(len(x.w)) # zero out the gradient of input Vol
        loss = 0.0

        if type(y) == list:
            for i in xrange(self.out_depth):
                dy = x.w[i] - y[i]
                x.dw[i] = dy
                loss += 2 * dy * dy
        else:
            # assume it is a dict with entries dim and val
            # and we pass gradient only along dimension dim to be equal to val
            i = y['dim']
            y_i = y['val']
            dy = x.w[i] - y_i
            x.dw[i] = dy
            loss += 2 * dy * dy

        return loss

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type,
            'num_inputs': self.num_inputs
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']
        self.num_inputs = json['num_inputs']

class SVMLayer(object):

    """Linear SVM classifier"""

    def __init__(self, opt={}):
        self.num_inputs = opt['in_sx'] * opt['in_sy'] * opt['in_depth']
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = 'svm'

    def forward(self, V, is_training):
        self.in_act = V
        self.out_act = V
        return V

    def backward(self, y):
        # compute and accumulate gradient wrt weights and bias of this layer
        x = self.in_act
        x.dw = zeros(len(x.w)) # zero out the gradient of input Vol

        yscore = x.w[y]
        margin = 1.0
        loss = 0.0
        for i in xrange(self.out_depth):
            if -yscore + x.w[i] + margin > 0:
                # Hinge loss: http://en.wikipedia.org/wiki/Hinge_loss
                x.dw[i] += 1
                x.dw[y] -= 1
                loss += -yscore + x.w[i] + margin

        return loss

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type,
            'num_inputs': self.num_inputs
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']
        self.num_inputs = json['num_inputs']