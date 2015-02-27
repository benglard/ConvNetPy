from util import getopt, zeros
from vol import Vol
from math import exp, tanh, log

# Helper functions

def relu(x):
    return max(0, x)

def sigmoid(x):
    try: 
        return 1.0 / (1 + exp(-x))
    except:
        return 0

class ReluLayer(object):

    """
    Implements ReLU nonlinearity elementwise
    x -> max(0, x)
    the output is in [0, inf)
    """

    def __init__(self, opt={}):
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth']
        self.layer_type = 'relu'

    def forward(self, V, is_training=False):
        self.in_act = V
        V2 = V.clone()
        V2.w = map(relu, V.w)
        self.out_act = V2
        return self.out_act

    def backward(self):
        V = self.in_act
        V2 = self.out_act
        N = len(V.w)
        V.dw = zeros(N) # zero out gradient wrt data
        for i in xrange(N):
            if V2.w[i] <= 0: # threshold
                V.dw[i] = 0
            else:
                V.dw[i] = V2.dw[i]

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

class SigmoidLayer(object):

    """
    Implements Sigmoid nnonlinearity elementwise
    x -> 1/(1+e^(-x))
    so the output is between 0 and 1
    """

    def __init__(self, opt={}):
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth']
        self.layer_type = 'sigmoid'

    def forward(self, V, is_training):
        self.in_act = V
        V2 = V.cloneAndZero()
        V2.w = map(sigmoid, V.w)
        self.out_act = V2
        return self.out_act

    def backward(self):
        V = self.in_act
        V2 = self.out_act
        N = len(V.w)
        V.dw = zeros(N) # zero out gradient wrt data
        for i in xrange(N):
            v2wi = V2.w[i]
            V.dw[i] = v2wi * (1.0 - v2wi) * V2.dw[i]

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

class MaxoutLayer(object):

    """
    Implements Maxout nnonlinearity that computes:
    x -> max(x)
    where x is a vector of size group_size. Ideally of course,
    the input size should be exactly divisible by group_size.
    """

    def __init__(self, opt={}):
        self.group_size = getopt(opt, 'group_size', 2)
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth'] / self.group_size
        self.layer_type = 'maxout'
        self.switches = zeros(self.out_sx * self.out_sy * self.out_depth)

    def forward(self, V, is_training):
        self.in_act = V
        N = self.out_depth
        V2 = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)

        if self.out_sx == 1 and self.out_sy == 1:
            for i in xrange(N):
                offset = i * self.group_size
                m = max(V.w[offset:])
                index = V.w[offset:].index(m)
                V2.w[i] = m
                self.switches[i] = offset + index 
        else:
            switch_counter = 0
            for x in xrange(V.sx):
                for y in xrange(V.sy):
                    for i in xrange(N):
                        ix = i * self.group_size
                        elem = V.get(x, y, ix)
                        elem_i = 0
                        for j in range(1, self.group_size):
                            elem2 = V.get(x, y, ix + j)
                            if elem2 > elem:
                                elem = elem2
                                elem_i = j
                        V2.set(x, y, i, elem)
                        self.switches[i] = ix + elem_i
                        switch_counter += 1

        self.out_act = V2
        return self.out_act

    def backward(self):
        V = self.in_act
        V2 = self.out_act
        N = self.out_depth
        V.dw = zeros(len(V.w)) # zero out gradient wrt data

        # pass the gradient through the appropriate switch
        if self.sx == 1 and self.sy == 1:
            for i in range(N):
                chain_grad = V2.dw[i]
                V.dw[self.switches[i]] = chain_grad
        else:
            switch_counter = 0
            for x in xrange(V2.sx):
                for y in xrange(V2.sy):
                    for i in xrange(N):
                        chain_grad = V2.get_grad(x,y,i)
                        V.set_grad(x, y, self.switches[n], chain_grad)
                        switch_counter += 1

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type,
            'group_size': self.group_size
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']
        self.group_size = json['group_size']
        self.switches   = zeros(self.group_size)

class TanhLayer(object):

    """
    Implements Tanh nnonlinearity elementwise
    x -> tanh(x) 
    so the output is between -1 and 1.
    """

    def __init__(self, opt={}):
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth']
        self.layer_type = 'tanh'

    def forward(self, V, is_training):
        self.in_act = V
        V2 = V.cloneAndZero()
        V2.w = map(tanh, V.w)
        self.out_act = V2
        return self.out_act

    def backward(self):
        V = self.in_act
        V2 = self.out_act
        N = len(V.w)
        V.dw = zeros(N) # zero out gradient wrt data
        for i in xrange(N):
            v2wi = V2.w[i]
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i]

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