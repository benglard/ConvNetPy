from util import getopt, zeros
from vol import Vol
from math import sqrt, log, exp

class SimilarityLayer(object):

    """
    Computes similarity measures, generalization of dot products 
    http://arxiv.org/pdf/1410.0781.pdf
    """

    def __init__(self, opt={}):
        self.out_depth = opt['num_neurons']
        self.l1_decay_mul = getopt(opt, 'l1_decay_mul', 0.0)
        self.l2_decay_mul = getopt(opt, 'l2_decay_mul', 1.0)

        self.num_inputs = opt['in_sx'] * opt['in_sy'] * opt['in_depth']
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = 'sim'

        bias = getopt(opt, 'bias_pref', 0.0)
        self.filters = [ Vol(1, 1, self.num_inputs) for i in xrange(self.out_depth) ]
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, in_training):
        self.in_act = V
        A = Vol(1, 1, self.out_depth, 0.0)
        Vw = V.w
        
        def norm(vec):
            return sqrt(sum(c * c for c in vec))
        
        normv = norm(Vw)

        # compute cos sim between V and filters
        for i in xrange(self.out_depth):
            sum_a = 0.0
            fiw = self.filters[i].w
            for d in xrange(self.num_inputs):
                sum_a += Vw[d] * fiw[d]
            sum_a += self.biases.w[i] # dot(W, v) + b
            
            normf = norm(fiw)
            try:
                A.w[i] = sum_a / (normv * normf)
            except:
                A.w[i] = 0

        self.out_act = A
        return self.out_act

    def backward(self):
        V = self.in_act
        V.dw = zeros(len(V.w)) # zero out gradient

        # compute gradient wrt weights and data
        for i in xrange(self.out_depth):
            fi = self.filters[i]
            chain_grad = self.out_act.dw[i]

            for d in xrange(self.num_inputs):
                V.dw[d] += fi.w[d] * chain_grad #grad wrt input data
                fi.dw[d] += V.w[d] * chain_grad #grad wrt params

            self.biases.dw[i] += chain_grad

    def getParamsAndGrads(self):
        response = []
        for d in xrange(self.out_depth):
            response.append({
                'params': self.filters[d].w,
                'grads': self.filters[d].dw,
                'l2_decay_mul': self.l2_decay_mul,
                'l1_decay_mul': self.l1_decay_mul
            })
        response.append({
            'params': self.biases.w,
            'grads': self.biases.dw,
            'l2_decay_mul': 0.0,
            'l1_decay_mul': 0.0
        })
        return response

    def toJSON(self):
        return {
            'out_depth'   : self.out_depth,
            'out_sx'      : self.out_sx,
            'out_sy'      : self.out_sy,
            'layer_type'  : self.layer_type,
            'num_inputs'  : self.num_inputs,
            'l1_decay_mul': self.l1_decay_mul,
            'l2_decay_mul': self.l2_decay_mul,
            'filters'     : [ f.toJSON() for f in self.filters ],
            'biases'      : self.biases.toJSON()
        }

    def fromJSON(self, json):
        self.out_depth    = json['out_depth']
        self.out_sx       = json['out_sx']
        self.out_sy       = json['out_sy']
        self.layer_type   = json['layer_type']
        self.num_inputs   = json['num_inputs']
        self.l1_decay_mul = json['l1_decay_mul']
        self.l2_decay_mul = json['l2_decay_mul']
        self.filters      = [ Vol(0, 0, 0, 0).fromJSON(f) for f in json['filters'] ]
        self.biases       = Vol(0, 0, 0, 0).fromJSON(json['biases'])

class MexLayer(object):

    """
    Computes log(sum(exp())) to fill "the role of activation
    functions, max or average pooling, and weights necessary
    for classification."
    http://arxiv.org/pdf/1410.0781.pdf
    """

    def __init__(self, opt={}):
        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth']

        self.zeta = getopt(opt, 'zeta', 1.0)
        if self.zeta == 0.0:
            print 'WARNING: zeta cannot equal 0'

        self.layer_type = 'mex'

    def forward(self, V, is_training):
        self.in_act = V
        V2 = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)

        sexp = 0.0
        for i in xrange(len(V2.w)):
            sexp += exp(V.w[i])
            V2.w[i] = log((sexp / (i + 1))) / self.zeta

        self.out_act = V2
        return self.out_act

    def backward(self):
        # same as relu

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