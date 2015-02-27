from util import zeros

class LocalResponseNormalizationLayer(object):

    """
    Local Response Normalization in window, along depths of volumes
    https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)
    but 1 is replaced with k. This creates local competition among neurons 
    along depth, independently at every particular location in the input volume.
    """

    def __init__(self, opt={}):
        self.k = opt['k']
        self.n = opt['n']
        self.alpha = opt['alpha']
        self.beta = opt['beta']

        self.out_sx = opt['in_sx']
        self.out_sy = opt['in_sy']
        self.out_depth = opt['in_depth']
        self.layer_type = 'lrn'

        if self.n % 2 == 0:
            print 'Warning: n should be odd for LRN layer.'

    def forward(self, V, in_training):
        self.in_act = V
        A = V.cloneAndZero()
        self.S_cache = V.cloneAndZero()
        n2 = self.n / 2

        for x in xrange(V.sx):
            for y in xrange(V.sy):
                for i in xrange(V.depth):
                    a_i = V.get(x, y, i)

                    # Normalize in a window of size n
                    den = 0.0
                    for j in xrange(max(0, i - n2), min(i + n2, V.depth - 1) + 1):
                        u_f = V.get(x, y, j)
                        den += u_f * u_f
                    den *= self.alpha / (float(self.n) ** 2)
                    den += self.k
                    self.S_cache.set(x, y, i, den) # will be useful for backprop
                    den = den ** self.beta

                    A.set(x, y, i, a_i / den)

        self.out_act = A
        return self.out_act

    def backward(self):
        # evaluate gradient wrt data
        V = self.in_act
        V.dw = zeros(len(V.w)) # zero out gradient wrt data
        A = self.out_act
        n2 = self.n / 2

        for x in xrange(V.sx):
            for y in xrange(V.sy):
                for i in xrange(V.depth):
                    chain_grad = self.out_act.get_grad(x, y, i)
                    S = self.S_cache.get(x, y, i)
                    S_b = S ** self.beta
                    S_b2 = S_b * S_b

                    # Normalize in a window of size n
                    for j in xrange(max(0, i - n2), min(i + n2, V.depth - 1) + 1):
                        a_j = V.get(x, y, j)
                        grad = -(a_j ** 2) * self.beta * (S ** (self.beta - 1)) * self.alpha / self.n * 2.0 
                        if j == i:
                            grad += S_b
                        grad /= S_b2
                        grad *= chain_grad
                        V.add_grad(x, y, j, grad)

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'k'         : self.k,
            'n'         : self.n,
            'alpha'     : self.alpha,
            'beta'      : self.beta,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'out_depth' : self.out_depth,
            'layer_type': self.layer_type
        }

    def fromJSON(self, json):
        self.k          = json['k']
        self.n          = json['n']
        self.alpha      = json['alpha']
        self.beta       = json['beta']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.out_depth  = json['out_depth']
        self.layer_type = json['layer_type']