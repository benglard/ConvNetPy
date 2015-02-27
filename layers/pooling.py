from util import getopt, zeros
from vol import Vol
from math import floor

class PoolLayer(object):

    """
    Max pooling layer: finds areas of max activation
    http://deeplearning.net/tutorial/lenet.html#maxpooling
    """

    def __init__(self, opt={}):
        self.sx = opt['sx'] # filter size
        self.in_depth = opt['in_depth']
        self.in_sx = opt['in_sx']
        self.in_sy = opt['in_sy']

        # optional
        self.sy = getopt(opt, 'sy', self.sx)
        self.stride = getopt(opt, 'stride', 2)
        self.pad = getopt(opt, 'pad', 0) # padding to borders of input volume

        self.out_depth = self.in_depth
        self.out_sx = int(floor((self.in_sx - self.sx + 2 * self.pad) / self.stride + 1))
        self.out_sy = int(floor((self.in_sy - self.sy + 2 * self.pad) / self.stride + 1))
        self.layer_type = 'pool'

        # Store switches for x,y coordinates for where the max comes from, for each output neuron
        switch_size = self.out_sx * self.out_sy * self.out_depth
        self.switch_x = zeros(switch_size)
        self.switch_y = zeros(switch_size)

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)
        switch_counter = 0

        for d in xrange(self.out_depth):
            x = -self.pad
            y = -self.pad
            for ax in xrange(self.out_sx):
                y = -self.pad
                for ay in xrange(self.out_sy):
                    # convolve centered at this particular location
                    max_a = -99999
                    win_x, win_y = -1, -1
                    for fx in xrange(self.sx):
                        for fy in xrange(self.sy):
                            off_x = x + fx
                            off_y = y + fy
                            if off_y >= 0 and off_y < V.sy \
                            and off_x >= 0 and off_x < V.sx:
                                v = V.get(off_x, off_y, d)
                                # max pool
                                if v > max_a:
                                    max_a = v
                                    win_x = off_x
                                    win_y = off_y

                    self.switch_x[switch_counter] = win_x
                    self.switch_y[switch_counter] = win_y
                    switch_counter += 1
                    A.set(ax, ay, d, max_a)

                    y += self.stride
                x += self.stride

        self.out_act = A
        return self.out_act

    def backward(self):
        # pooling layers have no parameters, so simply compute
        # gradient wrt data here
        V = self.in_act
        V.dw = zeros(len(V.w)) # zero out gradient wrt data
        A = self.out_act # computed in forward pass

        n = 0
        for d in xrange(self.out_depth):
            x = -self.pad
            y = -self.pad
            for ax in xrange(self.out_sx):
                y = -self.pad
                for ay in xrange(self.out_sy):
                    chain_grad = self.out_act.get_grad(ax, ay, d)
                    V.add_grad(self.switch_x[n], self.switch_y[n], d, chain_grad)
                    n += 1
                    y += self.stride
                x += self.stride

    def getParamsAndGrads(self): 
        return []

    def toJSON(self):
        return {
            'sx'        : self.sx,
            'sy'        : self.sy,
            'stride'    : self.stride,
            'in_depth'  : self.in_depth,
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'pad'       : self.pad,
            'layer_type': self.layer_type
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']
        self.sx         = json['sx']
        self.sy         = json['sy']
        self.stride     = json['stride']
        self.in_depth   = json['depth']
        self.pad        = json['pad']

        switch_size = self.out_sx * self.out_sy * self.out_depth
        self.switch_x = zeros(switch_size)
        self.switch_y = zeros(switch_size)