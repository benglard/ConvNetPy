from util import *
from math import sqrt

class Vol(object):

    """
    Vol is the basic building block of all data in a net.
    it is essentially just a 3D volume of numbers, with a
    width (sx), height (sy), and depth (depth).
    It is used to hold data for all filters, all volumes,
    all weights, and also stores all gradients w.r.t. 
    the data. c is optionally a value to initialize the volume
    with. If c is missing, fills the Vol with random numbers.
    """

    def __init__(self, sx, sy=None, depth=None, c=None):

        # if sx is a list
        if type(sx) in (list, tuple):
            # We were given a list in sx, assume 1D volume and fill it up
            self.sx = 1
            self.sy = 1
            self.depth = len(sx)

            self.w = zeros(self.depth)
            self.dw = zeros(self.depth)
            self.w = [ sx[i] for i in xrange(self.depth) ]
        else: 
            # We were given dimensions of the vol           
            self.sx = sx
            self.sy = sy
            self.depth = depth
            n = sx * sy * depth
            self.w = zeros(n)
            self.dw = zeros(n)

            if c == None:
                # Weight normalization is done to equalize the output
                # variance of every neuron, otherwise neurons with a lot
                # of incoming connections have outputs of larger variance
                scale = sqrt(1.0 / (self.sx * self.sy * self.depth))
                self.w = [ randn(0.0, scale) for i in xrange(n) ]
            else:
                self.w = [c] * n

    def __str__(self):
        return '\n{}{}\n{}{}\n'.format(
            'W:', self.w,
            'DW:', self.dw
        )

    __repr__ = __str__

    @property
    def size(self):
        return (self.sx, self.sy, self.depth)

    def get(self, x, y, d):
        ix = ((self.sx * y) + x) * self.depth + d
        return self.w[ix]

    def set(self, x, y, d, v):
        ix = ((self.sx * y) + x) * self.depth + d
        self.w[ix] = v

    def add(self, x, y, d, v):
        ix = ((self.sx * y) + x) * self.depth + d
        self.w[ix] += v

    def get_grad(self, x, y, d):
        ix = ((self.sx * y) + x) * self.depth + d
        return self.dw[ix]

    def set_grad(self, x, y, d, v):
        ix = ((self.sx * y) + x) * self.depth + d
        self.dw[ix] = v

    def add_grad(self, x, y, d, v):
        ix = ((self.sx * y) + x) * self.depth + d
        self.dw[ix] += v

    def cloneAndZero(self):
        return Vol(self.sx, self.sy, self.depth, 0.0)

    def clone(self): 
        V = Vol(self.sx, self.sy, self.depth, 0.0)
        n = len(self.w)
        for i in range(n):
            V.w[i] = self.w[i]
        return V

    def addFrom(self, V):
        for i in xrange(len(self.w)):
            self.w[i] += V.w[i]

    def addFromScaled(self, V, a):
        for i in xrange(len(self.w)):
            self.w[i] += a * V.w[i]

    def setConst(self, a):
        self.w = [a] * len(self.w)

    def toJSON(self):
        return {
            'sx' : self.sx,
            'sy' : self.sy,
            'depth' : self.depth,
            'w' : self.w
        }

    def fromJSON(self, json):
        self.sx = json['sx']
        self.sy = json['sy']
        self.depth = json['depth']

        n = self.sx * self.sy * self.depth
        self.w = zeros(n)
        self.dw = zeros(n)
        self.addFrom(json['w'])

        return self