from util import getopt, zeros
from vol import Vol
from math import floor

"""This file contains all layers that do dot products with input"""

class ConvLayer(object):

    """
    Performs convolutions: spatial weight sharing
    http://deeplearning.net/tutorial/lenet.html
    """

    def __init__(self, opt={}):
        self.out_depth = opt['filters']
        self.sx = opt['sx'] # filter size: should be odd if possible
        self.in_depth = opt['in_depth']
        self.in_sx = opt['in_sx']
        self.in_sy = opt['in_sy']

        # optional
        self.sy = getopt(opt, 'sy', self.sx)
        self.stride = getopt(opt, 'stride', 1) # stride at which we apply filters to input volume
        self.pad = getopt(opt, 'pad', 0) # padding to borders of input volume
        self.l1_decay_mul = getopt(opt, 'l1_decay_mul', 0.0)
        self.l2_decay_mul = getopt(opt, 'l2_decay_mul', 1.0)

        """
        Note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        final application.
        """
        self.out_sx = int(floor((self.in_sx - self.sx + 2 * self.pad) / self.stride + 1))
        self.out_sy = int(floor((self.in_sy - self.sy + 2 * self.pad) / self.stride + 1))
        self.layer_type = 'conv'

        bias = getopt(opt, 'bias_pref', 0.0)
        self.filters = [ Vol(self.sx, self.sy, self.in_depth) for i in xrange(self.out_depth) ]
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)

        v_sx = V.sx
        v_sy = V.sy
        xy_stride = self.stride

        for d in xrange(self.out_depth):
            f = self.filters[d]
            x = -self.pad
            y = -self.pad

            for ay in xrange(self.out_sy):
                x = -self.pad
                for ax in xrange(self.out_sx):
                    # convolve centered at this particular location
                    sum_a = 0.0
                    for fy in xrange(f.sy):
                        off_y = y + fy
                        for fx in xrange(f.sx):
                            # coordinates in the original input array coordinates
                            off_x = x + fx
                            if off_y >= 0 and off_y < V.sy and off_x >= 0 and off_x < V.sx:
                                for fd in xrange(f.depth):
                                    sum_a += f.w[((f.sx * fy) + fx) * f.depth + fd] \
                                    * V.w[((V.sx * off_y) + off_x) * V.depth + fd]

                    sum_a += self.biases.w[d]
                    A.set(ax, ay, d, sum_a)

                    x += xy_stride
                y += xy_stride

        self.out_act = A
        return self.out_act

    def backward(self):
        # compute gradient wrt weights, biases and input data
        V = self.in_act
        V.dw = zeros(len(V.w)) # zero out gradient

        V_sx = V.sx
        V_sy = V.sy
        xy_stride = self.stride
        
        for d in xrange(self.out_depth):
            f = self.filters[d]
            x = -self.pad
            y = -self.pad
            for ay in xrange(self.out_sy):
                x = -self.pad
                for ax in xrange(self.out_sx):
                    # convolve and add up the gradients
                    chain_grad = self.out_act.get_grad(ax, ay, d) # gradient from above, from chain rule
                    for fy in xrange(f.sy):
                        off_y = y + fy
                        for fx in xrange(f.sx):
                            off_x = x + fx
                            if off_y >= 0 and off_y < V_sy and off_x >= 0 and off_x < V_sx:
                                # forward prop calculated: a += f.get(fx, fy, fd) * V.get(ox, oy, fd)
                                #f.add_grad(fx, fy, fd, V.get(off_x, off_y, fd) * chain_grad)
                                #V.add_grad(off_x, off_y, fd, f.get(fx, fy, fd) * chain_grad)
                                for fd in xrange(f.depth):
                                    ix1 = ((V.sx * off_y) + off_x) * V.depth + fd
                                    ix2 = ((f.sx * fy) + fx) * f.depth + fd
                                    f.dw[ix2] += V.w[ix1] * chain_grad
                                    V.dw[ix1] += f.w[ix2] * chain_grad

                    self.biases.dw[d] += chain_grad
                    x += xy_stride
                y += xy_stride

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
            'sx'          : self.sx,
            'sy'          : self.sy,
            'stride'      : self.stride,
            'in_depth'    : self.in_depth,
            'out_depth'   : self.out_depth,
            'out_sx'      : self.out_sx,
            'out_sy'      : self.out_sy,
            'layer_type'  : self.layer_type,
            'l1_decay_mul': self.l1_decay_mul,
            'l2_decay_mul': self.l2_decay_mul,
            'pad'         : self.pad,
            'filters'     : [ f.toJSON() for f in self.filters ],
            'biases'      : self.biases.toJSON()
        }

    def fromJSON(self, json):
        self.sx           = json['sx']
        self.sy           = json['sy']
        self.stride       = json['stride']
        self.in_depth     = json['in_depth']
        self.out_depth    = json['out_depth']
        self.out_sx       = json['out_sx']
        self.out_sy       = json['out_sy']
        self.layer_type   = json['layer_type']
        self.l1_decay_mul = json['l1_decay_mul']
        self.l2_decay_mul = json['l2_decay_mul']
        self.pad          = json['pad']
        self.filters      = [ Vol(0, 0, 0, 0).fromJSON(f) for f in json['filters'] ]
        self.biases       = Vol(0, 0, 0, 0).fromJSON(json['biases'])

class FullyConnectedLayer(object):

    """
    Fully connected dot products, ie. multi-layer perceptron net
    Building block for most networks
    http://www.deeplearning.net/tutorial/mlp.html
    """

    def __init__(self, opt={}):
        self.out_depth = opt['num_neurons']
        self.l1_decay_mul = getopt(opt, 'l1_decay_mul', 0.0)
        self.l2_decay_mul = getopt(opt, 'l2_decay_mul', 1.0)

        self.num_inputs = opt['in_sx'] * opt['in_sy'] * opt['in_depth']
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = 'fc'

        bias = getopt(opt, 'bias_pref', 0.0)
        self.filters = [ Vol(1, 1, self.num_inputs) for i in xrange(self.out_depth) ]
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, in_training):
        self.in_act = V
        A = Vol(1, 1, self.out_depth, 0.0)
        Vw = V.w

        # dot(W, x) + b
        for i in xrange(self.out_depth):
            sum_a = 0.0
            fiw = self.filters[i].w
            for d in xrange(self.num_inputs):
                sum_a += Vw[d] * fiw[d]
            sum_a += self.biases.w[i]
            A.w[i] = sum_a

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