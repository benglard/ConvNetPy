from util import *
from time import time
from math import sqrt

class Trainer(object):

    def __init__(self, net, options={}):

        """
        Manages Trainers:
            1. Vanilla SGD
            2. Momentum
            3. Adagrad
            4. Adadelta
            5. Windowgrad
        """

        self.opt = options
        self.net = net
        self.learning_rate = getopt(options, 'learning_rate', 0.01)
        self.l1_decay = getopt(options, 'l1_decay', 0.0)
        self.l2_decay = getopt(options, 'l2_decay', 0.0)
        self.batch_size = getopt(options, 'batch_size', 1)
        self.method = getopt(options, 'method', 'sgd') # sgd/adagrad/adadelta/windowgrad

        self.momentum = getopt(options, 'momentum', 0.9)
        self.ro = getopt(options, 'ro', 0.95) # used in adadelta
        self.eps = getopt(options, 'eps', (10) ** (-6)) # used in adadelta

        self.k = 0 # iteration counter
        self.gsum = [] # last iteration gradients (used for momentum calculations)
        self.xsum = [] # used in adadelta

        self.win = Window()

    def __str__(self): return str(self.opt)
    __repr__ = __str__

    def train(self, x, y):
        self.k += 1
        
        start = time()
        self.net.forward(x, True) # we are training
        end = time()
        fwd_time = end - start

        if type(y) is not list:
            self.win.add(self.net.getPrediction() == y)

        start = time()
        cost_loss = self.net.backward(y)        
        l2_decay_loss = 0.0
        l1_decay_loss = 0.0
        end = time()
        bwd_time = end - start

        if self.k % self.batch_size == 0:
            pglist = self.net.getParamsAndGrads()

            # Initialize lists for accumulators. Will only be done once on first iteration
            if (len(self.gsum) == 0 and (self.method != 'sgd' or self.momentum > 0.0)):
                """
                Only vanilla sgd doesnt need either lists
                    momentum needs gsum
                    adagrad needs gsum
                    adadelta needs gsum and xsum
                """
                for elem in pglist:
                    self.gsum.append(zeros(len(elem['params'])))
                    if self.method == 'adadelta':
                        self.xsum.append(zeros(len(elem['params'])))
                    else:
                        self.xsum.append([])

            # Perform an update for all sets of weights
            for i in xrange(len(pglist)):
                pg = pglist[i] # param, gradient, other options in future (custom learning rate etc.)
                p = pg['params']
                g = pg['grads']

                # Learning rate for some parameters
                l2_decay_mul = getopt(pg, 'l2_decay_mul', 1.0)
                l1_decay_mul = getopt(pg, 'l1_decay_mul', 1.0)
                l2_decay = self.l2_decay * l2_decay_mul
                l1_decay = self.l1_decay * l1_decay_mul
                for j in xrange(len(p)):
                    l2_decay_loss += l2_decay * p[j] * p[j] / 2.0 # accumulate weight decay loss
                    l1_decay_loss += l1_decay * abs(p[j])
                    l1grad = l1_decay * (1 if p[j] > 0 else -1)
                    l2grad = l2_decay * p[j]
                    gij = (l2grad + l1grad + g[j]) / float(self.batch_size) # raw batch gradient

                    try:
                        gsumi = self.gsum[i]
                        xsumi = self.xsum[i]
                    except:
                        pass
                    if self.method == 'adagrad': # adagrad update
                        gsumi[j] += gij * gij
                        dx = - self.learning_rate / sqrt(gsumi[j] + self.eps) * gij
                        p[j] += dx
                    elif self.method == 'windowgrad':
                        """
                        This is adagrad but with a moving window weighted average
                        so the gradient is not accumulated over the entire history of the run. 
                        It's also referred to as Idea #1 in Zeiler paper on Adadelta.
                        """
                        gsumi[j] = self.ro * gsumi[j] + (1 - self.ro) * gij * gij
                        dx = - self.learning_rate / sqrt(gsumi[j] + self.eps) * gij
                        p[j] += dx
                    elif self.method == 'adadelta':
                        gsumi[j] = self.ro * gsumi[j] + (1 - self.ro) * gij * gij
                        dx = - sqrt((xsumi[j] + self.eps) / (gsumi[j] + self.eps)) * gij
                        xsumi[j] = self.ro * gsumi[j] + (1 - self.ro) * dx * dx
                        p[j] += dx
                    else: # SGD
                        if self.momentum > 0.0: # Momentum update
                            dx = self.momentum * gsumi[j] - self.learning_rate * gij # step
                            gsumi[j] = dx                                            # backup for next iteration
                            p[j] += dx                                               # apply gradient
                        else: # Vanilla SGD
                            p[j] += - self.learning_rate * gij
                    g[j] = 0.0 # zero out gradient so that we can begin accumulating anew

        return {
            'k': self.k,
            'fwd_time': fwd_time,
            'bwd_time': bwd_time,
            'time': fwd_time + bwd_time,
            'l2_decay_loss': l2_decay_loss,
            'l1_decay_loss': l1_decay_loss,
            'cost_loss': cost_loss,
            'loss': cost_loss + l1_decay_loss + l2_decay_loss,
            'accuracy': self.win.get_average()
        }