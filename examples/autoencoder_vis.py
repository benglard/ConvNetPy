from vol import Vol
from vol_util import augment
from net import Net 
from trainers import Trainer
from random import sample
from util import maxmin

import numpy, cv2 # requires numpy and opencv

import os, struct
from array import array as pyarray
from subprocess import call

training_data = None
testing_data = None
n = None
t = None

# Load mnist data
def load_data(training=True):
    """Adapted from http://g.sweyla.com/blog/2012/mnist-numpy/"""
    path = "./data"

    if training:
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    else:
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    # Inputs
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    imgs = pyarray("B", fimg.read())
    fimg.close()

    imgs = [imgs[n:n+784] for n in xrange(0, len(imgs), 784)]
    inputs = []
    for img in imgs:
        V = Vol(28, 28, 1, 0.0)
        V.w = [ (px / 255.0) for px in img ]
        inputs.append(V)

    # Outputs
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    labels = pyarray("b", flbl.read())
    flbl.close()

    return zip(inputs, labels)

def start():
    global training_data, testing_data, n, t

    training_data = load_data()
    testing_data = load_data(False)

    print 'Data loaded...'

    layers = []
    layers.append({'type': 'input', 'out_sx': 28, 'out_sy': 28, 'out_depth': 1})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'tanh'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'tanh'})
    layers.append({'type': 'fc', 'num_neurons': 2, 'activation': 'tanh'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'tanh'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'tanh'})
    layers.append({'type': 'regression', 'num_neurons': 28 * 28})
    print 'Layers made...'

    n = Net(layers)
    print 'Net made...'
    print n

    t = Trainer(n, {'method': 'adadelta', 'learning_rate': 1.0, 'batch_size': 50, 'l2_decay': 0.001, 'l1_decay': 0.001});
    print 'Trainer made...'

def train():
    global training_data, testing_data, n, t

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  '
    print '----------------------------------------------------'
    try:
        for x, y in training_data: 
            stats = t.train(x, x.w)
            print stats['k'], stats['time'], stats['loss']

            if stats['k'] % 1000 == 0: test()
    except:
        return

def test():
    global training_data, testing_data, n, t

    print 'In autoencoder testing'
    test_n = 100

    xcodes = []
    ycodes = []
    xs = []
    for x, y in sample(testing_data, test_n):
        n.forward(x)
        xcode, ycode = n.layers[5].out_act.w
        xcodes.append(xcode)
        ycodes.append(ycode)
        nx = numpy.array(x.w).reshape((28,28))
        xs.append(nx)
    mmx = maxmin(xcodes)
    mmy = maxmin(ycodes)

    dim = 500
    img = numpy.zeros((dim, dim))
    img[:] = 255 #white
    for xcode, ycode, nx in zip(xcodes, ycodes, xs):
        xpos = (dim - 28 * 2) * (xcode - mmx['minv']) / mmx['dv'] + 28
        ypos = (dim - 28 * 2) * (ycode - mmy['minv']) / mmy['dv'] + 28
        xpos = int(xpos)
        ypos = int(ypos)
        img[ypos:ypos + 28, xpos:xpos + 28] = nx
    cv2.imshow('MNIST', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()