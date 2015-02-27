from vol import Vol
from vol_util import augment
from net import Net 
from trainers import Trainer
from random import sample

import os, struct, sys
from array import array as pyarray

training_data = None
testing_data = None
n = None
t = None

training_data2 = None
n2 = None
t2 = None

# Load mnist data
def load_data(training=True):
    """Adapted from http://g.sweyla.com/blog/2012/mnist-numpy/"""
    path = './data'

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
    V = Vol(28, 28, 1, 0.0)
    for img in imgs:
        V.w = [ (px / 255.0) for px in img ]
        inputs.append(augment(V, 24))

    # Outputs
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    labels = pyarray("b", flbl.read())
    flbl.close()

    return zip(inputs, labels)

def run_big_net():
    global training_data, testing_data, n, t, training_data2

    training_data = load_data()
    testing_data = load_data(False)
    training_data2 = []

    print 'Data loaded...'

    layers = []
    layers.append({'type': 'input', 'out_sx': 24, 'out_sy': 24, 'out_depth': 1})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'relu', 'drop_prob': 0.5})
    #layers.append({'type': 'fc', 'num_neurons': 800, 'activation': 'relu', 'drop_prob': 0.5})
    layers.append({'type': 'softmax', 'num_classes': 10})
    print 'Layers made...'

    n = Net(layers)
    print 'Net made...'
    print n

    t = Trainer(n, {'method': 'sgd', 'momentum': 0.0})
    print 'Trainer made...'

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  ', 'training accuracy'
    print '----------------------------------------------------'
    try:
        for x, y in training_data: 
            stats = t.train(x, y)
            print stats['k'], stats['time'], stats['loss'], stats['accuracy']
            training_data2.append((x, n.getPrediction()))
    except: #hit control-c or other
        pass

    print 'In testing: 5000 trials'
    right = 0
    count = 5000
    for x, y in sample(testing_data, count):
        n.forward(x)
        right += n.getPrediction() == y
    accuracy = float(right) / count * 100
    print accuracy

def run_small_net():
    global training_data2, n2, t2, testing_data

    layers = []
    layers.append({'type': 'input', 'out_sx': 24, 'out_sy': 24, 'out_depth': 1})
    #layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'relu'})
    layers.append({'type': 'softmax', 'num_classes': 10})
    print 'Layers made...'

    n2 = Net(layers)
    print 'Smaller Net made...'
    print n2

    t2 = Trainer(n2, {'method': 'sgd', 'momentum': 0.0})
    print 'Trainer made for smaller net...'

    print 'In training of smaller net...'
    print 'k', 'time\t\t  ', 'loss\t  ', 'training accuracy'
    print '----------------------------------------------------'
    try:
        for x, y in training_data2: 
            stats = t2.train(x, y)
            print stats['k'], stats['time'], stats['loss'], stats['accuracy']
    except: #hit control-c or other
        pass

    print 'Testing smaller net: 5000 trials'
    right = 0
    count = 5000
    for x, y in sample(testing_data, count):
        n2.forward(x)
        right += n2.getPrediction() == y
    accuracy = float(right) / count * 100
    print accuracy