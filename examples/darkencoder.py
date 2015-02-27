from vol import Vol
from vol_util import augment
from net import Net 
from trainers import Trainer
from random import sample

import os, struct
from array import array as pyarray
from subprocess import call

training_data = None
training_data2 = None
testing_data = None
n = None
n2 = None
t = None
t2 = None

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
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'regression', 'num_neurons': 28 * 28})

    print 'Layers made...'

    n = Net(layers)

    print 'Net made...'
    print n

    t = Trainer(n, {'method': 'adadelta', 'batch_size': 20, 'l2_decay': 0.001});

    print 'Trainer made...'

def train():
    global training_data, n, t, training_data2

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  '
    print '----------------------------------------------------'
    training_data2 = []
    try:
        for x, y in training_data: 
            stats = t.train(x, x.w)
            print stats['k'], stats['time'], stats['loss']
            training_data2.append((Vol(n.forward(x).w), y))
    except: #hit control-c or other
        return

def train2():
    global training_data2, n2, t2

    layers = []
    layers.append({'type': 'input', 'out_sx': 28, 'out_sy': 28, 'out_depth': 1})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'softmax', 'num_classes': 10})
    print 'Layers made...'

    n2 = Net(layers)
    print 'Net made...'
    print n2

    t2 = Trainer(n2, {'method': 'adadelta', 'batch_size': 20, 'l2_decay': 0.001});
    print 'Trainer made...' 

    print 'In training of smaller net...'
    print 'k', 'time\t\t  ', 'loss\t  ', 'training accuracy'
    print '----------------------------------------------------'
    try:
        for x, y in training_data2: 
            stats = t2.train(x, y)
            print stats['k'], stats['time'], stats['loss'], stats['accuracy']
    except: #hit control-c or other
        return

def test():
    global testing_data, n2

    print 'Testing smaller net: 5000 trials'
    right = 0
    count = 5000
    for x, y in sample(testing_data, count):
        n2.forward(x)
        right += n2.getPrediction() == y
    accuracy = float(right) / count * 100
    print accuracy