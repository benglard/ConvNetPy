from vol import Vol
from vol_util import augment
from net import Net 
from trainers import Trainer
from random import sample
import numpy
import os
from subprocess import call

training_data = None
testing_data = None
network = None
t = None

def load_data(crop, gray, training=True):
    filename = './data/cifar10_'
    if training:
        filename += 'train.npz'
    else:
        filename += 'test.npz'

    data = numpy.load(filename)
    xs = data['x']
    ys = data['y']

    for i in xrange(len(xs)):
        V = Vol(32, 32, 3, 0.0)
        for d in xrange(3):
            for x in xrange(32):
                for y in xrange(32):
                    px = xs[i][x * 32 + y, d] / 255.0 #- 0.5
                    V.set(x, y, d, px)
        if crop:
            V = augment(V, 24, gray)

        y = ys[i]
        yield V, y

def start(conv, crop, gray):
    global training_data, testing_data, network, t

    training_data = load_data(crop, gray)
    testing_data = load_data(crop, gray, False)

    print 'Data loaded...'

    layers = []

    dim = 24 if crop else 32
    depth = 1 if gray else 3
    layers.append({'type': 'input', 'out_sx': dim, 'out_sy': dim, 'out_depth': depth})

    if conv:
        layers.append({'type': 'conv', 'sx': 5, 'filters': 16, 'stride': 1, 'pad': 2, 'activation': 'relu'}) #, 'drop_prob': 0.5})
        layers.append({'type': 'pool', 'sx': 3, 'stride': 2}) #, 'drop_prob': 0.5})
        layers.append({'type': 'conv', 'sx': 5, 'filters': 20, 'stride': 1, 'pad': 2, 'activation': 'relu'}) #, 'drop_prob': 0.5})
        layers.append({'type': 'pool', 'sx': 2, 'stride': 2}) #, 'drop_prob': 0.5})
        #layers.append({'type': 'lrn', 'alpha': 5 * (10 ** -5), 'beta': 0.75, 'k': 1, 'n': 3, 'drop_prob': 0.5})
    else:
        layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
        #layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
        #layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'regression', 'num_neurons': dim * dim * depth})

    print 'Layers made...'

    network = Net(layers)

    print 'Net made...'
    print network

    t = Trainer(network, {'method': 'sgd', 'batch_size': 4, 'l2_decay': 0.0001});

    print 'Trainer made...'
    print t

def train():
    global training_data, testing_data, network, t

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  '
    print '----------------------------------------------------'
    try:
        for x, y in training_data: 
            stats = t.train(x, x.w)
            print stats['k'], stats['time'], stats['loss']
    except:
        return

def display(python_path, pred, x):
    display_cmd = '; '.join([
        'import cv2, numpy',
        'predw = numpy.rot90(numpy.array({}, dtype=numpy.float64).reshape(32, 32, 3))'.format(pred),
        'cv2.imshow(\'P\', predw)',
        'xw = numpy.rot90(numpy.array({}, dtype=numpy.float64).reshape(32, 32, 3))'.format(x),
        'cv2.imshow(\'X\', xw)',
        'cv2.waitKey(0)',
        'cv2.destroyAllWindows()'
    ])

    cmd = '{} -c \"{}\"'.format(os.path.join(python_path, 'python'), display_cmd)
    call(cmd, shell=True)

def test(path=None, test_n=None):
    global training_data, testing_data, network, t

    print 'In autoencoder testing'

    path = path or '/opt/local/bin'
    test_n = test_n or 5

    #for x, y in sample(testing_data, test_n):
    c = 0
    while c < test_n: 
        x, y = next(testing_data)
        print y
        pred = network.forward(x).w
        display(path, pred, x.w)
        c += 1