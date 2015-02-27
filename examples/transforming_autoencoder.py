from vol import Vol
from vol_util import augment
from net import Net 
from trainers import Trainer
from random import sample

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
        pxs = [0.0] * 784
        for n in xrange(784):
            if n % 28 == 0:
                pxs[n] = 0
            else:
                pxs[n] = img[n - 1]
        V.w = pxs
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
    layers.append({
        'type': 'capsule', 'num_neurons': 30, 
        'num_recog': 3, 'num_gen': 4, 'num_pose': 2,
        'dx': 1, 'dy': 0
    })
    layers.append({'type': 'regression', 'num_neurons': 28 * 28})
    print 'Layers made...'

    n = Net(layers)

    print 'Net made...'
    print n

    t = Trainer(n, {'method': 'sgd', 'batch_size': 20, 'l2_decay': 0.001})
    print 'Trainer made...'

def train():
    global training_data, testing_data, n, t

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  '
    print '----------------------------------------------------'
    for x, y in training_data: 
        stats = t.train(x, x.w)
        print stats['k'], stats['time'], stats['loss']

def display(python_path, pred, x):
    display_cmd = '; '.join([
        'import cv2, numpy',
        'predw = numpy.array({}, dtype=numpy.float64).reshape(28, 28)'.format(pred),
        'cv2.imshow(\'P\', predw)',
        'xw = numpy.array({}, dtype=numpy.float64).reshape(28, 28)'.format(x),
        'cv2.imshow(\'X\', xw)',
        'cv2.waitKey(0)',
        'cv2.destroyAllWindows()'
    ])

    cmd = '{} -c \"{}\"'.format(os.path.join(python_path, 'python'), display_cmd)
    call(cmd, shell=True)

def test(path=None, test_n=None):
    global training_data, testing_data, n, t

    print 'In autoencoder testing'

    path = path or '/opt/local/bin'
    test_n = test_n or 5

    for x, y in sample(testing_data, test_n):
        print y
        pred = n.forward(x).w
        display(path, pred, x.w)