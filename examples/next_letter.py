from util import *
from collections import defaultdict

from vol import Vol
from net import Net 
from trainers import Trainer
from random import randint, choice, sample

N = 100000
training_data = None
n = None
t = None
frequencies = defaultdict(lambda: 0)

def normalize():
    global frequencies

    s = 1.0 * sum(frequencies[letter] for letter in frequencies)
    for letter in frequencies:
        frequencies[letter] /= s

def load_data(train=True):
    global N, frequencies

    with open('./data/big.txt', 'r') as infile:
        text = infile.read()
    
    skip = 3
    size = skip * N
    start = randint(0, len(text) - size)
    content = text[start:start+size]
    data = []

    for i in range(0, len(content), skip):
        x1, x2, y = content[i:i+skip]

        l1 = ord(x1)
        l2 = ord(x2)
        frequencies[l1] += 1
        frequencies[l2] += 1

        V = Vol(1, 1, 255, 0.0)
        V.w[l1] = 1.0
        V.w[l2] = 1.0
        label = ord(y)
        data.append((V, label))

    normalize()

    return data

def start():
    global training_data, n, t

    training_data = load_data()

    print 'Data loaded...'

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 255})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'softmax', 'num_classes': 255})

    print 'Layers made...'

    n = Net(layers)

    print 'Net made...'
    print n

    t = Trainer(n, {'method': 'adadelta', 'batch_size': 10, 'l2_decay': 0.0001});

    print 'Trainer made...'

def train():
    global training_data, n, t

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  '
    print '----------------------------------------------------'
    try:
        for x, y in training_data: 
            stats = t.train(x, y)
            print stats['k'], stats['time'], stats['loss']
    except:
        return

def test():
    global n, frequencies

    y = weightedSample(lst=frequencies, count=2)
    x = Vol(1, 1, 255, 0.0)
    x.w[y[0]] = 1.0
    x.w[y[1]] = 1.0

    s = ''
    for i in xrange(50): 
        n.forward(x)
        pred = n.getPrediction()
        pred2 = weightedSample(lst=frequencies, count=1)[0]

        s += chr(pred) + chr(pred2)

        x.w[pred] = 1.0

        x.w[y[0]] = 0.0
        x.w[y[1]] = 0.0

        y = (pred, pred2)
    print s