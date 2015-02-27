# Requires nltk, nltk data

from vol import Vol
from net import Net 
from trainers import Trainer

from nltk import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import gutenberg as corpus #can use others like inaugural

from random import shuffle, sample
from sys import exit

training_data = None
testing_data = None
network = None
t = None
N = 0
tokens = None

def load_data():
    global N, words

    raw = list(word 
            for fileid in corpus.fileids()
            for word in corpus.words(fileid))
    words = list(token for token in RegexpTokenizer('\w+').tokenize(' '.join(raw)))[100:1000]
    tokens = set(words)
    tokens_l = list(tokens)
    N = len(tokens)
    print 'Corpus size: {} words'.format(N)

    step = 4
    data = []
    for gram in ngrams(words, step):
        w1, w2, w3, pred = gram
        V = Vol(1, 1, N, 0.0)
        V.w[tokens_l.index(w1)] = 1
        V.w[tokens_l.index(w2)] = 1
        V.w[tokens_l.index(w3)] = 1
        label = tokens_l.index(pred)
        data.append((V, label))

    return data

def start():
    global training_data, testing_data, network, t, N

    all_data = load_data()
    shuffle(all_data)
    size = int(len(all_data) * 0.1)
    training_data, testing_data = all_data[size:], all_data[:size]
    print 'Data loaded, size: {}...'.format(len(all_data))

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': N})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 10, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'softmax', 'num_classes': N})

    print 'Layers made...'

    network = Net(layers)

    print 'Net made...'
    print network

    t = Trainer(network, {'method': 'adadelta', 'batch_size': 10, 'l2_decay': 0.0001});

def train():
    global training_data, network, t

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  ', 'training accuracy'
    print '----------------------------------------------------'
    try:
        for x, y in training_data: 
            stats = t.train(x, y)
            print stats['k'], stats['time'], stats['loss'], stats['accuracy']
    except KeyboardInterrupt:
        return

def test():
    global testing_data, network

    print 'In testing...'
    right = 0
    for x, y in testing_data:
        network.forward(x)
        right += network.getPrediction() == y
    accuracy = float(right) / len(testing_data)
    print accuracy