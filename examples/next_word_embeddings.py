from vol import Vol
from net import Net 
from trainers import Trainer
from util import *

import os
from random import shuffle, sample, random
from sys import exit

embeddings = None
training_data = None
testing_data = None
network = None
t = None
N = None
tokens_l = None

def load_data():
    global embeddings, N, tokens_l

    embeddings = {}
    raw = file('./data/word_projections-80.txt').read()
    raw = raw[9:]
    raw = raw.split('\n')
    for elem in raw:
        try:
            data = elem.split()
            word = data[0].lower()
            vector = [ float(v) for v in data[1:] ]
            embeddings[word] = vector
        except:
            continue

    path = './data/text/train_tiny'
    words = list(token 
        for fname in os.listdir(path)
        for token in file(os.path.join(path, fname)).read().split())
    tokens = set(words)
    tokens_l = list(tokens)
    N = len(tokens)
    print 'Corpus size: {} words'.format(N)

    step = 4
    data = []
    for n in xrange(0, len(words) - step):
        w1, w2, w3, pred = words[n:n+step]

        if not (w1 in embeddings and w2 in embeddings and w3 in embeddings
        and pred in embeddings and pred in tokens): continue

        V = Vol(embeddings[w1] + embeddings[w2] + embeddings[w3])
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
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 240})
    
    layers.append({'type': 'fc', 'num_neurons': 200, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 10, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    
    #layers.append({'type': 'conv', 'sx': 1, 'filters': 240, 'pad': 0}) #lookup table like
    #layers.append({'type': 'fc', 'num_neurons': 200, 'activation': 'tanh', 'drop_prob': 0.5})
    #layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'tanh', 'drop_prob': 0.5})
    
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
        pass
    finally:
        saveJSON('./models/next_word_embeddings/network.json', network.toJSON())

def test_text(text, ngenerate=10, delete=True):
    out = ''
    for n in xrange(ngenerate):
        x = []
        words = text.split()
        for word in words:
            if word not in embeddings:
                return 'word: {} not in corpus'.format(word)
            else:
                x.extend(embeddings[word])
        output = network.forward(Vol(x)).w
        pred = network.getPrediction()
        new = tokens_l[pred] if random() < 0.5 else \
            weightedSample(embeddings.keys(), output)

        out += ' ' + new
        text = ' '.join(words[1:] + [new])
    return out

def test():
    global testing_data, network

    try:
        print 'In testing...'
        right = 0
        for x, y in testing_data:
            network.forward(x)
            right += network.getPrediction() == y
        accuracy = float(right) / len(testing_data)
        print accuracy
    except KeyboardInterrupt:
        pass
    finally:
        print test_text('the answer is')
        print test_text('i did this')