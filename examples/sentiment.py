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

def load_data():
    global embeddings

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

    data = []
    raw = file('./data/sentiment-kaggle/train.tsv').read().split('\n')[1:]
    for line in raw:
        try: 
            values = line.split('\t')
            phrase = values[2]
            sentag = int(values[3]) - 1
            
            x = []
            for word in phrase.split():
                if word in embeddings:
                    x.append(embeddings[word])
            
            avgs = [0.0] * 80
            for n in xrange(80):
                for vec in x:
                    avgs[n] += vec[n]
                try:
                    avgs[n] /= float(len(x))
                except:
                    avgs[n] = 0.0

            V = Vol(avgs)
            data.append((V, sentag))
        except:
            continue

    return data

def start():
    global training_data, testing_data, network, t

    all_data = load_data()
    shuffle(all_data)
    size = int(len(all_data) * 0.1)
    training_data, testing_data = all_data[size:], all_data[:size]
    print 'Data loaded, size: {}...'.format(len(all_data))

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 80})
    layers.append({'type': 'fc', 'num_neurons': 200, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 10, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'softmax', 'num_classes': 4})

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

    try:
        print 'In testing...'
        right = 0
        for x, y in testing_data:
            network.forward(x)
            right += network.getPrediction() == y
        accuracy = float(right) / len(testing_data)
        print accuracy
    except KeyboardInterrupt:
        return

def fill():
    global embeddings

    output = 'PhraseId,Sentiment\n'
    raw = file('./data/sentiment-kaggle/test.tsv').read().split('\n')[1:]
    for idx, line in enumerate(raw):
        try: 
            values = line.split('\t')
            phrase_id = values[0]
            phrase = values[2]
            
            x = []
            for word in phrase.split():
                if word in embeddings:
                    x.append(embeddings[word])
            
            avgs = [0.0] * 80
            for n in xrange(80):
                for vec in x:
                    avgs[n] += vec[n]
                try:
                    avgs[n] /= float(len(x))
                except:
                    avgs[n] = 0.0

            network.forward(Vol(avgs))
            output += '{},{}\n'.format(phrase_id, network.getPrediction() + 1)

            print idx
        except:
            continue
    with open('./data/sentiment-kaggle/out1.csv', 'w') as outfile:
        outfile.write(output)

    print 'Done'