from vol import Vol
from net import Net 
from trainers import Trainer

import random

training_data = []
testing_data = []

def makeTerrainData(n_points=1000):
    global training_data, testing_data

###############################################################################
### from: https://github.com/udacity/ud120-projects/blob/master/choose_your_own/prep_terrain_data.py
### make the toy dataset
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    for x, y in zip(X_train, y_train):
        training_data.append(( Vol(x), int(y) ))
    for x, y in zip(X_test, y_test):
        testing_data.append(( Vol(x), int(y) ))

makeTerrainData(10000)

layers = []
layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 2})
layers.append({'type': 'fc', 'num_neurons': 45, 'activation': 'relu'})
layers.append({'type':'softmax', 'num_classes': 2})

n = Net(layers)

print 'Net made...'
print n

t = Trainer(n, {'momentum': 0.14})

print 'Trainer made...'

def train():
    print 'In training...'
    print 'k', 'time\t\t   ', 'loss\t    ', 'training accuracy'
    print '----------------------------------------------------'
    for x, y in training_data: 
        stats = t.train(x, y)
        print stats['k'], stats['time'], stats['loss'], stats['accuracy']

def test():
    print 'In testing...'
    right = 0
    for x, y in testing_data:
        n.forward(x)
        right += n.getPrediction() == y
    accuracy = float(right) / len(testing_data) * 100
    print accuracy