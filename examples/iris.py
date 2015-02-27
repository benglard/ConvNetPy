# Requires scikit-learn

from vol      import Vol
from net      import Net 
from trainers import Trainer

from sklearn.datasets import load_iris

iris_data = None
network = None
sgd = None
N_TRAIN = 120

def load_data():
    global iris_data

    data = load_iris()

    xs = data.data
    ys = data.target
    
    inputs = [ Vol(list(row)) for row in xs ]
    labels = list(ys)

    iris_data = zip(inputs, labels)
    print 'Data loaded...'

def start():
    global network, sgd

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 4})
    layers.append({'type': 'softmax', 'num_classes': 3}) #svm works too
    print 'Layers made...'

    network = Net(layers)
    print 'Net made...'
    print network

    sgd = Trainer(network, {'momentum': 0.1, 'l2_decay': 0.001})
    print 'Trainer made...'
    print sgd

def train():
    global iris_data, sgd

    print 'In training...'
    print 'k', 'time\t\t   ', 'loss\t    ', 'training accuracy'
    print '----------------------------------------------------'
    for x, y in iris_data[:N_TRAIN]: 
        stats = sgd.train(x, y)
        print stats['k'], stats['time'], stats['loss'], stats['accuracy']

def test():
    global iris_data, network

    print 'In testing...'
    right = 0
    for x, y in iris_data[N_TRAIN:]:
        network.forward(x)
        right += network.getPrediction() == y
    accuracy = float(right) / (150 - N_TRAIN) * 100
    print accuracy