from vol import Vol
from net import Net 
from trainers import Trainer
from random import randint, shuffle

def generate(n, train=True):
    data = []
    if train:
        for i in range(n/2): data.append(((2 + randint(-1, 1), 7 + randint(-1, 1)), 1))   # top right
        for i in range(n/2): data.append(((7.5 + randint(-2, 2), 2 + randint(-1, 1)), 0)) # bottom left
    else:
        for i in range(n/2): data.append(((2 + randint(-1, 1), 7 + randint(-1, 1)), 1)) # top right
        for i in range(n/2): data.append(((7 + randint(-1, 1), 2 + randint(-1, 1)), 0)) # bottom left
    return [ (Vol(x), label) for x, label in data ]

N_train = 1000
N_test = 30

training_data = generate(N_train)
shuffle(training_data)

testing_data = generate(N_test, False)
shuffle(testing_data)

print 'Data loaded...'

layers = []
layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 2})
layers.append({'type':'softmax', 'num_classes': 2})

n = Net(layers)

print 'Net made...'
print n

t = Trainer(n, {'momentum': 0.1, 'l2_decay': 0.001})

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
    accuracy = float(right) / N_test * 100
    print accuracy