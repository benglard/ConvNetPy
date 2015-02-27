# Requires scikit-learn

from vol_util import augment
from vol      import Vol
from net      import Net 
from trainers import Trainer

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people

training_data = None
testing_data = None
network = None
t = None

def load_data():
    global training_data, testing_data

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    xs = lfw_people.data
    ys = lfw_people.target

    inputs = []
    labels = list(ys)

    for face in xs:
        V = Vol(50, 37, 1, 0.0)
        V.w = list(face)
        inputs.append(augment(V, 30))

    x_tr, x_te, y_tr, y_te = train_test_split(inputs, labels, test_size=0.25)

    training_data = zip(x_tr, y_tr)
    testing_data = zip(x_te, y_te)

    print 'Dataset made...'

def start():
    global network, t

    layers = []
    layers.append({'type': 'input', 'out_sx': 30, 'out_sy': 30, 'out_depth': 1})
    layers.append({'type': 'fc', 'num_neurons': 100, 'activation': 'sigmoid'})
    layers.append({'type': 'softmax', 'num_classes': 7})
    print 'Layers made...'

    network = Net(layers)
    print 'Net made...'
    print network

    t = Trainer(network, {'method': 'adadelta', 'batch_size': 20, 'l2_decay': 0.001})
    print 'Trainer made...'
    print t

def train():
    global training_data, network, t

    print 'In training...'
    print 'k', 'time\t\t  ', 'loss\t  ', 'training accuracy'
    print '----------------------------------------------------'
    try:
        for x, y in training_data: 
            stats = t.train(x, y)
            print stats['k'], stats['time'], stats['loss'], stats['accuracy']
    except: #hit control-c or other
        return

def test():
    global training_data, testing_data, network, t

    print 'In testing'
    right = 0
    count = 0
    try: 
        for x, y in testing_data:
            network.forward(x)
            right += network.getPrediction() == y
            print count
            count += 1
    except:
        pass
    finally:
        accuracy = float(right) / count * 100
        print accuracy