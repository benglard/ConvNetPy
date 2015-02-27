from vol      import Vol
from net      import Net 
from trainers import Trainer

training_data = []
testing_data = []
network = None
sgd = None
N_TRAIN = 800

def load_data():
    global training_data, testing_data

    train = [ line.split(',') for line in
        file('./data/titanic-kaggle/train.csv').read().split('\n')[1:] ]
    for ex in train:
        PassengerId,Survived,Pclass,Name,NameRest,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked = ex
        
        # Fixing
        sex = 0.0 if Sex == 'male' else 1.0
        age = 0 if Age == '' else float(Age)
        Embarked = Embarked.replace('\r', '')
        if Embarked == 'C':
            emb = 0.0
        elif Embarked == 'Q':
            emb = 1.0
        else:
            emb = 2.0

        vec = [ float(Pclass), sex, age, float(SibSp), float(Parch), float(Fare), emb ]
        v = Vol(vec)
        training_data.append((v, int(Survived)))

    test = [ line.split(',') for line in
        file('./data/titanic-kaggle/test.csv').read().split('\n')[1:] ]
    for ex in test:
        PassengerId,Pclass,Name,NameRest,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked = ex
        
        # Fixing
        sex = 0.0 if Sex == 'male' else 1.0
        age = 0 if Age == '' else float(Age)
        Embarked = Embarked.replace('\r', '')
        if Embarked == 'C':
            emb = 0.0
        elif Embarked == 'Q':
            emb = 1.0
        else:
            emb = 2.0
        fare = 0 if Fare == '' else float(Fare)

        vec = [ float(Pclass), sex, age, float(SibSp), float(Parch), fare, emb ]
        testing_data.append(Vol(vec))

    print 'Data loaded...'

def start():
    global network, sgd

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 7})
    #layers.append({'type': 'fc', 'num_neurons': 30, 'activation': 'relu'})
    #layers.append({'type': 'fc', 'num_neurons': 30, 'activation': 'relu'})
    layers.append({'type': 'softmax', 'num_classes': 2}) #svm works too
    print 'Layers made...'

    network = Net(layers)
    print 'Net made...'
    print network

    sgd = Trainer(network, {'momentum': 0.2, 'l2_decay': 0.001})
    print 'Trainer made...'
    print sgd

def train():
    global training_data, sgd

    print 'In training...'
    print 'k', 'time\t\t   ', 'loss\t    ', 'training accuracy'
    print '----------------------------------------------------'
    for x, y in training_data[:N_TRAIN]: 
        stats = sgd.train(x, y)
        print stats['k'], stats['time'], stats['loss'], stats['accuracy']

def test():
    global training_data, network

    print 'In testing...'
    right = 0
    for x, y in training_data[N_TRAIN:]:
        network.forward(x)
        right += network.getPrediction() == y
    accuracy = float(right) / (len(training_data) - N_TRAIN) * 100
    print accuracy