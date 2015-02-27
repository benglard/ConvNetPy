# Requires nltk, nltk data

from vol import Vol
from net import Net 
from trainers import Trainer

from nltk import FreqDist
from nltk.corpus import nps_chat as corpus
from string import punctuation
from random import shuffle

training_data = None
testing_data = None
network = None
t = None
N = 0
words = None
labels = None

# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"
])

def load_data():
    global N, words, labels

    posts = corpus.xml_posts()[:10000]
    freqs = [ FreqDist(post.text) for post in posts ] 
    words = list(set(word 
                    for dist in freqs 
                    for word in dist.keys()
                    if word not in ENGLISH_STOP_WORDS and
                    word not in punctuation))

    labels = list(set([ post.get('class') for post in posts ]))

    data = []
    N = len(words)
    for post, dist in zip(posts, freqs):
        V = Vol(1, 1, N, 0.0)
        for i, word in enumerate(words):
            V.w[i] = dist.freq(word)
        data.append((V, labels.index(post.get('class'))))

    return data

def start():
    global training_data, testing_data, network, t, N, labels

    data = load_data()
    shuffle(data)
    size = int(len(data) * 0.01)
    training_data, testing_data = data[size:], data[:size]
    print 'Data loaded...'

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': N})
    layers.append({'type': 'fc', 'num_neurons': 10, 'activation': 'sigmoid'})
    layers.append({'type': 'softmax', 'num_classes': len(labels)})
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
    #while True:
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