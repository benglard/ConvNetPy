# Requires nltk, nltk data

from vol import Vol
from net import Net 
from trainers import Trainer

from nltk import FreqDist, RegexpTokenizer
from nltk.corpus import gutenberg as corpus #can use others like inaugural
from string import punctuation

training_data = None
network = None
t = None
N = 0
words = None

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
    global N, words

    freqs = [ FreqDist(corpus.words(fileid)) for fileid in corpus.fileids() ]
    words = list(set(word 
                    for dist in freqs 
                    for word in dist.keys()
                    if word not in ENGLISH_STOP_WORDS and
                    word not in punctuation))

    data = []
    N = len(words)
    for dist in freqs:
        V = Vol(1, 1, N, 0.0)
        for i, word in enumerate(words):
            V.w[i] = dist.freq(word)
        data.append((V, V.w))

    return data

def start():
    global training_data, network, t, N

    training_data = load_data()
    print 'Data loaded...'

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': N})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 10, 'activation': 'sigmoid'})
    layers.append({'type': 'fc', 'num_neurons': 50, 'activation': 'sigmoid'})
    layers.append({'type': 'regression', 'num_neurons': N})

    print 'Layers made...'

    network = Net(layers)

    print 'Net made...'
    print network

    t = Trainer(network, {'method': 'adadelta', 'batch_size': 4, 'l2_decay': 0.0001});

def train():
    global training_data, network, t

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
    global N, words, network

    print 'In testing.'

    gettysburg = """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."""
    tokenizer = RegexpTokenizer('\w+')
    gettysburg_tokens = tokenizer.tokenize(gettysburg) 

    samples = []
    for token in gettysburg_tokens:
        word = token.lower()
        if word not in ENGLISH_STOP_WORDS and word not in punctuation:
            samples.append(word)

    dist = FreqDist(samples)
    V = Vol(1, 1, N, 0.0)
    for i, word in enumerate(words):
        V.w[i] = dist.freq(word)

    pred = network.forward(V).w
    topics = []
    while len(topics) != 5:
        max_act = max(pred)
        topic_idx = pred.index(max_act)
        topic = words[topic_idx]

        if topic in gettysburg_tokens:
            topics.append(topic)
    
        del pred[topic_idx]

    print 'Topics of the Gettysburg Address:'
    print topics