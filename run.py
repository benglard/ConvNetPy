from sys import argv

# MNIST test #classic
if '--mnist' in argv:
    from examples.mnist import *
    start('--conv' in argv)
    train()
    try: 
        index = argv.index('-testn')
        n = int(argv[index + 1])
        test(n)
    except:  
        test(5000)

# random 2D data classification test
if '--toy2D' in argv:
    from examples.toy2D import *
    train()
    test()

# Cifar10 test
if '--cifar10' in argv:
    from examples.cifar.cifar10 import *
    start('-conv' in argv, '-crop' in argv, '-gray' in argv)
    train()
    test()

# Cifar10 autoencoder
if '--cifar10-ae' in argv:
    from examples.cifar.autoencoder import *
    start('-conv' in argv, '-crop' in argv, '-gray' in argv)
    train()
    test()

# Learns to predict the next letter in a sequence (trained on trigrams)
if '--nextletter' in argv:
    from examples.next_letter import *
    start()
    train()
    test()

# Autoencode mnist digits, display them with opencv
if '--autoencoder' in argv:
    from examples.autoencoder import *
    start()
    train()
    
    path, test_n = None, None
    if '-path' in argv:
        path = argv[argv.index('-path') + 1]

    if '-testn' in argv:
        test_n = int(argv[argv.index('-testn') + 1])

    test(path=path, test_n=test_n)

# Uses an autoencoder trained on frequency distributions 
# from project gutenberg to do topic modeling
# (using the assumption that words with highest activation == topics)
if '--topics' in argv:
    from examples.topics import *
    start()
    train()
    test()

# Uses an autoencoder trained on frequency distributions 
# from project gutenberg to do semantic similarity search
# cos(v, qv) for v in doc, v = wieghts of 10 neuron sigmoid ("compressed code of text")
if '--sim' in argv:
    from examples.similarity import *
    start()
    train()
    test()

# Classify iris dataset, requires scikit-learn
if '--iris' in argv:
    from examples.iris import *
    load_data()
    start()
    train()
    test()

# Classify labeled faces in the wild, requires scikit-learn
if '--faces' in argv:
    from examples.faces import *
    load_data()
    start()
    train()
    test()

# Predict next word from bigram model
if '--nextword' in argv:
    from examples.next_word import *
    start()
    train()
    test()

# Predict dialogue class from frequency distribution of text
if '--dialogue' in argv:
    from examples.dialogue import *
    start()
    train()
    test()

# Predict next word based on word embeddings
if '--nextworde' in argv:
    from examples.next_word_embeddings import *
    start()
    train()
    test()

# Predict 1 of 4 sentiment tags, kaggle challenge
# https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
if '--sentiment' in argv:
    from examples.sentiment import *
    start()
    train()
    test()
    fill()

# Based on Geoff Hinton's Dark Knowledge talk 
# https://www.youtube.com/watch?v=EK61htlw8hY
# Train large net with dropout and jitter, use outputs (soft targets)
# as inputs to smaller neural net -- pretty amazing results
# Larger net has already learned a sort of similarity function btw. inputs
if '--dark-knowledge' in argv:
    from examples.dark_knowledge import *
    run_big_net()
    run_small_net()

if '--udacity-terrain' in argv:
    from examples.udacity_terrain import *
    train()
    test()

if '--darkencoder' in argv:
    from examples.darkencoder import *
    start()
    train()
    train2()
    test()

# Autoencode mnist digits, display them in 2D with opencv
if '--autoencoder-vis' in argv:
    from examples.autoencoder_vis import *
    start()
    test()
    train()
    test()

if '--titanic' in argv:
    from examples.titanic import *
    load_data()
    start()
    train()
    test()

if '--mnist-n2i' in argv:
    from examples.num2img import *
    start()
    train()
    test()

if '--tae' in argv:
    from examples.transforming_autoencoder import start, train, test
    start()
    train()
    test()