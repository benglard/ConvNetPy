# This file transforms the cifar10 python dataset
# into a form readable by pypy. This program
# cannot be run with pypy, needs CPython

import cPickle, os
import numpy
from sys import argv, exit

if '-path' not in argv:
    raise Exception('Specify a local path to cifar data')

path = argv[argv.index('-path') + 1]

def unpickle(file):  
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

xs = []
ys = []
for i in range(1, 6):
    filename = 'data_batch_' + str(i)
    d = unpickle(os.path.join(path, filename))
    x = d['data']
    y = d['labels']
    xs.append(x)
    ys.append(y)

x = numpy.concatenate(xs)
x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
y = numpy.concatenate(ys)

numpy.savez('./data/cifar10_train.npz', x=x, y=y)

d = unpickle(os.path.join(path, 'test_batch'))
x = d['data']
x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
y = d['labels']

numpy.savez('./data/cifar10_test.npz', x=x, y=y)