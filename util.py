from random import random
from math import sqrt, log, floor
import json

# Random Number Utilities

return_v = False
v_val = 0.0
def gaussRandom():
    global return_v, v_val

    if return_v:
        return_v = False
        return v_val
    u = 2 * random() - 1
    v = 2 * random() - 1
    r = u * u + v * v
    if r == 0 or r > 1: 
        return gaussRandom()
    c = sqrt(-2.0 * log(r) / r)
    v_val = v * c #cache this
    return_v = True
    return u * c

def randf(a, b): 
    return random() * (b - a) + a

def randi(a, b): 
    return int(floor(random() * (b - a) + a))
    
def randn(mu, std): 
    return mu + gaussRandom() * std

# Array Utilities

import numpy
def zeros(n=None):
    if not n:
        return []
    else:
        return [0.0] * int(n)

def arrContains(arr, elt):
    for elem in arr:
        if elem == elt:
            return True
    return False

def arrUnique(arr):
    b = set(arr)
    return list(b)

# Return max and min of a given non-empty list
def maxmin(w):
    if len(w) == 0:
        return {}
    maxv = max(w)
    maxi = w.index(maxv)
    minv = min(w)
    mini = w.index(minv)
    return {
        'maxi': maxi,
        'maxv': maxv,
        'mini': mini,
        'minv': minv,
        'dv'  : maxv - minv
    }

# Create random permutations of numbers, in range [0 ... n-1]
def randperm(n):
    i, j, temp = n - 1, 0, None
    array = range(n)
    while i:
        j = int(floor(random() * (i + 1)))
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
        i -= 1
    return array

# Sample for list 'lst' according to probabilities in list probs
# the two lists are of same size, and probs adds up to 1
# or lst is a dictionary with keys and probabilities for values 
def weightedSample(lst=None, prob=None):
    if not (prob or lst): 
        return

    if type(lst) != list:
        try:
            prob = lst.values()
            lst = lst.keys()
        except:
            return

    p = randf(0, 1.0)
    cumprob = 0.0
    for k in xrange(len(lst)):
        cumprob += prob[k]
        if p < cumprob:
            return lst[k]

# Syntactic sugar function for getting default parameter values
def getopt(opt, field_name, default_value):
    return opt.get(field_name, default_value)

# Utilities for saving/loading json to/from a file

def saveJSON(filename, data):
    print 'Saving to: {}'.format(filename)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def loadJSON(path):
    with open(path, 'r') as infile:
        return json.load(path)

class Window(object):

    """
    a window stores _size_ number of values
    and returns averages. Useful for keeping running
    track of validation or training accuracy during SGD
    """

    def __init__(self, size=100, minsize=20):
        self.v = []
        self.size = size
        self.minsize = minsize
        self.sum = 0

    def add(self, x):
        self.v.append(x)
        self.sum += x
        if len(self.v) > self.size:
            xold = self.v.pop(0)
            self.sum -= xold

    def get_average(self):
        if len(self.v) < self.minsize:
            return -1
        else:
            return 1.0 * self.sum / len(self.v)

    def reset(self):
        self.v = []
        self.sum = 0