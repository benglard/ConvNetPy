from vol import Vol
from util import randi

def augment(V, crop, grayscale=False):
    # note assumes square outputs of size crop x crop
    # randomly sample a crop in the input volume
    if crop == V.sx: return V

    dx = randi(0, V.sx - crop)
    dy = randi(0, V.sy - crop)

    W = Vol(crop, crop, V.depth)
    for x in xrange(crop):
        for y in xrange(crop):
            if x + dx < 0 or x + dx >= V.sx or \
                y + dy < 0 or y + dy >= V.sy: continue
            for d in xrange(V.depth):
                W.set(x, y, d, V.get(x + dx, y + dy, d))

    if grayscale:
        #flatten into depth=1 array
        G = Vol(crop, crop, 1, 0.0)
        for i in xrange(crop):
            for j in xrange(crop):
                G.set(i, j, 0, W.get(i, j, 0))
        W = G

    return W