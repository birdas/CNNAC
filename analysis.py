import numpy as np
import time


def bin_to_int(bin_):
    bin_ = bin_.flatten()
    res = 0
    for b in bin_:
        res = (res << 1) | b
    return res


def int_to_bin(num, feature_size):
    size = feature_size[0] * feature_size[1]
    l = '{:0{size}b}'.format(num, size=size)
    #print(l)
    return np.asarray([int(i) for i in l]).reshape(feature_size)


def scan(fplan, feature_size):
    seen_features = {}
    feat = []
    print(fplan)
    print(len(fplan), len(fplan[0]))
    
    for i in range(len(fplan) - feature_size[0] + 1):
        for j in range(len(fplan[i]) - feature_size[1] + 1):
            # print(i, i+feature_size[0], j, j+feature_size[1])
            feat = fplan[i:i+feature_size[0],j:j+feature_size[1]]
            feat = np.asarray(feat).astype(int)
            # print(feat, len(feat))
            # print(feat, len(feat))
            # time.sleep(5)
            code = bin_to_int(feat)
            if code in seen_features:
                seen_features[code] += 1
            else:
                seen_features[code] = 1

    return seen_features


def most_common(seen_features, feature_size, amount=None, ret=False):
    sort = dict(sorted(seen_features.items(), key=lambda item: item[1])[::-1])
    # rint(sort)
    keys = list(sort.keys())
    if amount == None:
        amount = len(keys)
    ret_ = []
    for i in range(amount):
        ret_.append(int_to_bin(keys[i], feature_size))
        print(int_to_bin(keys[i], feature_size))
    if ret:
        return ret_


def analyze(fplan, feature_size):
    print(scan(fplan, feature_size))


def find(fplan, feat, feature_size):
    locs = []
    for i in range(len(fplan) - feature_size[0] + 1):
        for j in range(len(fplan[i]) - feature_size[1] + 1):
            check = fplan[i:i+feature_size[0],j:j+feature_size[1]]
            # print(check, feat)
            if check.all() == feat.all():
                locs.append((i,j))
    return locs


def find_all(fplan, feats, feature_size):
    locs = {}
    for f in feats:
        loc = find(fplan, f, feature_size)
        if bin_to_int(f) in locs:
            locs[bin_to_int(f)].append(loc)
        else:
            locs[bin_to_int(f)] = [loc]
    return locs


def mse(array1, array2):
    difference_array = np.subtract(array1, array2)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return mse


def brute_solve():
    pass





import os
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
from skimage import color, io


for s in os.listdir('phillip_data/imgs_with_intermediate/'):
    if 'pre' not in s:
        f = os.path.join('phillip_data/imgs_with_intermediate/', s)
        if os.path.isfile(f):
            # load image as pixel array
            data = io.imread(f, as_gray=True)
            # summarize shape of the pixel array
            # print(data.dtype)
            # print(data.shape)
            # print(data)
            # display the array of pixels as an image
            data = np.asarray(data).astype(int)
            # pyplot.imshow(data)
            # pyplot.show()
            feature_size = (3, 3)
            feats = scan(data, feature_size)
            mc_feats = most_common(seen_features=feats, feature_size=feature_size, ret=True)
            # print(find_all(data, [mc_feats[-1]], feature_size))
            break