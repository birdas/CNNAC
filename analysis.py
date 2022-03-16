import numpy as np


def bin_to_int(bin_):
    res = 0
    for b in bin_:
        res = (res << 1) | b
    return res


def int_to_bin(num):
    return [int(i) for i in bin(num)[2:]]


def scan(fplan, feature_size):
    seen_features = {}
    feat = []
    
    for i in range(len(fplan) - feature_size[0] + 1):
        for j in range(len(fplan[i]) - feature_size[1] + 1):
            feat = fplan[i:i+feature_size[0]][j:j+feature_size[1]]
            feat = np.asarray(feat)
            feat = feat.flatten()
            code = bin_to_int(feat)
            if code in seen_features:
                seen_features[code] += 1
            else:
                seen_features[code] = 1

    return seen_features


def analyze(fplan, feature_size):
    print(scan(fplan, feature_size))




fplan = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
feature_size = (2, 2)
analyze(fplan, feature_size)