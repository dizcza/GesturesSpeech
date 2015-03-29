# coding = utf-8

##########################################
# A gesture Weighted DTW implementation. #
##########################################

import numpy as np
from numpy.linalg import norm

# TODO implement fastdtw


def dist_measure(fdata1, fdata2, weights):
    """
    :param fdata1, fdata2: (#markers, 3) framed points
    :return: dist, w.r.t. the same markers
    """
    # weights = np.ones(fdata1.shape[0])
    return np.sum(norm(fdata1 - fdata2, axis=1) * weights)


def wdtw_windowed(data1, data2, weights):
    """ Computes the DTW of two sequences.

    :param ndarray data1: (#markers, #frames1, 3) data of the known gest
    :param ndarray data1: (#markers, #frames2, 3) data of the unknown gest
    :param weights: weights from the known gest
    :return the minimum distance between tho given gests

    """
    r, c = data1.shape[1], data2.shape[1]
    window = max(r // 2, abs(r-c))

    D = np.empty((r+1, c+1))
    D[:, :] = np.inf
    D[0, 0] = 0.

    for i in range(1, r+1):
        bottom = max(1, i-window)
        top = min(c+1, i+window)
        for j in range(bottom, top, 1):
            cost = dist_measure(data1[:,i-1,:], data2[:,j-1,:], weights)
            D[i, j] = cost + min(D[i-1, j-1], D[i-1, j], D[i, j-1])

    dist = D[-1, -1] / (r + c)
    # dist = D[-1, -1]

    return dist


def wdtw(data1, data2, weights):
    """ Computes the DTW of two sequences.

    :param ndarray data1: (#markers, #frames1, 3) data of the known gest
    :param ndarray data1: (#markers, #frames2, 3) data of the unknown gest
    :param weights: weights from the known gest
    :return the minimum distance between tho given gests

    """
    r, c = data1.shape[1], data2.shape[1]

    D = np.zeros((r+1, c+1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    for i in range(1, r+1):
        for j in range(1, c+1):
            cost = dist_measure(data1[:,i-1,:], data2[:,j-1,:], weights)
            D[i, j] = cost + min(D[i-1, j-1], D[i-1, j], D[i, j-1])

    dist = D[-1, -1] / (r + c)
    # dist = D[-1, -1]

    return dist
