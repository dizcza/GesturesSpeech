# coding = utf-8

"""
=====
A gesture DTW implementation.
=====
"""

import numpy as np
from numpy.linalg import norm


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

    # dist = D[-1, -1] / (r + c)
    dist = D[-1, -1]

    return dist


class DtwBackward(object):
    def __init__(self, seq1, seq2, distance_func=None):
        """
        seq1, seq2 are two lists,
        distance_func is a function for calculating
        the local distance between two elements.
        """
        self._seq1 = np.array(seq1)
        self._seq2 = np.array(seq2)
        if distance_func is None:
            self._distance_func = lambda seq1, seq2: norm(seq1 - seq2)
        else:
            self._distance_func = distance_func
        self._map = {(-1, -1): 0.0}
        self._distance_matrix = {}
        self._path = []

    def get_distance(self, i1, i2):
        ret = self._distance_matrix.get((i1, i2))
        if not ret:
            ret = self._distance_func(self._seq1[i1], self._seq2[i2])
            self._distance_matrix[(i1, i2)] = ret
        return ret

    def calculate_backward(self, i1, i2):
        """
        Calculate the dtw distance between
        seq1[:i1 + 1] and seq2[:i2 + 1]
        """
        global calls
        calls += 1

        if self._map.get((i1, i2)) is not None:
            return self._map[(i1, i2)]

        if i1 == -1 or i2 == -1:
            self._map[(i1, i2)] = float('inf')
            return float('inf')

        min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                             key=lambda x: self.calculate_backward(*x))

        self._map[(i1, i2)] = self.get_distance(i1, i2) + \
            self.calculate_backward(min_i1, min_i2)

        return self._map[(i1, i2)]

    def get_path(self):
        """
        Calculate the path mapping.
        Must be called after calculate()
        """
        i1, i2 = (len(self._seq1) - 1, len(self._seq2) - 1)
        while (i1, i2) != (-1, -1):
            self._path.append((i1, i2))
            min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                                 key=lambda x: self._map[x[0], x[1]])
            i1, i2 = min_i1, min_i2
        return self._path

    def calculate(self):
        return self.calculate_backward(len(self._seq1) - 1,
                                       len(self._seq2) - 1)


if __name__ == "__main__":
    global full_pairs, calls
    full_pairs = []
    calls = 0
    x = np.arange(20)
    y = np.arange(50)

    for xi in x:
        l = [(xi, yi) for yi in y]
        full_pairs += l

    d = DtwBackward(x, y)
    # print d.calculate_backward(5, 5)
    print(d.calculate())
    print(float(calls) / len(x) / len(y))
    # path = d.get_path()
    # x_path, y_path = zip(*d.get_path())
    # print x_path, y_path
    # print d.get_distance(4, 3)