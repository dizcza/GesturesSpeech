# coding=utf-8

###########################################################
# A gesture weighted fast DTW implementation.             #
# Source: https://github.com/slaypni/fastdtw              #
# FastDTW theory: http://cs.fit.edu/~pkc/papers/tdm04.pdf #
###########################################################

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.linalg import norm
from collections import defaultdict


def remove_nan(fdata1, fdata2, weights):
    """
    :param fdata1, fdata2: (#markers, #dim) frame data
    :param weights: (#markers,) joint weights (motion contribution)
    :return: aligned frame data and weights without any NaN
    """
    assert fdata1.shape[0] == fdata2.shape[0], "#markers should be the same"
    corrupted_markers = []
    for markerID in range(fdata1.shape[0]):
        is_bad = False
        is_bad = is_bad or np.isnan(fdata1[markerID, :]).any()
        is_bad = is_bad or np.isnan(fdata2[markerID, :]).any()
        is_bad = is_bad or np.isnan(weights[markerID])
        if is_bad:
            corrupted_markers.append(markerID)
    clean_data1 = np.delete(fdata1, corrupted_markers, axis=0)
    clean_data2 = np.delete(fdata2, corrupted_markers, axis=0)
    clean_weights = np.delete(weights, corrupted_markers)
    return clean_data1, clean_data2, clean_weights


def dist_measure(fdata1, fdata2, weights):
    """
    :param fdata1, fdata2: (#markers, #dim) frame data
    :param weights: (#markers,) joint weights (motion contribution)
    :return: (float) dist, w.r.t. the same markers
    """
    # weights = np.ones(fdata1.shape[0])
    if np.isnan(fdata1).any() or np.isnan(fdata2).any() or np.isnan(weights).any():
        fdata1, fdata2, weights = remove_nan(fdata1, fdata2, weights)
    return np.sum(norm(fdata1 - fdata2, axis=1) * weights)


def fastdtw(x, y, weights, radius=1):
    """
     Speeds up classic dtw algorithm.
    :param x: (#markers, #frames1, #dim) data of the known gest
    :param y: (#markers, #frames2, #dim) data of the unknown gest
    :param radius: constrain, defines window searching field
    :returns: dtw cost, dtw path
    """
    min_time_size = radius + 2

    if x.shape[1] < min_time_size or y.shape[1] < min_time_size:
        return _dtw(x, y, weights, None)

    x_shrunk = __reduce_by_half(x)
    y_shrunk = __reduce_by_half(y)
    distance, path = fastdtw(x_shrunk, y_shrunk, weights, radius)
    window = __expand_window(path, x.shape[1], y.shape[1], radius)
    return _dtw(x, y, weights, window)


def _dtw(x, y, weights, window=None):
    """
     Classic windowed dtw algorithm.
    :param x: (#markers, #frames1, #dim) data of the known gest
    :param y: (#markers, #frames2, #dim) data of the unknown gest
    :returns: dtw cost, dtw path
    """
    len_x, len_y = x.shape[1], y.shape[1]
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    for i, j in window:
        dt = dist_measure(x[:,i-1,:], y[:,j-1,:], weights)
        D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1), (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return D[len_x, len_y][0], path


def __reduce_by_half(x):
    """
    :param x: (#markers, #frames1, #dim) data
    :return: (#markers, #frames1 / 2, #dim) shrunk data
    """
    second_ind = np.arange(1, x.shape[1], 2)
    first_ind = second_ind - 1
    return (x[:,first_ind,:] + x[:,second_ind,:]) / 2.


def __expand_window(path, len_x, len_y, radius):
    """
    :param path: list of (i, j) cells path
    :param len_x: #frames1
    :param len_y: #frames2
    :param radius: constrain, defines a window
    :return: window, expanded by a radius
    """
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b) for a in range(-radius, radius+1) for b in range(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1), (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window