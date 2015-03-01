# coding = utf-8

"""
=====
A gesture DTW implementation.
=====
"""


from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import norm
from functools import partial


def dist_measure(fdata1, fdata2, weights=None):
    """
    :param fdata1, fdata2: (#markers, 3) framed points
    :return: dist, w.r.t. the same markers
    """
    if weights == None:
        weights = np.ones(fdata1.shape[0])
    return np.sum(norm(fdata1 - fdata2, axis=1) * weights)


def swap_first_two_cols(data):
    """
     Swaps markers with frames columns.
    :param data: (#markers, frames, 3) ndarray
    :return: (#frames, #markers, 3) data
    """
    new_shape = data.shape[1], data.shape[0], data.shape[2]
    swapped_data = np.empty(shape=new_shape)
    for frame in range(data.shape[1]):
        swapped_data[frame, ::] = data[:, frame, :]
    return swapped_data


def modify_weights(gest, thrown_labels):
    weights_ordered = []
    for marker in gest.labels:
        if marker not in thrown_labels:
            weights_ordered.append(gest.weights[marker])
    weights_ordered = np.array(weights_ordered)
    weights_ordered /= sum(weights_ordered)
    return weights_ordered


def align_data_shape(known_gest, unknown_gest):
    throw_labels_known = []
    throw_labels_unknown = []
    for known_label in known_gest.labels:
        if known_label not in unknown_gest.labels:
            throw_labels_known.append(known_label)
    for unknown_label in unknown_gest.labels:
        if unknown_label not in known_gest.labels:
            throw_labels_unknown.append(unknown_label)

    del_ids1 = known_gest.get_ids(*throw_labels_known)
    del_ids2 = unknown_gest.get_ids(*throw_labels_unknown)

    data1 = np.delete(known_gest.get_norm_data(), del_ids1, axis=0)
    data2 = np.delete(unknown_gest.get_norm_data(), del_ids2, axis=0)

    weights = modify_weights(known_gest, throw_labels_known)

    return data1, data2, weights


def compare(known_gest, unknown_gest):
    """
     Input gestures must have get_norm_data() and get_weights() methods!
    :param known_gest: sequence known to be in some gesture class
    :param unknown_gest: unknown test sequence
    :return: (float), similarity of the given gestures
    """
    data1 = known_gest.get_norm_data()
    data2 = unknown_gest.get_norm_data()
    weights = known_gest.get_weights()
    # print data1.shape, data2.shape, weights.shape

    if data1.shape[0] != data2.shape[0]:
        data1, data2, weights = align_data_shape(known_gest, unknown_gest)
        print "aligned to ", data1.shape, data2.shape


    if not data1.any() or not data2.any():
        print "Incompatible data dimensions."
        return np.inf

    # was: data.shape == (#markers, frames, 3)
    data1 = swap_first_two_cols(data1)
    data2 = swap_first_two_cols(data2)
    # now: data.shape == (#frames, #markers, 3)

    dist_measure_weighted = partial(dist_measure, weights=weights)
    dist, cost, path = dtw(data1, data2, dist=dist_measure_weighted)

    return dist, cost, path


def show_comparison(known_gest, unknown_gest):
    """
     Shows the result of gestures comparison.
    :param known_gest, unknown_gest: some 3d-gestures
    """
    dist, cost, path = compare(known_gest, unknown_gest)
    # print unknown_gest.shoulder_length * dist * cost.shape[0] / 20. * 100.
    print 'Minimum distance found: %.4f' % dist
    # print "x-path (first gesture frames):\n", path[0]
    # print "y-path (second gesture frames):\n", path[1]
    plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[1]-0.5))
    plt.xlabel("FRAMES #1: %s" % known_gest.name)
    plt.ylabel("FRAMES #2: (unknown)")
    plt.title("Weighted DTW frames path")
    plt.show()

