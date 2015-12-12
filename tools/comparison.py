# coding=utf-8

from functools import partial
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from dtw import dtw
from tools.fastdtw import dist_measure, fastdtw


def modify_weights(gest, thrown_labels):
    """
    :param gest: known gest
    :param thrown_labels: labels to be thrown out
    :return: aligned and re-normalized weights
    """
    weights_ordered = []
    for marker in gest.labels:
        if marker not in thrown_labels:
            weights_ordered.append(gest.weights[marker])
    weights_ordered = np.array(weights_ordered)
    weights_ordered /= sum(weights_ordered)
    return weights_ordered


def take_common_markers(known_gest, unknown_gest):
    """
    :param known_gest: sequence known to be in some gesture class
    :param unknown_gest: unknown test sequence
    :return: aligned both data to the same markers dimension
    """
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

    data1 = np.delete(known_gest.norm_data, del_ids1, axis=0)
    data2 = np.delete(unknown_gest.norm_data, del_ids2, axis=0)

    weights = modify_weights(known_gest, throw_labels_known)

    return data1, data2, weights


def align_by_first_frame(known_data, unknown_data):
    """
    Translates all markers in unknown_data so that their position in space
    matches with markers from known data in first frame.
    Offset is measured by the first marker of the first frame in both sequences.
    Consider the usage only when general pre-processing (sample-independent)
    shows a poor result.

    :param known_data: (#markers, #frames, #dim) data of a known gest
    :param unknown_data: (#markers, #frames, #dim) data of an unknown gest
    :return: aligned by first markers pos unknown data
    """
    if_error = "Invalid markers dim. Align data shapes first."
    assert known_data.shape[0] == unknown_data.shape[0], if_error
    offset = known_data[:, 0, :] - unknown_data[:, 0, :]
    offset = offset.reshape((known_data.shape[0], 1, known_data.shape[2]))
    unknown_data += offset
    return unknown_data


def compare(known_gest, unknown_gest, dtw_chosen=fastdtw, weighted=True):
    """
     Main comparison function for two gesture examples.
     NOTE:
        - input gestures must have get_norm_data() and get_weights() methods.
        - unknown gesture weights are NOT involved into comparison
          (only known gesture weights are used)
    :param known_gest: known train sample
    :param unknown_gest: unknown test sample
    :param dtw_chosen: fastdtw or _dtw (classic)
    :param weighted: use weighted FastDTW modification or just FastDTW
    :return: (float), similarity (cost) of the given gestures
    """
    if known_gest.labels == unknown_gest.labels:
        known_data = known_gest.get_norm_data()
        unknown_data = unknown_gest.get_norm_data()
        weights = known_gest.get_weights()
    else:
        # TODO apply fines for throwing out markers in test gest because of their absence in train gest
        known_data, unknown_data, weights = take_common_markers(known_gest, unknown_gest)

    # NOTE. Using this reduces level of generality,
    #       but in some cases yields better result.
    #       In our case, do not use it.
    # unknown_data = align_by_first_frame(known_data, unknown_data)

    if not weighted: weights = np.ones(known_data.shape[0])

    if not known_data.any() or not unknown_data.any():
        print("Incompatible data dimensions. Returned np.inf")
        return np.inf

    dist, path = dtw_chosen(known_data, unknown_data, weights)
    if dist == np.inf:
        print("WARNING: dtw comparison gave np.inf")

    # you can play around with cost normalization
    dist /= float(len(path))
    # NOTE. If you use precise dtw cost (without a normalization),
    #       you can end up with better recognition performance
    #       (lower out-of-sample error) in case of small variance of gesture lengths.
    #       BUT! It'll cause loss of generality.
    #       In our case, using normalization yields better results
    #       and thus without loss of generality.

    return dist


def show_comparison(known_gest, unknown_gest):
    """
     Shows the result of gestures comparison.
    :param known_gest: a BasicMotion example
    :param unknown_gest: a BasicMotion example
    """
    data1 = known_gest.get_norm_data()
    data2 = unknown_gest.get_norm_data()
    weights = known_gest.get_weights()

    if data1.shape[0] != data2.shape[0]:
        data1, data2, weights = take_common_markers(known_gest, unknown_gest)
        print("aligned to ", data1.shape, data2.shape)

    if not data1.any() or not data2.any():
        print("Incompatible data dimensions.")
        return np.inf

    # was: data.shape == (#markers, #frames, #dim)
    data1 = np.swapaxes(data1, 0, 1)
    data2 = np.swapaxes(data2, 0, 1)
    # now: data.shape == (#frames, #markers, #dim)

    dist_measure_weighted = partial(dist_measure, weights=weights)
    dist, cost, path = dtw(data1, data2, dist=dist_measure_weighted)

    print('Minimum distance found (unnormalized): %f' % (dist * (known_gest.frames + unknown_gest.frames)))
    plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[1]-0.5))
    plt.xlabel("FRAMES #1: %s" % known_gest.name)
    plt.ylabel("FRAMES #2: %s" % unknown_gest.name)
    plt.title("Weighted DTW frames path")
    plt.show()
