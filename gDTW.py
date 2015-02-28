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


def compare(known_gest, unknown_gest):
    """
     Input gestures must have get_norm_data() method!
    :param known_gest: sequence known to be in some gesture class
    :param unknown_gest: unknown test sequence
    :return: (float), similarity of the given gestures
    """
    data1 = known_gest.get_norm_data()
    data2 = unknown_gest.get_norm_data()
    if data1.shape[0] != data2.shape[0]:
        print "Incompatible data dimensions."
        return np.inf

    # was: data.shape == (#markers, frames, 3)
    data1 = swap_first_two_cols(data1)
    data2 = swap_first_two_cols(data2)
    # now: data.shape == (#frames, #markers, 3)

    dist_measure_weighted = partial(dist_measure, weights=known_gest.get_weights())
    dist, cost, path = dtw(data1, data2, dist=dist_measure_weighted)

    return dist, cost, path


def show_comparison(gest1, gest2):
    """
     Shows the result of gestures comparison.
    :param gest1, gest2: some 3d-gestures
    """
    dist, cost, path = compare(gest1, gest2)
    print 'Minimum distance found: %.4f' % dist
    print "x-path (first gesture frames):\n", path[0]
    print "y-path (second gesture frames):\n", path[1]
    plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[1]-0.5))
    plt.xlabel("FRAMES #1")
    plt.ylabel("FRAMES #2")
    plt.title("DTW frames path")
    plt.show()

