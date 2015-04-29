# coding=utf-8

import numpy as np
from numpy.linalg import norm


def diff(data, step=1):
    """
    :param data: (#markers, #frames, 3) ndarray of 3d points data
    :param step: number of frames per step
    :return: average differential of (data[i] - data[i-step]) / step
    """
    new_shape = list(data.shape)
    indices = np.arange(step, data.shape[1], step)
    new_shape[1] = len(indices)
    deriv_data = np.empty(shape=new_shape)
    for tick in range(len(indices)):
        frame = indices[tick]
        _deriv = data[:, frame-step+1:frame, :] - data[:, frame-step:frame-1, :]
        _deriv = np.average(_deriv, axis=1)
        deriv_data[:, tick, :] = _deriv
    return deriv_data


def moving_average_simple(xs, wsize=5):
    """
    :param xs: (n,) array of values
    :param wsize: (2n+1) window size for averaging
    :return: smooth array
    """
    step = int(wsize / 2)
    xs_smooth = []
    for i in range(step, len(xs) - step):
        # +1 because the right boundary is *excluded*
        xs_smooth.append(sum(xs[(i - step):(i + step + 1)]) / float(wsize))
    return xs_smooth


def moving_average(data, wsize=5):
    """
    :param data: (#markers, #frames, 3) 3d points data
    :param wsize: (2n+1) window size for averaging
    :return: (#markers, #frames, 3) smooth data
    """
    step = int(wsize / 2)
    new_shape = list(data.shape)
    new_shape[1] -= 2 * step
    data_smooth = np.empty(shape=new_shape)
    for marker in range(data.shape[0]):
        for ordinate in range(data.shape[2]):
            xs = data[marker, :, ordinate]
            data_smooth[marker, :, ordinate] = moving_average_simple(xs, wsize)
    return data_smooth