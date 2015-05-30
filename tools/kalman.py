# coding=utf-8

import numpy as np


def kalman_1d(x_noisy, k_stab=0.2):
    """
     Performs smoothing on a 1d-array.
    :param x_noisy: noisy 1d-array
    :param k_stab: kalman stable gain
    :return: optimal xs
    """
    first_visible_id = ~np.isnan(x_noisy)[0]
    start_val = x_noisy[first_visible_id]
    x_opt = np.ones(first_visible_id + 1) * start_val
    for frame in range(first_visible_id + 1, len(x_noisy)):
        if np.isnan(x_noisy[frame]):
            x_opt = np.append(x_opt, x_opt[-1])
        else:
            xi = k_stab * x_noisy[frame] + (1. - k_stab) * x_opt[frame-1]
            x_opt = np.append(x_opt, xi)
    return x_opt


def kalman_filter(data):
    """
    :param data: (#markers, #frames, #dim) gesture data
    :return: filtered (smooth) gesture data
    """
    if data.shape[1] > 1:
        for markerID in range(data.shape[0]):
            for dim in range(data.shape[2]):
                x_opt = kalman_1d(data[markerID, :, dim])

                # put NaNs back, if you want
                # x_opt[np.isnan(data[markerID, ::]).any(axis=1)] = np.nan

                data[markerID, :, dim] = x_opt
    return data
