# coding=utf-8

import numpy as np
from numpy import sqrt


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


def deviation(_dscr, mode="hands", step=1):
    """
    :param _dscr: description info
    :param mode: whether use only hands marker all full set of markers
    :param step: number of frames per step
    :return: list of offsets from current to init pos
    """
    data = _dscr["data"]

    if mode == "hands":
        data = data[_dscr["hands_ids"], :, :]
    init_pos = data[:, _dscr["init_frame"], :]
    # print init_pos.shape
    offset = []
    for frame in range(step, _dscr["frames"], step):
        # for snap_shot in range(step):
        #     data[:, frame-step:frame, :]
        # frames_excluded_dev = np.sum(sqrt((data[:, frame-step:frame, :] - init_pos) ** 2))

        coord_aver = np.average(data[:, frame-step:frame, :], axis=1)
        # if frame in range(1160, 1170):
        #     print coord_aver == np.array([np.nan] * 3)
            # print coord_aver.shape
        # shift = np.sum(sqrt((coord_aver - init_pos) ** 2)) / _dscr["markers"]
        shift = sqrt(np.sum((coord_aver - init_pos) ** 2)) / _dscr["markers"]
        offset.append(shift)
    return offset


def moving_average_simple(xs, wsize=5):
    """
    :param xs: (n,) array of values
    :param wsize: (2n+1) window size for averaging
    :return: smooth array
    """
    step = wsize / 2
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
    step = wsize / 2
    new_shape = list(data.shape)
    new_shape[1] -= 2 * step
    data_smooth = np.empty(shape=new_shape)
    for marker in range(data.shape[0]):
        for ordinate in range(3):
            xs = data[marker, :, ordinate]
            data_smooth[marker, :, ordinate] = moving_average_simple(xs, wsize)
    return data_smooth