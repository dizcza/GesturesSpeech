# coding=utf-8

from helper import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from numpy import sqrt
import os
import json

reader = btk.btkAcquisitionFileReader()


def get_hands_ids(labeling):
    lhand_labels = np.array([
        "LFSH",     "LBSH",     "LUPA",     "LELB",     "LIEL",
        "LOWR",     "LIWR",     "LWRE",     "LIHAND",   "LOHAND",
        "LIDX1",    "LIDX2",    "LIDX3",    "LMDL1",    "LMDL2",
        "LMDL3",    "LRNG1",    "LRNG2",    "LRNG3",    "LPNK1",
        "LPNK2",    "LPNK3",    "LTHM1",    "LTHM2",    "LTHM3",
    ])
    rhand_labels = ["R" + label[1:] for label in lhand_labels]
    hands_labels = np.concatenate([lhand_labels, rhand_labels])
    hands_ids = []
    for label in hands_labels:
        hands_ids.append(labeling[label])
    return hands_ids


def get_feet_ids(labeling):
    feet_labels = ["RANK", "LANK", "RTOE", "LTOE", "RHEL", "LHEL"]
    feet_ids = []
    for label in feet_labels:
        feet_ids.append(labeling[label])
    return feet_ids


def describe(c3d_file):
    reader.SetFilename(c3d_file)
    reader.Update()
    acq = reader.GetOutput()
    labeling = {}
    for i in range(acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(i).GetLabel().split(":")[-1]
        labeling[label] = i
    description = {
        "filename": c3d_file,
        "label": labeling
    }

    data = get_points_data(acq)
    feet_ids = get_feet_ids(labeling)
    feet_coord = data[feet_ids, :, :]
    feet_center = np.average(np.average(feet_coord, axis=1), axis=0)
    description["feet"] = {
        "ids": feet_ids,
        "center": feet_center.tolist()
    }
    description["hands_ids"] = get_hands_ids(labeling)
    description["init_frame"] = init_frame(c3d_file)
    description["frames"] = acq.GetPointFrameNumber()
    description["markers"] = acq.GetPoints().GetItemNumber()
    description["data"] = data
    description["freq"] = acq.GetPointFrequency()

    return description


def diff(data, step=1):
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


def get_points_data(acq):
    data = np.empty((3, acq.GetPointFrameNumber(), 1))
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(i).GetLabel()
        data = np.dstack((data, acq.GetPoint(label).GetValues().T))
    data = np.delete(data.T, 0, axis=0)  # first marker is noisy for this file
    data[data == 0] = np.NaN             # handle missing data (zeros)
    if any(data[data == 0]):
        print "\nThere is %d missing data (zeros)." % len(data[data == 0])
    return data


def animate(i):
    for pt, xi in zip(pts, data):
        x, y, z = xi[:i].T
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])
    return pts


def display_animation(_descr, speed_rate=1, frames_range=None):
    global pts, data

    chunks_num = int(speed_rate * _descr["freq"] / 24)
    if frames_range:
        frames = np.arange(frames_range[0], frames_range[1], chunks_num)
    else:
        frames = np.arange(0, _descr["frames"]-1, chunks_num)

    data = _descr["data"][:, frames, :]
    # data = data[_descr["feet"]["ids"], :, :]

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.view_init(10, 120)
    pts = []
    for i in range(data.shape[0]):
        pts += ax.plot([], [], [], 'o')

    ax.set_xlim3d(-700, 700)
    ax.set_ylim3d(-1000, -100)
    ax.set_zlim3d(0, 1500)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    anim = animation.FuncAnimation(fig,
                                   func=animate,
                                   frames=data.shape[1],
                                   interval=1./_descr["freq"],
                                   blit=True)
    plt.show()


def deviation(_dscr, mode="hands", step=1):
    data = _dscr["data"]

    if mode == "hands":
        data = data[_dscr["hands_ids"], :, :]
    init_pos = data[:, _dscr["init_frame"], :]

    offset = []
    for frame in range(step, _dscr["frames"], step):
        coord_aver = np.average(data[:, frame-step:frame, :], axis=1)
        shift = np.sum(sqrt((coord_aver - init_pos) ** 2)) / _dscr["markers"]
        offset.append(shift)
    return offset


def plot_relaxed_indices(_dscr, thr=100., plot=True):
    offset = deviation(_dscr)

    frame = 0
    relaxed_indices = []

    if offset[0] < thr:
        # reduce to the left-hollow-right form
        relaxed_indices.append(0)
        while frame < len(offset) and offset[frame] < thr:
            frame += 1

    while frame < len(offset):
        # handle left peak
        while frame < len(offset) and offset[frame] >= thr:
            frame += 1
        left_peak = frame

        while frame < len(offset) and offset[frame] < thr:
            frame += 1
        right_peak = frame

        aver = (left_peak + right_peak) // 2 - 1
        relaxed_indices.append(aver)

        frame += 1

    if plot:
        plt.plot(range(len(offset)), offset)
        relaxed_val = np.array(offset)[relaxed_indices]
        plt.plot(relaxed_indices, relaxed_val, 'ro')

        plt.xlabel("Frames")
        plt.ylabel("Aver offset per marker, mm")
        plt.title("#gesture samples: %d" % (len(relaxed_indices) - 1))
        plt.legend(["offset", "relaxed pos"], numpoints=1)

        plt.show()

    return relaxed_indices


def plot_them_all(folder):
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):

            try:
                _dscr = describe(folder + c3d_file)
                print "file: %s; \t frames: %d" % (c3d_file, _dscr["frames"])
            except:
                continue

            try:
                plot_relaxed_indices(_dscr, thr=70., plot=True)
                # display_animation(_dscr)
                # plot_deriv(acq)
            except:
                # print "bad"
                continue


def plot_deriv(_decr):
    data = moving_average(_decr["data"])
    deriv = diff(data, step=2)
    deriv = moving_average(deriv, wsize=7)

    dev = np.average(sqrt(np.sum(deriv ** 2, axis=2)), axis=0)
    dev = moving_average_simple(dev, wsize=7)
    plt.plot(range(len(dev)), dev)
    plt.plot(range(len(dev)), np.zeros(len(dev)), 'r--', lw=3)
    plt.show()


# reader.SetFilename("D:/GesturesDataset/Meet/M1_02_v2.c3d")
# reader.Update()
# acq_main = reader.GetOutput()
# print init_frame("D:/GesturesDataset/Meet/M1_02_v2.c3d")

_descr = describe("D:/GesturesDataset/Meet/M1_02_v2.c3d")
# display_animation(_descr, speed_rate=1)
# plot_relaxed_indices(_descr)
plot_them_all("D:/GesturesDataset/Meet/")
# plot_deriv(_descr)

