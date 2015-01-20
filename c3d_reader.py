# coding=utf-8

from helper import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from numpy import sqrt
import os
import json

reader = btk.btkAcquisitionFileReader()


def get_hands_ids(given_labels):
    hands_labels = get_hand_labels()
    hands_ids = []
    del given_labels["RIDX3"]
    del given_labels["RPNK3"]
    for label in hands_labels:
        if label in given_labels:
            hands_ids.append(given_labels[label])
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
        "acquisition": acq,
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
    description["freq"] = acq.GetPointFrequency()
    description["missed"] = check_for_missed(data)

    if description["missed"] > 0:
        data[data == 0] = np.NaN             # handle missing data (zeros)
    description["data"] = data

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


def check_for_missed(data):
    # for marker in range(data.shape[0]):
    #     for frame in range(data.shape[1]):
    #         for ordinate in range(data.shape[2]):
    #             if data[marker][frame][ordinate] == 0:
    #                 print marker, frame, ordinate

    missed = len(data[data == 0])
    if len(data[data == 0]) > 0:
        print "\tThere is %d missed data (zeros)." % missed
    return missed


def get_points_data(acq):
    data = np.empty((3, acq.GetPointFrameNumber(), 1))
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(i).GetLabel()
        data = np.dstack((data, acq.GetPoint(label).GetValues().T))
    data = np.delete(data.T, 0, axis=0)  # first marker is noisy for this file
    return data


def animate(i):
    for pt, xi in zip(pts, data):
        x, y, z = xi[:i].T
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])
    return pts


def display_animation(_dscr, speed_rate=1, frames_range=None):
    global pts, data

    chunks_num = int(speed_rate * _dscr["freq"] / 24)
    if frames_range:
        frames = np.arange(frames_range[0], frames_range[1], chunks_num)
    else:
        frames = np.arange(0, _dscr["frames"]-1, chunks_num)

    data = _dscr["data"][:, frames, :]
    # data = data[_dscr["feet"]["ids"], :, :]

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
                                   interval=1./_dscr["freq"],
                                   blit=True)
    plt.show()


def deviation(_dscr, mode="hands", step=1):
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


def get_relaxed_indices(_dscr, thr=10.):
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
        while frame < len(offset) and (offset[frame] >= thr or np.isnan(offset[frame])):
            frame += 1
        left_peak = frame

        while frame < len(offset) and (offset[frame] < thr or np.isnan(offset[frame])):
            frame += 1
        right_peak = frame

        if right_peak > left_peak:
            aver = (left_peak + right_peak) // 2 - 1
            relaxed_indices.append(aver)

        frame += 1

    return relaxed_indices


def plot_relaxed_indices(_dscr, thr=10.):
    offset = deviation(_dscr)
    relaxed_indices = get_relaxed_indices(_dscr, thr)

    plt.plot(range(len(offset)), offset)
    relaxed_val = np.array(offset)[relaxed_indices]
    plt.plot(relaxed_indices, relaxed_val, 'ro')

    unique_gestures = (len(relaxed_indices) - 1) / 2.
    plt.xlabel("Frames")
    plt.ylabel("Aver offset per marker, mm")
    plt.title("#gesture samples: 2 * %d" % unique_gestures)
    plt.legend(["offset", "relaxed pos"], numpoints=1)

    plt.show()


def split_file(filename):
    writer = btk.btkAcquisitionFileWriter()
    _dscr = describe(filename)
    short_name = filename.split('/')[-1]
    folder_path = filename[:-len(short_name)]
    short_name = short_name.split('.c3d')[0]
    folder_path += "split/" + short_name + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    relaxed_indices = get_relaxed_indices(_dscr)
    pairs = zip(relaxed_indices[:-1], relaxed_indices[1:])
    pairs_in_two = [(pairs[i], pairs[i+1]) for i in range(0, len(relaxed_indices)-1, 2)]

    for unique_id, two_the_same_samples in enumerate(pairs_in_two):
        # two_the_same_samples = pairs_in_two[unique_id]
        gesture = "_gest%d" % unique_id
        for sample_id, frame_borders in enumerate(two_the_same_samples):
            new_short_name = short_name + gesture + "_sample%d.c3d" % sample_id
            left_frame, right_frame = frame_borders

            # Copy original data
            clone = _dscr["acquisition"].Clone()

            # Crop the acquisition to keep only the ROI
            clone.ResizeFrameNumberFromEnd(_dscr["frames"] - left_frame + 1)
            clone.ResizeFrameNumber(right_frame - left_frame + 1)
            clone.SetFirstFrame(left_frame)

            # Make sure to left events to be empty
            # since they initially were empty
            clone.ClearEvents()

            # Create new C3D file
            writer.SetInput(clone)
            writer.SetFilename(folder_path + new_short_name)
            writer.Update()


def split_mult_files(folder_path):
    for c3d_file in os.listdir(folder_path):
        if c3d_file.endswith(".c3d"):
            split_file(folder_path + '/' + c3d_file)


def plot_them_all(folder):
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            try:
                _dscr = describe(folder + c3d_file)
                print "file: %s; \t frames: %d" % (c3d_file, _dscr["frames"])

                plot_relaxed_indices(_dscr, thr=10.)
                # display_animation(_dscr)
                # plot_deriv(acq)
            except:
                print "cannot describe %s" % c3d_file
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
# print_info("D:/GesturesDataset/Meet/M5_01.c3d")
# print init_frame("D:/GesturesDataset/Meet/M1_02_v2.c3d")

# _dscr = describe("D:/GesturesDataset/Meet/M3_01.c3d")
# display_animation(_dscr, speed_rate=1)
# plot_relaxed_indices(_dscr)
# plot_them_all("D:/GesturesDataset/Meet/")
# plot_deriv(_dscr)
# split_file("D:/GesturesDataset/Meet/M5_01.c3d")

split_mult_files("D:/GesturesDataset/Meet/")