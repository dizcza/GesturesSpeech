# coding=utf-8

from helper import *
from labeling import *
from files_modifier import *
from plotter import *
import os
import json


reader = btk.btkAcquisitionFileReader()


def describe(c3d_file):
    """
    :param c3d_file: .c3d-file to read info from
    :return: full description dic
    """
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


def get_points_data(acq):
    """
    :param acq: acquisition from .c3d-file
    :return: (#markers, #frames, 3) ndarray of 3d points data
    """
    data = np.empty((3, acq.GetPointFrameNumber(), 1))
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(i).GetLabel()
        data = np.dstack((data, acq.GetPoint(label).GetValues().T))
    data = np.delete(data.T, 0, axis=0)  # first marker is noisy for this file
    return data

def plot_relaxed_indices(_dscr):
    """
     Plots current deviations per frame.
    :param _dscr: description info
    """
    offset = deviation(_dscr)
    relaxed_indices = get_relaxed_indices(_dscr)

    plt.plot(range(len(offset)), offset)
    relaxed_val = np.array(offset)[relaxed_indices]
    plt.plot(relaxed_indices, relaxed_val, 'ro')

    unique_gestures = (len(relaxed_indices) - 1) / 2.
    plt.xlabel("Frames")
    plt.ylabel("Aver offset per marker, mm")
    plt.title("#gesture samples: 2 * %d" % unique_gestures)
    plt.legend(["offset", "relaxed pos"], numpoints=1)

    plt.show()


def get_relaxed_indices(_dscr, thr=10.):
    """
    :param _dscr: description info
    :param thr: the positions below that value are considered to be near relaxed (init) pos
    :return: possible relaxed (init) indices
    """
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
            # argMin = aver + np.argmin(offset[aver:right_peak])
            relaxed_indices.append(aver)

        frame += 1

    # check the last one for being min deviation
    if offset[-1] - offset[relaxed_indices[-1]] < -2:
        relaxed_indices[-1] = len(offset) - 1

    return relaxed_indices


def split_file(filename, split_thr):
    """
    Splits particular .c3d-file into unique examples.
    :param filename: .c3d-file
    :param split_thr: the positions below that value are considered to be near relaxed (init) pos
    """
    writer = btk.btkAcquisitionFileWriter()
    _dscr = describe(filename)
    short_name = filename.split('/')[-1]
    folder_path = filename[:-len(short_name)]
    short_name = short_name.split('.c3d')[0]
    folder_path += "split/" + short_name + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    relaxed_indices = get_relaxed_indices(_dscr, split_thr)
    pairs = zip(relaxed_indices[:-1], relaxed_indices[1:])
    pairs_in_two = [(pairs[i], pairs[i+1]) for i in range(0, len(relaxed_indices)-1, 2)]

    for unique_id, two_the_same_samples in enumerate(pairs_in_two):
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


def split_mult_files(folder_path, split_thr):
    """
     Splits all examples into their folders by unique ones.
    :param folder_path: folder with .c3d-examples from particular group
    :param split_thr: the positions below that value are considered to be near relaxed (init) pos
    """
    for c3d_file in os.listdir(folder_path):
        if c3d_file.endswith(".c3d"):
            split_file(folder_path + '/' + c3d_file, split_thr)
        print "%s is successfully split" % c3d_file


def plot_them_all(folder):
    """
     Plots the main info: deviation per frame.
    :param folder: path to folder with .c3d-files
    """
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            try:
                _dscr = describe(folder + c3d_file)
                print "file: %s; \t frames: %d" % (c3d_file, _dscr["frames"])

                plot_relaxed_indices(_dscr)
                # display_animation(_dscr, frames_range=[2725, 2900])
                # plot_deriv(acq)
            except:
                print "cannot describe %s" % c3d_file
                continue


# reader.SetFilename("D:/GesturesDataset/Meet/M1_02_v2.c3d")
# reader.Update()
# acq_main = reader.GetOutput()
# print_info("D:/GesturesDataset/Meet/M5_01.c3d")
# print init_frame("D:/GesturesDataset/Meet/M1_02_v2.c3d")

# _dscr = describe("D:/GesturesDataset/Hospital/H6_mcraw.c3d")
# display_animation(_dscr, speed_rate=1)
# plot_relaxed_indices(_dscr)
# plot_them_all("D:/GesturesDataset/School/")
# plot_deriv(_dscr)
# split_file("D:/GesturesDataset/Hospital/H6_mcraw.c3d")

# split_mult_files("D:/GesturesDataset/School/", split_thr=10.0)