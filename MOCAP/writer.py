# coding=utf-8

import btk
import os
from splitter import *


def rewrite_files_in(folder):
    """
     Just rewrites all .c3d-files in the folder to get rid of BTK-warnings.
    :param folder: path to folder with .c3d-files
    """
    reader = btk.btkAcquisitionFileReader()
    writer = btk.btkAcquisitionFileWriter()

    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            reader.SetFilename(folder + c3d_file)
            reader.Update()
            acq = reader.GetOutput()

            if not os.path.exists(folder + "/reWritten/"):
                os.makedirs(folder + "/reWritten/")

            writer.SetInput(acq)
            writer.SetFilename(folder + "/reWritten/" + c3d_file)
            writer.Update()


def change_orientation(filename, default_orient=(0, 2, 1)):
    """
    Changes invalid orientation to valid one: XYZ.
    :param filename: filename of .c3d-file
    :param default_orient: invalid orient (XZY by default)
    """
    reader = btk.btkAcquisitionFileReader()
    writer = btk.btkAcquisitionFileWriter()

    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()

    for i in range(acq.GetPoints().GetItemNumber()):
        point_data = acq.GetPoint(i).GetValues()
        point_data[:, [0, 1, 2]] = point_data[:, default_orient]
        # point_data[:, 1] = -point_data[:, 1]
        acq.GetPoint(i).SetValues(point_data)

    writer.SetInput(acq)
    writer.SetFilename(filename)
    writer.Update()


def fill_missed_frame_and_save(filename, markerID, frame_missed):
    reader = btk.btkAcquisitionFileReader()
    writer = btk.btkAcquisitionFileWriter()

    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()

    framesXYZ = acq.GetPoint(markerID).GetValues()

    point_prev_frame = acq.GetPoint(markerID).GetValues()[frame_missed-1, :]
    point_next_frame = acq.GetPoint(markerID).GetValues()[frame_missed+1, :]
    point_aver_frame = (point_prev_frame + point_next_frame) / 2.

    framesXYZ[frame_missed, :] = point_aver_frame

    acq.GetPoint(markerID).SetValues(framesXYZ)

    print "%s[%d] vals set to %s" % (acq.GetPoint(markerID).GetLabel(), frame_missed, framesXYZ[frame_missed, :])

    writer.SetInput(acq)
    writer.SetFilename(filename)
    writer.Update()


def modify_orient_in(folder):
    """
    :param folder: path to folder with .c3d-files
    """
    print "Reading..."
    for c3d in os.listdir(folder):
        if c3d.endswith(".c3d"):
            change_orientation(folder + c3d)
    print "Done."


def split_file(folder_path, filename, double_pairs):
    """
    Splits particular .c3d-file into unique examples.
    :param folder_path: .c3d-file
    :param filename: short filename
    :param double_pairs: list of double pairs of frame borders
    """
    reader = btk.btkAcquisitionFileReader()
    writer = btk.btkAcquisitionFileWriter()
    reader.SetFilename(folder_path + filename)
    reader.Update()
    acq = reader.GetOutput()
    FRAMES = acq.GetPointFrameNumber()

    short_name = filename.split('.c3d')[0]
    new_folder_path = folder_path + "split/" + short_name + "/"
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    for unique_id, two_the_same_samples in enumerate(double_pairs):
        gesture = "_gest%d" % unique_id
        for sample_id, frame_borders in enumerate(two_the_same_samples):
            new_short_name = short_name + gesture + "_sample%d.c3d" % sample_id
            left_frame, right_frame = frame_borders

            # Copy original data
            clone = acq.Clone()

            # Crop the acquisition to keep only the ROI
            clone.ResizeFrameNumberFromEnd(FRAMES - left_frame + 1)
            clone.ResizeFrameNumber(right_frame - left_frame + 1)
            clone.SetFirstFrame(left_frame)

            # Make sure to left events to be empty
            # since they initially were empty
            clone.ClearEvents()

            # Create new C3D file
            writer.SetInput(clone)
            writer.SetFilename(new_folder_path + new_short_name)
            writer.Update()
    print "%s was successfully split into 2x%d samples" % (short_name, len(double_pairs))


def split_mult_files(folder_path, split_thr):
    """
     Splits all examples into their folders by unique ones.
    :param folder_path: folder with .c3d-examples from particular group
    :param split_thr: the positions below that value are considered to be near relaxed (init) pos
    """
    for c3d_file in os.listdir(folder_path):
        if c3d_file.endswith(".c3d"):
            gest = HumanoidUkr(folder_path + c3d_file)
            double_pair = gest.get_double_border_frames(split_thr)
            split_file(folder_path, c3d_file, double_pair)