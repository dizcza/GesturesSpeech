# coding=utf-8

import os
import shutil
import warnings

import numpy as np

try:
    import btk
except ImportError:
    import MOCAP.local_tools.btk_fake as btk


def print_info(filename):
    """
     Prints the whole info about filename.c3d.
    :param filename: .c3d-file
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()
    print('Acquisition duration: %.2f s' % acq.GetDuration())
    print('Point frequency: %.2f Hz' % acq.GetPointFrequency())
    print('Number of frames: %d' % acq.GetPointFrameNumber())
    print('Point unit: %s' % acq.GetPointUnit())
    print('Analog frequency: %.2f Hz' % acq.GetAnalogFrequency())
    print('Number of analog channels: %d' % acq.GetAnalogNumber())
    print('Number of events: %d' % acq.GetEventNumber())
    print("Markers: %d" % acq.GetPoints().GetItemNumber())

    print("\nALL METADATA:")
    if hasattr(acq, "fake") and acq.fake is True:
        fake_btk_warn = "to print out acquisition metadata from %s, use native btk package" % filename
        warnings.warn(fake_btk_warn)
    for i in range(acq.GetMetaData().GetChildNumber()):
        print(acq.GetMetaData().GetChild(i).GetLabel() + ':')
        for j in range(acq.GetMetaData().GetChild(i).GetChildNumber()):
            print(acq.GetMetaData().GetChild(i).GetChild(j).GetLabel(),)
        print('\n')

    print("\nPOINT C3D info:")
    for i in range(acq.GetPoints().GetItemNumber()):
        print(acq.GetPoint(i).GetLabel())
        print(acq.GetPoint(i).GetDescription())
        # print(acq.GetPoint(i).GetValues())


def get_corrupted_frames(data):
    """
     Checks data values for being zeros.
    :param data: (#markers, #frames, 3) ndarray of 3d points data
    :return list of corrupted frame IDs
    """
    corrupted_frames = []
    for frame in range(data.shape[1]):
        if 0 in data[:, frame, :] or np.isnan(data[:, frame, :]).any():
            corrupted_frames.append(frame)
    return corrupted_frames


def init_frame(filename, verbose=False):
    """
    :param filename: .c3d-file
    :return: init (relaxed) frame pertains to the filename.c3d
    """
    short_name = filename.split('\\')[-1]
    initFrames = {
        "M1_02_v2.c3d": 280,
        "M2_02.c3d": 405,
        "M3_01.c3d": 785,
        "M4_01.c3d": 1470,
        "M5_01.c3d": 660,
        "M6_01.c3d": 400,
        "M7_01.c3d": 400,
        "M8_01.c3d": 500,
        "M9_01.c3d": 500,

        "C1_mcraw.c3d": 795,
        "C2_mcraw.c3d": 420,
        "C3_mcraw.c3d": 850,

        "D1_mcraw001.c3d": 740,
        "D2_mcraw001.c3d": 780,
        "D3_mcraw001.c3d": 740,
        "D4_mcraw001.c3d": 800,

        "F1_mcraw.c3d": 755,
        "F2_mcraw.c3d": 1150,
        "F3_mcraw.c3d": 815,
        "F4_mcraw.c3d": 385,
        "F5_mcraw.c3d": 1055,

        "H1_mcraw.c3d": 800,
        "H2_mcraw.c3d": 450,
        "H3_mcraw.c3d": 520,
        "H4_mcraw.c3d": 450,
        "H5_mcraw.c3d": 700,
        "H6_mcraw.c3d": 1000,

        "N1_mcraw002.c3d": 285,
        "N2_mcraw002.c3d": 440,
        "N3_mcraw001.c3d": 450,

        "S1_mcraw001.c3d": 570,
        "S2_mcraw001.c3d": 430,
        "S3_mcraw001.c3d": 580,
        "S4_mcraw001.c3d": 400,
        "S5_mcraw001.c3d": 500,
        "S6_mcraw001.c3d": 380,
        "S7_mcraw001.c3d": 450,
        "S8_mcraw001.c3d": 510,
    }

    if short_name in initFrames.keys():
        return initFrames[short_name]
    else:
        if verbose:
            message = "c3d filename %s has no init frame; returning 0" % short_name
            warnings.warn(message)
        return 0


def separate_dataset():
    """
     0's samples will be for training
     1's samples will be for testing
    """
    mocap_path = r"D:\GesturesDataset\MoCap\splitAll"
    trn_folder = os.path.join(mocap_path, "Training")
    tst_folder = os.path.join(mocap_path, "Testing")

    for folder in (trn_folder, tst_folder):
        shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)

    for c3d_file in os.listdir(mocap_path):
        gesture_name = c3d_file[:-12]
        src = os.path.join(mocap_path, c3d_file)
        if c3d_file.endswith("_sample0.c3d"):
            trn_subfolder = os.path.join(trn_folder, gesture_name)
            os.mkdir(trn_subfolder)
            shutil.copy(src, trn_subfolder)
        elif c3d_file.endswith("_sample1.c3d"):
            tst_subfolder = os.path.join(tst_folder, gesture_name)
            os.mkdir(tst_subfolder)
            shutil.copy(src, tst_subfolder)


if __name__ == "__main__":
    separate_dataset()