# coding=utf-8

import numpy as np
# from files_modifier import *
import btk


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
    print "Markers: %d" % acq.GetPoints().GetItemNumber()

    print "\nALL METADATA:"
    for i in range(acq.GetMetaData().GetChildNumber()):
        print(acq.GetMetaData().GetChild(i).GetLabel() + ':')
        for j in range(acq.GetMetaData().GetChild(i).GetChildNumber()):
            print acq.GetMetaData().GetChild(i).GetChild(j).GetLabel(),
        print('\n')

    for i in range(acq.GetPoints().GetItemNumber()):
        print acq.GetPoint(i).GetLabel()
        print acq.GetPoint(i).GetDescription()
        # print acq.GetPoint(i).GetValues()


def check_for_missed(data):
    """
     Checks for values in data being zeros.
    :param data: (#markers, #frames, 3) ndarray of 3d points data
    :return overall zero values number in data
    """
    # for marker in range(data.shape[0]):
    #     for frame in range(data.shape[1]):
    #         for ordinate in range(data.shape[2]):
    #             if data[marker][frame][ordinate] == 0:
    #                 print marker, frame, ordinate

    # missed = len(data[data == 0])
    corrupted_frames = []
    for frame in range(data.shape[1]):
        if 0 in data[:, frame, :] or np.isnan(data[:, frame, :]).any():
            corrupted_frames.append(frame)
    return corrupted_frames


if __name__ == "__main__":
    # change_orientation("D:/GesturesDataset/Family/F4_mcraw.c3d")
    pass