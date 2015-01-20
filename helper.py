import btk
import os
import numpy as np


def get_hand_labels():
    lhand_labels = np.array([
        "LFSH",     "LBSH",     "LUPA",     "LELB",     "LIEL",
        "LOWR",     "LIWR",     "LWRE",     "LIHAND",   "LOHAND",
        "LIDX1",    "LIDX2",    "LIDX3",    "LMDL1",    "LMDL2",
        "LMDL3",    "LRNG1",    "LRNG2",    "LRNG3",    "LPNK1",
        "LPNK2",    "LPNK3",    "LTHM1",    "LTHM2",    "LTHM3",
    ])
    rhand_labels = ["R" + label[1:] for label in lhand_labels]
    hands_labels = np.concatenate([lhand_labels, rhand_labels])
    return hands_labels


def init_frame(filename):
    short_name = filename.split('/')[-1]
    initFrames = {
        "M1_02_v2.c3d": 280,
        "M2_02.c3d": 405,
        "M3_01.c3d": 785,
        "M4_01.c3d": 1470,
        "M5_01.c3d": 660,
        "M6_01.c3d": 400,
        "M7_01.c3d": 400,
        "M8_01.c3d": 500,
        "M9_01.c3d": 500
    }
    return initFrames[short_name]


def print_info(filename):
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
    :return: smooth data
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


def rewrite_files_in(folder):
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


def change_orientation(c3d_file, default_orient=(0, 2, 1)):
    reader = btk.btkAcquisitionFileReader()
    writer = btk.btkAcquisitionFileWriter()

    reader.SetFilename(c3d_file)
    reader.Update()
    acq = reader.GetOutput()

    for i in range(acq.GetPoints().GetItemNumber()):
        point_data = acq.GetPoint(i).GetValues()
        point_data[:, [0, 1, 2]] = point_data[:, default_orient]
        acq.GetPoint(i).SetValues(point_data)

    writer.SetInput(acq)
    writer.SetFilename(c3d_file)
    writer.Update()


def modify_orient_in(folder):
    print "Reading..."
    for c3d in os.listdir(folder):
        if c3d.endswith(".c3d"):
            change_orientation(folder + c3d)
    print "Done."


def check_for_missed_hand_labels(given_labels):
    hands_labels = get_hand_labels()
    missed_labels = []
    for label in hands_labels:
        if label not in given_labels:
            missed_labels.append(label)
    return missed_labels