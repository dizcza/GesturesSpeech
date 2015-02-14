import btk
import os


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


