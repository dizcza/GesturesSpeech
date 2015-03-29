# coding=utf-8

"""
 This is a fake btk module to simulate native btk (https://code.google.com/p/b-tk)
 on Python 3.x, using c3d module from https://github.com/EmbodiedCognition/py-c3d.

 NOTE: This is a very cut btk package.
       Most of methods, used in native btk module, are omitted here.
       If you want to get full access onto .c3d files,
         switch to Python 2.7 version and use native btk module.
       The purpose of creating this fake btk module is to be able to
         switch easily from Python 2.7 onto Python 3.x,
         since native btk module uses Python 2.7, whereas blender package
         (its versions 2.5 and higher) works only on Python 3.x.
"""

import c3d
import numpy as np


def gather_labels(reader):
    """
    :param reader: c3d Reader object
    :return: gesture labels
    """
    markers = reader.header.point_count
    labelsParam = reader.get("POINT:LABELS")
    byte_length = labelsParam.dimensions[0]
    labels = []
    for i in range(markers):
        raw_str = labelsParam.bytes[i * byte_length:(i + 1) * byte_length].decode('utf-8')
        this_label = raw_str.strip(' ').split(':')[-1]
        labels.append(this_label)
    return tuple(labels)


def gather_data(reader):
    """
    :param reader: c3d Reader object
    :return array: (#markers, #frames, 3) data
    """
    frames = reader.last_frame() - reader.first_frame() + 1
    markers = reader.header.point_count
    data = np.empty(shape=(markers, frames, 3))
    for i, (_, points, _) in enumerate(reader.read_frames()):
        data[:, i, :] = points[:, :3]

    # dealing with mm --> m
    return data / 1e3


class btkAcquisitionFileReader(object):
    """
     Simulates native btk btkAcquisitionFileReader class.
    """
    def __init__(self):
        self.__name__ = "btk_fake_reader"
        self._fname = None
        self._acq = None

    def SetFilename(self, fname):
        """
        :param fname: filename.c3d
        """
        self._fname = fname

    def Update(self):
        self._acq = Acquisition(self._fname)

    def GetOutput(self):
        return self._acq


class Acquisition(object):
    """
     Simulates btk acquisition methods.
    """
    def __init__(self, fname):
        """
        :param fname: filename.c3d
        """
        self.fake = True
        with open(fname, 'rb') as handle:
            reader = c3d.Reader(handle)
            self.fps = reader.header.frame_rate
            self.frames = reader.last_frame() - reader.first_frame() + 1
            self.labels = gather_labels(reader)
            self.data = gather_data(reader)
            assert self.frames == self.data.shape[1], "#frames don't match with data.shape"

    def GetData(self):
        return self.data

    def GetMetaData(self):
        return MetaData()

    def GetPoints(self):
        """
        :return: AcqPoints to be able to print out number of markers
        """
        return AcqPoints(self.labels)

    def GetPoint(self, _id):
        return Point(self.labels[_id], _id)

    def GetPointFrequency(self):
        return self.fps

    def GetDuration(self):
        """
        :return: a gesture duration in seconds
        """
        return float(self.frames) / self.fps

    def GetPointFrameNumber(self):
        return self.frames

    def GetPointUnit(self):
        return

    def GetAnalogFrequency(self):
        return 0

    def GetAnalogNumber(self):
        return 0

    def GetEventNumber(self):
        return 0


class AcqPoints(object):
    def __init__(self, labels):
        self._markers_total = len(labels)

    def GetItemNumber(self):
        """
        :return: total number of markers in the acquisition
        """
        return self._markers_total


class Point(object):
    def __init__(self, label, index):
        """
        :param label: (str) point label
        """
        self._label = label
        self._index = index

    def GetLabel(self):
        return self._label

    def GetDescription(self):
        return self._index


class MetaData(object):
    def __init__(self):
        pass

    def GetChildNumber(self):
        return 0