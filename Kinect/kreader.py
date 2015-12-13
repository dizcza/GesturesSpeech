# coding=utf-8

"""
Kinect project is based upon database from the reference:
  @inproceedings{celebi2013gesture,
      author    = {Sait Celebi and Ali Selman Aydin and Talha Tarik Temiz and Tarik Arici},
      title     = {{Gesture Recognition Using Skeleton Data with Weighted Dynamic Time Warping}},
      booktitle = {Computer Vision Theory and Applications},
      publisher = {Visapp},
      year      = {2013},
}
"""

import warnings
import os
import numpy as np

from tools.humanoid import HumanoidBasic
from tools.anim_viewer import DataViewer
from Kinect.data_manager import load_database

# loads Kinect database if not loaded yet
# and returns a path to Kinect project data
KINECT_PATH = load_database()

# total number of present markers
MARKERS = 20


def convert_time(string):
    """
    :param string: HH:MM:SS time format
    :return: time in sec
    """
    time_event = 0.
    for tt, multiplier in zip(string.split(":"), [3600., 60., 1.]):
        time_event += float(tt) * multiplier
    return time_event


def gather_labels(rlines):
    """
    :param rlines: body lines from txt-file
    :return: dic: {[marker name]: id, ...}
    """
    label_indices = 7 + 4 * np.arange(MARKERS)
    label_names = np.array(rlines)[label_indices]
    labels = []
    for noisy_label in label_names:
        labels.append(noisy_label[1:-1])
    return tuple(labels)


def read_body(rlines):
    """
    :param rlines: body lines from txt-file
    :returns: - (20, frames, 3) data
              - fps
    """
    block_size = 82
    frames = int(rlines[4])
    data = np.zeros(shape=(MARKERS, frames, 3))
    chunks = int((len(rlines) / block_size))

    if frames != chunks:
        warnings.warn("frames != chunks; took min")
        frames = min(frames, chunks)

    block_begins = [5 + i * block_size for i in range(frames)]
    time_events = []
    for begin in block_begins:
        frame = int(rlines[begin][1:])
        time_events.append(convert_time(rlines[begin + 1]))
        for label_id in range(MARKERS):
            _start = begin + 2 + 4 * label_id
            _end = _start + 4
            x, z, y = np.array(rlines[_start + 1: _end], dtype=float)
            data[label_id, frame, :] = x, -y, z
    time_events = np.array(time_events)
    dt = time_events[1:] - time_events[:-1]
    fps = np.average(1. / dt)

    return data, fps


class HumanoidKinect(HumanoidBasic):
    """
        Provides an instruments to visualize and operate 3d data,
        taken with Microsoft Kinect sensor and written in simple txt-format.
        Data can be downloaded from the reference below:
            http://datascience.sehir.edu.tr/visapp2013/
    """

    def __init__(self, txt_path, fps=None):
        """
         Creates a gesture from a Kinect folder.
        :param txt_path: txt-file path
        :param fps: new fps to be set
        """
        HumanoidBasic.__init__(self, fps)
        self.project = "Kinect"
        self.fname = os.path.basename(txt_path)

        # dealing with hand marker names
        self.hand_markers = ["HandLeft", "WristLeft", "ElbowLeft",
                             "HandRight", "WristRight", "ElbowRight"]
        self.shoulder_markers = ["ShoulderLeft", "ShoulderCenter", "ShoulderRight"]

        # Note: 'prime' and 'free' hand params attribute the oneHand
        # training scenario, where only the prime hand (active one)
        # is relevant. Although they are set each time the gesture
        # is created, they aren't used (and not relevant at all)
        # during the testing.
        swap = {"left": "right", "right": "left"}
        self.prime_hand = txt_path.split(os.sep)[-1].split("Hand")[0].lower()
        self.free_hand = swap[self.prime_hand]

        with open(txt_path, 'rU') as rfile:
            rlines = rfile.readlines()
            self.name = rlines[3][1:-1]
            self.labels = gather_labels(rlines)
            self.data, self.fps = read_body(rlines)

        self.frames = self.data.shape[1]
        self.set_fps(fps)
        self.preprocessing()
        self.set_weights()

    def define_moving_markers(self, mode):
        """
         Defines moving markers, w.r.t. hand mode.
        :param mode: use both hands (by default) or only the prime one
        """
        self.moving_markers = np.array([])
        if mode == "oneHand":
            for marker in self.labels:
                if self.prime_hand in marker.lower() and marker in self.hand_markers:
                    self.moving_markers = np.append(self.moving_markers, marker)
        else:
            HumanoidBasic.define_moving_markers(self, mode)

    def animate_pretty(self):
        """
         Pretty 3d animation like in OpenGL.
        """
        feet_labels = "FootRight", "FootLeft"
        feet_ids = self.get_ids(*feet_labels)
        hip_to_floor_dist = -np.average(self.data[feet_ids, :, 2])
        _data = np.swapaxes(self.data, 0, 1)
        _data *= 1e3
        _data[:, :, 2] += 1e3 * hip_to_floor_dist
        shape = _data.shape[0], _data.shape[1], 1
        _data = np.append(_data, np.zeros(shape), axis=2)
        try:
            DataViewer(_data, self.fps).mainloop(slow_down=2.5)
        except StopIteration:
            pass


def demo_run():
    """
     Kinect project demo.
    """
    gest_path = os.path.join(KINECT_PATH, "Training", "RightHandPushUp", "RightHandPushUp_000.txt")
    assert os.path.exists(gest_path), "Unable to find the %s" % gest_path
    gest = HumanoidKinect(gest_path)
    print(gest)
    gest.show_displacements("bothHands", ("ElbowRight", "WristRight", "HandRight"))
    gest.animate_pretty()


if __name__ == "__main__":
    demo_run()
