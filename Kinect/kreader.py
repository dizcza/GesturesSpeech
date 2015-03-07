# coding = utf-8

"""
Reference:
  @inproceedings{celebi2013gesture,
      author    = {Sait Celebi and Ali Selman Aydin and Talha Tarik Temiz and Tarik Arici},
      title     = {{Gesture Recognition Using Skeleton Data with Weighted Dynamic Time Warping}},
      booktitle = {Computer Vision Theory and Applications},
      publisher = {Visapp},
      year      = {2013},
}
"""

from humanoid import HumanoidBasic
import matplotlib.pyplot as plt
import numpy as np

KINECT_PATH = "D:\GesturesDataset\KINECT\\"
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
    return labels


def read_body(rlines):
    """
    :param rlines: body lines from txt-file
    :returns: - (20, frames, 3) data
              - fps
    """
    BLOCK_SIZE = 82
    frames = int(rlines[4])
    data = np.zeros(shape=(MARKERS, frames, 3))
    block_begins = [5 + i * BLOCK_SIZE for i in range(len(rlines) / BLOCK_SIZE)]
    time_events = []
    for begin in block_begins:
        frameID = int(rlines[begin][1:])
        time_events.append(convert_time(rlines[begin+1]))
        for labelID in range(MARKERS):
            _start = begin + 2 + 4 * labelID
            _end = _start + 4
            # label = block[_start]
            x, z, y = np.array(rlines[_start+1:_end], dtype=float)
            data[labelID, frameID, :] = x, -y, z
    time_events = np.array(time_events)
    dt = time_events[1:] - time_events[:-1]
    fps = np.average(1./dt)

    return data, fps


class HumanoidKinect(HumanoidBasic):
    """
        Provides an instruments to visualize and operate 3d data,
        taken with Microsoft Kinect sensor and written in simple txt-format.
        Data can be downloaded from the reference below:
            http://datascience.sehir.edu.tr/visapp2013/
    """

    def __init__(self, filename, fps=None):
        """
         Creates a gesture from Kinect folder.
        :param filename: txt-file path
        """
        HumanoidBasic.__init__(self)
        self.project = "Kinect"

        # dealing with hand markers
        self.hand_markers = ["HandLeft",  "WristLeft",  "ElbowLeft",
                             "HandRight", "WristRight", "ElbowRight"]
        swap = {"left": "right", "right": "left"}
        # Note: 'prime' and 'free' hands are the attributes only for training.
        # Although they are determined each time the gesture is created,
        # they aren't used (and not relevant at all) during the testing.
        self.prime_hand = filename.split('\\')[-1].split("Hand")[0].lower()
        self.free_hand = swap[self.prime_hand]
        self.shoulder_markers = ["ShoulderLeft", "ShoulderCenter", "ShoulderRight"]

        with open(filename, 'r') as rfile:
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
         Sets moving markers, w.r.t. mode.
        :param mode: use both hand (by default) or only prime one
        """
        self.moving_markers = np.array([])
        if mode == "oneHand":
            for marker in self.labels:
                if self.prime_hand in marker.lower() and marker in self.hand_markers:
                    self.moving_markers = np.append(self.moving_markers, marker)
        else:
            HumanoidBasic.define_moving_markers(self, mode="bothHands")

    def show_displacement(self, mode="bothHands"):
        """
        :param mode: use both hand (by default) or only prime one
        """
        HumanoidBasic.plot_displacement(self, mode, rotation=0, fontsize=12, add_error=True)
        plt.show()


if __name__ == "__main__":
    gest = HumanoidKinect("D:\GesturesDataset\KINECT\Training\LeftHandWave\LeftHandWave_000.txt")
    gest.animate()
    print gest
    gest.show_displacement()
    gest.compute_weights(mode="oneHand", beta=1e-4)
