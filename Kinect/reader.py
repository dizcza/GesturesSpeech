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

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MOCAP.math_kernel import moving_average
from mpl_toolkits.mplot3d import Axes3D
import json


KINECT_PATH = "D:\GesturesDataset\KINECT\\"
MARKERS = 20
Tmin = .007
Tmax = .075
HAND_MARKERS = np.array(["HandLeft",  "WristLeft",  "ElbowLeft",
                         "HandRight", "WristRight", "ElbowRight"])


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


class Humanoid(object):
    """
        Provides an instruments to visualize and operate 3d data,
        taken with Microsoft Kinect sensor and written in simple txt-format.
        Data can be downloaded from the reference below:
            http://datascience.sehir.edu.tr/visapp2013/
    """

    def __init__(self, filename):
        """
         Creates a gesture from Kinect folder.
        :param filename: txt-file path
        """
        swap = {"left": "right", "right": "left"}
        self.prime_hand = filename.split('\\')[-1].split("Hand")[0].lower()
        self.free_hand = swap[self.prime_hand]
        with open(filename, 'r') as rfile:
            rlines = rfile.readlines()
            self.name = rlines[3][1:-1]
            self.labels = gather_labels(rlines)
            self.data, self.fps = read_body(rlines)
        self.frames = self.data.shape[1]
        self.estimate_human_height()
        self.preprocessing()
        self.set_weights()


    def __del__(self):
        plt.close()


    def __str__(self):
        s = "GESTURE: %s\n" % self.name
        s += "\t frames: \t %d\n" % self.frames
        s += "\t FPS: \t\t %d\n" % self.fps
        s += "\t height: \t %g m\n" % self.height
        s += "\t data std: \t %g m (or %.2f %%)\n" % (self.std, self.std/self.height * 100)
        return s


    def get_norm_data(self):
        """
        :return: (#markers, #frames, 3) normalized data
        """
        return self.norm_data


    def get_ids(self, *args):
        """
         Gets specific data ids by marker_names keys.
        :param marker_names: list of keys
        :return: data ids, w.r.t. marker_names
        """
        ids = []
        for marker in args:
            ids.append(self.labels.index(marker))
        if len(ids) == 1:
            return ids[0]
        else:
            return ids


    def get_weights(self):
        """
        :return: (#markers,) ravelled array of weights
        """
        weights_ordered = []
        for marker in self.labels:
            weights_ordered.append(self.weights[marker])
        return np.array(weights_ordered)


    def estimate_human_height(self):
        """
         Computes a human height.
        """
        feet_ids = self.get_ids("FootRight", "FootLeft")
        head_id = self.get_ids("Head")
        pentagone = self.data[head_id, ::] - self.data[feet_ids, ::]
        self.height = np.average(np.average(pentagone, axis=1), axis=0)[2]


    def preprocessing(self):
        """
         3 steps data pre-processing.
        """
        # step 0: moving average smoothing
        wsize = 5
        self.data = moving_average(self.data, wsize)
        self.frames -= wsize - 1

        # step 1: subtract shoulder center from all joints
        center_id = self.get_ids("ShoulderCenter")
        shoulder_center = np.average(self.data[center_id, ::], axis=0)
        center_std = norm(np.std(self.data[center_id, ::], axis=0))
        self.data -= shoulder_center

        # step 2: normalize by shoulder dist
        sh_ids = self.get_ids("ShoulderLeft", "ShoulderRight")
        sh_diff = self.data[sh_ids[0], ::] - self.data[sh_ids[1], ::]
        self.shoulder_length = np.average(norm(sh_diff, axis=1))
        sh_std = norm(np.std(sh_diff, axis=0))
        self.std = norm([center_std, sh_std])
        self.norm_data = self.data / self.shoulder_length


    def define_moving_markers(self, mode):
        """
         Sets moving markers, w.r.t. mode.
        :param mode: whether use one or both hands
        """
        self.moving_markers = np.array([])
        if mode == "oneHand":
            for marker in self.labels:
                if self.prime_hand in marker.lower() and marker in HAND_MARKERS:
                    self.moving_markers = np.append(self.moving_markers, marker)
        else:
            for marker in self.labels:
                if marker in HAND_MARKERS:
                    self.moving_markers = np.append(self.moving_markers, marker)


    def compute_displacement(self, mode=None):
        """
         Computes joints displacements.
        """
        self.define_moving_markers(mode)
        self.joint_displace = {}
        self.joint_std = {}
        for markerID in range(self.norm_data.shape[0]):
            marker = self.labels[markerID]
            if marker in self.moving_markers:
                # delta_per_frame.shape == (frames-1, 3)
                delta_per_frame = self.norm_data[markerID, 1:, :] - \
                                  self.norm_data[markerID, :-1, :]
                dist_per_frame = norm(delta_per_frame, axis=1)
                offset = np.average(dist_per_frame)
                j_std = np.std(dist_per_frame, dtype=np.float64)
            else:
                offset = 0
                j_std = 0
            self.joint_displace[marker] = offset
            self.joint_std[marker] = j_std


    def upd_thr(self, mode=None):
        """
         Updates global displacement thresholds
        """
        global Tmin, Tmax
        self.compute_displacement(mode)
        if min(self.joint_displace) < Tmin:
            Tmin = min(self.joint_displace)
        if max(self.joint_displace) > Tmax:
            Tmax = max(self.joint_displace)


    def show_displacement(self, mode=None):
        """
         Plots a chart bar of joints displacements.
        """
        if not hasattr(self, "joint_displace"):
            self.compute_displacement(mode)

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)

        offset_list, joint_std_list = [], []
        for marker in self.labels:
            if marker in self.moving_markers:
                offset_list.append(self.joint_displace[marker])
                joint_std_list.append(self.joint_std[marker])

        ind = np.arange(len(offset_list))
        width = 0.5
        offset_bar = ax.bar(ind, offset_list, width, yerr=joint_std_list,
                            error_kw=dict(elinewidth=2, ecolor='red'))
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xticks(ind+width/2)
        ax.set_title("%s joint displacements" % self.name)
        ax.set_ylabel("displacement, norm units")
        ax.set_xticklabels(self.moving_markers)
        plt.show()


    def init_3d(self):
        """
         Initialize empty 3d plots.
        """
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = Axes3D(self.fig)
        self.ax.view_init(15, 110)
        self.pts = []
        for marker in range(self.data.shape[0]):
            self.pts += self.ax.plot([], [], [], 'o', color='black', markersize=4)
        self.ax.set_xlim3d([np.nanmin(self.data[:, :, 0]), np.nanmax(self.data[:, :, 0])])
        self.ax.set_ylim3d([np.nanmin(self.data[:, :, 1]), np.nanmax(self.data[:, :, 1])])
        self.ax.set_zlim3d([np.nanmin(self.data[:, :, 2]), np.nanmax(self.data[:, :, 2])])


    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        for marker in range(self.data.shape[0]):
            x, y, z = self.data[marker, frame, :]
            self.pts[marker].set_data([x], [y])
            self.pts[marker].set_3d_properties([z])
        return []


    def animate(self):
        """
         Animates 3d data.
        :param mode: show all markers or prime only
        """
        self.init_3d()

        self.anim = animation.FuncAnimation(self.fig,
                                           func=self.next_frame,
                                           frames=self.frames,
                                           interval=1.,
                                           blit=True)
        plt.show(self.fig)


    def limit_displacement(self):
        """
         Limits joint displacements from the bottom and the top.
        """
        # FIXME 6 hand ids --> 3 primary ids
        for marker in self.moving_markers:
            if self.joint_displace[marker] < Tmin:
                self.joint_displace[marker] = Tmin
            elif self.joint_displace[marker] > Tmax:
                self.joint_displace[marker] = Tmax


    def compute_weights(self, beta=1e-4):
        """
         Computes weights to be used in DTW.
        :param beta: param to be chosen during the training
        """
        self.weights = {}
        displacements = np.array(self.joint_displace.values())
        denom = np.sum(1. - np.exp(-beta * displacements))
        for marker in self.labels:
            self.weights[marker] = (1. - np.exp(
                -beta * self.joint_displace[marker])) / denom


    def set_weights(self):
        """
         Sets loaded weihts from KINECT_INFO
        """
        self.weights = {}
        KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
        weights_aver_dic = KINECT_INFO["weights"]
        weights_arr = weights_aver_dic[self.name]
        for markerID, marker_name in enumerate(self.labels):
            self.weights[marker_name] = weights_arr[markerID]