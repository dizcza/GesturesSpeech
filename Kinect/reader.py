# coding = utf-8

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MOCAP.math_kernel import moving_average
from mpl_toolkits.mplot3d import Axes3D


KINECT_PATH = "D:\GesturesDataset\KINECT\\"
MARKERS = 20
BLOCK_SIZE = 82


def convert_time(string):
    """
    :param string: HH:MM:SS time format
    :return: time in sec
    """
    time_event = 0.
    for tt, multiplier in zip(string.split(":"), [3600, 60, 1]):
        time_event += float(tt) * multiplier
    return time_event


def gather_labels(rlines):
    """
    :param rlines: body lines from txt-file
    :return: dic: {[marker name]: id, ...}
    """
    label_indices = 7 + 4 * np.arange(MARKERS)
    label_names = np.array(rlines)[label_indices]
    labels = {}
    for labelID, marker_name in enumerate(label_names):
        labels[marker_name[1:-1]] = labelID
    return labels


def read_body(rlines):
    """
    :param rlines: body lines from txt-file
    :returns: - (20, frames, 3) data
              - fps
    """
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
    def __init__(self, filename):
        """
         Creates a gesture from Kinect folder.
        :param filename: txt-file path
        """
        with open(KINECT_PATH + filename, 'r') as rfile:
            rlines = rfile.readlines()
            self.name = rlines[3][1:-1]
            self.labels = gather_labels(rlines)
            self.data, self.fps = read_body(rlines)
        self.frames = self.data.shape[1]
        self.estimate_height()
        self.preprocessing()
        self.init_3d()

    def __str__(self):
        s = "GESTURE: %s\n" % self.name
        s += "\t frames: \t %d\n" % self.frames
        s += "\t FPS: \t\t %d\n" % self.fps
        s += "\t height: \t %g m\n" % self.height
        s += "\t data std: \t %g m (or %.2f %%)\n" % (self.std, self.std/self.height * 100)
        return s

    def get_ids(self, marker_names):
        """
         Gets specific data ids by marker_names keys.
        :param marker_names: list of keys
        :return: data ids, w.r.t. marker_names
        """
        ids = []
        for marker in marker_names:
            ids.append(self.labels[marker])
        return ids

    def estimate_height(self):
        """
         Computes a human height.
        """
        self.feet_ids = self.get_ids(["FootRight", "FootLeft"])
        self.feet_data = self.data[self.feet_ids, ::]
        pentagone = self.data[self.labels["Head"], ::] - self.feet_data
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
        center_id = self.labels["ShoulderCenter"]
        shoulder_center = np.average(self.data[center_id, ::], axis=0)
        center_std = norm(np.std(self.data[center_id, ::], axis=0))
        self.data -= shoulder_center

        # step 2: normalize by shoulder dist
        sh_ids = self.get_ids(["ShoulderLeft", "ShoulderRight"])
        sh_diff = self.data[sh_ids[0], ::] - self.data[sh_ids[1], ::]
        sh_dist = np.average(norm(sh_diff, axis=1))
        sh_std = norm(np.std(sh_diff, axis=0))
        self.std = norm([center_std, sh_std])
        self.norm_data = self.data / sh_dist

        # step 3: creature feature map
        self.primary_markers = ["HandLeft", "HandRight",
                                "WristLeft", "WristRight",
                                "ElbowLeft", "ElbowRight"]
        primary_ids = self.get_ids(self.primary_markers)
        self.primary_data = self.data[primary_ids, ::]
        # self.feature_map = self.primary_data.reshape(3 * len(self.primary_ids), self.frames) / sh_dist
        self.feature_map = self.primary_data / sh_dist

    def compute_displacement(self):
        """
         Computes joints displacements.
        """
        self.joint_displace = []
        self.joint_std = []
        for jointID, joint in enumerate(self.feature_map):
            # joint.shape == (frames, 3)
            delta_per_frame = joint[1:, :] - joint[:-1, :]
            dist_per_frame = norm(delta_per_frame, axis=1)
            offset = np.sum(dist_per_frame)
            self.joint_displace.append(offset)
            self.joint_std = np.std(dist_per_frame)
        self.joint_std = np.array(self.joint_std)

    def show_displacement(self):
        """
         Plots a chart bar of joints displacements.
        """
        self.compute_displacement()
        plt.clf()
        ax = self.fig.add_subplot(111)
        ind = np.arange(len(self.joint_displace))
        width = 0.5
        offsets = ax.bar(ind, self.joint_displace, width, yerr=self.joint_std,
                         error_kw=dict(elinewidth=2, ecolor='red'))
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xticks(ind+width/2)
        ax.set_title("%s joints displacement" % self.name)
        ax.set_ylabel("displacement, norm units")
        ax.set_xticklabels(self.primary_markers)
        plt.show()

    def init_3d(self):
        """
         Initialize empty 3d plots.
        """
        self.fig = plt.figure(figsize=(10, 10))
        # self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d')
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

    def animate(self, mode=None):
        """
         Animates 3d data.
        :param mode: show all markers or prime only
        """
        if mode == "prime":
            self.data = self.primary_data

        self.anim = animation.FuncAnimation(self.fig,
                                           func=self.next_frame,
                                           frames=self.frames,
                                           interval=1.,
                                           blit=True)
        plt.show(self.fig)



def main():
    gest = Humanoid("Training/LeftHandSwipeRight/LeftHandSwipeRight_007.txt")
    print gest
    gest.show_displacement()
    # gest.animate("prime")


main()



