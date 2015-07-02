# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

from tools.basic import BasicMotion


class HumanoidBasic(BasicMotion):
    """
     Constructs a humanoid with empty fields and basic methods.
    """

    def __init__(self, fps):
        """
         Necessary fields declaration.
        """
        BasicMotion.__init__(self, fps)
        self.hand_markers = []
        self.shoulder_markers = "", "", ""
        self.shoulder_width = 0.

    def __str__(self):
        """
        :return: string representation of gesture
        """
        s = BasicMotion.__str__(self)
        s += "\n\t shoulder width: \t %.3f m" % self.shoulder_width
        return s

    def preprocessing(self):
        """
         2 steps data pre-processing.
        """
        # step 1: subtract shoulder center from all joints
        sh_left, sh_center, sh_right = self.shoulder_markers
        center_id = self.get_ids(sh_center)[0]
        shoulder_center = np.average(self.data[center_id, ::], axis=0)
        center_std = norm(np.std(self.data[center_id, ::], axis=0))
        self.data -= shoulder_center

        # step 2: normalize by shoulder width
        sh_ids = self.get_ids(sh_left, sh_right)
        sh_diff = self.data[sh_ids[0], ::] - self.data[sh_ids[1], ::]
        self.shoulder_width = np.average(norm(sh_diff, axis=1))
        sh_std = norm(np.std(sh_diff, axis=0))
        self.std = norm([center_std, sh_std])
        self.norm_data = self.data / self.shoulder_width

    def define_moving_markers(self, mode):
        """
         Defines moving markers, w.r.t. mode.
        :param mode: use both hands (by default) or only prime one
        """
        self.moving_markers = []
        if mode == "bothHands":
            for marker in self.labels:
                if marker in self.hand_markers:
                    self.moving_markers.append(marker)
        else:
            BasicMotion.define_moving_markers(self, None)

    def init_3dbox(self):
        self.xmin = 0.
        self.xmax = 0.
        self.ymin = 0.1
        self.ymax = 0.1

    def init_animation(self):
        """
         Initialize empty 3d plots.
        """
        self.init_3dbox()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = Axes3D(self.fig)
        self.ax.view_init(15, 110)
        self.pts = []
        for marker in range(self.data.shape[0]):
            self.pts += self.ax.plot([], [], [], 'o', color='black', markersize=4)

        self.ax.set_xlim3d([np.nanmin(self.data[:, :, 0]) - self.xmin,
                            np.nanmax(self.data[:, :, 0]) + self.xmax])

        self.ax.set_ylim3d([np.nanmin(self.data[:, :, 1]) - self.ymin,
                            np.nanmax(self.data[:, :, 1]) + self.ymax])

        self.ax.set_zlim3d([np.nanmin(self.data[:, :, 2]),
                            np.nanmax(self.data[:, :, 2])])

    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        for marker in range(self.data.shape[0]):
            x, y, z = self.data[marker, self.faster * frame, :]
            self.pts[marker].set_data([x], [y])
            self.pts[marker].set_3d_properties([z])
        return []
