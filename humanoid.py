# coding = utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import json
from basic import BasicMotion

# TODO set confidence measure


# noinspection PyTypeChecker
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
        self.shoulder_length = 0.

    def __str__(self):
        """
        :return: string representation of gesture
        """
        s = BasicMotion.__str__(self)
        s += "\t shoulder length: \t %.3f m\n" % self.shoulder_length
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

        # step 2: normalize by shoulder dist
        sh_ids = self.get_ids(sh_left, sh_right)
        sh_diff = self.data[sh_ids[0], ::] - self.data[sh_ids[1], ::]
        self.shoulder_length = np.average(norm(sh_diff, axis=1))
        sh_std = norm(np.std(sh_diff, axis=0))
        self.std = norm([center_std, sh_std])
        self.norm_data = self.data / self.shoulder_length

    def define_moving_markers(self, mode):
        """
         Sets moving markers, w.r.t. mode.
        :param mode: use both hand (by default) or only prime one
        """
        self.moving_markers = []
        if mode == "bothHands":
            for marker in self.labels:
                if marker in self.hand_markers:
                    self.moving_markers.append(marker)

    def get_internal_energy(self, mode):
        """
         Internal energy is computed as overall joint displacements
         along the frames.
        :param mode: bothHands or oneHand
        :return: internal energy in normal units
        """
        self.compute_displacement(mode)
        d = np.array(list(self.joint_displace.values()))
        w = np.array(list(self.weights.values()))
        U = np.sum(d * w)
        return U

    def init_3dbox(self):
        self.xmin = 0.
        self.xmax = 0.
        self.ymin = 0.1
        self.ymax = 0.1

    def init_3d(self):
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

    def animate(self, faster=1):
        """
         Animates 3d data.
        """
        self.init_3d()
        self.faster = faster

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=int(self.frames/faster),
                                       interval=1.,     # in ms
                                       blit=True)
        try:
            plt.show(self.fig)
        except AttributeError:
            pass

    def get_weights_discrepancy(self, other, mode):
        """
         Tells whether self joint displacement snapshot is
         alike to the other joint displacement snapshot.
        :param other: other gesture
        :param mode: use both hand (by default) or only prime one
        :return: weights discrepancy between self and other
        """
        self.compute_displacement(mode)
        other.compute_displacement(mode)

        present_markers = [mar for mar in self.labels if mar in other.labels]
        if not any(present_markers):
            return np.inf

        active_self_displ = [self.joint_displace[marker] for marker in present_markers]
        active_other_displ = [other.joint_displace[marker] for marker in present_markers]
        diff = norm(np.subtract(active_self_displ, active_other_displ))
        the_lowest_sum = min(sum(active_self_displ), sum(active_other_displ))
        diff_rate = diff / the_lowest_sum
        return diff_rate

    def is_comparable_with(self, other, mode="bothHands", thr=0.16):
        """
         Tells whether self joint displacement snapshot is
         alike to the other joint displacement snapshot.
        :param other: other gesture
        :param mode: use both hand (by default) or only prime one
        :param thr: rate of the lowest displacement sum as a similarity measure
        :return: if self-gest and other-gest can pertain to the same class
                (DTW should go next to specify the answer)
        """
        diff_rate = self.get_weights_discrepancy(other, mode)
        return diff_rate < thr


def align_gestures(self, other):
    """
     Total gestures alignment of their data shapes by throwing out missed labels.
    :param self: one gesture
    :param other: another gesture
    """
    throw_labels = {
        "self": [],
        "other": []
    }
    instance = {
        "self": self,
        "other": other
    }
    for known_label in self.labels:
        if known_label not in other.labels:
            throw_labels["self"].append(known_label)
    for unknown_label in other.labels:
        if unknown_label not in self.labels:
            throw_labels["other"].append(unknown_label)

    for me in instance:
        del_ids = instance[me].get_ids(*throw_labels[me])
        instance[me].data = np.delete(instance[me].data, del_ids, axis=0)
        instance[me].norm_data = np.delete(instance[me].norm_data, del_ids, axis=0)
        for missed_marker in throw_labels[me]:
            instance[me].labels.remove(missed_marker)
            del instance[me].weights[missed_marker]
            del instance[me].joint_displace[missed_marker]
            del instance[me].joint_std[missed_marker]
            if missed_marker in instance[me].moving_markers:
                instance[me].moving_markers.remove(missed_marker)

    # update weights
    for me in instance:
        weights_new_sum = sum(instance[me].weights.values())
        # instance labels are already updated
        for marker in instance[me].labels:
            instance[me].weights[marker] /= weights_new_sum

    # return aligned self and other gestures
    return self, other