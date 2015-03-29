# coding = utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import json


# TODO set confidence measure


# noinspection PyTypeChecker
class HumanoidBasic(object):
    """
     Constructs a humanoid with empty fields and basic methods.
    """

    def __init__(self):
        """
         Necessary fields declaration.
        """
        self.project = ""
        self.name = ""
        self.hand_markers = []
        self.labels = []
        self.weights = {}
        self.data = np.array([])
        self.norm_data = np.array([])
        self.fps = 0.
        self.frames = 0
        self.std = 0.
        self.shoulder_markers = "", "", ""
        self.shoulder_length = 0.
        self.moving_markers = []
        self.joint_displace = {}
        self.joint_std = {}

    def __str__(self):
        """
        :return: string representation of gesture
        """
        s = "GESTURE: %s\n" % self.name
        s += "\t shoulder length: \t %.3f m\n" % self.shoulder_length
        s += "\t frames: \t\t\t %d\n" % self.frames
        s += "\t FPS: \t\t\t\t %d\n" % self.fps
        s += "\t data std: \t\t\t %.5f m " % self.std
        return s

    def get_norm_data(self):
        """
        :return: (#markers, #frames, 3) normalized data
        """
        return self.norm_data

    def set_fps(self, new_fps):
        """
            Modify data, w.r.t. new fps.
        :param new_fps: fps (or points frequency) to be made in the data
        """
        if new_fps is None or new_fps >= self.fps:
            # does nothing
            return
        step_to_throw = float(self.fps) / (self.fps - new_fps)
        indices_thrown = np.arange(self.data.shape[1]) * step_to_throw
        indices_thrown = indices_thrown.astype(dtype="int")
        self.data = np.delete(self.data, indices_thrown, axis=1)
        self.frames = self.data.shape[1]
        self.fps = new_fps

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
        :return array: (#markers,) ravelled array of weights
        """
        weights_ordered = []
        for marker in self.labels:
            weights_ordered.append(self.weights[marker])
        return np.array(weights_ordered)

    def preprocessing(self):
        """
         2 steps data pre-processing.
        """
        # step 1: subtract shoulder center from all joints
        sh_left, sh_center, sh_right = self.shoulder_markers
        center_id = self.get_ids(sh_center)
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

    def compute_displacement(self, mode):
        """
         Computes joints displacements.
        :param mode: use both hand (by default) or only prime one
        """
        self.define_moving_markers(mode)
        for markerID in range(self.norm_data.shape[0]):
            marker = self.labels[markerID]
            if marker in self.moving_markers:
                # delta_per_frame.shape == (frames-1, 3)
                delta_per_frame = np.subtract(self.norm_data[markerID, 1:, :],
                                              self.norm_data[markerID, :-1, :])
                dist_per_frame = norm(delta_per_frame, axis=1)

                # offset: average --> sum
                offset = np.sum(dist_per_frame)

                j_std = np.std(dist_per_frame, dtype=np.float64)
            else:
                offset = 0
                j_std = 0
            self.joint_displace[marker] = offset
            self.joint_std[marker] = j_std

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

    def plot_displacement(self, mode):
        """
         Plots a chart bar of joints displacements.
        :param mode: use both hand (by default) or only prime one
        """
        self.define_plot_style()
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
        if self.add_error:
            ax.bar(ind, offset_list, width, yerr=joint_std_list,
                   error_kw=dict(elinewidth=2, ecolor='red'))
        else:
            ax.bar(ind, offset_list, width)
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xticks(ind+width/2)
        ax.set_title("%s joint displacements" % self.name)
        ax.set_ylabel("marker motion measure,  norm units")
        xtickNames = ax.set_xticklabels(self.moving_markers)
        plt.setp(xtickNames, rotation=self.rotation, fontsize=self.fontsize)

    def show_displacements(self, mode="bothHands"):
        """
        :param mode: use both hand (by default) or only prime one
        """
        self.plot_displacement(mode)
        plt.show()

    def define_plot_style(self):
        """
         Setting bar char plot style.
        """
        self.rotation = 0
        self.fontsize = 12
        self.add_error = False

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

    def compute_weights(self, mode, beta):
        """
         Computes weights to be used in DTW.
        :param beta: param to be chosen during the training
        """
        self.compute_displacement(mode)
        displacements = np.array(list(self.joint_displace.values()))
        denom = np.sum(1. - np.exp(-beta * displacements))
        for marker in self.labels:
            self.weights[marker] = (1. - np.exp(
                -beta * self.joint_displace[marker])) / denom

    def set_weights(self):
        """
         Sets loaded weights from _INFO.json
        """
        json_file = self.project.upper() + "_INFO.json"
        dic_info = json.load(open(json_file, 'r'))
        weights_aver_dic = dic_info["weights"]
        if self.name in weights_aver_dic:
            # if _INFO file provides weights for current gesture
            weights_arr = weights_aver_dic[self.name]
            for markerID, marker_name in enumerate(self.labels):
                self.weights[marker_name] = weights_arr[markerID]
        else:
            # compute weights for unknown gesture
            self.compute_weights(mode="bothHands", beta=0.)

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