# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import json
from matplotlib import rc
import os


font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)


class BasicMotion(object):
    def __init__(self, fps):
        self.fps = fps
        self.project = ""
        self.name = ""
        self.labels = []
        self.weights = {}
        self.data = np.array([], dtype=np.float64)
        self.norm_data = np.array([], dtype=np.float64)
        self.frames = 0
        self.std = 0.
        self.joint_displace = {}
        self.joint_std = {}
        self.moving_markers = []
        self.fig = None
        self.ax = None

    def __str__(self):
        """
        :return: string representation of gesture
        """
        s = "Motion name: %s\n" % self.name
        s += "\t data.shape:\t%s\n" % str(self.data.shape)
        s += "\t FPS: \t\t\t %d\n" % self.fps
        s += "\t data std: \t\t %.5f m " % self.std
        return s

    def is_good(self):
        """
        :return: tells whether it contains nan weights (bad guy) or not (good guy)
        """
        return True

    def define_moving_markers(self, mode):
        msg = "moving markers should be defined properly or set by default"
        assert mode is None, msg
        self.moving_markers = tuple(self.labels)

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
        return tuple(ids)

    def get_weights(self):
        """
        :return array: (#markers,) ravelled array of weights
        """
        if not any(self.weights):
            self.compute_weights(None, 1e-6)
        weights_ordered = []
        for marker in self.labels:
            weights_ordered.append(self.weights[marker])
        return np.array(weights_ordered)

    def compute_displacement(self, mode):
        """
         Computes joints displacements.
        :param mode: use both hand (by default) or only prime one
        """
        self.define_moving_markers(mode)
        for markerID in range(self.norm_data.shape[0]):
            marker = self.labels[markerID]
            if marker in self.moving_markers:
                # its shape == (frames-1, dim)
                frames_xyz_delta = np.diff(self.norm_data[markerID, ::], axis=0)
                frames_xyz_delta = frames_xyz_delta[~np.isnan(frames_xyz_delta).any(axis=1)]
                if frames_xyz_delta.shape[0] > 0:
                    dist_per_frame = norm(frames_xyz_delta, axis=1)
                else:
                    # that marker isn't involved in DTW comparison anymore
                    dist_per_frame = 0

                # offset: average --> sum
                offset = np.sum(dist_per_frame)

                j_std = np.std(dist_per_frame, dtype=np.float64)
            else:
                offset = 0
                j_std = 0
            self.joint_displace[marker] = offset
            self.joint_std[marker] = j_std

    def define_plot_style(self):
        """
         Setting bar char plot style.
        """
        self.rotation = 0
        self.fontsize = 12
        self.add_error = False

    def plot_displacement(self, mode, highlight):
        """
         Plots a chart bar of joints displacements.
        :param mode: defines moving markers
        :param highlight: which markers to highlight
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
            barlist = ax.bar(ind, offset_list, width, yerr=joint_std_list,
                             error_kw=dict(elinewidth=2, ecolor='red'))
        else:
            barlist = ax.bar(ind, offset_list, width)
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xticks(ind+width/2)
        ax.set_title("%s joint displacements" % self.name)
        ax.set_ylabel("marker motion measure,  norm units")
        xtickNames = ax.set_xticklabels(self.moving_markers)
        plt.setp(xtickNames, rotation=self.rotation, fontsize=self.fontsize)
        for lighted_marker in highlight:
            lighted_id = self.moving_markers.index(lighted_marker)
            barlist[lighted_id].set_color("#99CCFF")

    def show_displacements(self, mode, highlight=()):
        """
        :param mode: use both hand (by default) or only prime one
        """
        self.plot_displacement(mode, highlight)
        plt.show()

    def compute_weights(self, mode, beta):
        """
         Computes weights to be used in DTW.
        :param beta: param to be chosen during the training
        """
        self.compute_displacement(mode)
        if beta is None:
            denom = np.sum(list(self.joint_displace.values()))
            for marker in self.labels:
                self.weights[marker] = self.joint_displace[marker] / denom
        else:
            displacements = np.array(list(self.joint_displace.values()))
            denom = np.sum(1. - np.exp(-beta * displacements))
            for marker in self.labels:
                self.weights[marker] = (1. - np.exp(-beta * self.joint_displace[marker])) / denom

    def set_weights(self):
        """
         Loads weights from _INFO.json
        """
        json_file = self.project.upper() + "_INFO.json"
        if not os.path.exists(json_file):
            json_file = "../" + json_file
        try:
            dic_info = json.load(open(json_file, 'r'))
        except FileNotFoundError:
            dic_info = {"weights": ()}

        weights_aver_dic = dic_info["weights"]
        if self.name in weights_aver_dic:
            # if _INFO file provides weights for current gesture
            weights_arr = weights_aver_dic[self.name]
            for markerID, marker_name in enumerate(self.labels):
                self.weights[marker_name] = weights_arr[markerID]
        else:
            # compute weights for unknown gesture
            self.compute_weights(None, None)

    def set_gesture_length(self, length):
        """
         Makes gesture length to be fixed.
        :param length: (int), new number of frames to be set
        """
        step = self.frames / float(length)
        keep_frames = np.array(np.arange(length) * step, dtype=int)
        self.data = self.data[:, keep_frames, :]
        self.norm_data = self.norm_data[:, keep_frames, :]
        self.frames = length
