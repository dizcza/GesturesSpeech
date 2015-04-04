# coding=utf-8

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
import pickle
import os

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)


class Emotion(object):
    def __init__(self, obj_path, fps=None):
        """
        :param obj_path: path to pickled data
        :param fps: new fps to be set
        """
        self.project = "Emotion"
        self.name = os.path.basename(obj_path).strip(".pkl")

        # not for sure
        # TODO check it out from blender
        self.fps = 30

        # loading data from a pickle
        info = pickle.load(open(obj_path, 'rb'))
        self.data = info["data"]
        self.norm_data = np.copy(self.data)
        self.author = info["author"]
        self.emotion = info["emotion"]
        self.labels = info["labels"]
        self.frames = self.data.shape[1]

        self.set_fps(fps)
        self.preprocessor()

    def __str__(self):
        s = "File name:\t\t%s\n" % self.name
        s += "\temotion:\t%s\n" % self.emotion
        s += "\tauthor:\t\t%s\n" % self.author
        s += "\tdata.shape:\t%s\n" % str(self.data.shape)
        return s

    def get_ids(self, *args):
        """
         Gets specific data ids by marker_names keys.
        :param marker_names: list of keys
        :return: data ids, w.r.t. marker_names
        """
        ids = []
        for marker in args:
            ids.append(self.labels.index(marker))
        return ids

    def preprocessor(self):
        # step 1: subtracting nose pos
        nose_index = self.get_ids("p0")[0]
        nose_pos = np.average(self.data[nose_index,::], axis=0)
        self.data -= nose_pos
        print(nose_pos)

        # step 2: slope aligning
        eyebrow_ids = self.get_ids("ebr_or", "ebr_ir", "ebr_il", "ebr_ol")
        eye_ids = self.get_ids("eup_r", "edn_r", "eup_l", "edn_l")
        cheek_ids = self.get_ids("chr", "wr", "wl", "chl")
        lip_ids = self.get_ids("lir", "liup", "lidn", "lil")
        jaw_id = self.get_ids("jaw")
        median_points = []
        for ids in (eyebrow_ids, eye_ids, cheek_ids, lip_ids, jaw_id):
            xy_aver = np.average(np.average(self.data[ids, ::], axis=1), axis=0)
            median_points.append(xy_aver)
            print(xy_aver)
        median_points = np.resize(median_points, new_shape=(5, 2))
        self.coef = np.polyfit(median_points[:, 0], median_points[:, 1], deg=1)

        # step 3: dividing data by ? dist

    def accumulate_noise(self):
        """
         Data processing: accumulates noisy displacement unless action potential burst.
        """
        markers = len(self.labels)
        accumulated = np.zeros(markers)
        prev_frame = np.zeros(markers, dtype="int")

        act_pot = 1.
        for frame in range(1, self.data.shape[1]):
            accumulated += norm(self.data[:, frame, :] - self.data[:, frame-1, :], axis=1)
            for jointID in range(accumulated.shape[0]):
                if accumulated[jointID] > act_pot:
                    self.norm_data[:, frame, :] = self.data[:, frame, :]
                    accumulated[jointID] = 0.
                    prev_frame[jointID] = frame
                else:
                    self.norm_data[:, frame, :] = self.data[:, prev_frame[jointID], :]

    def set_fps(self, new_fps):
        """
            Modify data, w.r.t. new fps (taken from HumanoidBasic class).
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

    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        self.scat.set_offsets(self.data[:, frame, :])
        a, b = self.coef
        xs = np.arange(50)
        ys = a * xs + b
        plt.plot(xs, ys)
        plt.plot(self.data[:, frame, 0], self.data[:, frame, 1], 'bo')
        return []

    def animate(self):
        """
         Animates 3d data.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scat = plt.scatter(self.data[:, 0, 0], self.data[:, 0, 1])
        self.ax.grid()
        self.ax.set_title(self.emotion)
        # self.data = self.norm_data

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=self.frames,
                                       interval=50.,     # in ms
                                       blit=True)
        try:
            plt.show(self.fig)
        except AttributeError:
            pass


if __name__ == "__main__":
    em = Emotion(r"D:\GesturesDataset\Emotion\pickles\28-4-1.pkl")
    # print(em)
    # em.preprocessor()
    em.animate()
