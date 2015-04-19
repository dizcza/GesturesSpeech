# coding=utf-8

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from basic import BasicMotion


class Emotion(BasicMotion):
    def __init__(self, obj_path, fps=None):
        """
        :param obj_path: path to pickled data
        :param fps: new fps to be set
        """
        BasicMotion.__init__(self, fps=24)
        self.project = "Emotion"
        self.name = os.path.basename(obj_path).strip(".pkl")

        # loading data from a pickle
        info = pickle.load(open(obj_path, 'rb'))
        self.data = info["data"]
        self.norm_data = None
        self.author = info["author"]
        self.emotion = info["emotion"]
        self.labels = info["labels"]
        self.frames = self.data.shape[1]

        self.set_fps(fps)
        self.preprocessor()
        self.set_weights()

    def __str__(self):
        s = BasicMotion.__str__(self)
        s += "\n\t emotion:\t\t %s" % self.emotion
        s += "\n\t author:\t\t %s" % self.author
        return s

    def preprocessor(self):
        # TODO think about better preprocessor
        # step 0: dealing with first frame bug
        self.data = self.data[:,1:,:]
        self.frames -= 1
        self.norm_data = self.data.copy()

        # step 1: subtract nose pos of the first frame
        nose, jaw, ebr_ir, ebr_il = self.get_ids("p0", "jaw", "ebr_ir", "ebr_il")
        # nose_pos = np.average(self.data[nose_ind,::], axis=0)
        first_frame = 0
        while np.isnan(self.data[nose, first_frame, :]).any():
            first_frame += 1
        first_frame = min(first_frame, self.data.shape[1] - 1)
        self.norm_data -= self.data[nose, first_frame, :]

        # step 2: divide data by base line dist
        top_point = (self.data[ebr_ir,0,:] + self.data[ebr_il,0,:]) / 2.
        base_line = norm(top_point - self.data[jaw,0,:])
        self.norm_data /= base_line

    def slope_align(self):
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

    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        self.scat.set_offsets(self.data[:, frame, :])
        # a, b = self.coef
        # xs = np.arange(-50, 0)
        # ys = a * xs + b
        # plt.plot(xs, ys)
        # plt.plot(self.data[:, frame, 0], self.data[:, frame, 1], 'bo')
        return []

    def visual_help(self):
        eyebrow_ids = self.get_ids("ebr_or", "ebr_ir", "ebr_il", "ebr_ol")
        eye_ids = self.get_ids("eup_r", "edn_r", "eup_l", "edn_l")
        cheek_ids = self.get_ids("chr", "wr", "wl", "chl")
        lip_ids = self.get_ids("lir", "liup", "lidn", "lil")
        jaw_id = self.get_ids("jaw")
        self.data = self.data[eyebrow_ids, ::]
        print(eyebrow_ids)
        print(list(zip(range(len(self.labels)), self.labels)))
        print("ebr_or", "ebr_ir", "ebr_il", "ebr_ol")

    def animate(self):
        """
         Animates 3d data.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scat = plt.scatter(self.data[:, 0, 0], self.data[:, 0, 1])
        self.ax.grid()
        self.ax.set_title(self.emotion)

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=self.frames,
                                       interval=1e3/self.fps,     # in ms
                                       blit=True)
        try:
            plt.show(self.fig)
        except AttributeError:
            pass


if __name__ == "__main__":
    em = Emotion(r"D:\GesturesDataset\Emotion\pickles\33-3-1.pkl")
    # a = pickle.load(open(r"D:\GesturesDataset\Emotion\pickles\33-3-1.pkl", 'rb'))
    # print(a["data"].shape)
    # em.preprocessor()
    em.animate()
