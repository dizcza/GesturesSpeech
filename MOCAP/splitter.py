# coding=utf-8

import MOCAP.helper as helper
from MOCAP.mreader import HumanoidUkr
from MOCAP.helper import get_corrupted_frames
import numpy as np
import matplotlib.pyplot as plt
import os


class HumanoidUkrSplitter(HumanoidUkr):
    def __init__(self, filepath):
        HumanoidUkr.__init__(self, filepath, fps=None)
        self.markers_total = len(self.labels)
        self.corrupted = get_corrupted_frames(self.data)
        self.offset = []


    def compute_offset(self, mode="bothHands", step=1):
        """
         Computes and stores offset (deviation) from current to init pos for each frame.
        :param mode: whether use only hands marker or full set of markers
        :param step: number of frames per step
        """
        relaxed_frame = helper.init_frame(self.fpath)
        init_pos = self.data[:, relaxed_frame, :]
        if mode == "bothHands":
            hand_ids = self.get_ids(self.hand_markers)
            data = self.data[hand_ids, ::]
            init_pos = init_pos[hand_ids, :]
        else:
            data = self.data
        offset = []
        for frame in range(step, self.frames, step):
            coord_aver = np.average(data[:, frame-step:frame, :], axis=1)
            shift = np.sqrt(np.sum((coord_aver - init_pos) ** 2)) / data.shape[0]
            offset.append(shift)
        self.offset = offset


    def compute_relaxed_indices(self, thr=10.):
        """
         Computes and stores relaxed indices to be split then.
        :param thr: the positions below that value are considered to be
                    near relaxed (init) pos
        """
        frame = 0
        relaxed_indices = []

        if not hasattr(self, "offset"):
            self.compute_offset()

        if self.offset[0] < thr:
            # reduce to the left-hollow-right form
            relaxed_indices.append(0)
            while frame < len(self.offset) and self.offset[frame] < thr:
                frame += 1

        while frame < len(self.offset):
            # handle left peak
            while frame < len(self.offset) and (self.offset[frame] >= thr or np.isnan(self.offset[frame])):
                frame += 1
            left_peak = frame

            while frame < len(self.offset) and (self.offset[frame] < thr or np.isnan(self.offset[frame])):
                frame += 1
            right_peak = frame

            if right_peak > left_peak:
                aver = (left_peak + right_peak) // 2 - 1
                # argMin = aver + np.argmin(offset[aver:right_peak])
                relaxed_indices.append(aver)

            frame += 1

        # check the last one for being min deviation
        if self.offset[-1] - self.offset[relaxed_indices[-1]] < -2:
            relaxed_indices[-1] = len(self.offset) - 1

        self.relaxed_indices = relaxed_indices


    def get_double_border_frames(self, split_thr):
        """
        :param split_thr: the positions below that value are considered to be
                          near relaxed (init) pos
        :return list of double pairs, each of them contains 2 pairs of border frames
                (for each sample per gest)
        """
        self.compute_relaxed_indices(split_thr)
        pairs = zip(self.relaxed_indices[:-1], self.relaxed_indices[1:])
        double_pairs = [(pairs[i], pairs[i + 1]) for i in range(0, len(self.relaxed_indices) - 1, 2)]

        return double_pairs


    def plot_relaxed_indices(self):
        """
         Plots current deviations per frame.
        """
        if not hasattr(self, "relaxed_indices"):
            self.compute_relaxed_indices()

        plt.plot(self.offset)
        relaxed_val = np.array(self.offset)[self.relaxed_indices]
        plt.plot(self.relaxed_indices, relaxed_val, 'ro')

        unique_gestures = (len(self.relaxed_indices) - 1) / 2.
        plt.xlabel("Frames")
        plt.ylabel("Aver offset per marker, mm")
        plt.title("#gesture samples: 2 * %d" % unique_gestures)
        plt.legend(["offset", "relaxed pos"], numpoints=1)

        plt.show()


def plot_them_all(folder):
    """
     Plots relaxed indices with their deviation from relaxed frame.
    :param folder: path to folder with .c3d-files
    """
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            try:
                fname = os.path.join(folder, c3d_file)
                gest = HumanoidUkrSplitter(fname)
                print("file: %s; \t frames: %d" % (c3d_file, gest.frames))
                gest.plot_relaxed_indices()
            except:
                print("cannot describe %s" % c3d_file)
                continue


if __name__ == "__main__":
    gest = HumanoidUkrSplitter("D:\GesturesDataset\MoCap\Meet\M7_01.c3d")
    # gest.animate()
    gest.plot_relaxed_indices()