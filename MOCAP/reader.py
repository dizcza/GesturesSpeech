# coding=utf-8

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from helper import *
from labelling import *
from math_kernel import *


def gather_points_data(acq):
    """
    :param acq: acquisition from .c3d-file
    :return: (#markers, #frames, 3) ndarray of 3d points data
    """
    data = np.empty((3, acq.GetPointFrameNumber(), 1))
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(i).GetLabel()
        data = np.dstack((data, acq.GetPoint(label).GetValues().T))
    data = np.delete(data.T, 0, axis=0)  # first marker is noisy for this file (truly)
    return data


class HumanoidUkr(object):
    """
     Creates an instance of Ukrainian Motion Capture gesture, saved in .c3d-format.
    """

    def __init__(self, c3d_file):
        self.filename = c3d_file
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(c3d_file)
        reader.Update()

        self.acq = reader.GetOutput()
        self.labels = gather_labels(self.acq)
        self.data = gather_points_data(self.acq)
        
        self.markers_total = self.acq.GetPoints().GetItemNumber()
        self.hand_ids = get_hands_ids(self.labels)
        self.hand_data = self.data[self.hand_ids, :, :]
        relaxed_frame = init_frame(c3d_file)
        self.init_pos = self.data[:, relaxed_frame, :]
        self.frames = self.acq.GetPointFrameNumber()
        self.freq = self.acq.GetPointFrequency()
        self.missed = check_for_missed(self.data)

        if any(self.missed):
            self.data[self.data == 0] = np.NaN


    def __del__(self):
        plt.close()


    def compute_offset(self, mode="hands", step=1):
        """
         Computes and stores offset (deviation) from current to init pos for each frame.
        :param mode: whether use only hands marker or full set of markers
        :param step: number of frames per step
        """
        if mode == "hands":
            data = self.hand_data
            init_pos = self.init_pos[self.hand_ids, :]
        else:
            data = self.data
            init_pos = self.init_pos
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


    def animate(self, save=None):
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

        if save:
            mp4_file = self.filename.split('/')[-1][:-4] + '.mp4'
            self.anim.save(mp4_file, writer='ffmpeg', fps=50)
            print "Saved in %s" % mp4_file
        else:
            plt.show(self.fig)


if __name__ == "__main__":
    gest = HumanoidUkr("D:\GesturesDataset\Meet\M7_01.c3d")
    gest.plot_relaxed_indices()
    plt.show()