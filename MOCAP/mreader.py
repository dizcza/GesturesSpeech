# coding=utf-8

import btk
from humanoid import HumanoidBasic
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from labelling import *
from helper import init_frame


def gather_points_data(acq):
    """
    :param acq: acquisition from .c3d-file
    :return: (#markers, #frames, 3) ndarray of 3d points data
    """
    data = np.empty((3, acq.GetPointFrameNumber(), 1))
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label_id = acq.GetPoint(i).GetLabel()
        data = np.dstack((data, acq.GetPoint(label_id).GetValues().T))
    data = np.delete(data.T, 0, axis=0)  # first marker is noisy for this file (truly)

    # dealing with mm --> m
    return data / 1e3


def parse_fname(fname):
    """
    :param fname: path to .c3d-file
    :return: class name of the instance
    """
    short_name = fname.split('\\')[-1]
    if "_sample" in short_name:
        return short_name[:-12]
    else:
        return short_name


class HumanoidUkr(HumanoidBasic):
    """
     Creates an instance of Ukrainian Motion Capture gesture, saved in .c3d-format.
    """
    def __init__(self, c3d_file, fps=None):
        HumanoidBasic.__init__(self)
        # TODO think about self.name: parse xlsx info
        self.project = "MoCap"

        # setting unique gesture name
        self.name = parse_fname(c3d_file)

        # setting up BTK reader to gather acquisition
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(c3d_file)
        reader.Update()
        acq = reader.GetOutput()

        # initial fps (should be 120)
        self.fps = acq.GetPointFrequency()

        # dealing with markers
        self.labels = gather_labels(acq)
        self.hand_markers = get_hand_labels(self.labels)
        self.shoulder_markers = ["LBSH", "CLAV", "RBSH"]

        # dealing with data
        self.data = gather_points_data(acq)
        self.frames = self.data.shape[1]
        self.set_fps(fps)
        relaxed_frame = init_frame(c3d_file)
        self.init_pos = self.data[:, relaxed_frame, :]

        self.preprocessing()
        self.set_weights()

    def __str__(self):
        """
        :return: string representation of gesture
        """
        s = HumanoidBasic.__str__(self)
        self.estimate_hand_contribution()
        rhand = self.rhand_contib * 100.
        lhand = 100. - rhand
        s += "\n\t rhand x lhand: \t %.1f%% x %.1f%%" % (rhand, lhand)
        return s

    def estimate_hand_contribution(self):
        """
         Estimates how much each hand has been moving along the prev positions.
        """
        if not any(self.joint_displace):
            self.compute_displacement(mode="bothHands")
        rhand_contrib = 0.
        lhand_contrib = 0.
        for marker, deviation in self.joint_displace.iteritems():
            if marker[0] == "R":
                rhand_contrib += deviation
            elif marker[0] == "L":
                lhand_contrib += deviation
        self.rhand_contib = rhand_contrib / (rhand_contrib + lhand_contrib)

    def show_displacement(self, mode="bothHands", rotation=80, fontsize=7, add_error=True):
        """
            Plots a chart bar of joints displacements.
        :param mode: use both hand (by default) or only prime one
        :param rotation: labeled bar text rotation (Ox axis)
        :param fontsize: labeled bar text font size (Ox axis)
        :param add_error: whether to add bar error on plot or not
        """
        HumanoidBasic.plot_displacement(self, mode, rotation, fontsize, add_error)
        plt.show()

    def save_displacement(self, mode="bothHands", rotation=80, fontsize=7, add_error=False):
        """
            Saves joint displacements in png.
        :param mode: use both hand (by default) or only prime one
        :param rotation: labeled bar text rotation (Ox axis)
        :param fontsize: labeled bar text font size (Ox axis)
        :param add_error: whether to add bar error on plot or not
        """
        HumanoidBasic.plot_displacement(self, mode, rotation, fontsize, add_error)
        plt.savefig("joint_displacements.png")

    def init_3dbox(self):
        self.xmin = 0.4
        self.xmax = 0.4
        self.ymin = 0.4
        self.ymax = 0.4

    def animate(self, faster=7):
        """
         Animates 3d data.
         :param faster: how much faster animate it
        """
        HumanoidBasic.animate(self, faster)

    def save_anim(self, faster=7):
        """
         Saves animation in mp4 video file.
        :param faster: how much faster animate it
        """
        self.init_3d()
        self.faster = faster

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=int(self.frames/faster),
                                       interval=1.,     # in ms
                                       blit=True)
        mp4_file = self.name + '.mp4'
        anim.save(mp4_file, writer='ffmpeg', fps=50)
        print "Animation is saved in %s" % mp4_file


if __name__ == "__main__":
    gest = HumanoidUkr("D:\GesturesDataset\splitAll\C1_mcraw_gest1_sample0.c3d")
    gest.set_fps(24)
    print gest
    gest.animate(faster=1)