# coding=utf-8

import btk
from humanoid import HumanoidBasic
import matplotlib.animation as animation
from labelling import *


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


class HumanoidUkr(HumanoidBasic):
    """
     Creates an instance of Ukrainian Motion Capture gesture, saved in .c3d-format.
    """

    def __init__(self, c3d_file):
        HumanoidBasic.__init__(self)
        # TODO think about self.name: parse xlsx info
        self.project = "MOCAP"
        self.name = c3d_file.split('\\')[-1][:-12]
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(c3d_file)
        reader.Update()

        # set info
        self.acq = reader.GetOutput()
        self.frames = self.acq.GetPointFrameNumber()
        self.fps = self.acq.GetPointFrequency()

        # dealing with markers
        self.labels = gather_labels(self.acq)
        self.markers_total = len(self.labels)
        self.hand_markers = get_hand_labels()
        self.shoulder_markers = ["LBSH", "CLAV", "RBSH"]

        # dealing with data
        self.data = gather_points_data(self.acq)
        relaxed_frame = init_frame(c3d_file)
        self.init_pos = self.data[:, relaxed_frame, :]

        # dealing with animation
        self.miny = 400
        self.maxy = 400

        self.preprocessing()
        self.set_weights()

        # dealing with mm --> m
        self.shoulder_length /= 1e3
        self.std /= 1e3


    def estimate_hand_contribution(self):
        if not any(self.joint_displace):
            self.compute_displacement(self)
        rhand_contrib = 0.
        lhand_contrib = 0.
        for marker, deviation in self.joint_displace.iteritems():
            if marker[0] == "R":
                rhand_contrib += deviation
            elif marker[0] == "L":
                lhand_contrib += deviation
        self.rhand_contib = rhand_contrib / (rhand_contrib + lhand_contrib)


    def __str__(self):
        s = HumanoidBasic.__str__(self)
        self.estimate_hand_contribution()
        rhand = self.rhand_contib * 100.
        lhand = 100. - rhand
        s += "\n\t rhand x lhand: \t %.1f%% x %.1f%%" % (rhand, lhand)
        return s


    def show_displacement(self, mode=None, rotation=0, fontsize=12):
        """
         Plots a chart bar of joints displacements.
        """
        HumanoidBasic.show_displacement(self, rotation=80, fontsize=7)


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
                                       interval=1.,
                                       blit=True)
        mp4_file = self.name + '.mp4'
        anim.save(mp4_file, writer='ffmpeg', fps=50)
        print "Saved in %s" % mp4_file


if __name__ == "__main__":
    gest = HumanoidUkr("D:\GesturesDataset\splitAll\C1_mcraw_gest0_sample0.c3d")
    # gest.animate()
    print gest
    gest.show_displacement()
