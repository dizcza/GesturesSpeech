# coding=utf-8

import os

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from tools.humanoid import HumanoidBasic
import MOCAP.labelling as labelling
import MOCAP.helper as helper


try:
    import btk
except ImportError:
    import MOCAP.btk_fake as btk

MOCAP_PATH = r"D:\GesturesDataset\MoCap\splitAll"


def gather_points_data(acq):
    """
    :param acq: acquisition from .c3d-file
    :return: (#markers, #frames, 3) ndarray of 3d points data
    """
    if hasattr(acq, "fake") and acq.fake is True:
        # returning data read by c3d module
        return acq.GetData()
    else:
        # returning data read by btk module
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
        """
         Reads Motion Capture C3D file.
        :param c3d_file: path to file.c3d
        :param fps: new fps to be set
        """
        HumanoidBasic.__init__(self, fps)
        self.project = "MoCap"

        # setting unique gesture name
        self.name = parse_fname(c3d_file)

        # setting up BTK reader to gather acquisition
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(c3d_file)
        reader.Update()
        acq = reader.GetOutput()

        # default fps (should be 120)
        self.fps = acq.GetPointFrequency()

        # dealing with markers
        self.labels = labelling.gather_labels(acq)
        self.hand_markers = labelling.get_hand_labels(self.labels)
        self.shoulder_markers = "LBSH", "CLAV", "RBSH"

        # dealing with data
        self.data = gather_points_data(acq)
        self.frames = self.data.shape[1]
        relaxed_frame = helper.init_frame(c3d_file)
        self.init_pos = self.data[:, relaxed_frame, :]

        self.set_fps(fps)
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

    def define_plot_style(self):
        """
         Setting bar char plot style.
        """
        self.rotation = 80
        self.fontsize = 7
        self.add_error = False

    def init_3dbox(self):
        self.xmin = 0.4
        self.xmax = 0.4
        self.ymin = 0.4
        self.ymax = 0.4

    def estimate_hand_contribution(self):
        """
         Estimates how much each hand has been moving along the prev positions.
        """
        if not any(self.joint_displace):
            self.compute_displacement(mode="bothHands")
        rhand_contrib = 0.
        lhand_contrib = 0.
        for marker, deviation in self.joint_displace.items():
            if marker[0] == "R":
                rhand_contrib += deviation
            elif marker[0] == "L":
                lhand_contrib += deviation
        self.rhand_contib = rhand_contrib / (rhand_contrib + lhand_contrib)

    def save_displacement(self, mode="bothHands"):
        """
            Saves joint displacements in png.
        :param mode: use both hand (by default) or only prime one
        """
        HumanoidBasic.plot_displacement(self, mode)
        plt.savefig("joint_displacements.png")

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
        print("Animation is saved in %s" % mp4_file)


def test_nan_weights():
    """
     Tests each sample for having nan weights.
    """
    for log_c3d in os.listdir(MOCAP_PATH):
        if log_c3d.endswith(".c3d"):
            log_path = os.path.join(MOCAP_PATH, log_c3d)
            gest = HumanoidUkr(log_path)
            w = gest.get_weights()
            assert not np.isnan(w).any(), "nan weights in %s" % log_c3d


if __name__ == "__main__":
    test_nan_weights()
    gest = HumanoidUkr(r"D:\GesturesDataset\MoCap\splitAll\Training\C1_mcraw_gest0\C1_mcraw_gest0_sample0.c3d")
    print(gest)
    # gest.show_displacements("bothHands")
    # gest.animate()
