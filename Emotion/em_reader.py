# coding=utf-8

import os
import pickle

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from tools.basic import BasicMotion
from tools.kalman import kalman_filter

# path to Emotion project data
EMOTION_PATH = os.path.join(os.path.dirname(__file__), "_data")


class Emotion(BasicMotion):
    def __init__(self, pkl_path, fps=None):
        """
        :param pkl_path: path to pickled data
        :param fps: new fps to be set
        """
        BasicMotion.__init__(self, fps=24)
        self.project = "Emotion"
        self.fname = os.path.basename(pkl_path).strip(".pkl")

        # loading data from a pickle
        info = pickle.load(open(pkl_path, 'rb'))
        self.data = info["data"]
        self.norm_data = None
        self.author = info["author"]
        self.emotion = info["emotion"]
        self.labels = tuple(info["labels"])
        self.frames = self.data.shape[1]
        self.name = self.emotion
        self.slope = 0

        self.set_fps(fps)
        self.preprocessor()
        self.set_weights()

    def __str__(self):
        s = BasicMotion.__str__(self)
        s += "\n\t slope:\t\t\t %.3g degrees" % self.slope
        s += "\n\t file name:\t\t %s" % self.fname
        s += "\n\t author:\t\t %s" % self.author
        return s

    def preprocessor(self):
        """
         5-steps pre-processor.
         NOTE. All Emotion samples should begin with relaxed state.
        """
        # TODO deal with camera jerking
        # step 0: dealing with first frame bug
        self.data = self.data[:, 1:, :]
        self.frames -= 1
        self.norm_data = self.data.copy()

        # step 1: subtract nose pos of the first frame
        nose, jaw, ebr_ir, ebr_il = self.get_ids("p0", "jaw", "ebr_ir", "ebr_il")
        first_frame = 0
        while np.isnan(self.data[nose, first_frame, :]).any():
            first_frame += 1
        first_frame = min(first_frame, self.data.shape[1] - 1)
        self.norm_data -= self.data[nose, first_frame, :]

        # step 2: divide data by base line length
        top_point = np.average(self.data[[ebr_ir, ebr_il], 0, :], axis=0)
        base_line = norm(top_point - self.data[jaw, 0, :])
        self.norm_data /= base_line

        # step 3: slope aligning
        self.slope_align()

        # step 4: kalman filter
        self.norm_data = kalman_filter(self.norm_data)

        # step 5: deal with eyes winking
        self.deal_with_winking()

    def define_moving_markers(self, mode):
        """
        :param mode: defines moving markers
        """
        if mode == "no_eyes":
            self.moving_markers = list(self.labels)
            for eye_marker in ("eup_r", "edn_r", "eup_l", "edn_l"):
                self.moving_markers.remove(eye_marker)
        else:
            BasicMotion.define_moving_markers(self, mode)

    def slope_align(self):
        """
         Aligns slope of the whole picture of markers.
         The resulted picture is vertically aligned.
        """
        left_marks = "eup_l", "edn_l", "lil", "wl", "chl", "ebr_ol", "ebr_il"
        right_marks = [mark[:-1] + "r" for mark in left_marks]
        left_ids = self.get_ids(*left_marks)
        right_ids = self.get_ids(*right_marks)
        angl = 0.
        for lid, rid in zip(left_ids, right_ids):
            dx, dy = self.norm_data[lid, 0, :] - self.norm_data[rid, 0, :]
            angl += np.arctan2(dy, dx)
        angl /= len(left_marks)
        self.slope = angl * 180 / np.pi
        rot_matrix = np.array([[np.cos(angl), -np.sin(angl)],
                               [np.sin(angl), np.cos(angl)]])
        self.norm_data = self.norm_data.dot(rot_matrix)

    def deal_with_winking(self):
        """
         A wizard to deal with eye winking.
         It's known, that a human wink duration lies within the range of [300, 600] ms.
         Taking that into account, we can find out winking frames,
         skip them and approximate the gap instead.
         Uncomment plotting stuff to see the approximation result.
        """
        wink_window = int(0.6 * self.fps)

        # make sure wink window is big enough
        if wink_window < 2: return

        to_be_wink_threshold = 0.05
        eup_r, edn_r, eup_l, edn_l = self.get_ids("eup_r", "edn_r", "eup_l", "edn_l")
        both_eyes = (eup_r, edn_r), (eup_l, edn_l)
        up_down_dist = []
        for eye in both_eyes:
            eye_vector = self.norm_data[eye[0], ::] - self.norm_data[eye[1], ::]
            up_down_dist.append(norm(eye_vector, axis=1))
        eyes_wink = np.sum(up_down_dist, axis=0)

        for frame in range(1, len(eyes_wink) - 1):
            start = max(0, frame - wink_window // 2)
            end = min(len(eyes_wink), frame + wink_window // 2)
            left_bound = max(eyes_wink[start:frame])
            right_bound = max(eyes_wink[frame:end])
            if eyes_wink[frame] < eyes_wink[frame - 1] and eyes_wink[frame] < eyes_wink[frame + 1]:
                deep_left = (left_bound - eyes_wink[frame]) / max(eyes_wink)
                deep_right = (right_bound - eyes_wink[frame]) / max(eyes_wink)
                if deep_left > to_be_wink_threshold and deep_right > to_be_wink_threshold:
                    start = start + np.argmax(eyes_wink[start:frame])
                    end = frame + np.argmax(eyes_wink[frame:end])
                    for eye in both_eyes:
                        for eye_marker in eye:
                            for dim in range(self.norm_data.shape[2]):
                                x_begin = self.norm_data[eye_marker, start, dim]
                                x_end = self.norm_data[eye_marker, end, dim]
                                line = np.linspace(x_begin, x_end, end - start)
                                self.norm_data[eye_marker, start:end, dim] = line

    def gaussian_filter(self):
        """
         Applies gaussian smoothing filter (also called low-pass filter)
         to the sequence of XY for each marker independently.
         (NOT USED ANYMORE)
        """
        track_next = 7
        frames_total = self.norm_data.shape[1]

        # accumulates total sigma_x and sigma_y
        sigmas = np.zeros(2)
        for markerID in range(self.norm_data.shape[0]):
            # its shape == (frames-1, dim)
            frames_xyz_delta = np.subtract(self.norm_data[markerID, 1:, :],
                                           self.norm_data[markerID, :-1, :])
            # dealing with nan
            frames_xyz_delta = frames_xyz_delta[~np.isnan(frames_xyz_delta).any(axis=1)]
            if frames_xyz_delta.shape[0] > 0:
                sigmas += np.std(frames_xyz_delta, axis=0)

        for markerID in range(self.norm_data.shape[0]):
            for frame in range(1, frames_total):
                end = min(frame + track_next, frames_total)
                dX_next = np.subtract(self.norm_data[markerID, frame:end, :],
                                      self.norm_data[markerID, frame - 1, :])
                weights = np.linspace(0, 1, end - frame)
                accumulated_offset = np.sum(norm(dX_next, axis=1) * weights)
                _dx = np.subtract(self.norm_data[markerID, frame, :],
                                  self.norm_data[markerID, frame - 1, :])
                blur_factor = 1. - np.exp(- accumulated_offset / (2 * sigmas))
                _dx *= blur_factor
                self.norm_data[markerID, frame, :] = self.norm_data[markerID, frame - 1, :] + _dx

    def define_plot_style(self):
        """
         Setting bar char plot style.
        """
        self.rotation = 50
        self.fontsize = 11
        self.add_error = False

    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        self.scat.set_offsets(self.data[:, frame, :])
        return []

    def init_animation(self):
        """
         Animates 2d data.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scat = plt.scatter(self.data[:, 0, 0], self.data[:, 0, 1])


def test_nan_weights():
    """
     Tests each sample for having no nan weights.
    """
    for log_pkl in os.listdir(EMOTION_PATH):
        if log_pkl.endswith(".pkl"):
            log_path = os.path.join(EMOTION_PATH, log_pkl)
            gest = Emotion(log_path)
            w = gest.get_weights()
            assert not np.isnan(w).any(), "nan weights in %s" % log_pkl


def show_all_emotions():
    """
     Animates Emotion instances.
    """
    for i, pkl_log in enumerate(os.listdir(EMOTION_PATH)):
        if pkl_log.endswith(".pkl"):
            pkl_path = os.path.join(EMOTION_PATH, pkl_log)
            em = Emotion(pkl_path)
            if em.emotion != "undefined":
                em.data = em.norm_data
                em.animate()


def demo_run():
    """
     Emotion project demo.
    """
    smile_folder = os.path.join(EMOTION_PATH, "Training", "smile")
    smile_file_name = os.listdir(smile_folder)[0]
    em_path = os.path.join(smile_folder, smile_file_name)
    assert os.path.exists(em_path), "Unable to find the %s" % em_path
    em = Emotion(em_path)
    print(em)
    em.show_displacements(None, ("liup", "lidn", "jaw", "lir", "lil"))
    em.animate()


if __name__ == "__main__":
    demo_run()
