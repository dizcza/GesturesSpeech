# coding=utf-8

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import json
from basic import BasicMotion

EMOTION_PATH_PICKLES = r"D:\GesturesDataset\Emotion\pickles"

# TODO deal with camera jerking (58-1-1) --> blur


class Emotion(BasicMotion):
    def __init__(self, obj_path, fps=None):
        """
        :param obj_path: path to pickled data
        :param fps: new fps to be set
        """
        BasicMotion.__init__(self, fps=24)
        self.project = "Emotion"
        self.fname = os.path.basename(obj_path).strip(".pkl")

        # loading data from a pickle
        info = pickle.load(open(obj_path, 'rb'))
        self.data = info["data"]
        self.norm_data = None
        self.author = info["author"]
        self.emotion = info["emotion"]
        self.labels = tuple(info["labels"])
        self.frames = self.data.shape[1]
        self.name = self.emotion

        self.set_fps(fps)
        self.preprocessor()
        self.set_weights()

    def __str__(self):
        s = BasicMotion.__str__(self)
        s += "\n\t file name:\t\t %s" % self.fname
        s += "\n\t author:\t\t %s" % self.author
        return s

    def preprocessor(self):
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
        top_point = (self.data[ebr_ir,0,:] + self.data[ebr_il,0,:]) / 2.
        base_line = norm(top_point - self.data[jaw,0,:])
        self.norm_data /= base_line

        # step 3: gaussian blurring filter
        self.gaussian_filter()

        # step 4: deal eye winking
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


    def deal_with_winking(self):
        """
         A wizard to deal with eye winking.
         It's known, that a human wink duration lies within the range of [300, 500] ms.
         Taking that into account, we can find out winking frames,
         skip them and approximate the gap instead.
        """
        wink_window = int(0.5 * self.fps)
        to_be_wink_threshold = 0.05
        eup_r, edn_r, eup_l, edn_l = self.get_ids("eup_r", "edn_r", "eup_l", "edn_l")
        both_eyes = (eup_r, edn_r), (eup_l, edn_l)
        up_down_dist = []
        for eye in both_eyes:
            dX = self.norm_data[eye[0],::] - self.norm_data[eye[1],::]
            up_down_dist.append(norm(dX, axis=1))
        eyes_wink = np.sum(up_down_dist, axis=0)

        # plt.plot(eyes_wink, 'b')
        for frame in range(1, len(eyes_wink)-1):
            start = max(0, frame-wink_window//2)
            end = min(len(eyes_wink), frame+wink_window//2)
            left_bound = max(eyes_wink[start:frame])
            right_bound = max(eyes_wink[frame:end])
            if eyes_wink[frame] < eyes_wink[frame-1] and eyes_wink[frame] < eyes_wink[frame+1]:
                deep_left = (left_bound - eyes_wink[frame]) / max(eyes_wink)
                deep_right = (right_bound - eyes_wink[frame]) / max(eyes_wink)
                if deep_left > to_be_wink_threshold and deep_right > to_be_wink_threshold:
                    start = start + np.argmax(eyes_wink[start:frame])
                    end = frame + np.argmax(eyes_wink[frame:end])
                    # print(start, frame, end, deep_left, deep_right)
                    # y = np.linspace(left_bound, right_bound, end - start)
                    # plt.plot(np.arange(start, end, 1), y, 'go')
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
                                      self.norm_data[markerID, frame-1, :])
                weights = np.linspace(0, 1, end - frame)
                accumulated_offset = np.sum(norm(dX_next, axis=1) * weights)
                _dx = np.subtract(self.norm_data[markerID, frame, :],
                                  self.norm_data[markerID, frame-1, :])
                blur_factor = 1. - np.exp(- accumulated_offset / (2 * sigmas))
                _dx *= blur_factor
                self.norm_data[markerID,frame,:] = self.norm_data[markerID,frame-1,:] + _dx
        # self.data = self.norm_data

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

    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        self.scat.set_offsets(self.data[:, frame, :])
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

    def define_plot_style(self):
        """
         Setting bar char plot style.
        """
        self.rotation = 50
        self.fontsize = 11
        self.add_error = False

    def animate(self):
        """
         Animates 2d data.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scat = plt.scatter(self.data[:, 0, 0], self.data[:, 0, 1])
        self.ax.grid()
        self.ax.set_title("%s: %s" % (self.emotion, self.fname))

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=self.frames,
                                       interval=1e3/self.fps,     # in ms
                                       blit=True)
        try:
            plt.show(self.fig)
            # plt.draw()
        except AttributeError:
            pass


def test_nan_weights():
    """
     Tests each sample for having nan weights.
    """
    for log_c3d in os.listdir(EMOTION_PATH_PICKLES):
        if log_c3d.endswith(".pkl"):
            log_path = os.path.join(EMOTION_PATH_PICKLES, log_c3d)
            gest = Emotion(log_path)
            w = gest.get_weights()
            assert not np.isnan(w).any(), "nan weights in %s" % log_c3d


def show_all_emotions():
    """
     Animates Emotion instances.
    """
    for i, pkl_log in enumerate(os.listdir(EMOTION_PATH_PICKLES)):
        if pkl_log.endswith(".pkl"):
            pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
            em = Emotion(pkl_path)
            # print(em.fname, em.emotion)
            # em.animate()
            if em.emotion != "undefined":
                # em.show_displacements(None)
                if em.emotion == u"улыбка":
                    em.animate()


if __name__ == "__main__":
    test_nan_weights()
    em = Emotion(r"D:\GesturesDataset\Emotion\pickles\46-3-2.pkl")
    em.data = em.norm_data
    # em.show_displacements(None)
    # em.deal_with_winking()
    # plt.show()
    # print(em)
    em.compute_weights(None, None)
    em.animate()
