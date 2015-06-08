# coding=utf-8

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from Emotion.em_reader import Emotion, EMOTION_PATH
from tools.kalman import kalman_1d, kalman_filter


def estimate_sensory_noise():
    """
    :return: deviation of sensory noise in silent samples
    """
    chillout_files = "26-1-1", "26-1-2", "37-2-1", "50-3-2", "54-5-2", "57-1-2"
    eyes_markers = {"eup_r", "edn_r", "eup_l", "edn_l"}
    stds = np.empty((0, 2), dtype=float)
    for fname in chillout_files:
        fpath = os.path.join(EMOTION_PATH, fname + ".pkl")
        em = Emotion(fpath)
        labels = set(em.labels)
        no_eyes = labels.difference(eyes_markers)
        ids = em.get_ids(*no_eyes)
        sigmas = np.std(em.data[ids, ::], axis=1)
        stds = np.append(stds, sigmas, axis=0)
    sigma_xy = norm(stds, axis=0) / len(chillout_files)
    sigma_sensor = np.average(sigma_xy)
    return sigma_sensor


def demo_kalman():
    """
     Kalman filter visualizer.
    """
    smile_folder = os.path.join(EMOTION_PATH, "Training", "smile")
    smile_file_name = os.listdir(smile_folder)[0]
    em_path = os.path.join(smile_folder, smile_file_name)
    em = Emotion(em_path)
    for markerID, marker in enumerate(em.labels):
        plt.ylabel("Xs")
        for dim in range(2):
            plt.subplot(2, 1, dim + 1)
            x_real = em.data[markerID, :, dim]
            x_opt = kalman_1d(x_real)
            plt.plot(x_real, linewidth=2)
            plt.plot(x_opt, linewidth=2)
            plt.legend(["noisy (real)", "kalman"], loc=2)
        plt.ylabel("Ys")
        plt.xlabel("frame")
        plt.suptitle("marker: %s" % marker)
        plt.show()


if __name__ == "__main__":
    demo_kalman()
