# coding=utf-8

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from copy import deepcopy

from Emotion.emotion import Emotion, EMOTION_PATH_PICKLES
from tools.kalman import kalman_1d, kalman_filter


def estimate_sensory_noise():
    chillout_files = "26-1-1", "26-1-2", "37-2-1", "50-3-2", "54-5-2", "57-1-2"
    eyes_markers = {"eup_r", "edn_r", "eup_l", "edn_l"}
    stds = np.empty((0, 2), dtype=float)
    for fname in chillout_files:
        fpath = os.path.join(EMOTION_PATH_PICKLES, fname + ".pkl")
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
    em = Emotion(r"D:\GesturesDataset\Emotion\pickles\31-2-2.pkl")
    for markerID, marker in enumerate(em.labels):
        plt.subplot(211)
        x_real = em.data[markerID, :, 0]
        x_real = x_real[~np.isnan(x_real)]
        x_real_opt = kalman_1d(x_real, 0.15)
        # x_real_opt = kalman_1d(x_real_opt)
        plt.plot(x_real, linewidth=2)
        plt.plot(x_real_opt, linewidth=2)
        plt.ylabel("Xs")
        # plt.legend(["noisy (real)", "kalman", "mov aver"])

        plt.subplot(212)
        x_real = em.data[markerID, :, 1]
        x_real = x_real[~np.isnan(x_real)]
        x_real_opt = kalman_1d(x_real, 0.3)
        plt.plot(x_real, linewidth=2)
        plt.plot(x_real_opt, linewidth=2)
        plt.ylabel("Ys")
        plt.xlabel("frame")
        plt.legend(["noisy (real)", "kalman", "mov aver"], loc=2)

        plt.suptitle("marker: %s" % marker)
        plt.show()



if __name__ == "__main__":
    demo_kalman()
    # sigma_sensor = estimate_sensory_noise()
    # print(sigma_sensor)
