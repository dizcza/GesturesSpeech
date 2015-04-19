# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from MOCAP.math_kernel import moving_average
import pickle
from Emotion.emotion import Emotion

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)


EMOTION_PATH_PICKLES = r"D:\GesturesDataset\Emotion\pickles"


def _update_plot(i, fig, scat, data):
    """
    :param i: frame ID
    :param fig:
    :param scat: 2d array
    :param 2darray data: (#markers, frames, 2)
    """
    scat.set_offsets(data[:, i, :])
    return []


def show_all_data():
    """
     Animates pickled data.
    """
    for i, pkl_log in enumerate(os.listdir(EMOTION_PATH_PICKLES)):
        fig = plt.figure()
        fname = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
        info = pickle.load(open(fname, 'rb'))
        data = info["data"]
        # data = moving_average(data, wsize=3)

        ax = fig.add_subplot(111)
        ax.grid(True, linestyle='-', color='0.75')
        plt.title(pkl_log.strip(".pkl"))

        scat = plt.scatter(data[:, 0, 0], data[:, 0, 1])
        anim = animation.FuncAnimation(fig,
                                       func=_update_plot,
                                       frames=data.shape[1],
                                       fargs=(fig, scat, data),
                                       interval = 10,
                                       blit=True)
        try:
            plt.show()
        except AttributeError:
            continue


def show_all_emotions():
    """
     Animates Emotion instances.
    """
    for i, pkl_log in enumerate(os.listdir(EMOTION_PATH_PICKLES)):
        if pkl_log.endswith(".pkl"):
            pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
            em = Emotion(pkl_path)
            # em.animate()


if __name__ == "__main__":
    show_all_emotions()