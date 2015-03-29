# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MOCAP.math_kernel import moving_average
import pickle


def _update_plot(i, fig, scat, data):
    """
    :param i: frame ID
    :param fig:
    :param scat: 2d array
    :param 2darray data: (#markers, frames, 2)
    """
    scat.set_offsets(data[:, i, :])
    return []


def one_step_show():
    """
     Animation.
    """
    dir_path = "D:\GesturesDataset\Emotion\\txt\data"
    for i, log in enumerate(os.listdir(dir_path)[10:]):
        fig = plt.figure()
        print(log)
        fname = os.path.join(dir_path, log)
        info = pickle.load(open(fname, 'rb'))
        data = info["data"]
        data = moving_average(data, wsize=3)

        ax = fig.add_subplot(111)
        ax.grid(True, linestyle = '-', color = '0.75')
        plt.title(log.strip(".pkl"))

        scat = plt.scatter(data[:, 0, 0], data[:, 0, 1])
        anim = animation.FuncAnimation(fig,
                                       func=_update_plot,
                                       frames=data.shape[1],
                                       fargs=(fig, scat, data),
                                       interval = 10,
                                       blit=True)
        try:
            plt.show()
        except:
            continue


if __name__ == "__main__":
    one_step_show()