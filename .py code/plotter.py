# coding=utf-8

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from math_kernel import *


def plot_deriv(_decr):
    """
     Plots the derivatives (by frame, not time) of data offsets.
    :param _decr: description dic
    """
    data = moving_average(_decr["data"])
    deriv = diff(data, step=2)
    deriv = moving_average(deriv, wsize=7)

    dev = np.average(sqrt(np.sum(deriv ** 2, axis=2)), axis=0)
    dev = moving_average_simple(dev, wsize=7)
    plt.plot(range(len(dev)), dev)
    plt.plot(range(len(dev)), np.zeros(len(dev)), 'r--', lw=3)
    plt.show()






def animate(i):
    """
    :param i: frame
    :return: points (markers) pos in the current frame
    """
    for pt, xi in zip(pts, data):
        x, y, z = xi[:i].T
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])
    return pts


def display_animation(_dscr, speed_rate=1., frames_range=None):
    """
    :param _dscr: description dic
    :param speed_rate: animation speed (1.0 is by default)
    :param frames_range: frames range to animate markers (if it's set)
    """
    global pts, data

    chunks_num = int(speed_rate * _dscr["freq"] / 24)
    if frames_range:
        frames = np.arange(frames_range[0], frames_range[1], chunks_num)
    else:
        frames = np.arange(0, _dscr["frames"]-1, chunks_num)

    data = _dscr["data"][:, frames, :]
    # data = data[_dscr["feet"]["ids"], :, :]

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.view_init(10, 120)
    pts = []
    for i in range(data.shape[0]):
        pts += ax.plot([], [], [], 'o')

    ax.set_xlim3d(-700, 700)
    ax.set_ylim3d(-1000, -100)
    ax.set_zlim3d(0, 1500)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    anim = animation.FuncAnimation(fig,
                                   func=animate,
                                   frames=data.shape[1],
                                   interval=1./_dscr["freq"],
                                   blit=True)
    plt.show()