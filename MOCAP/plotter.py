# coding=utf-8

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from math_kernel import *
import time


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


def animate(i, data, pts):
    """
    :param i: frame
    :return: points (markers) pos in the current frame
    """
    global prev_time
    # print (time.time() - prev_time) * 1e3
    prev_time = time.time()

    for marker in range(data.shape[0]):
        x, y, z = data[marker, i, :]
        pts[marker].set_data([x], [y])
        pts[marker].set_3d_properties([z])

    # for pt, xi in zip(pts, data):
    #     x, y, z = xi[:i].T
    #     pt.set_data(x[-1:], y[-1:])
    #     pt.set_3d_properties(z[-1:])
    return []


def display_animation(_dscr, speed_rate=1., frames_range=None, save=False):
    """
    :param _dscr: description dic
    :param speed_rate: animation speed (1.0 is by default)
    :param frames_range: frames range to animate markers (if it's set)
    """
    if save:
        frames_step = 2
    else:
        frames_step = int(speed_rate * 5)

    if frames_range:
        frames = np.arange(frames_range[0], frames_range[1], frames_step)
    else:
        frames = np.arange(0, _dscr["frames"]-1, frames_step)

    # data = _dscr["data"]
    data = _dscr["data"][:, frames, :]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.view_init(15, 110)
    pts = []
    for i in range(data.shape[0]):
        # sets 83 empty plots for 83 markers
        pts += ax.plot([], [], [], 'o', color='black', markersize=4)

    # ax.set_xlim3d(-700, 700)
    # ax.set_ylim3d(-1000, -500)
    # ax.set_zlim3d(0, 1500)
    ax.set_axis_off()

    ax.set_xlim3d([np.nanmin(_dscr["data"][:, :, 0])+100, np.nanmax(_dscr["data"][:, :, 0])-100])
    ax.set_ylim3d([np.nanmin(_dscr["data"][:, :, 1])-400, np.nanmax(data[:, :, 1])+400])
    ax.set_zlim3d([np.nanmin(_dscr["data"][:, :, 2]), np.nanmax(_dscr["data"][:, :, 2])])

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    global prev_time
    prev_time = time.time()
    anim = animation.FuncAnimation(fig,
                                   func=animate,
                                   fargs=(data, pts),
                                   frames=data.shape[1],
                                   interval=1000./_dscr["freq"],
                                   blit=True)

    if save:
        mp4_file = _dscr['filename'].split('/')[-1][:-4] + '.mp4'
        anim.save(mp4_file, writer='ffmpeg', fps=50)
        print "Saved in %s" % mp4_file
    else:
        plt.show()
        pass