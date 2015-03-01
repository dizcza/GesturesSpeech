# coding=utf-8

from splitter import *
import os


def add_bar_plane(corrupted_frames, frames_num, init_fr=0):
    """
     Adds bar plane for check_them_all() func.
    :param corrupted_frames: list of corrupted frames in the acq
    :param frames_num: number of frames in the gesture
    :param init_fr: =0 by default
    """
    fig, ax = plt.subplots()
    missed = init_fr + np.array(corrupted_frames)
    heights = np.ones(len(corrupted_frames))
    ax.bar(missed, heights, width=0.8, color='black')
    ax.set_xlim([init_fr, frames_num + init_fr])


def check_them_all(folder):
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            try:
                gest = HumanoidUkr(folder + c3d_file)
                if gest.markers_total < 83:
                    print "Not enough markers in %s: \t %d < 83" % (gest.filename, gest.markers_total)

                if any(gest.missed):
                    corrupted_share = float(len(gest.missed)) / gest.frames * 100.
                    add_bar_plane(gest.missed, gest.frames)
                    print "%d (%.2f%%) corrupted frames in %s\n" % (len(gest.missed),
                                                                    corrupted_share,
                                                                    gest.filename)
                    print np.array(gest.missed)
                    plt.title(c3d_file)
                    plt.show()
            except:
                print "cannot describe %s" % c3d_file
                continue


def plot_them_all(folder):
    """
     Plots the main info: deviation per frame.
    :param folder: path to folder with .c3d-files
    """
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            try:
                gest = HumanoidUkr(folder + c3d_file)
                print "file: %s; \t frames: %d" % (c3d_file, gest.frames)
                gest.plot_relaxed_indices()
            except:
                print "cannot describe %s" % c3d_file
                continue