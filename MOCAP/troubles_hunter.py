# coding=utf-8

from MOCAP.splitter import *
import os
import matplotlib.pyplot as plt


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
    """
     Checks for having troubles in the folder.
    :param folder: path to folder with c3d files
    """
    for c3d_file in os.listdir(folder):
        if c3d_file.endswith(".c3d"):
            try:
                fname = os.path.join(folder, c3d_file)
                gest = HumanoidUkrSplitter(fname)
                if gest.markers_total < 83:
                    print("Not enough markers in %s: \t %d < 83" % (c3d_file, gest.markers_total))

                if any(gest.corrupted):
                    corrupted_share = float(len(gest.corrupted)) / gest.frames * 100.
                    add_bar_plane(gest.corrupted, gest.frames)
                    print("%d (%.2f%%) corrupted frames in %s\n" % (len(gest.corrupted),
                                                                    corrupted_share,
                                                                    gest.name))
                    print(np.array(gest.corrupted))
                    plt.title(c3d_file)
                    plt.show()
            except:
                print("cannot describe %s" % c3d_file)
                continue


if __name__ == "__main__":
    # NOTE: splitAll folder status is OK (01.03.15)
    check_them_all(MOCAP_PATH)