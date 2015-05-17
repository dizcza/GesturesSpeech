# coding=utf-8

############################################################
# This file maintains tools for multiple projects testing. #
############################################################

from tools.instruments import Testing
from Kinect.kreader import HumanoidKinect
from MOCAP.mreader import HumanoidUkr

import matplotlib.pyplot as plt


def plot_error_vs_fps():
    """
     Plots the out-of-sample error VS fps for both projects:
     Kinect and MOCAP (in comparison)
    """
    fps_range = range(2, 11, 1)
    mark_size = 10
    for class_name in (HumanoidKinect, HumanoidUkr):
        proj = Testing(class_name)
        test_errors = []
        for fps in fps_range:
            inf, sup, tot = proj.the_worst_comparison(fps, verbose=False)
            Etest = float(sup) / tot
            test_errors.append(Etest)
        plt.plot(fps_range, test_errors, 'o--', ms=mark_size)
        mark_size -= 3
    plt.ylabel("Etest")
    plt.xlabel("FPS")
    plt.ylim(ymin=-0.01)
    plt.xlim(1, 14)
    plt.title("out-of-sample error VS fps")
    plt.grid()
    plt.legend(["Kinect", "MoCap"], numpoints=1)
    plt.savefig("../png/error_vs_fps.png")
    plt.show()


if __name__ == "__main__":
    plot_error_vs_fps()