# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_rrate_both_proj():
    """
     Plots connected results from both Kinect and MoCap projects.
    """
    # Kinect
    total_samples = 160.
    fps = 2, 3, 4, 5, 10, 15, 20, 25, 30
    misclassified = tuple([34, 10, 1] + [0] * (len(fps) - 3))
    rrate = 1. - np.array(misclassified, dtype="float") / total_samples
    plt.plot(fps, rrate, linestyle='--', marker='o', ms=9, color='blue')

    # MoCap
    plot_rrate_vs_fps()


def plot_rrate_vs_fps():
    """
     Plot recognition rate VS fps
    """
    total_samples = 139.

    fps = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30
    misclassified = tuple([57, 15, 10, 5, 1, 2, 1, 1] + [0] * (len(fps) - 8))
    rrate = 1. - np.array(misclassified, dtype="float") / total_samples
    plt.plot(fps, rrate, linestyle='--', marker='o', color='green')
    plt.legend(["Kinect", "MoCap"], numpoints=1, loc=0)

    # plt.plot(5, 1, 'bo', ms=12, fillstyle='none')
    plt.plot(9, 1, 'go', ms=10, fillstyle='none')

    plt.ylim(ymax=1.02)
    plt.ylabel("recognition rate")
    plt.xlabel("fps")
    plt.title("rRate VS fps (WDTD is used)")

    plt.grid()
    plt.show()


plot_rrate_both_proj()


def plot_bvar_vs_beta():
    zipped = [(1e-06, 0.17508874776345451, 0.0647448), (0.0001, 0.1750887441645072, 0.0647448), (0.01, 0.17508838448171357, 0.0647446), (0.1, 0.1750851147626187, 0.0647432), (1.0, 0.17505242793231682, 0.0647286), (10.0, 0.1747263457118941, 0.0645836), (100.0, 0.17160113631910878, 0.0632738), (1000.0, 0.15821415968124972, 0.0593719)]
    betas, bvars, stds = np.array(zip(*zipped))

    # plt.errorbar(np.log(betas), bvars, stds, marker='^', ms=8)
    plt.plot(np.log(betas), bvars, linestyle='-', marker='^', ms=8)
    plt.xlabel("log(beta)")
    rstd_mean = np.average(stds / bvars) * 100.
    plt.ylabel("Db")
    plt.legend(["between variance Db\nstd=%.1f%%" % rstd_mean], loc=3, numpoints=1)
    plt.title("Choosing the best beta")
    plt.grid()

    plt.savefig("choosing_beta.png")
    # plt.show()
