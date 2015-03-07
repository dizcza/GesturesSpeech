# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np


def plot_rrate_vs_fps():
    """
     Plot recognition rate VS fps
    """
    total_samples = 160.

    fps = 2, 3, 4, 5, 10, 15, 20, 25, 30
    misclassified = tuple([34, 10, 1] + [0] * (len(fps) - 3))
    rrate = 1. - np.array(misclassified, dtype="float") / total_samples
    plt.plot(fps, rrate, linestyle='-', marker='o', color='blue')
    plt.plot(5, 1, 'ro', ms=10, fillstyle='none')

    plt.ylim(ymax=1.02)
    plt.ylabel("recognition rate")
    plt.xlabel("fps")
    plt.title("rRate VS fps (WDTD is used)")
    plt.grid()
    plt.show()


plot_rrate_vs_fps()


def plot_ratio_vs_beta():
    zipped = [(1e-06, 14.081139305558757, 7.6406484891271473),
              (0.0001, 14.081137778713124, 7.6406477655985388),
              (0.01, 14.080985117717137, 7.6405754294798482),
              (0.1, 14.079597717387001, 7.6399180239295399),
              (1.0, 14.065763741894175, 7.6333572368619675),
              (10.0, 13.932369376871822, 7.5693745244272828),
              (100.0, 13.21349848348074, 7.1269725019718235),
              (1000.0, 13.022094467981216, 6.9309437215372967)]

    betas, ratios, ratio_std = np.array(zip(*zipped))

    within_var, within_std = np.array(zip(*[
        (0.0422084, 0.0141697),
        (0.0422084, 0.0141697),
        (0.0422083, 0.0141696),
        (0.0422072, 0.0141688),
        (0.0421965, 0.0141613),
        (0.0420915, 0.0140886),
        (0.0414313, 0.0136786),
        (0.0411712, 0.0135355)
    ]))

    between_var, between_std = np.array(zip(*[
        (0.594343, 0.253369),
        (0.594343, 0.253369),
        (0.594334, 0.253366),
        (0.594261, 0.253345),
        (0.593527, 0.253126),
        (0.586434, 0.25096),
        (0.547452, 0.233501),
        (0.536135, 0.22441)
    ]))

    # plt.errorbar(np.log(betas), ratios, stds, marker='^', ms=8)
    plt.suptitle("Choosing the best beta")
    plt.subplot(311)
    plt.plot(np.log(betas), ratios, color='grey', linestyle='-', marker='s', ms=5)
    rstd_mean = np.average(ratio_std / ratios) * 100.
    plt.ylabel("R")
    plt.legend(["discriminant ratio R\nstd=%.1f%%" % rstd_mean], loc=3, numpoints=1)
    plt.grid()

    plt.subplot(312)
    plt.plot(np.log(betas), between_var, color='blue', linestyle='-', marker='^', ms=5)
    bstd_mean = np.average(between_std / between_var) * 100.
    plt.ylabel("Db")
    plt.legend(["between variance Db\nstd=%.1f%%" % bstd_mean], loc=3, numpoints=1)
    plt.grid()

    plt.subplot(313)
    plt.plot(np.log(betas), within_var, color='green', linestyle='-', marker='o', ms=5)
    plt.xlabel("log(beta)")
    wstd_mean = np.average(within_std / within_var) * 100.
    plt.ylabel("Dw")
    plt.legend(["within variance Dw\nstd=%.1f%%" % wstd_mean], loc=3, numpoints=1)
    plt.grid()

    plt.savefig("choosing_beta.png")
    # plt.show()