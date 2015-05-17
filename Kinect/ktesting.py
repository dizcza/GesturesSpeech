# coding=utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import Testing, Training


def plot_error_vs_fps(mode="bothHands", beta=None):
    """
     Plots the out-of-sample error VS fps, w.r.t. mode and beta params.
    """
    Testing(HumanoidKinect).error_vs_fps(mode, beta)


def run_the_worst_comparison(mode="bothHands", beta=None, fps=None):
    """
     Runs the worst comparison scenario after computing weights with given params.
    :param mode: mode for tracking only hand markers or all of them
    :param beta: (float), needed for computing weights
    :param fps: frames per sec
    """
    # fps >= 6 gives 100 % accuracy with wdtw cost normalization
    # and the same for fps >= 12 without cost normalization
    Training(HumanoidKinect).compute_weights(mode, beta, fps)
    Testing(HumanoidKinect).the_worst_comparison(fps)


if __name__ == "__main__":
    # plot_error_vs_fps()
    run_the_worst_comparison()
