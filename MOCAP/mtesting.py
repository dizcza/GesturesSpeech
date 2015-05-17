# coding=utf-8

from MOCAP.mreader import HumanoidUkr
from tools.instruments import Testing, Training


def plot_error_vs_fps(mode="bothHands", beta=None):
    """
     Plots the out-of-sample error VS fps, w.r.t. mode and beta params.
    """
    Testing(HumanoidUkr).error_vs_fps(mode, beta)


def run_the_worst_comparison(mode="bothHands", beta=None, fps=None):
    """
     Runs the worst comparison scenario after computing weights with given params.
    :param mode: mode for tracking only hand markers or all of them
    :param beta: (float), needed for computing weights
    :param fps: frames per sec
    """
    # fps >= 8 gives 100 % accuracy independently of
    # whether wdtw cost normalization is ON or OFF.
    Training(HumanoidUkr).compute_weights(mode, beta, fps)
    Testing(HumanoidUkr).the_worst_comparison(fps)


if __name__ == "__main__":
    run_the_worst_comparison()
