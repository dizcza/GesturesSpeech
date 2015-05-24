# coding=utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import Testing, Training


def plot_error_vs_fps(mode="bothHands", beta=1e2):
    """
     Plots the out-of-sample error VS fps, w.r.t. mode and beta params.
    """
    Testing(HumanoidKinect).error_vs_fps(mode, beta)


def run_the_worst_comparison(mode="bothHands", beta=None, fps=None, weighted=True):
    """
     Runs the worst and the best comparison scenarios after computing weights with given params.
    :param mode: mode for tracking only hand markers or all of them
    :param beta: (float), defines weights activity;
                  the best 'beta' value is around zero (or None);
                  set it to None to model when 'beta' vanishes;
                  set it to a very large number (e.g., 1e6) to model
                    unweighted (the same as weighted=False) scenario;
                  warn: setting 'beta' to 0 causes an error.
    :param fps: frames per sec
                'fps' >= 6 gives 100 % accuracy with WDTW cost normalization
                and the same accuracy for 'fps' >= 12 without cost normalization
    :param weighted: use weighted FastDTW modification or just FastDTW
    """
    Training(HumanoidKinect).compute_weights(mode, beta, fps)
    Testing(HumanoidKinect).the_worst_comparison(fps, weighted=weighted)


if __name__ == "__main__":
    # plot_error_vs_fps()
    run_the_worst_comparison()
