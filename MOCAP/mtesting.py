# coding=utf-8

from MOCAP.mreader import HumanoidUkr
from tools.instruments import Testing, Training


def plot_error_vs_fps(mode="bothHands", beta=None):
    """
     Plots the out-of-sample error VS fps, w.r.t. mode and beta params.
    """
    Testing(HumanoidUkr).error_vs_fps(mode, beta)


def run_the_worst_comparison(mode="bothHands", beta=None, fps=None, weighted=True):
    """
     Runs the worst and the best comparison scenarios after computing weights
     with the given params.
    :param mode: mode for tracking only hand markers ("bothHands")
                 or all of them (None)
    :param beta: (float), needed for computing weights;
                 for MoCap project you can set whatever you want
                 positive 'beta' and all test will be passed successfully;
                 set it to a very large number (e.g., 1e6) to model
                    unweighted (the same as weighted=False) scenario;
    :param fps: frames per sec
                'fps' >= 8 gives 100 % accuracy.
    :param weighted: use weighted FastDTW modification or just FastDTW
                     unweighted scenario doesn't require computing the weights,
                      cause all weights are set to be 1.
    """
    Training(HumanoidUkr).compute_weights(mode, beta, fps)
    Testing(HumanoidUkr).the_worst_comparison(fps, weighted=weighted)


if __name__ == "__main__":
    run_the_worst_comparison()
