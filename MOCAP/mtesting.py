# coding=utf-8

from MOCAP.mreader import HumanoidUkr
from tools.instruments import Testing, Training


def plot_error_vs_fps(mode="bothHands", beta=1e2):
    """
     Plots the out-of-sample error VS fps, w.r.t. mode and beta params.
    """
    Testing(HumanoidUkr).error_vs_fps(mode, beta)


def run_the_worst_comparison(mode="bothHands", beta=1e2, fps=None, weighted=True):
    """
     Runs the worst and the best comparison scenarios after computing weights with given params.
    :param mode: mode for tracking only hand markers or all of them
    :param beta: (float), needed for computing weights;
                 actually, for MoCap project, you can set whatever
                 you want 'beta' and all test will be passed successfully;
    :param fps: frames per sec
                'fps' >= 8 gives 100 % accuracy independently of
                whether WDTW cost normalization is ON or OFF.
    :param weighted: use weighted FastDTW modification or just FastDTW
    """
    Training(HumanoidUkr).compute_weights(mode, beta, fps)
    Testing(HumanoidUkr).the_worst_comparison(fps, weighted=weighted)


if __name__ == "__main__":
    run_the_worst_comparison()
