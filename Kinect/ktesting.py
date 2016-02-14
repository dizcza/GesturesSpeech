# coding=utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import InstrumentCollector, Testing, Training


def plot_error_vs_fps(mode="bothHands", beta=None):
    """
     Plots the out-of-sample error VS fps, w.r.t. mode and beta params.
    """
    Testing(HumanoidKinect).error_vs_fps(mode, beta)


def run_the_worst_comparison(mode="bothHands", beta=None, fps=None, weighted=True):
    """
     Runs the worst and the best comparison scenarios after computing weights
     with the given params.
    :param mode: mode for tracking only hand markers ("bothHands")
                 or all of them (None)
    :param beta: (float), defines weights activity;
                  the best 'beta' value is around zero (or None);
                  set it to None to model when 'beta' vanishes;
                  set it to a very large number (e.g., 1e6) to model
                    unweighted (the same as weighted=False) scenario;
    :param fps: frames per sec
                'fps' >= 6 gives 100 % accuracy
    :param weighted: use weighted FastDTW modification or just FastDTW
                     unweighted scenario doesn't require computing the weights,
                      cause all weights are set to be 1.
    """
    Training(HumanoidKinect).compute_weights(mode, beta, fps)
    Testing(HumanoidKinect).the_worst_comparison(fps, weighted=weighted)


def print_average_duration():
    """
     Prints average gestures duration in seconds.
    """
    duration = InstrumentCollector(HumanoidKinect).compute_average_duration()
    print("Average gesture duration (sec): %g " % duration)


if __name__ == "__main__":
    run_the_worst_comparison()
