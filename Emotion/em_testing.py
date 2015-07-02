# coding=utf-8

from Emotion.em_reader import Emotion
from tools.instruments import Testing, Training


def run_the_worst_comparison(mode=None, beta=None, fps=None, weighted=True):
    """
     Runs the worst and the best comparison scenarios after computing weights
     with the given params.
    :param mode: mode for tracking special markers;
                 set it to "no_eyes" not to track eyes markers
    :param beta: (float), defines weights activity;
                  the best 'beta' value is around zero (or None);
                  set it to None to model when 'beta' vanishes;
                  set it to a very large number (e.g., 1e6) to model
                    unweighted (the same as weighted=False) scenario;
    :param fps: frames per sec
    :param weighted: use weighted FastDTW modification or just FastDTW
                     unweighted scenario doesn't require computing the weights,
                      cause all weights are set to be 1.
    """
    Training(Emotion).compute_weights(mode, beta, fps)
    Testing(Emotion).the_worst_comparison(fps, weighted=weighted)


if __name__ == "__main__":
    run_the_worst_comparison()
