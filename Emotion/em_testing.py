# coding=utf-8

from Emotion.emotion import Emotion
from tools.instruments import Testing, Training


def run_the_worst_comparison(mode=None, beta=None, fps=None):
    """
     Runs the worst comparison scenario after computing weights with given params.
    :param mode: mode for tracking special markers;
                 set it to "no_eyes" not to track eyes markers
    :param beta: (float), needed for computing weights
    :param fps: frames per sec
    """
    Training(Emotion).compute_weights(mode, beta, fps)
    Testing(Emotion).the_worst_comparison(fps)


if __name__ == "__main__":
    Testing(Emotion).show_a_comparison()
    # run_the_worst_comparison()
