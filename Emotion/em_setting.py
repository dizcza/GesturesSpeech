# coding=utf-8

from Emotion.emotion import Emotion
from tools.instruments import Training


def plot_ratio_vs_fps(mode=None, beta=1e2):
    """
     Plots discriminant ratio VS fps.
    """
    Training(Emotion).ratio_vs_fps(mode, beta, 1, 24)


def choose_beta(mode=None, fps=None, reset=False):
    """
     Chooses the best beta (which yields the biggest discriminant ratio R),
     w.r.t. mode and fps params.
     Set mode to None to use all markers.
     Set fps to None not to change default fps
    """
    Training(Emotion).choose_beta_pretty(mode, fps, reset)


if __name__ == "__main__":
    choose_beta()
