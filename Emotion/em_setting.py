# coding=utf-8

from Emotion.emotion import Emotion
from tools.instruments import Training


def plot_ratio_vs_fps(mode=None, beta=None):
    """
     Plots discriminant ratio VS fps.
    """
    Training(Emotion).ratio_vs_fps(mode, beta, 1, 24)


def upd_ratio(mode=None, beta=None, fps=None):
    """
     Updates discriminant ratio, w.r.t. mode, beta and fps params.
    """
    Training(Emotion).update_ratio(mode, beta, fps, verbose=True)


def choose_beta(mode=None, fps=None):
    """
     Chooses the best beta (which yields the biggest discriminant ratio R),
     w.r.t. mode and fps params.
    """
    Training(Emotion).choose_beta_pretty(mode, fps)


if __name__ == "__main__":
    choose_beta()
