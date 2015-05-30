# coding=utf-8

from MOCAP.mreader import HumanoidUkr
from tools.instruments import Training


def plot_ratio_vs_fps(hand_mode="bothHands", beta=None):
    """
     Plots discriminant ratio VS fps.
    """
    Training(HumanoidUkr).ratio_vs_fps(hand_mode, beta, 1, 120, 2)


def choose_beta(hand_mode="bothHands", fps=None, reset=False):
    """
     Chooses the best beta (which yields the biggest discriminant ratio R),
     w.r.t. hand mode and fps params.
    """
    Training(HumanoidUkr).choose_beta_pretty(hand_mode, fps, reset)


if __name__ == "__main__":
    # plot_ratio_vs_fps()
    choose_beta()
