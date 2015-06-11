# coding=utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import Training


def plot_ratio_vs_fps(hand_mode="bothHands", beta=None, reset=False):
    """
     Plots discriminant ratio VS fps.
    """
    Training(HumanoidKinect).ratio_vs_fps(hand_mode, beta, 1, 30, 1, reset)


def choose_beta(hand_mode="bothHands", fps=None, reset=False):
    """
     Chooses the best beta (which yields the biggest discriminant ratio R),
     w.r.t. hand mode and fps params.
    """
    Training(HumanoidKinect).choose_beta_pretty(hand_mode, fps, reset)


if __name__ == "__main__":
    # plot_ratio_vs_fps()
    choose_beta()
