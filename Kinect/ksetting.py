# coding=utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import Training


def plot_ratio_vs_fps(hand_mode="bothHands", beta=None):
    """
     Plots discriminant ratio VS fps.
    """
    Training(HumanoidKinect).ratio_vs_fps(hand_mode, beta, 2, 30)


def upd_ratio(hand_mode="bothHands", beta=None, fps=None):
    """
     Updates discriminant ratio, w.r.t. hand mode, beta and fps params.
    """
    Training(HumanoidKinect).update_ratio(hand_mode, beta, fps, verbose=True)


def choose_beta(hand_mode="bothHands", fps=None):
    """
     Chooses the best beta (which yields the biggest discriminant ratio R),
     w.r.t. hand mode and fps params.
    """
    Training(HumanoidKinect).choose_beta_pretty(hand_mode, fps)


if __name__ == "__main__":
    # plot_ratio_vs_fps()
    choose_beta()
