# coding=utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import Testing


def plot_error_vs_fps():
    """
     Plots the out-of-sample error VS fps.
    """
    Testing(HumanoidKinect).error_vs_fps()


def run_the_worst_comparison(fps=None):
    Testing(HumanoidKinect).the_worst_comparison(fps)


if __name__ == "__main__":
    plot_error_vs_fps()
