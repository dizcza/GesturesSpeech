# coding = utf-8

from MOCAP.mreader import HumanoidUkr
from instruments import Testing


if __name__ == "__main__":
    testInstruments = Testing(HumanoidUkr)
    testInstruments.the_worst_comparison(fps=10)
