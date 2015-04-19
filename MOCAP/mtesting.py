# coding = utf-8

from MOCAP.mreader import HumanoidUkr
from instruments import Testing


if __name__ == "__main__":
    tstInstr = Testing(HumanoidUkr)
    tstInstr.the_worst_comparison(fps=5)
