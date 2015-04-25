# coding = utf-8

from Kinect.kreader import HumanoidKinect
from instruments import Testing


if __name__ == "__main__":
    testInstrument = Testing(HumanoidKinect)
    testInstrument.the_worst_comparison(None)
    # testInstrument.compare_with_first(None)