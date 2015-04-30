# coding = utf-8

from Kinect.kreader import HumanoidKinect
from tools.instruments import Training


if __name__ == "__main__":
    trainInstruments = Training(HumanoidKinect)
    trainInstruments.compute_weights("bothHands", None, None)
    # trainInstruments.compute_within_variance(None)
    # trainInstruments.compute_between_variance(None)
