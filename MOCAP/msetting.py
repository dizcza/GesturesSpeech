# coding = utf-8

from MOCAP.mreader import HumanoidUkr
from tools.instruments import Training


if __name__ == "__main__":
    trainInstruments = Training(HumanoidUkr)
    trainInstruments.compute_weights("bothHands", None, None)
    trainInstruments.compute_between_variance(10)
