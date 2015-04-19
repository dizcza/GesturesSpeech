# coding=utf-8

from instruments import Training
from Emotion.emotion import Emotion


if __name__ == "__main__":
    trainInstruments = Training(Emotion)
    # trainInstruments.compute_within_variance(None)
    # trainInstruments.compute_between_variance(None)
    # trainInstruments.choose_beta(None, None)
    trainInstruments.update_ratio(None, 1e5, None)