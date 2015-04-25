# coding=utf-8

from Emotion.emotion import Emotion
from instruments import Training


if __name__ == "__main__":
    trainInstruments = Training(Emotion, "EntireFace")
    # trainInstruments.compute_weights(None, None, None)
    trainInstruments.compute_within_variance(None)
    trainInstruments.compute_between_variance(None)
    # trainInstruments.choose_beta(None, None)
    # trainInstruments.update_ratio(None, 1e5, None)