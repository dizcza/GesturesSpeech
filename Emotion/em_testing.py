# coding=utf-8

from Emotion.emotion import Emotion
from instruments import Testing


if __name__ == "__main__":
    testInstruments = Testing(Emotion)
    testInstruments.the_worst_comparison(None)
    # testInstruments.show_a_comparison()
