# coding=utf-8

from Emotion.emotion import Emotion
from tools.instruments import Testing, Training


def entire_face_test():
    # set mode to "no_eyes" not to track eyes movements
    Training(Emotion).compute_weights(None, 1e2, None)
    Testing(Emotion).the_worst_comparison(None)


if __name__ == "__main__":
    entire_face_test()
