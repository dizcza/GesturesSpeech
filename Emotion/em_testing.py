# coding=utf-8

from Emotion.emotion import Emotion
from instruments import Testing, Training


def entire_face_test():
    # set mode to "no_eyes" not to track eyes movements
    Training(Emotion, suffix="EntireFace").compute_weights(None, None, None)
    Testing(Emotion, suffix="EntireFace").the_worst_comparison(None)


if __name__ == "__main__":
    entire_face_test()
