# coding=utf-8

from Emotion.emotion import Emotion
from instruments import Testing, Training


def entire_face_test():
    # set mode to "no_eyes" not to track eyes movements
    Training(Emotion, "EntireFace").compute_weights("no_eyes", None, None)
    Testing(Emotion, "EntireFace").the_worst_comparison(None)


def face_area_test():
    pass


if __name__ == "__main__":
    entire_face_test()
