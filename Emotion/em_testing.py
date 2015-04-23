# coding=utf-8

from Emotion.emotion import Emotion
from Emotion.emotion_area import EmotionArea
from Emotion.preparation import get_face_areas, EMOTION_PATH_PICKLES
from instruments import Testing, Training
from Emotion.em_setting import compute_areas_weights
import os
import numpy as np


def face_areas_test(fps):
    # compute_areas_weights(None, None, None)
    face_areas = get_face_areas()
    for area in face_areas:
        area_folder = os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", area)
        Testing(EmotionArea, prefix=area_folder).the_worst_comparison(fps)


def entire_face_test():
    # set mode to "no_eyes" not to track eyes movements
    Training(Emotion, suffix="EntireFace").compute_weights("no_eyes", None, None)
    Testing(Emotion, suffix="EntireFace").the_worst_comparison(None)


if __name__ == "__main__":
    # entire_face_test()
    # compute_areas_weights(None, None, None)
    face_areas_test(None)
