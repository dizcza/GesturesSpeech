# coding=utf-8

import os

import numpy as np

from Emotion.emotion import EMOTION_PATH_PICKLES
from Emotion.FaceLocations.emotion_area import EmotionArea
from Emotion.FaceLocations.preparation import get_face_areas
from tools.instruments import Testing


def face_areas_test(fps):
    """
     Run the worst comparison explicitly for each face area.
    :param fps: (24 by default) pass as None not to change default fps
    """
    print("WITHOUT POST PROCESSOR")
    face_areas = get_face_areas()
    the_best_the_worst_the_total = []
    for area in face_areas:
        print("*** TOOK FACE AREA: %s" % area)
        area_folder = os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", area)
        instruments = Testing(EmotionArea, prefix=area_folder)
        instruments.compute_weights(None, None, fps)
        one_area_case = instruments.the_worst_comparison(fps)
        the_best_the_worst_the_total.append(one_area_case)
    the_best, the_worst, total = np.sum(the_best_the_worst_the_total, axis=0)
    print("*** FACE AREA TEST IS COMPLETED. ***")
    print(" (SUMMARY) THE BEST: %d \t THE WORST: %d \t TOTAL SAMPLES: %d" % (
        the_best, the_worst, total
    ))


if __name__ == "__main__":
    face_areas_test(None)