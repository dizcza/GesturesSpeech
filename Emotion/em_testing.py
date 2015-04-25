# coding=utf-8

from Emotion.emotion import Emotion
from Emotion.emotion_area import EmotionArea
from Emotion.preparation import get_face_areas, EMOTION_PATH_PICKLES
from instruments import Testing, Training
from Emotion.em_setting import compute_areas_weights
import os
import numpy as np


def face_areas_test(fps):
    print("WITHOUT POST PROCESSOR")
    # compute_areas_weights(None, None, None)
    face_areas = get_face_areas()
    the_best_the_worst_the_total = []
    for area in face_areas:
        print("*** TOOK FACE AREA: %s" % area)
        area_folder = os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", area)
        instruments = Testing(EmotionArea, prefix=area_folder)
        instruments.compute_weights(None, None, None)
        one_area_case = instruments.the_worst_comparison(fps)
        the_best_the_worst_the_total.append(one_area_case)
    the_best, the_worst, total = np.sum(the_best_the_worst_the_total, axis=0)
    print("*** FACE AREA TEST IS COMPLETED. ***")
    print(" (SUMMARY) THE BEST: %d \t THE WORST: %d \t TOTAL SAMPLES: %d" % (
        the_best, the_worst, total
    ))


def entire_face_test():
    # set mode to "no_eyes" not to track eyes movements
    Training(Emotion, suffix="EntireFace").compute_weights(None, None, None)
    Testing(Emotion, suffix="EntireFace").the_worst_comparison(None)


if __name__ == "__main__":
    # entire_face_test()
    # compute_areas_weights(None, None, None)
    # face_areas_test(None)
    instr = Testing(EmotionArea, prefix=os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", "eyes"))
    instr.compute_weights(None, None, None)
    print(instr.proj_info["weights"]["eyes"])