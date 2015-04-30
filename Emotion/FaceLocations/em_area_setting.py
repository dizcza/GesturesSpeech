# coding=utf-8

import os

import numpy as np

from Emotion.FaceLocations.emotion_area import EmotionArea
from Emotion.FaceLocations.preparation import get_face_areas
from Emotion.emotion import EMOTION_PATH_PICKLES
from tools.instruments import Training


def face_area_compute_variance(fps):
    """
     Computes within and between variance explicitly for each face area.
    :param fps: (24 by default) pass as None not to change default fps
    """
    print("WITHOUT POST PROCESSOR")
    face_areas = get_face_areas()
    within_vars, between_vars = [], []
    for area in face_areas:
        print("*** TOOK FACE AREA: %s" % area)
        area_folder = os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", area)
        instruments = Training(EmotionArea, prefix=area_folder)
        instruments.compute_weights(None, None, fps)
        wvar = instruments.compute_within_variance(fps)
        bvar = instruments.compute_between_variance(fps)
        within_vars.append(wvar)
        between_vars.append(bvar)
    wvar_aver = np.average(within_vars)
    bvar_aver = np.average(between_vars)
    print(" (AVERAGED) WITHIN VAR: %f \t BETWEEN VAR: %f" % (
        wvar_aver, bvar_aver
    ))


if __name__ == "__main__":
    face_area_compute_variance(None)