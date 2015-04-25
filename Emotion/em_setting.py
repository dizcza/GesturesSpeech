# coding=utf-8

from Emotion.emotion import Emotion
from Emotion.emotion_area import EmotionArea
from Emotion.preparation import EMOTION_PATH_PICKLES
from instruments import Training
from comparison import compare
import json
import os
import numpy as np


def compute_areas_weights(mode, beta, fps):
    """
     Computes aver weights from the Training dataset.
    :param mode: defines moving markers
    :param beta: to be choosing to yield the biggest ratio
    :param fps: frames per second to be set
    """
    proj_info = json.load(open("EMOTION_AREAS_INFO.json", 'r'))
    trn_path = os.path.join(EMOTION_PATH_PICKLES, "Training", "FaceAreas")
    proj_info["beta"] = beta

    global_weights = {}

    for face_area in os.listdir(trn_path):
        global_weights[face_area] = {}
        face_area_dir = os.path.join(trn_path, face_area)
        for act_name in os.listdir(face_area_dir):
            act_dir = os.path.join(face_area_dir, act_name)
            current_act_weights = []
            for log in os.listdir(act_dir):
                log_path = os.path.join(act_dir, log)
                gest = EmotionArea(log_path, fps)
                if gest.action != "(undef)":
                    gest.compute_weights(mode, beta)
                    weights = gest.get_weights()
                    if not np.isnan(weights).any():
                        current_act_weights.append(weights)
            if len(current_act_weights) > 0:
                global_weights[face_area][act_name] = np.average(current_act_weights, axis=0).tolist()

    proj_info["weights"] = global_weights
    json.dump(proj_info, open("EMOTION_AREAS_INFO.json", 'w'))
    print("New weights are saved in EMOTION_AREAS_INFO.json")


def compute_within_variance(fps):
    """
     Computes aver within variance from the Training dataset.
     :param fps: frames per second to be set
    """
    proj_info = json.load(open("EMOTION_AREAS_INFO.json", 'r'))
    print("EmotionArea: COMPUTING WITHIN VARIANCE")
    trn_path = os.path.join(EMOTION_PATH_PICKLES, "Training", "FaceAreas")

    one_vs_the_same_var = []

    for face_area in os.listdir(trn_path):
        print(face_area)
        face_area_dir = os.path.join(trn_path, face_area)
        for act_name in os.listdir(face_area_dir):
            print(act_name)
            act_dir = os.path.join(face_area_dir, act_name)
            log_examples = os.listdir(act_dir)
            current_act_var = []
            while len(log_examples) > 1:
                fpath_trn = os.path.join(act_dir, log_examples[0])
                firstGest = EmotionArea(fpath_trn, fps)
                if firstGest.action == "(undef)":
                    log_examples.pop(0)
                    continue
                for another_log in log_examples[1:]:
                    another_fpath = os.path.join(act_dir, another_log)
                    goingGest = EmotionArea(another_fpath, fps)
                    if goingGest.action != "(undef)":
                        dist, path = compare(firstGest, goingGest)
                        dist /= float(len(path))
                        current_act_var.append(dist)
                log_examples.pop(0)
            if any(current_act_var):
                one_vs_the_same_var.append(np.average(current_act_var))

    if any(one_vs_the_same_var):
        within_var = np.average(one_vs_the_same_var)
        within_std = np.std(one_vs_the_same_var)
    else:
        within_var = None
        within_std = None

    proj_info["within_variance"] = within_var
    proj_info["within_std"] = within_std
    json.dump(proj_info, open("EMOTION_AREAS_INFO.json", 'w'))

    info = "Done with: \n\t within-var: %s \n\t " % within_var
    info += "within-std: %s\n" % within_std
    print(info)



if __name__ == "__main__":
    trainInstruments = Training(Emotion, "EntireFace")
    trainInstruments.compute_weights(None, None, None)
    # trainInstruments.compute_within_variance(None)
    # trainInstruments.compute_between_variance(None)
    # trainInstruments.choose_beta(None, None)
    # trainInstruments.update_ratio(None, 1e5, None)