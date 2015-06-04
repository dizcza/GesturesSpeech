# coding=utf-8

import numpy as np
import os
import shutil
import json
import pickle
from Emotion.em_reader import Emotion, EMOTION_PATH_PICKLES


def get_face_markers():
    face_markers = {
        "mouth": ("lir", "liup", "lil", "lidn", "jaw"),
        "eyes": ("eup_r", "edn_r", "eup_l", "edn_l"),
        "eyebrows": ("ebr_or", "ebr_ir", "ebr_il", "ebr_ol"),
        "cheeks": ("chr", "chl"),
        "nostrils": ("wr", "wl")
    }
    return face_markers


def get_face_areas():
    return "mouth", "eyes", "eyebrows", "cheeks", "nostrils"


def define_valid_face_actions():
    valid_actions = {
        "mouth": ("default", "smile", "open", "shifted", "(undef)"),
        "eyes": ("default", "closed", "(undef)"),
        "eyebrows": ("default", "down", "up", "(undef)"),
        "cheeks": ("default", "down", "up", "(undef)"),
        "nostrils": ("default", "down", "up", "(undef)")
    }
    return valid_actions


def split_face_areas_tricky(trn_rate=0.5):
    """
     Splits filenames:
     each valid action of each face area into training and testing folders
     Don't even try to follow the logic.
    :param trn_rate: how many files go for training
    """
    face_areas = get_face_areas()
    valid_actions = define_valid_face_actions()
    loaded_dic = json.load(open("face_structure_merged.json", 'r'))
    face_labels = get_face_markers()

    for face_ar in valid_actions:
        # delete undef actions
        valid_actions[face_ar] = valid_actions[face_ar][:-1]

    for area in face_areas:
        area_folder = os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", area)
        shutil.rmtree(area_folder, ignore_errors=True)
        os.mkdir(area_folder)
        trn_path = os.path.join(area_folder, "Training")
        tst_path = os.path.join(area_folder, "Testing")
        for _path in (trn_path, tst_path):
            os.mkdir(_path)
            for val_act in valid_actions[area]:
                _area_act_path = os.path.join(_path, val_act)
                os.mkdir(_area_act_path)

    facearea_action_filename = {}
    for area in face_areas:
        facearea_action_filename[area] = {}
        for val_act in valid_actions[area]:
            facearea_action_filename[area][val_act] = []

    for fname in loaded_dic:
        for _area in face_areas:
            for this_act in loaded_dic[fname][_area]:
                if this_act != "(undef)":
                    facearea_action_filename[_area][this_act].append(fname)

    filename_pickle = {}
    for pkl_log in os.listdir(EMOTION_PATH_PICKLES):
        if pkl_log.endswith(".pkl"):
            pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
            pkl_obj = pickle.load(open(pkl_path, 'rb'))
            filename_pickle[pkl_log[:-4]] = pkl_obj

    objects_dic = {}
    for pkl_log in os.listdir(EMOTION_PATH_PICKLES):
        if pkl_log.endswith(".pkl"):
            pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
            objects_dic[pkl_log[:-4]] = Emotion(pkl_path)

    for area in face_areas:
        area_folder = os.path.join(EMOTION_PATH_PICKLES, "FaceAreas", area)
        trn_path = os.path.join(area_folder, "Training")
        tst_path = os.path.join(area_folder, "Testing")
        for val_act in valid_actions[area]:
            files = facearea_action_filename[area][val_act]
            if len(files) < 2:
                for _path in (trn_path, tst_path):
                    area_act_path = os.path.join(_path, val_act)
                    shutil.rmtree(area_act_path)
                continue
            np.random.shuffle(files)
            trn_size = int(trn_rate * len(files))
            trn_files, tst_files = files[:trn_size], files[trn_size:]

            for _path, _files in ((trn_path, trn_files), (tst_path, tst_files)):
                _area_act_path = os.path.join(_path, val_act)
                for a_file in _files:
                    a_labels = face_labels[area]
                    em = objects_dic[a_file]
                    hot_ids = em.get_ids(*a_labels)
                    a_data = filename_pickle[a_file]["data"][hot_ids, ::]
                    an_author = filename_pickle[a_file]["author"]
                    an_emotion = filename_pickle[a_file]["emotion"]
                    a_norm_data = em.norm_data[hot_ids, ::]
                    obj = {
                        "data": a_data,
                        "norm_data": a_norm_data,
                        "labels": a_labels,
                        "author": an_author,
                        "emotion": an_emotion,
                        "face_area": area,
                        "action": val_act
                    }
                    obj_path = os.path.join(_area_act_path, "%s.pkl" % a_file)
                    pickle.dump(obj, open(obj_path, 'wb'), protocol=2)


def resave_face_structure_json():
    # TODO if eyes both open and closed -- > new class
    face_structure = json.load(open(r"face_structure_merged.json", 'r'))
    modif = {
        "49-1-2": [("cheeks", ["up"]), ("mouth", ["smile"])],
        "37-1-1": [("cheeks", ["default"]), ("mouth", ["smile"])],
        "56-2-2": [("cheeks", ["up"])],
        "46-4-1": [("eyebrows", ["down"]), ("mouth", ["smile", "open"])],
        "48-3-2": [("mouth", ["open"])],
        "57-2-1": [("mouth", ["smile"])],
        "30-3-2": [("mouth", ["open"])],
        "52-3-1": [("mouth", ["(undef)"])],
        "49-2-2": [("eyes", ["(undef)"])],
        "46-3-1": [("eyes", ["(undef)"])],
        "31-1-2": [("mouth", ["smile"])],
        "33-4-1": [("mouth", ["smile", "default"])],
        "33-4-2": [("mouth", ["smile"])],
        "46-3-2": [("mouth", ["smile"])],
        "58-1-1": [("mouth", ["smile", "open"])],
        "31-2-2": [("eyes", ["(undef)"])],
        "54-5-1": [("eyes", ["closed"])],
        "61-4-1": [("eyes", ["default", "closed"])],
        "50-3-1": [("eyes", ["default", "closed"])]
    }
    for fname in modif:
        for pairID in range(len(modif[fname])):
            face_area, ok_act = modif[fname][pairID]
            emotion = face_structure[fname]["emotion"]
            em_json_dic = json.load(open(r"inspector_cache/%s.json" % emotion, 'r'))
            if fname in em_json_dic:
                em_json_dic[fname][face_area] = ok_act
                json.dump(em_json_dic, open(r"inspector_cache/%s.json" % emotion, 'w'))
            face_structure[fname][face_area] = ok_act
    json.dump(face_structure, open(r"face_structure_merged.json", 'w'))


if __name__ == "__main__":
    split_face_areas_tricky()
    resave_face_structure_json()
