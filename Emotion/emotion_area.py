# coding=utf-8

from Emotion.emotion import Emotion
import pickle
import json
import os
import numpy as np
from Emotion.preparation import get_face_areas


isString = lambda item: type(item).__name__ in ("str", "unicode")
isList = lambda item: type(item).__name__ in ("list", "tuple")
isMultipleActions = lambda item: type(item).__name__ == "MultipleActions"


class MultipleActions(object):
    def __init__(self, possible_actions):
        self.valid_names = tuple(possible_actions)
        assert 0 < len(possible_actions) <= 2, "invalid number of actions"

    def __str__(self):
        s = ""
        if len(self.valid_names) == 1:
            s = self.valid_names[0]
        elif len(self.valid_names) == 2:
            s = "{%s OR %s}" % (self.valid_names[0], self.valid_names[1])
        return s

    def __eq__(self, _other):
        """
        :param _other: instance of class "str" or "MultipleActionName"
        :return: valid names intersection
        """
        intersection = False
        if isList(_other):
            other = MultipleActions(_other)
        else:
            other = _other
        for self_name in self.valid_names:
            if isMultipleActions(other):
                intersection = intersection or self_name in other.valid_names
            elif isString(other):
                intersection = intersection or self_name == other
        return intersection

    def __neg__(self, other):
        return not self == other


class EmotionArea(Emotion):
    def __init__(self, obj_path, fps=None):
        obj_info = pickle.load(open(obj_path, 'rb'))
        face_structure = json.load(open("face_structure_merged.json", 'r'))
        self.face_area = obj_info["face_area"]
        self.action = obj_info["action"]
        Emotion.__init__(self, obj_path, fps)
        self.norm_data = obj_info["norm_data"]
        self.frames = self.norm_data.shape[1]
        self.action = face_structure[self.fname][self.face_area]
        self.name = MultipleActions(self.action)
        # self.postprocessor()

    def __str__(self):
        s = Emotion.__str__(self)
        s += "\n\t face area: \t%s" % self.face_area
        s += "\n\t action: \t\t%s" % self.action
        return s

    def preprocessor(self):
        pass

    def postprocessor(self):
        self.norm_data = np.subtract(
            self.norm_data[:, 1:, :], self.norm_data[:, :-1, :]
        )
        self.frames -= 1

    def set_weights(self):
        proj_info = json.load(open("EMOTION_AREAS_INFO.json", 'r'))
        if self.action in proj_info["weights"][self.face_area]:
            if self.action == "(undef)":
                weights_arr = np.ones(len(self.labels)) * np.nan
            else:
                weights_arr = proj_info["weights"][self.face_area][self.action]
            for markerID, marker_name in enumerate(self.labels):
                self.weights[marker_name] = weights_arr[markerID]

    def simple_preproc(self):
        self.data = self.data[:, 1:, :]
        self.frames -= 1

        # step 1: subtract centroid from the clean data
        clean_data = self.data[~np.isnan(self.data).any(axis=2)]
        centroid = np.average(clean_data, axis=(0, 1))
        self.norm_data = self.data - centroid

        # step 2: divide data by box size
        # skipped

        # step 3: gaussian blurring filter
        self.gaussian_filter()

        # step 4: deal eye winking
        if self.face_area == "eyes":
            self.deal_with_winking()

        self.data = self.norm_data


def test_face_area():
    smiles = r"D:\GesturesDataset\Emotion\pickles\Training\FaceAreas\mouth\smile"
    first_smile = os.listdir(smiles)[0]
    first_smile = os.path.join(smiles, first_smile)
    emArea = EmotionArea(first_smile)
    emArea.data = emArea.norm_data
    print(emArea)
    emArea.animate()


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


def test_multiple_names():
    face_structure = json.load(open(r"face_structure_merged.json", 'r'))
    for fname in face_structure:
        for face_area in get_face_areas():
            actions_num = len(face_structure[fname][face_area])
            if actions_num > 2:
                print("%s --> %s --> %s" % (fname, face_area, face_structure[fname][face_area]))
            # assert 0 < actions_num <= 2, "invalid number of actions"


def my_test():
    pkl_folder = r"D:\GesturesDataset\Emotion\pickles"
    for pkl_log in os.listdir(pkl_folder):
        pkl_path = os.path.join(pkl_folder, pkl_log)
        em = Emotion(pkl_path)


if __name__ == "__main__":
    resave_face_structure_json()
    test_multiple_names()
    # test_face_area()
    face_structure = json.load(open(r"face_structure_merged.json", 'r'))
    for fname in face_structure:
        eyes = face_structure[fname]["eyes"]
        if "(undef)" in eyes and len(eyes) > 1:
            print(fname, face_structure[fname]["emotion"], eyes)