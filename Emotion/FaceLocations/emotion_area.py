# coding=utf-8

import pickle
import json
import os
import numpy as np
import warnings
from Emotion.em_reader import Emotion, EMOTION_PATH
from Emotion.FaceLocations.preparation import get_face_areas


isString = lambda item: type(item).__name__ in ("str", "unicode")
isList = lambda item: type(item).__name__ in ("list", "tuple")
isMultipleActions = lambda item: type(item).__name__ == "MultipleActions"


warnings.warn("Emotion Area project is not finished yet!")


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
        self.norm_data = np.diff(self.norm_data, axis=1)
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
    smiles_dir = os.path.join(EMOTION_PATH, r"FaceAreas\mouth\Training\smile")
    first_smile_path = os.listdir(smiles_dir)[0]
    first_smile_path = os.path.join(smiles_dir, first_smile_path)
    emArea = EmotionArea(first_smile_path)
    emArea.data = emArea.norm_data
    print(emArea)
    emArea.show_displacements(None)
    emArea.animate()


def test_multiple_names():
    face_structure = json.load(open(r"face_structure_merged.json", 'r'))
    for fname in face_structure:
        for face_area in get_face_areas():
            actions_num = len(face_structure[fname][face_area])
            assert 0 < actions_num <= 2, "invalid number of actions"


if __name__ == "__main__":
    # test_multiple_names()
    test_face_area()
