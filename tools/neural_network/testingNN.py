# coding=utf-8

import numpy as np
import json
from pybrain.tools.customxml.networkreader import NetworkReader
from tools.neural_network.trainingNN import extract_features, collect_gestures

from Emotion.emotion import Emotion
from Kinect.kreader import HumanoidKinect
from MOCAP.mreader import HumanoidUkr
from tools.instruments import InstrumentCollector


def test_project(instr):
    print("%s: NETWORK TESTING" % instr.MotionClass.__name__)
    trn_samples, tst_samples, names_convention = collect_gestures(instr)

    project_name = tst_samples[0].project
    net = NetworkReader.readFrom(r"weights/%s.xml" % project_name)
    moving_marks = json.load(open("common_markers.json"))[project_name]
    use_frames = 20

    misclassified = [0, 0]
    sizes = len(trn_samples), len(tst_samples)
    for i, patch_set in enumerate([trn_samples, tst_samples]):
        for sample in patch_set:
            features_map, status = extract_features(sample, moving_marks, use_frames)
            if status == "OK":
                prob = net.activate(features_map)
                ind_got = np.argmax(prob)
                ind_shouldbe = names_convention[sample.name]
                if ind_got != ind_shouldbe:
                    misclassified[i] += 1
            else:
                sizes[i] -= 1
    errors = 100. * np.divide(misclassified, sizes)
    msg = "in-sample error: %f%% (%d / %d)\n" % (errors[0], misclassified[0], sizes[0])
    msg += "out-of-sample error: %f%% (%d / %d)" % (errors[1], misclassified[1], sizes[1])
    print(msg)


def test_kinect():
    instr = InstrumentCollector(HumanoidKinect)
    test_project(instr)

def test_mocap():
    instr = InstrumentCollector(HumanoidUkr)
    test_project(instr)

def test_emotion():
    instr = InstrumentCollector(Emotion)
    test_project(instr)


if __name__ == "__main__":
    # test_emotion()
    test_kinect()
