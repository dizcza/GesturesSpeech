# coding=utf-8

import sys
import os
import numpy as np
import json
import time
from pybrain.tools.customxml.networkreader import NetworkReader
from tools.neural_network.trainingNN import extract_features, collect_gestures
from tools.instruments import InstrumentCollector

from Emotion.em_reader import Emotion
from Kinect.kreader import HumanoidKinect
from MOCAP.mreader import HumanoidUkr


def compute_margin(_probabilities):
    """
    :param _probabilities: list of output probabilities
    :return: (float in range of 0..1) margin between
             the chosen positive and the first negative results
    """
    probabilities = list(_probabilities)
    ind_max = np.argmax(probabilities)
    max_prob = probabilities[ind_max]
    probabilities.pop(ind_max)
    upcoming = max(probabilities)
    margin = max_prob - upcoming
    return margin


def test_project(instr):
    """
    Evaluates in-sample and out-of-sample error on training and testing
    samples respectively. Also provides margin and a duration info.
    Margin computes as the difference between maximum output probability
    and the upcoming probability.
    :param instr: InstrumentCollector for a particular class
    """
    start = time.time()
    print("%s: NETWORK TESTING" % instr.MotionClass.__name__)
    trn_samples, tst_samples, names_convention = collect_gestures(instr)

    project_name = tst_samples[0].project
    net = NetworkReader.readFrom(r"weights/%s.xml" % project_name)
    common_markers_path = os.path.join(os.path.dirname(sys.argv[0]), "common_markers.json")
    moving_marks = json.load(open(common_markers_path))[project_name]
    use_frames = 20

    misclassified = [0, 0]
    margin = 0
    sizes = len(trn_samples), len(tst_samples)
    for i, patch_set in enumerate([trn_samples, tst_samples]):
        for sample in patch_set:
            features_map = extract_features(sample, moving_marks, use_frames)
            prob = net.activate(features_map)
            ind_got = np.argmax(prob)
            ind_shouldbe = names_convention[sample.name]
            if ind_got != ind_shouldbe:
                misclassified[i] += 1
            elif i == 1:
                margin += compute_margin(prob)
    errors = 100. * np.divide(misclassified, sizes)
    margin *= 100. / sizes[1]
    duration_per_sample = 1000. * (time.time() - start) / sum(sizes)
    msg = "in-sample error: %f%% (%d / %d)\n" % (errors[0], misclassified[0], sizes[0])
    msg += "out-of-sample error: %f%% (%d / %d)\n" % (errors[1], misclassified[1], sizes[1])
    msg += "margin: %.3g%%\n" % margin
    msg += "duration per sample: %d ms" % duration_per_sample
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
    test_emotion()
    # test_kinect()
    # test_mocap()
