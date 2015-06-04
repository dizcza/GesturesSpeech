# coding=utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from copy import deepcopy

from pybrain.structure import TanhLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter

from Emotion.em_reader import Emotion
from Kinect.kreader import HumanoidKinect
from MOCAP.mreader import HumanoidUkr
from tools.instruments import InstrumentCollector


def extract_features(_gest, moving_marks, use_frames):
    """
     Neural network preprocessor.
     Extracts features before feeding them into a NN.
    :param _gest: a gesture sample
    :param moving_marks: list of moving markers
    :param use_frames: fixed number of frames to work with
    :return: flattened np.array of XYZs for each moving marker for each frame
    """
    gest = deepcopy(_gest)
    gest.set_gesture_length(use_frames)
    hot_ids = gest.get_ids(*moving_marks)
    data = gest.norm_data[hot_ids, ::]
    visibleMarker_lastFrame = np.zeros(data.shape[0], dtype=int)
    for markerID in range(data.shape[0]):
        for frame in range(data.shape[1]):
            for dim in range(data.shape[2]):
                if np.isnan(data[markerID, frame, :]).any():
                    last_ok_frame = visibleMarker_lastFrame[markerID]
                    data[markerID, frame, dim] = data[markerID, last_ok_frame, dim]
                else:
                    visibleMarker_lastFrame[markerID] = frame

    assert not np.isnan(data).any(), "do smth with NaNs"

    x = np.ravel(data)
    delta = np.max(x) - np.min(x)
    return tuple(x / delta)


def get_input_layer_dim(gest, moving_marks, use_frames):
    """
    :param gest: a gesture sample
    :param moving_marks: list of moving markers
    :param use_frames: fixed number of frames to work with
    :return: NN input layer dimension
    """
    how_many_markers = len(moving_marks)
    ord_dim = gest.norm_data.shape[2]
    input_layer_dim = how_many_markers * use_frames * ord_dim
    print("Input layer size: %d x %d x %d ==> %d" % (
        how_many_markers, use_frames, ord_dim, input_layer_dim
    ))
    return input_layer_dim


def get_common_moving_markers(gestures, mode):
    """
    :param gestures: list of gestures
    :return: common moving markers for all given gestures
    """
    first_sample = deepcopy(gestures[0])
    first_sample.define_moving_markers(mode)
    moving_marks = list(first_sample.moving_markers)
    for gest in gestures[1:]:
        waste_labels = []
        for label in moving_marks:
            if label not in gest.labels:
                waste_labels.append(label)
        for bad_label in waste_labels:
            moving_marks.remove(bad_label)
    return moving_marks


def visualize(results_perc, proj_name):
    """
    Visualizes iteration process.
    :param results_perc: list of (in-sample, out-of-sample) errors
    """
    Ein, Eout = zip(*results_perc)
    plt.plot(Ein, 'bo-')
    plt.plot(Eout, 'go-')
    plt.legend(["in-sample", "out-of-sample"], numpoints=1)
    plt.title("Iteration process")
    plt.xlabel("# epochs")
    plt.ylabel("Error, %")
    png_path = os.path.join(os.path.dirname(sys.argv[0]), "png/%s.png" % proj_name)
    plt.savefig(png_path)
    plt.show()


def dump_common_markers(moving_marks, proj_name):
    common_markers_path = os.path.join(os.path.dirname(sys.argv[0]), "common_markers.json")
    if os.path.exists(common_markers_path):
        cm = json.load(open(common_markers_path))
    else:
        cm = {}
    cm[proj_name] = moving_marks
    json.dump(cm, open(common_markers_path, 'w'))


def dump_timing(duration, proj_name):
    timings_path = os.path.join(os.path.dirname(sys.argv[0]), "timings.json")
    if os.path.exists(timings_path):
        timing = json.load(open(timings_path))
    else:
        timing = {}
    timing[proj_name] = int(duration)
    json.dump(timing, open(timings_path, 'w'))


def run_network(trn_samples, tst_samples, names_convention, mov_mark_mode=None,
                use_frames=20, hidden_neurons=20, lrn_rate=0.005, iters_per_epoch=1):
    """
        1) creates a simple perceptron,
        2) feeds it with trn_samples,
        3) evaluates the results.
    :param trn_samples: list of training samples
    :param tst_samples: list of testing samples
    :param names_convention: a dic, containing integer labels for each
                             gesture class
    :param mov_mark_mode: moving markers mode (defines NN dimensionality)
    :param use_frames: fixed number of frames to work with
    """
    start = time.time()
    print("Running network \n")
    moving_marks = get_common_moving_markers(trn_samples + tst_samples,
                                             mov_mark_mode)
    proj_name = trn_samples[0].project
    dump_common_markers(moving_marks, proj_name)

    input_layer_dim = get_input_layer_dim(trn_samples[0],
                                          moving_marks,
                                          use_frames)
    two_datasets = []
    for dataset in (trn_samples, tst_samples):
        pybrn_data = ClassificationDataSet(input_layer_dim, 1,
                                           nb_classes=len(names_convention))
        for sample in dataset:
            features_map = extract_features(sample, moving_marks, use_frames)
            letter_class = names_convention[sample.name]
            pybrn_data.addSample(features_map, [letter_class])
        pybrn_data._convertToOneOfMany()
        two_datasets.append(pybrn_data)
    trndata, tstdata = two_datasets

    trn_size = trndata.getLength()
    tst_size = tstdata.getLength()
    print("Train size: %d \t test size: %d" % (trn_size, tst_size))

    print("Input x hidden x output dimensions: ",
          trndata.indim, hidden_neurons, trndata.outdim, '\n')

    fnn = buildNetwork(trndata.indim, hidden_neurons, trndata.outdim,
                       hiddenclass=TanhLayer, outclass=SoftmaxLayer,
                       bias=True)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.0,
                              verbose=False, weightdecay=1e-3,
                              learningrate=lrn_rate)

    results_perc = []
    tst_error = 100
    trn_error = 100
    weights_path = r"weights/%s.xml" % proj_name
    while trn_error > 1e-3:
        trainer.trainEpochs(iters_per_epoch)
        trn_error = percentError(trainer.testOnClassData(), trndata['class'])
        tst_error = percentError(trainer.testOnClassData(tstdata), tstdata['class'])
        epoch_status = "epoch: %d \t train error: %f%% \t test error: %f%% " % (
            trainer.totalepochs,
            trn_error,
            tst_error
        )
        print(epoch_status)
        results_perc.append((trn_error, tst_error))
    results_perc.append((trn_error, tst_error))
    NetworkWriter.writeToFile(fnn, weights_path)
    misclassified = tst_error / 100. * tst_size
    print("Eout_min: %f%% (%g / %d)" % (tst_error, misclassified, tst_size))
    duration = time.time() - start
    dump_timing(duration, proj_name)
    visualize(results_perc, proj_name)


def collect_gestures(instr):
    """
     Collects gestures with help of instrument collector.
    :returns:
        - list of gestures,
        - labeled gesture classes
    """
    trn_samples = instr.load_train_samples(fps=None)
    tst_samples = instr.load_test_samples(fps=None)

    names_convention = {
        folder: int_label for int_label, folder in
        enumerate(os.listdir(instr.trn_path))
    }

    return trn_samples, tst_samples, names_convention


def train_emotion():
    instr = InstrumentCollector(Emotion)
    trn_samples, tst_samples, names_convention = collect_gestures(instr)
    run_network(trn_samples, tst_samples, names_convention, None, iters_per_epoch=10)


def train_kinect():
    instr = InstrumentCollector(HumanoidKinect)
    trn_samples, tst_samples, names_convention = collect_gestures(instr)
    run_network(trn_samples, tst_samples, names_convention, "bothHands")


def train_mocap():
    instr = InstrumentCollector(HumanoidUkr)
    trn_samples, tst_samples, names_convention = collect_gestures(instr)
    run_network(trn_samples, tst_samples, names_convention, "bothHands", iters_per_epoch=10)


if __name__ == "__main__":
    train_emotion()
    # train_kinect()
    # train_mocap()