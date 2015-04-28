# coding=utf-8

from pybrain.structure import SigmoidLayer, TanhLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

import numpy as np
import os
import itertools
from copy import deepcopy

from Emotion.excel_parser import parse_xls
from Emotion.emotion import EMOTION_PATH_PICKLES, Emotion
from Kinect.kreader import HumanoidKinect
from MOCAP.mreader import MOCAP_PATH, HumanoidUkr


def net_preproc(_gest, moving_marks, use_frames):
    """
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

    if np.isnan(data).any():
        return [], "FAIL"

    x = np.ravel(data)
    delta = np.max(x) - np.min(x)
    return tuple(x / delta), "OK"


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


def run_network(all_samples, names_convention, mov_mark_mode=None,
                use_frames=50, hidden_neurons=30, tst_split=0.3):
    """
        1) creates a simple perceptron,
        2) feeds it with all_samples,
        3) evaluates the results.
    :param all_samples: list of read samples
    :param names_convention: a dic, containing integer labels for each
                             gesture class
    :param mov_mark_mode: moving markers mode (defines NN dimensionality)
    :param use_frames: fixed number of frames to work with
    :param tst_split: (0..1) how many samples leave for testing
    :returns:
        Eout_min - the lowest our-of-sample error,
        results_perc - list of (Ein, Eout) pairs per epochs,
        epochs - list of iterations during learning the NN,
        tstdata - pybrain testing data set (part of all_samples).
    """
    print("Running network \n")
    moving_marks = get_common_moving_markers(all_samples, mov_mark_mode)

    input_layer_dim = get_input_layer_dim(all_samples[0],
                                          moving_marks,
                                          use_frames)
    input_layer = ClassificationDataSet(input_layer_dim, 1,
                                        nb_classes=len(names_convention))

    for sample in all_samples:
        features_map, status = net_preproc(sample, moving_marks, use_frames)
        if status == "OK":
            letter_class = names_convention[sample.name]
            input_layer.addSample(features_map, [letter_class])

    tstdata_temp, trndata_temp = input_layer.splitWithProportion(tst_split)
    tst_size = tstdata_temp.getLength()
    trn_size = trndata_temp.getLength()
    print("Train size: %d \t test size: %d" % (trn_size, tst_size))

    tstdata = ClassificationDataSet(input_layer_dim, 1,
                                    nb_classes=len(names_convention))
    for n in range(0, tstdata_temp.getLength()):
        tstdata.addSample(
            tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1]
        )

    trndata = ClassificationDataSet(input_layer_dim, 1,
                                    nb_classes=len(names_convention))
    for n in range(0, trndata_temp.getLength()):
        trndata.addSample(
            trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1]
        )

    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    print("Input x hidden x output dimensions: ",
          trndata.indim, hidden_neurons, trndata.outdim, '\n')

    fnn = buildNetwork(trndata.indim, hidden_neurons, trndata.outdim,
                       hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.0,
                              verbose=False, weightdecay=1e-3, learningrate=1e-2)

    results_perc = []
    Eout_min = 1.0
    N_epochs = 500
    iters_per_epoch = 1
    weights_path = r"weights/%s.xml" % all_samples[0].project
    epochs = np.linspace(iters_per_epoch, iters_per_epoch*N_epochs, N_epochs)
    for i in range(N_epochs):
        trainer.trainEpochs(iters_per_epoch)
        trn_error = percentError(trainer.testOnClassData(), trndata['class'])
        tst_error = percentError(trainer.testOnClassData(
            dataset=tstdata), tstdata['class'])

        if tst_error / 100.0 < Eout_min:
            Eout_min = tst_error / 100.0
            NetworkWriter.writeToFile(fnn, weights_path)

        if i % 10 == 0:
            epoch_status = "epoch: %4d \t train error: %f%% \t test error: %f%% " % (
                trainer.totalepochs,
                trn_error,
                tst_error
            )
            print(epoch_status)
        results_perc.append((trn_error, tst_error))
    print("Eout_min: %f" % Eout_min)

    return Eout_min, results_perc, epochs, tstdata


def collect_gestures_em():
    """
     Collects Emotion 2D gestures.
    :returns:
        - list of Emotion gestures,
        - labeled gesture classes
    """
    em_basket = parse_xls()[0]
    names_convention = {em: its_id for its_id, em in enumerate(em_basket)}
    gestures = []
    okay_logs = list(itertools.chain(*em_basket.values()))
    for pkl_log in os.listdir(EMOTION_PATH_PICKLES):
        if pkl_log.endswith(".pkl") and pkl_log[:-4] in okay_logs:
            pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
            em = Emotion(pkl_path)
            gestures.append(em)
    return gestures, names_convention


def collect_gestures_kin():
    """
     Collects HumanoidKinect 3D gestures.
    :returns:
        - list of HumanoidKinect gestures,
        - labeled gesture classes
    """
    gestures = []
    dir_path = r"D:\temp\kin"
    for pkl_log in os.listdir(dir_path):
        if pkl_log.endswith(".txt"):
            pkl_path = os.path.join(dir_path, pkl_log)
            em = HumanoidKinect(pkl_path)
            gestures.append(em)

    names_convention = {
        "LeftHandPullDown": 0,
        "LeftHandPushUp": 1,
        "LeftHandSwipeRight": 2,
        "LeftHandWave": 3,
        "RightHandPullDown": 4,
        "RightHandPushUp": 5,
        "RightHandSwipeLeft": 6,
        "RightHandWave": 7
    }

    return gestures, names_convention


def collect_gestures_mocap():
    """
     Collects HumanoidUkr 3D gestures.
    :return: list of HumanoidUkr gestures
    """
    gestures = []
    names_convention = {}
    class_label = 0
    for c3d_log in os.listdir(MOCAP_PATH):
        if c3d_log.endswith(".c3d"):
            c3d_path = os.path.join(MOCAP_PATH, c3d_log)
            ukr_gest = HumanoidUkr(c3d_path)
            if ukr_gest.name not in names_convention:
                names_convention[ukr_gest.name] = class_label
                class_label += 1
            gestures.append(ukr_gest)

    return gestures, names_convention


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


def train_emotion():
    gestures, names_convention = collect_gestures_em()
    print("Read %d gestures" % len(gestures))
    run_network(gestures, names_convention, None, 50, 30, 0.3)


def train_kinect():
    gestures, names_convention = collect_gestures_kin()
    print("Read %d gestures" % len(gestures))
    run_network(gestures, names_convention, "bothHands", 50, 30, 0.3)


def train_mocap():
    gestures, names_convention = collect_gestures_mocap()
    print("Read %d gestures" % len(gestures))
    run_network(gestures, names_convention, "bothHands", 50, 30, 0.5)


if __name__ == "__main__":
    train_emotion()