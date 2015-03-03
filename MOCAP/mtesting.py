# coding = utf-8

from mreader import HumanoidUkr
from gDTW import compare, show_comparison, wdtw_windowed, wdtw
import os
import numpy as np


def compare_workout():
    """
     A workout.
    """
    trn_folder = "D:\GesturesDataset\splitAll\Training"
    trn_logs = os.listdir(trn_folder)

    first, second = trn_logs[0], trn_logs[1:]
    first = os.path.join(trn_folder, first)
    firstGest = HumanoidUkr(first, frame_step=1)

    for log in second:
        log = os.path.join(trn_folder, log)
        secondGest = HumanoidUkr(log, frame_step=1)
        show_comparison(firstGest, secondGest)


def compare_them_all(frame_step):
    """
     TESTING func
    """
    trn_folder = "D:\GesturesDataset\splitAll\Training"
    tst_folder = "D:\GesturesDataset\splitAll\Testing"
    misclassified = 0.
    total_samples = len(os.listdir(trn_folder))

    patterns = []
    for trn_log in os.listdir(trn_folder):
        trn_filename = os.path.join(trn_folder, trn_log)
        patterns.append(HumanoidUkr(trn_filename, frame_step))

    print "Testing with frame step: %d ..." % frame_step

    for tst_log in os.listdir(tst_folder):
        # print "\tComparing %s..." % tst_log
        tst_filename = os.path.join(tst_folder, tst_log)
        unknown = HumanoidUkr(tst_filename, frame_step)
        costs = []
        for knownGest in patterns:
            dist = compare(knownGest, unknown, dtw_chosen=wdtw)
            costs.append(dist)

        ind = np.argmin(costs)
        possibleGest = patterns[ind]

        if possibleGest.name != unknown.name:
            print "\t\tgot %s, should be %s" % (possibleGest.name, unknown.name)
            misclassified += 1.

    Etest = misclassified / total_samples
    print "Etest: %.2f" % Etest


# compare_workout()
compare_them_all(frame_step=12)