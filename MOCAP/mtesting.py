# coding = utf-8

from mreader import HumanoidUkr
from gDTW import compare, show_comparison
import os
import numpy as np


def compare_workout():
    """
     A workout.
    """
    trn_folder = "D:\GesturesDataset\splitAll\Training"
    trn_logs = os.listdir(trn_folder)

    first, second = trn_logs[0], trn_logs[74]
    first = os.path.join(trn_folder, first)
    second = os.path.join(trn_folder, second)

    firstGest = HumanoidUkr(first)
    secondGest = HumanoidUkr(second)

    show_comparison(firstGest, secondGest)


def compare_them_all():
    """
     TESTING func
    """
    trn_folder = "D:\GesturesDataset\splitAll\Training"
    tst_folder = "D:\GesturesDataset\splitAll\Testing"
    misclassified = 0.

    total_samples = len(os.listdir(trn_folder))

    patterns = []
    for trn_log in os.listdir(trn_folder)[:total_samples]:
        trn_filename = os.path.join(trn_folder, trn_log)
        patterns.append(HumanoidUkr(trn_filename))

    print "Testing..."

    for tst_log in os.listdir(tst_folder)[:total_samples]:
        print "\tComparing %s..." % tst_log
        tst_filename = os.path.join(tst_folder, tst_log)
        unknown = HumanoidUkr(tst_filename)
        costs = []
        for knownGest in patterns:
            dist = compare(knownGest, unknown)
            costs.append(dist)

        ind = np.argmin(costs)
        possibleGest = patterns[ind]

        if possibleGest.name != unknown.name:
            print "\t\tgot %s, should be %s" % (possibleGest.name, unknown.name)
            misclassified += 1.

    Etest = misclassified / total_samples
    print "Etest: %.2f" % Etest

# compare_workout()
compare_them_all()