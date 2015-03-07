# coding = utf-8

from mreader import HumanoidUkr
from gDTW import wdtw_windowed, wdtw
from comparison import compare, show_comparison
import os
import numpy as np
import matplotlib.pyplot as plt
from msetting import compute_between_variance


def compare_workout():
    """
     A workout.
    """
    trn_folder = "D:\GesturesDataset\splitAll\Training"
    trn_logs = os.listdir(trn_folder)

    first, second = trn_logs[0], trn_logs[1:]
    first = os.path.join(trn_folder, first)
    firstGest = HumanoidUkr(first)
    firstGest.compute_displacement(mode="bothHands")
    print sum(firstGest.joint_displace.values())

    for log in second:
        log = os.path.join(trn_folder, log)
        secondGest = HumanoidUkr(log)
        show_comparison(firstGest, secondGest)


def pick_two_min(array):
    """
    :param array: dtw costs array
    :return: argMin, the smallest cost, next upcoming cost
    """
    argMin = 0
    the_smallest = np.inf
    upcoming = np.inf

    for itemID in range(len(array)):
        if array[itemID] <= the_smallest:
            argMin = itemID
            upcoming = the_smallest
            the_smallest = array[itemID]
        elif array[itemID] < upcoming:
            # but also greater than the_smallest
            upcoming = array[itemID]
    return argMin, the_smallest, upcoming


def compare_them_all(fps):
    """
     TESTING func
    """
    print "(MoCap project) comparing them all ..."

    trn_folder = "D:\GesturesDataset\splitAll\Training"
    tst_folder = "D:\GesturesDataset\splitAll\Testing"
    misclassified = 0.
    total_samples = len(os.listdir(trn_folder))

    patterns = []
    for trn_log in os.listdir(trn_folder):
        trn_filename = os.path.join(trn_folder, trn_log)
        patterns.append(HumanoidUkr(trn_filename, fps))

    print "Testing with fps: %s ..." % fps
    confidence = []
    for tst_log in os.listdir(tst_folder):
        # print "\tComparing %s..." % tst_log
        tst_filename = os.path.join(tst_folder, tst_log)
        unknown = HumanoidUkr(tst_filename, fps)
        costs = []
        for knownGest in patterns:
            dist = compare(knownGest, unknown, dtw_chosen=wdtw)
            costs.append(dist)

        argMin, the_smallest_cost, upcoming = pick_two_min(costs)
        confidence.append(1. - the_smallest_cost / upcoming)
        possibleGest = patterns[argMin]

        if possibleGest.name != unknown.name:
            print "\t\tgot %s, should be %s" % (possibleGest.name, unknown.name)
            misclassified += 1.

    Etest = misclassified / total_samples
    print "Etest: %g <----> (%d / %d)" % (Etest, misclassified, total_samples)
    # print "Etest: %.2f; \t confidence: %.2f" % (Etest, np.average(confidence))
    return Etest, np.average(confidence)


def betweenVar_vs_fps():
    """
     Plots between variance VS fps dependency.
    """
    fps_range = np.arange(5, 12, 1)
    between_vars, stds = [], []
    for fps in fps_range:
        var, std = compute_between_variance(fps)
        between_vars.append(var)
        stds.append(std)
    plt.errorbar(fps_range, between_vars, stds, marker='^', ms=8)
    plt.xlabel("data freq (fps), 1/s")
    mean_std = 100. * np.average(stds)
    plt.ylabel("between variance, std=%.1f%%" % mean_std)
    plt.title("Between variance VS fps")
    plt.savefig("png/Db_fps.png")
    plt.show()


def error_vs_fps():
    """
     Plots the E-test VS fps dependency.
    """
    fps_range = range(5, 13, 1)
    test_errors = []
    for fps in fps_range:
        Etest, conf = compare_them_all(fps)
        test_errors.append(Etest)

    fps_range += [20, 120]
    test_errors += [0., 0.]

    plt.plot(fps_range, test_errors, marker='o', ms=8)
    plt.xlabel("data freq (fps), 1/s")
    plt.ylabel("Etest")
    plt.title("out-of-sample error VS fps")
    plt.savefig("png/Etest_fps.png")
    plt.show()


if __name__ == "__main__":
    # error_vs_fps()
    # compare_workout()
    compare_them_all(fps=1)
