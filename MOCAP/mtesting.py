# coding = utf-8

from MOCAP.mreader import HumanoidUkr, MOCAP_PATH
from MOCAP.msetting import compute_between_variance
from gDTW import wdtw_windowed, wdtw
from comparison import compare, show_comparison
import os
import numpy as np
import matplotlib.pyplot as plt
import time


def compare_workout():
    """
     A workout.
    """
    trn_folder = MOCAP_PATH + "Training"
    trn_logs = os.listdir(trn_folder)

    first, second = trn_logs[0], trn_logs[1:]
    first = os.path.join(trn_folder, first)
    firstGest = HumanoidUkr(first)
    firstGest.compute_displacement(mode="bothHands")
    print(sum(firstGest.joint_displace.values()))

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


def probAnd(array):
    """ 
    :param array: (A, B) list
    :return: P(A and B)
    """
    if len(array) == 1:
        # exit
        return array[0]
    return probOr(array[:-1]) * array[-1]


def probOr(array):
    """ 
    :param array: (A, B) list
    :return: P(A orB)
    """
    if len(array) == 1:
        # exit
        return array[0]
    return probOr(array[:-1]) + array[-1] - probAnd(array)


def test_inclusion_exclusion():
    a, b, c = 0.99, 0.99, 0.9
    should_be = a + b + c - a*b - a*c - b*c + a*b*c
    array = [a, b, c]
    got = probOr(array)
    print(should_be, got)


def estimate_activation_prob(U, costs):
    """
    :param U: internal energy of the unknown-known gestures system
    :param costs: energy levels
    :return: (float) activation probability
    """
    costs = np.sort(costs)
    argMin = np.argmin(costs)
    higher_energies = np.delete(costs, argMin)
    single_activation_prob = np.exp((costs[argMin] - higher_energies) / U)
    activation_prob = probOr(single_activation_prob[:10])
    return activation_prob


def compare_them_all(fps):
    """
     TESTING func
    """
    print("(MoCap project) comparing them all ...")

    trn_folder = MOCAP_PATH + "Training"
    tst_folder = MOCAP_PATH + "Testing"
    misclassified = 0.
    total_samples = len(os.listdir(tst_folder))

    patterns = []
    for trn_log in os.listdir(trn_folder):
        trn_filename = os.path.join(trn_folder, trn_log)
        patterns.append(HumanoidUkr(trn_filename, fps))

    print("Testing with fps: %s ..." % fps)
    confidence = [0.]
    for tst_log in os.listdir(tst_folder):
        print("\tComparing %s..." % tst_log)
        tst_filename = os.path.join(tst_folder, tst_log)
        unknownGest = HumanoidUkr(tst_filename, fps)
        costs = []
        for knownGest in patterns:
            dist = compare(knownGest, unknownGest, dtw_chosen=wdtw)
            costs.append(dist)

        argMin = np.argmin(costs)
        possibleGest = patterns[argMin]

        # U = unknownGest.get_internal_energy(mode="bothHands")
        # act_prob = estimate_activation_prob(U, costs)
        # confidence.append(1. - act_prob)
        # print "Pact: %f" % act_prob

        if possibleGest.name != unknownGest.name:
            print("\t\tgot %s, should be %s" % (possibleGest.name, unknownGest.name))
            misclassified += 1.
    Etest = misclassified / total_samples
    print("Etest: %g <----> (%d / %d)" % (Etest, misclassified, total_samples))

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

    plt.plot(fps_range, test_errors, marker='o', ms=8)
    plt.xlabel("data freq (fps), 1/s")
    plt.ylabel("Etest")
    plt.title("out-of-sample error VS fps")
    plt.savefig("png/Etest_fps.png")
    plt.show()


def how_many_incomparable(fps):
    print("(MoCap project) how many incomparable gestures?")
    begin = time.time()

    trn_folder = MOCAP_PATH + "Training"
    tst_folder = MOCAP_PATH + "Testing"
    incomparable_total = 0
    total_samples = len(os.listdir(tst_folder))

    patterns = []
    for trn_log in os.listdir(trn_folder):
        trn_filename = os.path.join(trn_folder, trn_log)
        patterns.append(HumanoidUkr(trn_filename, fps))

    print("Running with fps: %s" % fps)
    for i, tst_log in enumerate(os.listdir(tst_folder)):
        # print "\tComparing %s..." % tst_log
        print("\rProgress: {0}%".format((float(i)/(total_samples-1))*100),)
        tst_filename = os.path.join(tst_folder, tst_log)
        unknownGest = HumanoidUkr(tst_filename, fps)
        for knownGest in patterns:
            if not unknownGest.is_comparable_with(knownGest, thr=0.1):
                incomparable_total += 1.
        print(incomparable_total)

    got_rid_of = incomparable_total / total_samples
    perc = got_rid_of / total_samples
    print("got rid of: %g %% <----> (%d / %d)" % (perc, got_rid_of, total_samples))

    end = time.time()
    print("\t Duration: ~%g sec" % (end - begin))


if __name__ == "__main__":
    # error_vs_fps()
    # compare_workout()
    # how_many_incomparable(fps=None)
    compare_them_all(fps=None)
    # test_inclusion_exclusion()