# coding = utf-8

from wdtw import wdtw_windowed, wdtw
from comparison import compare, show_comparison
from Kinect.kreader import *
import os


def compare_workout():
    """
     Compares some two gestures from the Training set.
    """
    folder = KINECT_PATH + "Training\\LeftHandSwipeRight\\"
    gestures = []
    for log in os.listdir(folder):
        if log.endswith(".txt"):
            gest = HumanoidKinect(folder + log)
            gestures.append(gest)

    show_comparison(gestures[5], gestures[3])


def collect_patterns(fps):
    """
    :return: patters from Training set (taken as the first file in each folder)
    """
    pattern_gestures = []
    for root, _, logs in os.walk(KINECT_PATH + "Training\\"):
        if any(logs) and logs[0].endswith(".txt"):
            full_filename = os.path.join(root, logs[0])
            gest = HumanoidKinect(full_filename, fps)
            pattern_gestures.append(gest)
    print("Took %d patterns as the first log in each training dir." % \
          len(pattern_gestures))
    return pattern_gestures


def compare_them_all(fps):
    """
     Main testing function.
    :return: out-of-sample error
    """
    print("(Kinect project) comparing them all with fps = %s" % fps)
    patterns = collect_patterns(fps)
    misclassified = 0.
    total_samples = 0
    for root, _, logs in os.walk(KINECT_PATH + "Testing\\"):
        for test_log in logs:
            full_filename = os.path.join(root, test_log)
            if full_filename.endswith(".txt"):
                unknown_gest = HumanoidKinect(full_filename, fps)
                offset = []
                for known_gest in patterns:
                    dist = compare(known_gest, unknown_gest)
                    offset.append(dist)
                ind = np.argmin(offset)
                possible_gest = patterns[ind]
                # show_comparison(possible_gest, unknown_gest)

                if possible_gest.name != unknown_gest.name:
                    print("got %s, should be %s" % (possible_gest.name, unknown_gest.name))
                    misclassified += 1.

                total_samples += 1

    Etest = misclassified / float(total_samples)
    print("Etest: %g <----> (%d / %d)" % (Etest, misclassified, total_samples))

    return Etest


def the_worst_between_comparison(fps):
    """
     Computes the worst and the best out-of-sample error.
    :param fps: fps to be set in each gesture
    :return: supremum and infimum numbers of misclassified test samples
    """
    # total_samples == 160
    trn_path = KINECT_PATH + "Training\\"

    patterns = {}
    supremum = {}
    infimum = {}

    for dir in os.listdir(trn_path):
        patterns[dir] = []
        infimum[dir] = 0.
        supremum[dir] = 0.
        trn_subdir = os.path.join(trn_path, dir)
        for short_name in os.listdir(trn_subdir):
            fname = os.path.join(trn_subdir, short_name)
            knownGest = HumanoidKinect(fname, fps)
            patterns[dir].append(knownGest)

    for dir in os.listdir(trn_path):
        tst_subfolder = os.path.join(KINECT_PATH + "Testing\\", dir)
        for test_name in os.listdir(tst_subfolder):
            fname = os.path.join(tst_subfolder, test_name)
            unknownGest = HumanoidKinect(fname, fps)

            the_same_costs = []
            other_costs = []

            for theSamePattern in patterns[dir]:
                dist = compare(theSamePattern, unknownGest)
                the_same_costs.append(dist)

            for class_name, gestsLeft in patterns.items():
                if class_name != dir:
                    for knownGest in gestsLeft:
                        dist = compare(knownGest, unknownGest)
                        other_costs.append(dist)
            min_other_cost = min(other_costs)

            if max(the_same_costs) >= min_other_cost:
                supremum[dir] += 1.
            if min(the_same_costs) >= min_other_cost:
                infimum[dir] += 1

    for dir in supremum.keys():
        print(dir, supremum[dir], infimum[dir])
    total_supremum = sum(supremum.values())
    total_infimum = sum(infimum.values())
    print("inf: %d; \tsupr: %d" % (total_infimum, total_supremum))

    return total_infimum, total_supremum


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
    # compare_workout()
    # error_vs_fps()
    # compare_them_all(fps=None)
    # TODO compute the worst out-of-sample error
    the_worst_between_comparison(1)