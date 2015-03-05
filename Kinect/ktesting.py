# coding = utf-8

from gDTW import wdtw_windowed, wdtw
from comparison import compare, show_comparison
from kreader import *
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


def collect_patterns():
    """
    :return: patters from Training set (taken as the first file in each folder)
    """
    pattern_gestures = []
    for root, _, logs in os.walk(KINECT_PATH + "Training\\"):
        if any(logs) and logs[0].endswith(".txt"):
            full_filename = os.path.join(root, logs[0])
            gest = HumanoidKinect(full_filename)
            pattern_gestures.append(gest)
    print "Took %d patterns as the first log in each training dir." % \
          len(pattern_gestures)
    return pattern_gestures


def compare_them_all():
    """
     Main testing function.
    :return: out-of-sample error
    """
    print "(Kinect project) comparing them all ..."
    patterns = collect_patterns()
    misclassified = 0.
    total_samples = 0
    for root, _, logs in os.walk(KINECT_PATH + "Testing\\"):
        for test_log in logs:
            full_filename = os.path.join(root, test_log)
            if full_filename.endswith(".txt"):
                unknown_gest = HumanoidKinect(full_filename)
                offset = []
                for known_gest in patterns:
                    dist = compare(known_gest, unknown_gest)
                    offset.append(dist)
                ind = np.argmin(offset)
                possible_gest = patterns[ind]
                # show_comparison(possible_gest, unknown_gest)

                if possible_gest.name != unknown_gest.name:
                    print "got %s, should be %s" % (possible_gest.name, unknown_gest.name)
                    misclassified += 1.

                total_samples += 1

    Etest = misclassified / float(total_samples)
    print "Etest: %g" % Etest

    return Etest


if __name__ == "__main__":
    compare_workout()
