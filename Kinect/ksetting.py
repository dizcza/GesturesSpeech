# coding = utf-8

from kreader import HumanoidKinect, KINECT_PATH
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from gDTW import wdtw_windowed, wdtw
from comparison import compare, show_comparison
import os
import json
import time
from pprint import pprint


def compute_weights(mode, beta):
    """
     Computes aver weights from the Training dataset.
    """
    try:
        KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    except ValueError:
        KINECT_INFO = init_info()

    weights_aver = {}
    for root, _, logs in os.walk(KINECT_PATH + "Training\\"):
        weights_within = []
        gesture_class = ""
        for log in logs:
            full_filename = os.path.join(root, log)
            if full_filename.endswith(".txt"):
                gest = HumanoidKinect(full_filename)
                gesture_class = gest.name
                gest.compute_displacement(mode)
                gest.compute_weights(mode, beta)
                w = gest.get_weights()
                weights_within.append(w)

        if gesture_class != "":
            weights_aver[gesture_class] = np.average(weights_within, axis=0).tolist()
    KINECT_INFO["weights"] = weights_aver

    json.dump(KINECT_INFO,  open("KINECT_INFO.json",  'w'))
    print "New weights are saved in KINECT_INFO.json"


def compute_within_variance():
    """
     Computes aver within variance from the Training dataset.
    """
    
    print "(Kinect project) COMPUTING WITHIN VARIANCE"
    
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    one_vs_the_same_var = []
    for root, _, logs in os.walk(KINECT_PATH + "Training\\", topdown=False):
        log_examples = [_log for _log in logs if _log.endswith(".txt")]

        # if not any(log_examples):
        #     continue

        while len(log_examples) > 1:
            first_log = os.path.join(root, log_examples[0])
            firstGest = HumanoidKinect(first_log)

            for another_log in log_examples[1:]:
                full_filename = os.path.join(root, another_log)
                goingGest = HumanoidKinect(full_filename)
                # print first_log, another_log
                dist = compare(firstGest, goingGest)
                one_vs_the_same_var.append(dist)

            log_examples.pop(0)

    WITHIN_VAR = np.average(one_vs_the_same_var)
    within_std = np.std(one_vs_the_same_var)

    KINECT_INFO["within_variance"] = WITHIN_VAR
    KINECT_INFO["within_std"] = within_std
    json.dump(KINECT_INFO, open("KINECT_INFO.json", 'w'))

    info = "Done with: \n\t within-var: %g \n\t " % WITHIN_VAR
    info += "within-std: %g\n" % within_std
    print info


def compute_between_variance():
    """
     Computes aver between variance from the Training dataset.
    """
    
    print "(Kinect project) COMPUTING BETWEEN VARIANCE"
    
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    one_vs_others_var = []
    dirs = os.listdir(KINECT_PATH + "Training")
    roots = [KINECT_PATH + "Training\\" + one for one in dirs]

    while len(roots) > 1:
        first_dir = roots[0]
        dirs_left = roots[1:]
        # print "\tComparing %s with the others ..." % first_dir.split("\\")[-1]

        for first_log in os.listdir(first_dir):
            first_log_full = os.path.join(first_dir, first_log)
            firstGest = HumanoidKinect(first_log_full)

            for other_dir in dirs_left:
                for other_log in os.listdir(other_dir):
                    full_filename = os.path.join(other_dir, other_log)
                    goingGest = HumanoidKinect(full_filename)
                    dist = compare(firstGest, goingGest)
                    one_vs_others_var.append(dist)

        roots.pop(0)

    BETWEEN_VAR = np.average(one_vs_others_var)
    between_std = np.std(one_vs_others_var)

    KINECT_INFO["between_variance"] = BETWEEN_VAR
    KINECT_INFO["between_std"] = between_std
    json.dump(KINECT_INFO, open("KINECT_INFO.json", 'w'))

    info = "Done with: \n\t between-var: %g \n\t " % BETWEEN_VAR
    info += "between-std: %g\n" % between_std
    print info


def init_info():
    """
     Initializes empty MOCAP_INFO.
    """
    _INFO = {
        "weights": {},
        "beta": 1e-2,
        "within_variance": 1.,
        "between_variance": 1.,
        "within_std": 0.,
        "between_std": 0.,
        "d-ratio": None,
        "d-ratio-std": None
    }
    return _INFO


def update_ratio(beta):
    """
     Updates weights, within and between variance for the given beta param.
    :param beta: to be choosing to yield the biggest ratio
    :return: discriminant ratio
    """
    compute_weights(mode="oneHand", beta=beta)
    compute_within_variance()
    compute_between_variance()
    _INFO = json.load(open("KINECT_INFO.json", 'r'))

    within_var = _INFO["within_variance"]
    within_std = _INFO["within_std"]
    between_var = _INFO["between_variance"]
    between_std = _INFO["between_std"]

    sigma_b = between_std / within_var
    sigma_w = within_std * between_var / within_var ** 2
    ratio_std = norm([sigma_b, sigma_w])

    _INFO["d-ratio"] = between_var / within_var
    _INFO["d-ratio-std"] = ratio_std

    print "(!) New discriminant ratio: %g" % _INFO["d-ratio"]
    json.dump(_INFO, open("KINECT_INFO.json", 'w'))

    return _INFO["d-ratio"], ratio_std


def choose_beta(fps):
    """
     Choosing the best beta to yield the biggest discriminant ratio.
    """
    print "(Kinect project) choosing the beta with fps = %s" % fps
    begin = time.time()
    beta_range = [1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    gained_ratios = []
    gained_stds = []
    for beta in beta_range:
        print "BETA: %f" % beta
        ratio, std = update_ratio(beta)
        gained_ratios.append(ratio)
        gained_stds.append(std)

    print zip(beta_range, gained_ratios, gained_stds)
    best_ratio = max(gained_ratios)
    ind = np.argmax(gained_ratios)
    best_beta = beta_range[ind]
    print "BEST RATIO: %g, w.r.t. beta = %g" % (best_ratio, best_beta)
    end = time.time()
    print "\t Duration: ~%d m" % ((end - begin) / 60.)

    plt.errorbar(np.log(beta_range), gained_ratios, gained_stds,
                 linestyle='None', marker='^', ms=8)
    plt.xlabel("beta, log")
    plt.ylabel("discriminant ratio")
    plt.title("Choosing the best beta")
    plt.savefig("png/choosing_beta.png")
    plt.show()


def compute_lowest_weights_discrepancy(fps):
    """
     Computes the lowest weights discrepancy threshold for KINECT project
     as the biggest one from the training folder.
    :param fps: data fps to be set during reading the files
    """
    print "(Kinect project) compute_weights_discrepancy_infimum"
    weights_discr = {}
    for root, dirs, logs in os.walk(KINECT_PATH + "Training\\", topdown=False):
        for class_name in dirs:
            weights_discr[class_name] = []

    for root, _, logs in os.walk(KINECT_PATH + "Training\\", topdown=False):
        log_examples = [_log for _log in logs if _log.endswith(".txt")]
        while len(log_examples) > 1:
            first_log = os.path.join(root, log_examples[0])
            firstGest = HumanoidKinect(first_log, fps)
            gesture_class = firstGest.name
            for another_log in log_examples[1:]:
                full_filename = os.path.join(root, another_log)
                goingGest = HumanoidKinect(full_filename, fps)
                if gesture_class != goingGest.name:
                    raise KeyError
                discr_snapshot = firstGest.get_weights_discrepancy(goingGest, mode="bothHands")
                weights_discr[gesture_class].append(discr_snapshot)
            log_examples.pop(0)

    LOWEST_WEIGHTS_DISCR_THR = {}
    for class_name, discr_array in weights_discr.iteritems():
        LOWEST_WEIGHTS_DISCR_THR[class_name] = max(discr_array)
    pprint(LOWEST_WEIGHTS_DISCR_THR)
    print "The lowest weights dicr thr: %f" % max(LOWEST_WEIGHTS_DISCR_THR.values())


if __name__ == "__main__":
    # update_ratio(beta=1e-2)
    # choose_beta(fps=None)
    compute_lowest_weights_discrepancy(fps=None)
