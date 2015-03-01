# coding = utf-8

from kreader import *
from gDTW import compare, show_comparison
import os
import time
import json
import matplotlib.pyplot as plt


def compute_weights(beta, mode="oneHand"):
    """
     Computes aver weights from the Training dataset.
    """
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
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
                gest.compute_weights(beta)
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
    
    print "COMPUTING WITHIN VARIANCE"
    
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    one_vs_the_same_var = []
    for root, _, logs in os.walk(KINECT_PATH + "Training\\", topdown=False):
        log_examples = [_log for _log in logs if _log.endswith(".txt")]

        if not any(log_examples):
            continue

        while len(log_examples) > 1:
            first_log = os.path.join(root, log_examples[0])
            firstGest = HumanoidKinect(first_log)

            for another_log in log_examples[1:]:
                full_filename = os.path.join(root, another_log)
                goingGest = HumanoidKinect(full_filename)
                dist = compare(firstGest, goingGest)[0]
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
    
    print "COMPUTING BETWEEN VARIANCE"
    
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    one_vs_others_var = []
    dirs = os.listdir(KINECT_PATH + "Training")
    roots = [KINECT_PATH + "Training\\" + one for one in dirs]

    while len(roots) > 1:
        first_dir = roots[0]
        dirs_left = roots[1:]
        print "\tComparing %s with the others ..." % first_dir.split("\\")[-1]

        for first_log in os.listdir(first_dir):
            first_log_full = os.path.join(first_dir, first_log)
            firstGest = HumanoidKinect(first_log_full)

            for other_dir in dirs_left:
                indir_var = []
                for other_log in os.listdir(other_dir):
                    full_filename = os.path.join(other_dir, other_log)
                    goingGest = HumanoidKinect(full_filename)
                    dist = compare(firstGest, goingGest)[0]
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
     Initializes empty KINECT_INFO.
    """
    KINECT_INFO = {
        "path": KINECT_PATH,
        "weights": {},
        "beta": 0.1,
        "within_variance": 0.0407470426572,
        "between_variance": 0.665986610366,
        "within_std": .0,
        "between_std": .0,
        "d-ratio": 16.34441586273796
    }
    json.dump(KINECT_INFO, open("KINECT_INFO.json", 'w'))


def print_ratio():
    """
     Prints out discriminant ration from KINECT_INFO.json
    """
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    print "Loaded discriminant ratio: %g" % KINECT_INFO["d-ratio"]


def update_ratio(beta=1e-4):
    """
     Updates weights, within and between variance for the given beta param.
    :param beta: to be choosing to yield the biggest ratio
    :return: discriminant ratio
    """
    if not os.path.exists("KINECT_INFO.json"):
        init_info()

    compute_weights(beta)
    compute_within_variance()
    compute_between_variance()
    KINECT_INFO = json.load(open("KINECT_INFO.json", 'r'))
    between_var = KINECT_INFO["between_variance"]
    within_var = KINECT_INFO["within_variance"]

    KINECT_INFO["d-ratio"] = between_var / within_var
    print "(!) New discriminant ratio: %g" % KINECT_INFO["d-ratio"]
    json.dump(KINECT_INFO, open("KINECT_INFO.json", 'w'))

    return KINECT_INFO["d-ratio"]


def choose_beta():
    """
     Choosing the best beta to yield the biggest discriminant ratio.
    """
    begin = time.time()
    possible_betas = [1e-6, 1e-4, 1e-2, 1., 1e1, 1e2, 1e3]
    obtained_ratios = []
    for beta in possible_betas:
        ratio = update_ratio(beta)
        obtained_ratios.append(ratio)

    print zip(possible_betas, obtained_ratios)
    best_ratio = max(obtained_ratios)
    ind = np.argmax(obtained_ratios)
    best_beta = possible_betas[ind]
    print "BEST RATIO: %g, w.r.t. beta = %g" % (best_ratio, best_beta)
    end = time.time()
    print "\t Duration: ~%d m" % (end - begin) / 60.

    plt.plot(np.log(possible_betas), obtained_ratios, 'o')
    plt.xlabel("betas, log")
    plt.ylabel("discriminant ratio")
    plt.title("Picking up the best beta, which yields the biggest ratio")
    plt.show()


# update_ratio()
choose_beta()


