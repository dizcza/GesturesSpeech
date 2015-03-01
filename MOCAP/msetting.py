# coding = utf-8

from mreader import *
from gDTW import compare, show_comparison
import os
import time
import json
import matplotlib.pyplot as plt


def compute_weights(beta=1e-2):
    """
     Computes aver weights from the Training dataset.
    """
    try:
        MOCAP_INFO = json.load(open("MOCAP_INFO.json", 'r'))
    except ValueError:
        MOCAP_INFO = init_info()

    trn_folder = "D:\GesturesDataset\splitAll\Training"
    weights_info = {}
    
    for trn_file in os.listdir(trn_folder):
        if trn_file.endswith(".c3d"):
            filename = os.path.join(trn_folder, trn_file)
            gest = HumanoidUkr(filename)
            gest.compute_weights(beta)
            w = gest.get_weights().tolist()
            weights_info[gest.name] = w

    MOCAP_INFO["weights"] = weights_info

    json.dump(MOCAP_INFO,  open("MOCAP_INFO.json",  'w'))
    print "New weights are saved in MOCAP_INFO.json"


def compute_between_variance():
    """
     Computes aver between variance from the Training dataset.
    """
    
    print "COMPUTING BETWEEN VARIANCE"
    
    MOCAP_INFO = json.load(open("MOCAP_INFO.json", 'r'))
    trn_folder = "D:\GesturesDataset\splitAll\Training"
    one_vs_others_var = []

    samples = []
    trn_logs = os.listdir(trn_folder)
    for log in trn_logs:
        fname = os.path.join(trn_folder, log)
        samples.append(HumanoidUkr(fname))

    for first_id in range(len(trn_logs)):
        for other_id in range(first_id, len(trn_folder)):
            dist = compare(samples[first_id], samples[other_id])
            one_vs_others_var.append(dist)

    BETWEEN_VAR = np.average(one_vs_others_var)
    between_std = np.std(one_vs_others_var)

    MOCAP_INFO["between_variance"] = BETWEEN_VAR
    MOCAP_INFO["between_std"] = between_std
    json.dump(MOCAP_INFO, open("MOCAP_INFO.json", 'w'))

    info = "Done with: \n\t between-var: %g \n\t " % BETWEEN_VAR
    info += "between-std: %g\n" % between_std
    print info


def init_info():
    """
     Initializes empty MOCAP_INFO.
    """
    MOCAP_INFO = {
        "path": None,
        "weights": {},
        "beta": 1e-2,
        "within_variance": 1.,
        "between_variance": 1.,
        "within_std": 0.,
        "between_std": 0.,
        "d-ratio": None
    }
    json.dump(MOCAP_INFO, open("MOCAP_INFO.json", 'w'))
    return MOCAP_INFO


def print_ratio():
    """
     Prints out discriminant ration from MOCAP_INFO.json
    """
    MOCAP_INFO = json.load(open("MOCAP_INFO.json", 'r'))
    print "Loaded discriminant ratio: %g" % MOCAP_INFO["d-ratio"]


def update_ratio(beta=1e-4):
    """
     Updates weights, within and between variance for the given beta param.
    :param beta: to be choosing to yield the biggest ratio
    :return: discriminant ratio
    """
    if not os.path.exists("MOCAP_INFO.json"):
        init_info()

    compute_weights(beta)
    compute_between_variance()
    MOCAP_INFO = json.load(open("MOCAP_INFO.json", 'r'))
    between_var = MOCAP_INFO["between_variance"]

    MOCAP_INFO["d-ratio"] = between_var
    print "(!) New discriminant ratio: %g" % MOCAP_INFO["d-ratio"]
    json.dump(MOCAP_INFO, open("MOCAP_INFO.json", 'w'))

    return MOCAP_INFO["d-ratio"]


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
# choose_beta()
# compute_weights()
compute_between_variance()