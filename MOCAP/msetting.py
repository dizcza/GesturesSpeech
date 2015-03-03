# coding = utf-8

from mreader import *
from numpy.linalg import norm
from Kinect.ksetting import init_info
from gDTW import compare, show_comparison, wdtw, wdtw_windowed
import os
import time
import json
import matplotlib.pyplot as plt


def compute_weights(beta):
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
            gest = HumanoidUkr(filename, frame_step=5)
            gest.compute_weights(mode="bothHands", beta=beta)
            w = gest.get_weights().tolist()
            weights_info[gest.name] = w

    MOCAP_INFO["weights"] = weights_info

    json.dump(MOCAP_INFO,  open("MOCAP_INFO.json",  'w'))
    print "New weights are saved in MOCAP_INFO.json"


def compute_between_variance(frame_step):
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
        samples.append(HumanoidUkr(fname, frame_step))

    for first_id in range(len(trn_logs)):
        for other_id in range(first_id, len(trn_folder)):
            dist = compare(samples[first_id], samples[other_id], dtw_chosen=wdtw)
            one_vs_others_var.append(dist)

    BETWEEN_VAR = np.average(one_vs_others_var)
    between_std = np.std(one_vs_others_var)

    MOCAP_INFO["between_variance"] = BETWEEN_VAR
    MOCAP_INFO["between_std"] = between_std
    json.dump(MOCAP_INFO, open("MOCAP_INFO.json", 'w'))

    info = "Done with: \n\t between-var: %g \n\t " % BETWEEN_VAR
    info += "between-std: %g\n" % between_std
    print info

    return BETWEEN_VAR, between_std


def fps_dependency():
    frame_step_range = np.arange(5, 100, 5)
    between_vars, stds = [], []
    for step in frame_step_range:
        print "FRAME STEP: \t %d" % step
        var, std = compute_between_variance(step)
        between_vars.append(var)
        stds.append(std)
    freq = 120. / frame_step_range
    plt.errorbar(freq, between_vars, stds, marker='^', ms=8)
    plt.xlabel("data freq (fps), 1/s")
    mean_std = 100. * np.average(stds)
    plt.ylabel("between variance, std=%.1f%%" % mean_std)
    plt.title("FPS dependency")
    plt.savefig("fps.png")
    plt.show()


def choose_beta(frame_step):
    """
     Choosing the best beta to yield the biggest discriminant ratio.
    """
    begin = time.time()
    beta_range = [1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    between_vars = []
    between_stds = []
    for beta in beta_range:
        print "BETA: %1.e" % beta
        compute_weights(beta)
        var, std = compute_between_variance(frame_step)
        between_vars.append(var)
        between_stds.append(std)

    print zip(beta_range, between_vars, between_stds)
    best_ratio = max(between_vars)
    ind = np.argmax(between_vars)
    best_beta = beta_range[ind]
    print "BEST RATIO: %g, w.r.t. beta = %g" % (best_ratio, best_beta)
    end = time.time()
    print "\t Duration: ~%d m" % ((end - begin) / 60.)

    plt.errorbar(np.log(beta_range), between_vars, between_stds, marker='^', ms=8)
    plt.xlabel("beta, log")
    std_mean = np.average(np.array(between_stds)/np.array(between_vars)) * 100.
    plt.ylabel("between variance,  std=%.1f%%" % std_mean)
    plt.title("Choosing the best beta")
    plt.savefig("choosing_beta_stepxxx.png")
    plt.show()


if __name__ == "__main__":
    # update_ratio()
    choose_beta(frame_step=30)
    # fps_dependency()