# coding = utf-8
from reader import *


def compute_thresholds():
    """
     Updates bottom and top displacement thresholds.
    """
    for root, _, logs in os.walk(KINECT_PATH + "Training\\", topdown=False):
        for log in logs:
            full_filename = os.path.join(root, log)
            if full_filename.endswith(".txt"):
                gest = Humanoid(full_filename)
                gest.upd_thr("oneHand")
    print "Tmin: %f; \t Tmax: %f" % (Tmin, Tmax)


def compute_aver_weights():
    weights_aver = {}
    for root, _, logs in os.walk(KINECT_PATH + "Training\\", topdown=False):
        weights_within = []
        gesture_class = ""
        for log in logs:
            full_filename = os.path.join(root, log)
            if full_filename.endswith(".txt"):
                gest = Humanoid(full_filename)
                gesture_class = gest.name
                gest.compute_displacement(mode="oneHand")
                gest.compute_weights()
                w = gest.get_weights()
                weights_within.append(w)

        if gesture_class != "":
            weights_aver[gesture_class] = np.average(weights_within, axis=0).tolist()
    json.dump(weights_aver, open("WEIGHTS_AVER.json", 'w'))
    print "Saved in WEIGHTS_AVER.json"


# compute_aver_weights()