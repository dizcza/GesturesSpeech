# coding = utf-8
from gDTW import compare, show_comparison
from reader import *


def in_folder():
    """
     Reads all logs in the chosen folder.
    """
    folder = KINECT_PATH + "Training\\LeftHandSwipeRight\\"
    for log in os.listdir(folder):
        if log.endswith(".txt"):
            gest = Humanoid(folder + log)
            # gest.show_displacement("oneHand")
            gest.compute_displacement("oneHand")
            gest.compute_weights()
            gest.set_weights()
            # gest.show_displacement("oneHand")
            # gest.animate()
    # print "Tmin: %f; \t Tmax: %f" % (Tmin, Tmax)


def compare_test():
    folder = KINECT_PATH + "Training\\LeftHandSwipeRight\\"
    gestures = []
    for log in os.listdir(folder):
        if log.endswith(".txt"):
            gest = Humanoid(folder + log)
            gest.set_weights()
            gestures.append(gest)

    show_comparison(gestures[5], gestures[3])


in_folder()