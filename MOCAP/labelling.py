# coding=utf-8

import numpy as np


def gather_labels(acq):
    """
    :param acq: BTK acquisition
    :return: label (marker) names
    """
    labels = []
    for i in range(acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(i).GetLabel().split(":")[-1]
        labels.append(label)
    return labels


def get_hand_labels(labels):
    """
    :return: (n,) array of hand labels' names
    """
    lhand_labels = np.array([
        "LFSH",     "LBSH",     "LUPA",     "LELB",     "LIEL",
        "LOWR",     "LIWR",     "LWRE",     "LIHAND",   "LOHAND",
        "LIDX1",    "LIDX2",    "LIDX3",    "LMDL1",    "LMDL2",
        "LMDL3",    "LRNG1",    "LRNG2",    "LRNG3",    "LPNK1",
        "LPNK2",    "LPNK3",    "LTHM1",    "LTHM2",    "LTHM3",
    ])
    lhand_labels = [marker for marker in lhand_labels if marker in labels]
    rhand_labels = ["R" + label[1:] for label in lhand_labels]
    hands_labels = np.concatenate([lhand_labels, rhand_labels])
    return hands_labels


def save_labelling(gest):
    """
     Saves current gesture labelling.
    :param gest: HumanoidUkr instance
    """
    print "%d markers have been saved in valid_labels.txt" % len(gest.labels)
    np.savetxt("valid_labels.txt", gest.labels, fmt="%s", delimiter='\n')

