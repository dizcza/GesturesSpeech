# coding=utf-8

import numpy as np


def get_hand_labels():
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
    rhand_labels = ["R" + label[1:] for label in lhand_labels]
    hands_labels = np.concatenate([lhand_labels, rhand_labels])
    return hands_labels


def init_frame(filename):
    """
    :param filename: .c3d-file
    :return: init (relaxed) frame pertains to the filename.c3d
    """
    short_name = filename.split('/')[-1]
    initFrames = {
        "M1_02_v2.c3d": 280,
        "M2_02.c3d": 405,
        "M3_01.c3d": 785,
        "M4_01.c3d": 1470,
        "M5_01.c3d": 660,
        "M6_01.c3d": 400,
        "M7_01.c3d": 400,
        "M8_01.c3d": 500,
        "M9_01.c3d": 500,
        "C1_mcraw.c3d": 795,
        "C2_mcraw.c3d": 420,
        "C3_mcraw.c3d": 850,
        "F1_mcraw.c3d": 755,
        "F2_mcraw.c3d": 1150,
        "F3_mcraw.c3d": 815,
        "F4_mcraw.c3d": 385,
        "F5_mcraw.c3d": 1055,
        "H1_mcraw.c3d": 800,
        "H2_mcraw.c3d": 450,
        "H3_mcraw.c3d": 520,
        "H4_mcraw.c3d": 450,
        "H5_mcraw.c3d": 700,
        "H6_mcraw.c3d": 1000,
        "N1_mcraw002.c3d": 285,
        "N2_mcraw002.c3d": 440,
        "N3_mcraw001.c3d": 450,
        "S1_mcraw001.c3d": 570,
        "S2_mcraw001.c3d": 430,
        "S3_mcraw001.c3d": 580,
        "S4_mcraw001.c3d": 400,
        "S5_mcraw001.c3d": 500,
        "S6_mcraw001.c3d": 380,
        "S7_mcraw001.c3d": 450,
        "S8_mcraw001.c3d": 510,
    }
    return initFrames[short_name]


def get_hands_ids(given_labels):
    """
    :param given_labels: given markers' labels
    :return: hands ids pertains to the given labels
    """
    hands_labels = get_hand_labels()
    hands_ids = []
    del given_labels["RIDX3"]
    del given_labels["RPNK3"]
    for label in hands_labels:
        if label in given_labels:
            hands_ids.append(given_labels[label])
    return hands_ids


def get_feet_ids(given_labels):
    """
    :param given_labels: given markers' labels
    :return: feet ids pertains to the given labels
    """
    feet_labels = ["RANK", "LANK", "RTOE", "LTOE", "RHEL", "LHEL"]
    feet_ids = []
    for label in feet_labels:
        feet_ids.append(given_labels[label])
    return feet_ids


def check_for_missed_hand_labels(given_labels):
    """
    :param given_labels: markers' labels list
    :return: missed labels from its default list
    """
    hands_labels = get_hand_labels()
    missed_labels = []
    for label in hands_labels:
        if label not in given_labels:
            missed_labels.append(label)
    return missed_labels
