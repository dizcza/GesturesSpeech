# coding=utf-8

import numpy as np
import os
from writer import split_file

UKR_ALPHABET = u"абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'"


def file_layout(filename):
    """
    :param filename: short .c3d file-name
    :return: double pairs of border frames
    """
    BORDERS_LAYOUT = {
        "D1_mcraw001": [470, (780, 1060), (1415, 1775),
                             (2100, 2480), (2790, 3100),
                             (3440, 3770), (4100, 4400),
                             (4820, 5220), (5550, 5900)],

        "D2_mcraw001": (500, (805, 1115), (1420, 1740),
                             (2050, 2345), (2710, 3045),
                             (3355, 3660), (3985, 4315),
                             (4655, 5020), (5330, 5690)),

        "D3_mcraw001": (420, (720, 1045), (1335, 1625),
                             (1935, 2250), (2570, 2875),
                             (3210, 3530), (3840, 4170),
                             (4480, 4745), (5135, 5441)),

        "D4_mcraw001": (500, (835, 1135), (1470, 1790),
                             (2100, 2420), (2745, 3080),
                             (3355, 3655), (3980, 4290),
                             (4640, 5000), (5355, 5710),
                             (6035, 6330), (6605, 6890))
    }

    DACTYL_NAMES = {
        "D1_mcraw001": u"абвгґдеє",
        "D2_mcraw001": u"жзиіїйкл",
        "D3_mcraw001": u"мнопрсту",
        "D4_mcraw001": u"фхцчшщьюя'"
    }

    short_name = filename.split('.c3d')[0]
    layout = BORDERS_LAYOUT[short_name]
    indices = np.zeros(shape=(len(layout)-1, 3), dtype=int)
    indices[0, :] = [layout[0]] + list(layout[1])
    for pairID in range(1, indices.shape[0], 1):
        indices[pairID, :] = [indices[pairID-1, 2]] + list(layout[pairID+1])

    double_pairs = []
    for sampleID in range(indices.shape[0]):
        tripple = indices[sampleID]
        double_pairs.append(zip(tripple[:-1], tripple[1:]))

    return double_pairs


def split_dactyl(folder_path):
    """
    :param folder_path: folder with dactyl's .c3d file-names
    """
    for c3d_filename in os.listdir(folder_path):
        if c3d_filename.endswith(".c3d"):
            double_pairs = file_layout(c3d_filename)
            split_file(folder_path, c3d_filename, double_pairs)


dactyl_folder = "D:\GesturesDataset\Dactyl\\"
# split_dactyl(dactyl_folder)