# coding=utf-8

import os
import shutil


def separate_dataset():
    """
     0's samples will be for training
     1's samples will be for testing
    """
    splitAll = "D:\GesturesDataset\splitAll"
    trn_folder = splitAll + "\\Training"
    tst_folder = splitAll + "\\Testing"

    if not os.path.exists(trn_folder):
        os.mkdir(trn_folder)
    if not os.path.exists(tst_folder):
        os.mkdir(tst_folder)

    for c3d_file in os.listdir(splitAll):
        src = os.path.join(splitAll, c3d_file)
        if c3d_file.endswith("_sample0.c3d"):
            shutil.copy(src, trn_folder)
        elif c3d_file.endswith("_sample1.c3d"):
            shutil.copy(src, tst_folder)


separate_dataset()