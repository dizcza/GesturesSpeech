# coding=utf-8

from Emotion.excel_parser import parse_xls
from Emotion.csv_reader import EMOTION_PATH_PICKLES
import shutil
import os
import numpy as np


def split_data(trn_rate=0.5):
    """
     Splits pickled data into trn and tst data, w.r.t. training rate.
    :param trn_rate: how many files go for training
    """
    emotion_basket, _, _ = parse_xls(only_interest=True)
    trn_path = os.path.join(EMOTION_PATH_PICKLES, "Training")
    tst_path = os.path.join(EMOTION_PATH_PICKLES, "Testing")
    for _path in (trn_path, tst_path):
        shutil.rmtree(_path, ignore_errors=True)
        os.mkdir(_path)

    for class_name in emotion_basket.keys():
        all_files = np.array(emotion_basket[class_name])
        np.random.shuffle(all_files)
        trn_size = trn_rate * all_files.shape[0]
        trn_files, tst_files = all_files[:trn_size], all_files[trn_size:]
        for (_path, _files) in ((trn_path, trn_files), (tst_path, tst_files)):
            class_dirpath = os.path.join(_path, class_name)
            os.mkdir(class_dirpath)
            for fname in emotion_basket[class_name]:
                src = os.path.join(EMOTION_PATH_PICKLES, fname + ".pkl")
                shutil.copy(src, class_dirpath)


def compute_weights():
    pass
    # TODO do me


if __name__ == "__main__":
    split_data()