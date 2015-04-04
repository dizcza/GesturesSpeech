# coding=utf-8
import csv


# coding=utf-8

import os
import pickle
import numpy as np
from numpy import genfromtxt
from pprint import pprint
import itertools
from Emotion.excel_parser import parse_xls, upd_column
import win32com.client as win32

EMOTION_PATH_CSV = r"D:\GesturesDataset\Emotion\csv"
EMOTION_PATH_PICKLES = r"D:\GesturesDataset\Emotion\pickles"
EMOTION_PATH_ACTORS = r"D:\GesturesDataset\Emotion\txt\actors"
EMOTION_PATH_EMOTIONS = r"D:\GesturesDataset\Emotion\txt\emotions"
MARKERS = 18


def clean_labels():
    """
     Removes tracked marker log files, if they aren't in valid_labels.txt
    """
    valid_labels = np.genfromtxt('valid_labels.txt', dtype='str')
    for directory in os.listdir(EMOTION_PATH_CSV):
        dir_path = os.path.join(EMOTION_PATH_CSV, directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".csv"):
                label = marker_log[:-4]
                file_path = os.path.join(dir_path, marker_log)
                if label not in valid_labels:
                    os.remove(file_path)
                    print("Removed %s from %s." % (marker_log, directory))


def verify_labels():
    """
     Verifies all samples to have the same labels.
    """
    labels_casket = {}
    valid_labels = np.genfromtxt('valid_labels.txt', dtype='str')
    for directory in os.listdir(EMOTION_PATH_CSV):
        labels_casket[directory] = []
        dir_path = os.path.join(EMOTION_PATH_CSV, directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".csv"):
                label = marker_log[:-4]
                labels_casket[directory].append(label)
        okay = True
        for us_label in labels_casket[directory]:
            okay *= us_label in valid_labels
        for their_label in valid_labels:
            okay *= their_label in labels_casket[directory]
        if not okay:
            print("shit in %s" % directory)

    gathered_labels = np.array(list(labels_casket.values()))
    the_same = gathered_labels == gathered_labels[0, :]
    assert the_same.all(), "Shit"
    print("verify_labels: \tOkay. Ready for dumping the data.")
    # valid_labels = set(gathered_labels[0, :])
    # np.savetxt("valid_labels.txt", np.array(list(valid_labels)), fmt="%s")


def dump_pickles():
    """
     Dumps data.pkl
    """
    verify_labels()

    emotions, writers = parse_xls(only_interest=True)
    check_uniqueness(writers)
    check_uniqueness(emotions)

    for directory in os.listdir(EMOTION_PATH_CSV):
        file_info = {}
        data_dic = {}
        dir_path = os.path.join(EMOTION_PATH_CSV, directory)
        labels = set([])
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".csv"):
                log_path = os.path.join(dir_path, marker_log)
                data_dic[marker_log] = np.genfromtxt(log_path, delimiter=',')
                labels.add(marker_log[:-4])
        file_info["data"] = to_array(data_dic)
        file_info["author"] = find_key_by_val(writers, directory)
        file_info["emotion"] = find_key_by_val(emotions, directory)
        file_info["labels"] = list(labels)
        fpath = os.path.join(EMOTION_PATH_PICKLES, directory + ".pkl")
        pickle.dump(file_info, open(fpath, 'wb'))
    print("DONE DUMPING.")
    upd_excel()


def to_array(data_dic):
    """
     Converts dic representation into an array.
    :param data_dic: dic of 18 listed 2d data frames for each csv marker
    :return array: (markers, frames, 2) the same 2D data
    """
    # handle missed frames
    frames = np.inf
    for vals in data_dic.values():
        if vals.shape[0] < frames:
            frames = vals.shape[0]

    array = np.empty(shape=(MARKERS, frames, 2))
    for markerID, marker in enumerate(data_dic):
        array[markerID, ::] = data_dic[marker][:frames, :]

    return array


def create_dict(folder_path):
    """
    :param folder_path: path with txt files
    :return: container of the read data from the path
    """
    casket = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            logs = np.genfromtxt(fpath, dtype='str')
            casket[fname[:-4]] = [log.strip('"') for log in logs]
    return casket


def find_key_by_val(casket, this_val):
    """
    :param casket: some dict
    :param this_val: value to find in the dict
    :return: corresponding key
    """
    for key, values in casket.items():
        if this_val in values:
            return key
    return "unknown"


def check_uniqueness(casket):
    """
    :param casket: some dict
    :return: if dict values are unique or not
    """
    junk = list(itertools.chain(*casket.values()))
    assert len(set(junk)) == len(junk), "the dict values aren't unique"


def check_missed(casket, casket_name, written_files):
    """
    :param casket: some dict
    :param casket_name: dict name
    :param written_files: given txt files
    """
    print("Checking missed logs in %s" % casket_name)
    files_in_casket = list(itertools.chain(*casket.values()))
    for fname in files_in_casket:
        if fname not in written_files:
            print("\t got %s in %s not in given files" % (fname, casket_name))
    missed = []
    for fname in written_files:
        if fname not in files_in_casket:
            missed.append(fname)

    col_name = "D"
    if casket_name == "emotions":
        col_name = "B"
    elif casket_name == "writers":
        col_name = "C"
    upd_column(col_name, missed)

    print("%d files are missed in %s" % (len(missed), casket_name))


def check_data_shapes():
    """
     Modifies excel info file with incompatible data shapes csv files.
    """
    incompatible_shapes_in = []
    for directory in os.listdir(EMOTION_PATH_CSV):
        data_dic = {}
        dir_path = os.path.join(EMOTION_PATH_CSV, directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".csv"):
                log_path = os.path.join(dir_path, marker_log)
                data_dic[marker_log] = np.genfromtxt(log_path, delimiter=',')
        shapes = np.array(list(map(len, data_dic.values())))
        the_same_shape = shapes == shapes[0]
        if not the_same_shape.all():
            incompatible_shapes_in.append(directory)
    upd_column("A", incompatible_shapes_in)


def upd_excel():
    """
     Updates missed_data.xlsx info file.
    """
    print("Updating missed_data.xlsx")
    emotions, writers = parse_xls(only_interest=False)
    check_uniqueness(writers)
    check_uniqueness(emotions)
    given_csv_files = os.listdir(EMOTION_PATH_CSV)
    check_data_shapes()
    check_missed(emotions, "emotions", given_csv_files)
    check_missed(writers, "writers", given_csv_files)

    excel = win32.gencache.EnsureDispatch('Excel.Application')
    path = os.path.join(os.getcwd(), r"missed_data.xlsx")
    wb = excel.Workbooks.Open(path)
    ws = wb.Worksheets("missed")
    ws.Range("A1").Value = "incompatible data shape (different number of frames)"
    ws.Range("B1").Value = "unknown emotion in:"
    ws.Range("C1").Value = "unknown author in:"
    wb.Save()
    wb.Close()

if __name__ == "__main__":
    # clean_labels()
    # verify_labels()
    upd_excel()
    # dump_pickles()
