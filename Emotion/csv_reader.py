# coding=utf-8

import os
import shutil
import pickle
import numpy as np
from numpy import genfromtxt
from pprint import pprint
import itertools
from Emotion.excel_parser import parse_xls, upd_column, parse_whole_xls
import win32com.client as win32

EMOTION_PATH_CSV = r"D:\GesturesDataset\Emotion\csv"
EMOTION_PATH_PICKLES = r"D:\GesturesDataset\Emotion\pickles"
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
    print("verify_labels: \tOKAY. Ready to dump data.")
    # valid_labels = set(gathered_labels[0, :])
    # np.savetxt("valid_labels.txt", np.array(list(valid_labels)), fmt="%s")


def cut_frames_if_needed(gathered_data, boundaries_basket, file_name):
    """
    :param np.array gathered_data: (#markers, #frames, 2) gathered data from csv
    :param boundaries_basket: a dic with (begin, end) frames to be cut
                              for each file name
    :param file_name: XX-X-X str format
    :return: updated data
    """
    if file_name in boundaries_basket and boundaries_basket[file_name] is not None:
        a_comment = str(boundaries_basket[file_name])
        if "begin from " in a_comment:
            a_comment = a_comment.strip("begin from ")
            begin = int(a_comment.split(' ')[0])
        else:
            begin = 0
        if "stop on " in a_comment:
            a_comment = a_comment.split("stop on ")[-1]
            end = int(a_comment)
        else:
            end = gathered_data.shape[1]
        print("cut %s: \t %s" % (file_name, str(boundaries_basket[file_name])))
        gathered_data = gathered_data[:, begin:end, :]

    return gathered_data


def dump_pickles():
    """
     Dumps data.pkl
    """
    msg = "#################################################################\n" \
          "#                   Prepare to dump pickles                     #\n" \
          "#################################################################"
    print(msg)
    clean_labels()
    verify_labels()

    emotions, writers, boundaries = parse_xls()
    check_uniqueness(writers)
    check_uniqueness(emotions)
    print("*** Found %d unique emotions." % len(emotions))

    msg = "#################################################################\n" \
          "#                 Dumping the data: csv --> pkl                 #\n" \
          "#################################################################"
    print(msg)
    for directory in os.listdir(EMOTION_PATH_CSV):
        file_info = {}
        data_dic = {}
        dir_path = os.path.join(EMOTION_PATH_CSV, directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".csv"):
                log_path = os.path.join(dir_path, marker_log)
                data_dic[marker_log[:-4]] = np.genfromtxt(log_path, delimiter=',')
        gathered_data = to_array(data_dic)
        file_info["data"] = cut_frames_if_needed(gathered_data, boundaries, directory)
        file_info["author"] = find_key_by_val(writers, directory)
        file_info["emotion"] = find_key_by_val(emotions, directory)
        file_info["labels"] = list(data_dic.keys())
        fpath = os.path.join(EMOTION_PATH_PICKLES, directory + ".pkl")
        pickle.dump(file_info, open(fpath, 'wb'))
    upd_excel()
    split_data()


def to_array(data_dic):
    """
     Converts dic representation into an array.
    :param data_dic: dic of 18 listed 2d data frames for each csv marker
    :return array: (#markers, #frames, 2) the same 2D data
    """
    frames = next(iter(data_dic.values())).shape[0]
    array = np.empty(shape=(MARKERS, frames, 2), dtype=np.float64)
    for jointID, marker in enumerate(data_dic):
        array[jointID, ::] = data_dic[marker]
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
    return "undefined"


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

    my_scope = {"emotions": "B", "writers": "C"}
    col_name = my_scope[casket_name]
    upd_column(col_name, missed)

    print("%d files are missed in %s" % (len(missed), casket_name))


def check_data_shapes():
    """
     Modifies excel info file with incompatible data shapes csv files.
    """
    incompatible_shapes_in = set([])
    for directory in os.listdir(EMOTION_PATH_CSV):
        data_dic = {}
        dir_path = os.path.join(EMOTION_PATH_CSV, directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".csv"):
                log_path = os.path.join(dir_path, marker_log)
                data_dic[marker_log] = np.genfromtxt(log_path, delimiter=',')
        for array in data_dic.values():
            if np.isnan(array).any():
                incompatible_shapes_in.add(directory)
    msg = "check_data_shapes: "
    if any(incompatible_shapes_in):
        msg += "found %d csv files with np.nan" % len(incompatible_shapes_in)
    else:
        msg += "OKAY. All data shapes have the same dimension."
    print(msg)
    upd_column("A", incompatible_shapes_in)


def upd_excel():
    """
     Updates missed_data.xlsx info file.
    """
    msg = "#################################################################\n" \
          "#                 Updating missed_data.xlsx                     #\n" \
          "#################################################################"
    print(msg)
    emotions, writers, _ = parse_whole_xls()
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


def split_data(trn_rate=0.5):
    """
     Splits pickled data into trn and tst data, w.r.t. training rate.
    :param trn_rate: how many files go for training
    """
    emotion_basket, _, _ = parse_xls()
    trn_path = os.path.join(EMOTION_PATH_PICKLES, "Training")
    tst_path = os.path.join(EMOTION_PATH_PICKLES, "Testing")
    for _path in (trn_path, tst_path):
        shutil.rmtree(_path, ignore_errors=True)
        os.mkdir(_path)

    for class_name in emotion_basket.keys():
        all_files = np.array(emotion_basket[class_name])
        np.random.shuffle(all_files)
        trn_size = int(trn_rate * all_files.shape[0])
        trn_files, tst_files = all_files[:trn_size], all_files[trn_size:]
        for (_path, _files) in ((trn_path, trn_files), (tst_path, tst_files)):
            class_dirpath = os.path.join(_path, class_name)
            os.mkdir(class_dirpath)
            for fname in _files:
                src = os.path.join(EMOTION_PATH_PICKLES, fname + ".pkl")
                shutil.copy(src, class_dirpath)


if __name__ == "__main__":
    # clean_labels()
    # verify_labels()
    # upd_excel()
    dump_pickles()
    # check_data_shapes()
