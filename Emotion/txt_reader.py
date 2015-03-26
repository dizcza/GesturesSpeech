# coding=utf-8

import os
import cPickle as pickle
import numpy as np
from pprint import pprint
import itertools


EMOTION_PATH_TXT = "D:\GesturesDataset\Emotion\\txt"
MARKERS = 18


def clean_invalid_txt():
    """
     Cleans duplicated and corrupted txt data.
    """
    for directory in os.listdir(EMOTION_PATH_TXT + "\\old_format"):
        data = {}
        dir_path = os.path.join(EMOTION_PATH_TXT, directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".txt"):
                log_path = os.path.join(dir_path, marker_log)
                if " (2)" in marker_log:
                    os.remove(log_path)
                    continue
                try:
                    data[marker_log] = np.loadtxt(log_path, delimiter='\n')
                except ValueError:
                    os.remove(log_path)


def verify_labels():
    """
     Verifies all samples to have the same labels.
    """
    labels_casket = {}
    for directory in os.listdir(EMOTION_PATH_TXT + "\\old_format"):
        labels_casket[directory] = []
        dir_path = os.path.join(EMOTION_PATH_TXT, "old_format", directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".txt"):
                label = marker_log[:-6]
                labels_casket[directory].append(label)
    gathered_labels = np.array(list(labels_casket.values()))
    the_same = gathered_labels == gathered_labels[0, :]
    okay = the_same.all()
    if okay:
        print("Okay")
    else:
        print("Shit")
        raise ValueError
    valid_labels = set(gathered_labels[0, :])
    np.savetxt("valid_labels.txt", np.array(list(valid_labels)), fmt="%s")


def to_array(data_dic):
    """
     Converts dic representation into an array.
    :param data_dic: dic of listed data frames for each marker
    :return array: (markers, frames, 2) the same 2D data
    """
    # handle missed frames
    frames = np.inf
    for vals in data_dic.itervalues():
        if len(vals) < frames:
            frames = len(vals)

    array = np.empty(shape=(MARKERS, frames, 2))

    keys_x = [key for key in data_dic.keys() if key[-6:] == "_x.txt"]
    keys_y = [key for key in data_dic.keys() if key[-6:] == "_y.txt"]
    key_pairs = zip(keys_x, keys_y)
    for markerID, (marker_x, marker_y) in enumerate(key_pairs):
        array[markerID, :, 0] = data_dic[marker_x][:frames]
        array[markerID, :, 1] = data_dic[marker_y][:frames]

    return array


def dump_data():
    """
     Dumps data.npy
    """
    data_folder = os.path.join(EMOTION_PATH_TXT, "data")
    for directory in os.listdir(EMOTION_PATH_TXT + "\\old_format"):
        data_dic = {}
        dir_path = os.path.join(EMOTION_PATH_TXT, "old_format", directory)
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".txt"):
                log_path = os.path.join(dir_path, marker_log)
                data_dic[marker_log] = np.loadtxt(log_path, delimiter='\n')
        data = to_array(data_dic)
        file_path = os.path.join(data_folder, directory)
        np.save(file_path, data)


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
    if len(set(junk)) != len(junk):
        print("the dict values isn't unique")
        raise AttributeError


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
            print("got %s in %s not in given files" % (fname, casket_name))
    for fname in written_files:
        if fname not in files_in_casket:
            print(fname)


def dump_pickles():
    """
     Dumps data.pkl
    """
    writers = create_dict("D:\GesturesDataset\Emotion\\txt\\actors")
    emotions = create_dict("D:\GesturesDataset\Emotion\\txt\emotions")
    check_uniqueness(writers)
    check_uniqueness(emotions)
    txt_dirs = os.listdir(EMOTION_PATH_TXT + "\\old_format")
    # check_missed(writers, "writers", txt_dirs)
    check_missed(emotions, "emotions", txt_dirs)

    pickle_folder = os.path.join(EMOTION_PATH_TXT, "data")
    for directory in txt_dirs:
        file_info = {}
        data_dic = {}
        dir_path = os.path.join(EMOTION_PATH_TXT, "old_format", directory)
        labels = set([])
        for marker_log in os.listdir(dir_path):
            if marker_log.endswith(".txt"):
                log_path = os.path.join(dir_path, marker_log)
                data_dic[marker_log] = np.loadtxt(log_path, delimiter='\n')
                labels.add(marker_log[:-6])
        file_info["data"] = to_array(data_dic)
        file_info["author"] = find_key_by_val(writers, directory)
        file_info["emotion"] = find_key_by_val(emotions, directory)
        file_info["labels"] = list(labels)
        fpath = os.path.join(pickle_folder, directory + ".pkl")
        pickle.dump(file_info, open(fpath, 'wb'))


if __name__ == "__main__":
    dump_pickles()