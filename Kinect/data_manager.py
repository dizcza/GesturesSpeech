# coding=utf-8

import os
import rarfile
from tools.downloader import download_url

KINECT_DATABASE_URL = "http://datascience.sehir.edu.tr/visapp2013/WeightedDTW-Visapp2013-DB.rar"
OUTPUT_DATA_DIR = "_data"


def unrar(rar_path, destination=None):
    """
     Unpacks an archive.

    :param rar_path: a path to .rar archive
    :param destination: destination folder
    """
    rf = rarfile.RarFile(rar_path)
    rf.extractall(path=destination)


def load_database():
    """
     Loads Kinect database if not loaded yet.
     Before loading database check if the url still exists.

    :return: a path to Kinect project data
    """
    _data_path = os.path.dirname(__file__)
    _data_path = os.path.join(_data_path, OUTPUT_DATA_DIR)
    if not os.path.exists(_data_path):
        fname = download_url(KINECT_DATABASE_URL)
        unrar(fname, destination=_data_path)
        os.remove(fname)
        print("Kinect data is stored in %s" % _data_path)
    return _data_path


if __name__ == "__main__":
    load_database()
