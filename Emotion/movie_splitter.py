# coding=utf-8

import os
import shutil
from Emotion.excel_parser import parse_xls


def find_emotion(emotion_basket, src_path):
    """
    :param emotion_basket: {emotion: [fnames], }
    :param src_path: fname path
    :return: corresponding emotion (w.r.t. fname)
    """
    fname = src_path[-10:-4].replace('s', '-')
    fname = fname.replace('e', '-')
    for emotion, files in emotion_basket.items():
        if fname in files:
            return emotion
    return None


def split_movies():
    """
     Splits movies into emotion folders.
    """
    emotion_basket = parse_xls()[0]
    emotion_folder = r"D:\videos_mocap_18_11\emotions"
    shutil.rmtree(emotion_folder)

    for emotion in emotion_basket:
        em_path = os.path.join(emotion_folder, emotion)
        shutil.rmtree(em_path, ignore_errors=True)
        os.mkdir(em_path)

    movies_path = r"D:\videos_mocap_18_11\alexandr\cut"
    for mov in os.listdir(movies_path):
        if mov.endswith(".MOV"):
            src = os.path.join(movies_path, mov)
            emotion_name = find_emotion(emotion_basket, src)
            if emotion_name is not None:
                dst = os.path.join(emotion_folder, emotion_name)
                shutil.copy(src, dst)


if __name__ == "__main__":
    split_movies()
