# coding=utf-8

import cv2
from Emotion.OpenCV.window import create_window
from Emotion.OpenCV.processor import process_frame


# mp4_path = r"D:\GesturesDataset\videos_mocap_18_11\alexandr\cut\HPIM0029s3e1.MOV"
# mp4_path = r"D:\GesturesDataset\videos_mocap_18_11\alexandr\cut\HPIM0026s1e1.MOV"
mp4_path = r"d:\GesturesDataset\videos_mocap_18_11\emotions\боль\HPIM0029s1e2.MOV"


def run(index):
    capture = cv2.VideoCapture(index)
    success, frame = capture.read()

    while success:
        process_frame(frame, index)
        success, frame = capture.read()

        key = cv2.waitKey(20)
        if key == ord(' '):
            cv2.waitKey(0)
        elif key == 27:
            cv2.destroyAllWindows()
            quit()


def main(index):
    create_window()
    repeat = True

    while repeat:
        run(index)
        repeat = False if index == 0 else True

    cv2.destroyAllWindows()


# 540x360
# marker diameter (px): 13..17
if __name__ == "__main__":
    main(mp4_path)
    # main(0)
