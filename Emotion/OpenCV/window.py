# coding=utf-8

import cv2
import numpy as np
from Emotion.OpenCV.constants import HSV, PI, MAX_COLOR_VAL


def nothing(*args, **kwargs):
    pass


def create_window():
    """
     Creates a window with HSV thresholds.
    """
    cv2.namedWindow(HSV.WIN_NAME, 1)

    cv2.createTrackbar(HSV.HUE_MIN_NAME, HSV.WIN_NAME, 0, PI, nothing)
    cv2.createTrackbar(HSV.HUE_MAX_NAME, HSV.WIN_NAME, PI, PI, nothing)

    cv2.createTrackbar(HSV.SAT_MIN_NAME, HSV.WIN_NAME, 0, MAX_COLOR_VAL, nothing)
    cv2.createTrackbar(HSV.SAT_MAX_NAME, HSV.WIN_NAME, MAX_COLOR_VAL, MAX_COLOR_VAL, nothing)

    cv2.createTrackbar(HSV.VAL_MIN_NAME, HSV.WIN_NAME, 0, MAX_COLOR_VAL, nothing)
    cv2.createTrackbar(HSV.VAL_MAX_NAME, HSV.WIN_NAME, 40, MAX_COLOR_VAL, nothing)


def get_lower_upper():
    """
    :return: lower & upper black ranges from a trackbar
    """
    hue_lower = cv2.getTrackbarPos(HSV.HUE_MIN_NAME, HSV.WIN_NAME)
    sat_lower = cv2.getTrackbarPos(HSV.SAT_MIN_NAME, HSV.WIN_NAME)
    val_lower = cv2.getTrackbarPos(HSV.VAL_MIN_NAME, HSV.WIN_NAME)

    hue_upper = cv2.getTrackbarPos(HSV.HUE_MAX_NAME, HSV.WIN_NAME)
    sat_upper = cv2.getTrackbarPos(HSV.SAT_MAX_NAME, HSV.WIN_NAME)
    val_upper = cv2.getTrackbarPos(HSV.VAL_MAX_NAME, HSV.WIN_NAME)

    lower_black = np.array([hue_lower, sat_lower, val_lower])
    upper_black = np.array([hue_upper, sat_upper, val_upper])

    return lower_black, upper_black


def separate_channels(hsv):
    """
    :param hsv: hsv image
    :return: a sequence of h, s, v images
    """
    lower_black, upper_black = get_lower_upper()

    channel_streams = []
    for ch in range(3):
        lower_ch = lower_black[ch]
        upper_ch = upper_black[ch]
        ch_frame = hsv[:,:,ch]
        ret, mask_ch = cv2.threshold(ch_frame, lower_ch, upper_ch, cv2.THRESH_BINARY)
        ch_masked = cv2.bitwise_and(ch_frame, ch_frame, mask=mask_ch)
        channel_streams.append(ch_masked)
    channel_streams = np.hstack(channel_streams)

    return channel_streams
