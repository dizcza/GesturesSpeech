# coding=utf-8

import cv2
import numpy as np
from numpy.linalg import norm
import json
import os

# mp4_path = r"D:\GesturesDataset\videos_mocap_18_11\alexandr\cut\HPIM0029s3e1.MOV"
mp4_path = r"D:\GesturesDataset\videos_mocap_18_11\alexandr\cut\HPIM0026s1e1.MOV"

BOUNDARIES = (
    (150, 180),
    (0, 255),
    (0, 255)
)

MAX_COLOR_VAL = 255

BLUE = (MAX_COLOR_VAL, 0, 0)
GREEN = (0, MAX_COLOR_VAL, 0)
RED = (0, 0, MAX_COLOR_VAL)

COLORS = BLUE, GREEN, RED

PI = 180

both_eyes_cascade = cv2.CascadeClassifier('haar_both_eyes.xml')
eye_cascade = cv2.CascadeClassifier('haar_eye.xml')

MIN_MARKER_RADIUS_SOFT = 4
MIN_MARKER_RADIUS_HARD = 4
MAX_MARKER_RADIUS = 7

MIN_MARKER_AREA_SOFT = 3.14 * MIN_MARKER_RADIUS_SOFT ** 2
MIN_MARKER_AREA_HARD = 3.14 * MIN_MARKER_RADIUS_HARD ** 2
MAX_MARKER_AREA = 3.14 * MAX_MARKER_RADIUS ** 2

MARKERS_NUM = 18

V_MAX_LOWER = 200


class HSV(object):
    WIN_NAME = "hsv mask"

    HUE_MIN_NAME = "hue min"
    SAT_MIN_NAME = "sat min"
    VAL_MIN_NAME = "val min"

    HUE_MAX_NAME = "hue max"
    SAT_MAX_NAME = "sat max"
    VAL_MAX_NAME = "val max"

    HUE_DEFAULT = 150
    SAT_DEFAULT = 0
    VAL_DEFAULT = 0


class HoughCircles(object):
    WIN_NAME = "hough circles"

    DP_NAME = "dp"
    MIN_DIST_NAME = "minDist"
    PARAM_1_NAME = "param1"
    PARAM_2_NAME = "param2"
    MIN_RADIUS_NAME = "minRadius"
    MAX_RADIUS_NAME = "maxRadius"

    DP_DEFAULT = 1
    MIN_DIST_DEFAULT = 20
    PARAM_1_DEFAULT = 100
    PARAM_2_DEFAULT = 3
    MIN_RADIUS_DEFAULT = 4
    MAX_RADIUS_DEFAULT = 8





def resize_in_half(frame):
    return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)





def separate_channels(hsv):
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


def get_black_pixels(hsv):
    lower_black, upper_black = get_lower_upper()
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    return mask_black


def get_black_pixel_indices(hsv, inverse=False):
    mask_black = get_black_pixels(hsv)
    if inverse:
        return np.where(mask_black == 0)
    else:
        return np.where(mask_black > 0)


def hough_circles(frame, gray):
    dp, minDist, param1, param2, minRadius, maxRadius = get_hough_params()
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               dp=dp,
                               minDist=minDist,
                               param1=param1,
                               param2=param2,
                               minRadius=minRadius,
                               maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],GREEN,2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,RED,3)

        cv2.imshow('detected circles', frame)


def get_lower_upper():
    hue_lower = cv2.getTrackbarPos(HSV.HUE_MIN_NAME, HSV.WIN_NAME)
    sat_lower = cv2.getTrackbarPos(HSV.SAT_MIN_NAME, HSV.WIN_NAME)
    val_lower = cv2.getTrackbarPos(HSV.VAL_MIN_NAME, HSV.WIN_NAME)

    hue_upper = cv2.getTrackbarPos(HSV.HUE_MAX_NAME, HSV.WIN_NAME)
    sat_upper = cv2.getTrackbarPos(HSV.SAT_MAX_NAME, HSV.WIN_NAME)
    val_upper = cv2.getTrackbarPos(HSV.VAL_MAX_NAME, HSV.WIN_NAME)

    lower_black = np.array([hue_lower, sat_lower, val_lower])
    upper_black = np.array([hue_upper, sat_upper, val_upper])

    return lower_black, upper_black


def get_hough_params():
    dp = cv2.getTrackbarPos(HoughCircles.DP_NAME, HoughCircles.WIN_NAME)
    minDist = cv2.getTrackbarPos(HoughCircles.MIN_DIST_NAME, HoughCircles.WIN_NAME)
    param1 = cv2.getTrackbarPos(HoughCircles.PARAM_1_NAME, HoughCircles.WIN_NAME)
    param2 = cv2.getTrackbarPos(HoughCircles.PARAM_2_NAME, HoughCircles.WIN_NAME)
    minRadius = cv2.getTrackbarPos(HoughCircles.MIN_RADIUS_NAME, HoughCircles.WIN_NAME)
    maxRadius = cv2.getTrackbarPos(HoughCircles.MAX_RADIUS_NAME, HoughCircles.WIN_NAME)

    return dp, minDist, param1, param2, minRadius, maxRadius


def create_hough_circles_window():
    cv2.namedWindow(HoughCircles.WIN_NAME, 1)

    cv2.createTrackbar(HoughCircles.DP_NAME, HoughCircles.WIN_NAME, HoughCircles.DP_DEFAULT, 10, nothing)
    cv2.createTrackbar(HoughCircles.MIN_DIST_NAME, HoughCircles.WIN_NAME, HoughCircles.MIN_DIST_DEFAULT, 50, nothing)
    cv2.createTrackbar(HoughCircles.PARAM_1_NAME, HoughCircles.WIN_NAME, HoughCircles.PARAM_1_DEFAULT, MAX_COLOR_VAL, nothing)
    cv2.createTrackbar(HoughCircles.PARAM_2_NAME, HoughCircles.WIN_NAME, HoughCircles.PARAM_2_DEFAULT, 50, nothing)
    cv2.createTrackbar(HoughCircles.MIN_RADIUS_NAME, HoughCircles.WIN_NAME, HoughCircles.MIN_RADIUS_DEFAULT, 20, nothing)
    cv2.createTrackbar(HoughCircles.MAX_RADIUS_NAME, HoughCircles.WIN_NAME, HoughCircles.MAX_RADIUS_DEFAULT, 50, nothing)


def create_window():
    cv2.namedWindow(HSV.WIN_NAME, 1)

    cv2.createTrackbar(HSV.HUE_MIN_NAME, HSV.WIN_NAME, 0, PI, nothing)
    cv2.createTrackbar(HSV.HUE_MAX_NAME, HSV.WIN_NAME, PI, PI, nothing)

    cv2.createTrackbar(HSV.SAT_MIN_NAME, HSV.WIN_NAME, 0, MAX_COLOR_VAL, nothing)
    cv2.createTrackbar(HSV.SAT_MAX_NAME, HSV.WIN_NAME, MAX_COLOR_VAL, MAX_COLOR_VAL, nothing)

    cv2.createTrackbar(HSV.VAL_MIN_NAME, HSV.WIN_NAME, 0, MAX_COLOR_VAL, nothing)
    cv2.createTrackbar(HSV.VAL_MAX_NAME, HSV.WIN_NAME, 40, MAX_COLOR_VAL, nothing)


def get_skin_img_gray(hsv):
    sv_lower = 30, 100
    sv_upper = 150, MAX_COLOR_VAL
    hue_ranges = (0, 30), (145, PI)
    mask_skin = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_min, h_max in hue_ranges:
        lower_skin = tuple([h_min]) + sv_lower
        upper_skin = tuple([h_max]) + sv_upper
        _mask_local = cv2.inRange(hsv, lower_skin, upper_skin)
        mask_skin = cv2.bitwise_xor(mask_skin, _mask_local)

    _, thresh = cv2.threshold(mask_skin, 0, MAX_COLOR_VAL, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    skin_img = np.zeros(shape=hsv.shape[:2], dtype=np.uint8)

    areas = [cv2.contourArea(c) for c in contours]
    max_area_index = np.argmax(areas)

    cv2.drawContours(skin_img, contours, max_area_index, color=MAX_COLOR_VAL, thickness=cv2.FILLED)
    return skin_img


def haar_eye(gray, skinny, show=True):
    x, y, w, h = cv2.boundingRect(skinny)
    roi_face = gray[y:y+h, x:x+w]
    try:
        ex, ey, ew, eh = both_eyes_cascade.detectMultiScale(roi_face)[0]
        cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), MAX_COLOR_VAL / 2, thickness=2)
        # roi_eyes = roi_face[ey:ey+eh, ex:ex+ew]
        # eyes = eye_cascade.detectMultiScale(roi_eyes)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_eyes, (ex, ey), (ex + ew, ey + eh), MAX_COLOR_VAL, thickness=2)
    except:
        pass
    if show:
        cv2.imshow("haar eyes", gray)


def contour_center(contour):
    """
    :param contour: a list of XY, pertains to a contour
    :return: (tuple) contour's center
    """
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
    empty_img = np.zeros((rect_w, rect_h))
    rect_pos = [rect_x, rect_y]
    contour_local = contour - rect_pos
    cv2.drawContours(empty_img, [contour_local], -1, color=MAX_COLOR_VAL, thickness=cv2.FILLED)
    center_local = get_img_active_center(empty_img)
    xc, yc = np.add(center_local, rect_pos)
    return xc, yc


def get_img_active_center(img):
    """
    :param img: an image (gray or colored)
    :return: (tuple) image center
    """
    xc, yc = np.average(np.where(img > 0)[:2], axis=1).astype(np.uint32)
    return xc, yc


def refine_eye_contour(roi_soft, eye_hard_contour, min_intersect=0.9, max_delta=0.2):
    hard_center = contour_center(eye_hard_contour)
    intersect_rate = 0
    eye_center_prev = eye_center_refined = hard_center
    delta_relative = MIN_MARKER_RADIUS_SOFT / MIN_MARKER_RADIUS_SOFT

    while intersect_rate < min_intersect and delta_relative > max_delta:
        colorful_img = np.zeros(roi_soft.shape + tuple([3]), dtype=np.uint8)
        cv2.circle(colorful_img, eye_center_refined, MIN_MARKER_RADIUS_SOFT, color=BLUE, thickness=cv2.FILLED)
        # colorful_img = cv2.bitwise_and(colorful_img, colorful_img, mask=roi_soft)
        eye_center_refined = get_img_active_center(colorful_img)
        gray = cv2.cvtColor(colorful_img, cv2.COLOR_BGR2GRAY)
        intersect_rate = cv2.countNonZero(gray) / MIN_MARKER_AREA_SOFT
        delta_relative = norm(np.subtract(eye_center_refined, eye_center_prev)) / MIN_MARKER_RADIUS_SOFT
        eye_center_prev = eye_center_refined
        print(eye_center_refined, delta_relative, intersect_rate)

        cv2.imshow("colorful_img", colorful_img)
    print("done")

    return eye_hard_contour



def deal_with_eye_overlap(gray_soft, big_contours):
    gray_hard = gray_soft.copy()
    gray_hard[np.where(gray_hard < V_MAX_LOWER)] = 0
    gray_hard = cv2.blur(gray_hard, ksize=(3,3))

    eye_contours = []

    for stuck_contour in big_contours:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(stuck_contour)
        roi_hard = gray_hard[rect_x: rect_x + rect_w, rect_y: rect_y + rect_h]
        roi_soft = gray_soft[rect_x: rect_x + rect_w, rect_y: rect_y + rect_h]

        roi_hard_contours = get_contours(roi_hard)
        areas = [cv2.contourArea(c) for c in roi_hard_contours]
        roi_hard_contours = [roi_hard_contours[i] for i in range(len(roi_hard_contours)) if areas[i] > MIN_MARKER_AREA_HARD]
        if roi_hard_contours:
            eye_contour_index = np.argmin([cv2.contourArea(c) for c in roi_hard_contours])
            eye_hard_contour = roi_hard_contours[eye_contour_index]
            # eye_hard_contour = refine_eye_contour(roi_soft, eye_hard_contour)

            eye_hard_contour[:, :, 0] += rect_x
            eye_hard_contour[:, :, 1] += rect_y


            eye_contours.append(eye_hard_contour)

        cv2.rectangle(gray_hard, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), MAX_COLOR_VAL / 2, thickness=2)
        cv2.imshow("gray_hard", gray_hard)

    return eye_contours


def get_contours(gray, maxval=MAX_COLOR_VAL):
    _, thresh = cv2.threshold(gray, 0, MAX_COLOR_VAL, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def filter_by_contour_area(gray):
    gray_hard = gray.copy()
    gray_hard[np.where(gray < V_MAX_LOWER)] = 0

    contours = get_contours(gray)

    big_contours = []
    filtered_contours = []

    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area > MAX_MARKER_AREA:
            big_contours.append(c)
        elif c_area > MIN_MARKER_AREA_SOFT:
            filtered_contours.append(c)

    # if len(filtered_contours) < 18:
    #     # print(len(filtered_contours))
    #     eye_contours_additional = deal_with_eye_overlap(gray, big_contours)
    #     filtered_contours += eye_contours_additional
    # filtered_contours += big_contours
    if len(filtered_contours) != 18:
        print(len(filtered_contours))

    filtered_img = np.zeros(shape=gray.shape, dtype=np.uint8)
    cv2.drawContours(filtered_img, filtered_contours, -1, color=MAX_COLOR_VAL, thickness=cv2.FILLED)

    return filtered_img


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def take_smaller_part_hough(stuck_contour, gray):
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(stuck_contour)
    roi = gray[rect_x : rect_x + rect_w, rect_y : rect_y + rect_h]
    circles = cv2.HoughCircles(roi,
                               cv2.HOUGH_GRADIENT,
                               dp=2,
                               minDist=MAX_MARKER_RADIUS,
                               param1=MAX_COLOR_VAL,
                               param2=4,
                               minRadius=MAX_MARKER_RADIUS,
                               maxRadius=MAX_MARKER_RADIUS + MIN_MARKER_RADIUS_SOFT)

    bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(bgr,(i[0],i[1]),i[2],GREEN,2)
            # draw the center of the circle
            cv2.circle(bgr,(i[0],i[1]),2,RED,3)

        cv2.imshow('hough eyes dealer', bgr)
    else:
        print("shit")


def take_smaller_part(stuck_contour):
    cluster_n = 2
    pixel_resolution = 0.1
    steps = 30
    attempts = 10

    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(stuck_contour)
    shape = (rect_x + rect_w, rect_y + rect_h)
    filled_c = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(filled_c, [stuck_contour], -1, MAX_COLOR_VAL, thickness=cv2.FILLED)
    xy_filled = cv2.findNonZero(filled_c)

    points_converted = xy_filled.squeeze().astype(np.float32)
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, steps, pixel_resolution)
    ret, _, centers = cv2.kmeans(points_converted, cluster_n, None, term_crit, attempts, cv2.KMEANS_RANDOM_CENTERS)

    split_contours = [[] for _ in range(cluster_n)]

    for point_c in stuck_contour:
        distances = [dist(point_c, cnt) for cnt in centers]
        label = np.argmin(distances)
        split_contours[label].append(point_c)


    for label in range(cluster_n):
        shape = tuple( [len(split_contours[label])] ) + stuck_contour.shape[1:]
        split_contours[label] = np.resize(split_contours[label], shape)

    split_areas = [cv2.contourArea(c) for c in split_contours]
    smaller_contour = split_contours[np.argmin(split_areas)]

    return smaller_contour, split_contours, centers.astype(np.uint32)


def to_tuple(ndarray):
    return tuple(ndarray.tolist())


def process_frame(frame, index):
    if index == 0:
        frame = resize_in_half(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinny = get_skin_img_gray(hsv)

    channel_streams = separate_channels(hsv)

    dots_img = get_black_pixels(hsv)
    dots_img = cv2.bitwise_and(dots_img, skinny)

    # haar_eye(hsv[:, :, 2].astype(np.uint8), skinny)
    dots_img = cv2.blur(dots_img, ksize=(2,1))
    # dots_img = cv2.blur(dots_img, ksize=(1,1))
    # dots_img = cv2.blur(dots_img, ksize=(1,1))
    # dots_img = cv2.blur(dots_img, ksize=(1,1))

    # dots_img[ np.where(dots_img > 0) ] = MAX_COLOR_VAL

    dots_img = filter_by_contour_area(dots_img)
    cv2.imshow("markers", dots_img)

    # hough_circles(frame, dots_img)

    # cv2.imshow("skinny", skinny)
    cv2.imshow(HSV.WIN_NAME, channel_streams)


def nothing(*args, **kwargs):
    pass


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
    # create_hough_circles_window()
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
