# coding=utf-8

import cv2
import numpy as np
from Emotion.OpenCV.window import get_lower_upper, separate_channels
from Emotion.OpenCV.constants import *


def process_frame(frame, index):
    if index == 0:
        frame = resize_in_half(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinny = get_skin_img_gray(hsv)
    channel_streams = separate_channels(hsv)
    cv2.imshow(HSV.WIN_NAME, channel_streams)

    dots_img = get_black_pixels(hsv)
    dots_img = cv2.bitwise_and(dots_img, skinny)

    dots_hard = cv2.blur(dots_img, ksize=(2,1))

    # edges = cv2.Canny(hsv[:,:,1], 70, 150)
    # edges_ind = np.where(edges > 0)
    # dots_bgr = cv2.cvtColor(dots_hard, cv2.COLOR_GRAY2BGR)
    # dots_bgr[edges_ind] = BLUE
    # cv2.imshow("canny", dots_bgr)

    # dots_hard = np.copy(dots_img)
    dots_soft = cv2.blur(dots_img, ksize=(3,3))

    dots_hard = filter_by_contour_area(dots_hard, MODE.HARD)
    # dots_soft = filter_by_contour_area(dots_soft, MODE.SOFT)
    #
    # dots_soft = lay_on(dots_hard, dots_soft)
    #
    # for mode, _dots_gray in zip((MODE.HARD, MODE.SOFT), (dots_hard, dots_soft)):
    #     dots_bgr = draw_centers(_dots_gray)
    #     cv2.imshow("mode " + mode, dots_bgr)
    cv2.imshow("dots_hard",dots_hard)


def resize_in_half(frame):
    """
    :param frame: image
    :return: image, reduced by half
    """
    return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)



def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def does_intersect(eye_center_hard, possible_eyes_center_soft):
    """
    :param eye_center_hard: eye center with hard threshold
    :param possible_eyes_center_soft: points where can be eyes with soft threshold
    :return: if hard and soft eyes intersects
    """
    intersect = [dist(eye_center_hard, eye_soft) < MAX_MARKER_RADIUS_SOFT for eye_soft in possible_eyes_center_soft]
    return np.any(intersect)


def lay_on(dots_hard, dots_soft):
    """
    :param dots_hard: img obtained with hard threshold
    :param dots_soft: img obtained with soft threshold
    :return: dots_soft with both eyes from dots_hard if necessary
    """
    marker_centers_hard = get_marker_centers(dots_hard)
    marker_centers_soft = get_marker_centers(dots_soft)
    if len(marker_centers_soft) > MARKERS_NUM:
        cv2.imshow("too much markers", dots_soft)
    if len(marker_centers_soft) < MARKERS_NUM:
        eyes_hard = marker_centers_hard[EYE_UP_INDICES]
        possible_eyes_soft = marker_centers_soft[EYE_UP_INDICES]
        for eye_index, eye_hard in zip(EYE_UP_INDICES, eyes_hard):
            if not does_intersect(eye_hard, possible_eyes_soft):
                marker_centers_soft = np.insert(marker_centers_soft, eye_index, eye_hard, axis=0)
                dots_soft = copy_hard_to_soft(dots_hard, dots_soft, eye_hard)
    return dots_soft


def copy_hard_to_soft(dots_hard, dots_soft, pos):
    """
    :param dots_hard: img obtained with hard threshold
    :param dots_soft: img obtained with soft threshold
    :param pos: eye hard pos
    :return: dots_soft with inserted eye from dots_hard
    """
    y, x = pos
    x_left = x - MAX_MARKER_RADIUS_SOFT
    x_right = x + MAX_MARKER_RADIUS_SOFT
    y_top = y - MAX_MARKER_RADIUS_SOFT
    y_bottom = y + MAX_MARKER_RADIUS_SOFT
    dots_soft[x_left:x_right, y_top:y_bottom] = dots_hard[x_left:x_right, y_top:y_bottom]
    return dots_soft


def arrange_markers(contours):
    """
    :param contours: unarranged contours
    :return: contours arranged w.r.t. MARKER_NAMES
    """
    contours = np.array(contours)
    cheeks_nose = contours[6:10]
    cheeks_nose_center_xs = [get_contour_center(c)[0] for c in cheeks_nose]
    aligned_indices = np.argsort(cheeks_nose_center_xs)[::-1]
    contours[6:10] = cheeks_nose[aligned_indices]
    return contours


def filter_by_contour_area(dots_gray, mode):
    """
    :param dots_gray: gray image with dots (markers) left
    :param mode: hard or soft mode
    :return: dots image, filtered by area and size w.r.t. mode
    """
    contours = get_contours(dots_gray)

    filtered_contours = []

    for c in contours:
        if match_contour_area(c, mode) and match_contour_size(c, mode):
            filtered_contours.append(c)

    if mode == MODE.HARD and len(filtered_contours) != MARKERS_NUM:
        print(mode, len(filtered_contours))

    filtered_img = np.zeros(shape=dots_gray.shape, dtype=np.uint8)
    cv2.drawContours(filtered_img, filtered_contours, -1, color=MAX_COLOR_VAL, thickness=cv2.FILLED)

    return filtered_img


def match_contour_area(contour, mode):
    """
    :param contour: returned by get_contours(dots_gray)
    :param mode: hard of soft mode
    :return: if the contour fits in a possible area range
    """
    min_marker_area = np.pi * RADIUS_RANGE[mode]["min"] ** 2
    max_marker_area = np.pi * RADIUS_RANGE[mode]["max"] ** 2
    return min_marker_area < cv2.contourArea(contour) < max_marker_area


def get_black_pixels(hsv):
    """
    :param hsv: hsv image
    :return: its black mask
    """
    lower_black, upper_black = get_lower_upper()
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    return mask_black


def get_black_pixel_indices(hsv, inverse=False):
    mask_black = get_black_pixels(hsv)
    if inverse:
        return np.where(mask_black == 0)
    else:
        return np.where(mask_black > 0)


def get_skin_img_gray(hsv):
    """
    :param hsv: hsv colored image
    :return: skinny image of an hsv image
    """
    sv_lower = 30, 90
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


def get_contour_center(contour):
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


def get_contours(gray):
    """
    :param gray: gray image
    :return: its contours (all of them)
    """
    _, thresh = cv2.threshold(gray, 0, MAX_COLOR_VAL, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def match_contour_size(contour, mode, sigma=0.3):
    """
    :param contour: returned by get_contours(dots_gray)
    :param mode: hard of soft mode
    :param sigma: deviation from a circle to an ellipse
    :return: if the contour fits in a possible box size
    """
    lowest_box_size = (1 - sigma) * 2 * RADIUS_RANGE[mode]["min"]
    biggest_box_size = (1 + sigma) * 2 * RADIUS_RANGE[mode]["max"]
    width, height = cv2.boundingRect(contour)[2:]
    ratio = float(width) / height
    fit_in = ratio < MAX_CIRCLE_RATIO
    for c_size in (width, height):
        fit_in *= lowest_box_size < c_size < biggest_box_size
    return fit_in


def get_marker_centers(dots_gray):
    """
    :param dots_gray: gray img of markers
    :return: marker centers
    """
    contours = get_contours(dots_gray)
    contours = arrange_markers(contours)
    markers_centers = []
    for c_index, c in enumerate(contours):
        c_center = get_contour_center(c)
        markers_centers.append(c_center)
    return np.array(markers_centers)


def draw_centers(dots_gray):
    """
    :param dots_gray: dots gray img
    :return: dots BGR img with marker centers and text
    """
    marker_centers = get_marker_centers(dots_gray)
    dots_bgr = cv2.cvtColor(dots_gray, cv2.COLOR_GRAY2BGR)
    for m_index, m_center in enumerate(marker_centers):
        cv2.circle(dots_bgr, tuple(m_center), radius=2, color=GREEN, thickness=2)
        x_right = m_center[0] + MIN_MARKER_RADIUS_HARD
        y_top = m_center[1] - MIN_MARKER_RADIUS_HARD // 2
        cv2.putText(dots_bgr, MARKER_NAMES[m_index], (x_right, y_top), cv2.FONT_ITALIC, 0.5, BLUE, thickness=1)
    return dots_bgr
