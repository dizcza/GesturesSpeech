# coding=utf-8

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

MIN_MARKER_RADIUS_HARD = 3
MAX_MARKER_RADIUS_HARD = 70
MIN_MARKER_RADIUS_SOFT = 5
MAX_MARKER_RADIUS_SOFT = 8
MAX_CIRCLE_RATIO = 10.8

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


class MODE(object):
    HARD = "hard"  # noisy but without eyes overlapping
    SOFT = "soft"  # less noise but with eyes overlapping


RADIUS_RANGE = {
    MODE.SOFT: {"min": MIN_MARKER_RADIUS_SOFT, "max": MAX_MARKER_RADIUS_SOFT},
    MODE.HARD: {"min": MIN_MARKER_RADIUS_HARD, "max": MAX_MARKER_RADIUS_HARD}
}

MARKER_NAMES = {
    0: "jaw",
    1: "lidn",
    2: "lil",
    3: "lir",
    4: "liup",
    5: "chl",
    6: "wl",
    7: "p0",
    8: "wr",
    9: "chr",
    10: "edn_l",
    11: "edn_r",
    12: "eup_l",
    13: "eup_r",
    14: "ebr_ol",
    15: "ebr_il",
    16: "ebr_ir",
    17: "ebr_or"
}

EYE_UP_INDICES = [12, 13]
