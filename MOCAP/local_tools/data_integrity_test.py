# coding=utf-8

import os
import numpy as np
from MOCAP.mreader import MOCAP_PATH, HumanoidUkr


def test_nan_weights():
    """
     Tests each sample for having nan weights.
    """
    for log_c3d in os.listdir(MOCAP_PATH):
        if log_c3d.endswith(".c3d"):
            log_path = os.path.join(MOCAP_PATH, log_c3d)
            gest = HumanoidUkr(log_path)
            w = gest.get_weights()
            assert not np.isnan(w).any(), "nan weights in %s" % log_c3d


if __name__ == "__main__":
    test_nan_weights()
