# coding=utf-8

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os
import sys
import time
import json
from comparison import compare, show_comparison
from Kinect.kreader import KINECT_PATH
from MOCAP.mreader import MOCAP_PATH
from Emotion.csv_reader import EMOTION_PATH_PICKLES


########################################################################################################################
#                                              T E S T I N G                                                           #
########################################################################################################################

class Testing(object):
    def __init__(self, MotionClass):
        self.MotionClass = MotionClass
        _paths = {
            "HumanoidUkr": MOCAP_PATH,
            "HumanoidKinect": KINECT_PATH,
            "Emotion": EMOTION_PATH_PICKLES
        }
        self.proj_path = _paths[MotionClass.__name__]
        names_collection = {
            "HumanoidUkr": "MOCAP_INFO.json",
            "HumanoidKinect": "KINECT_INFO.json",
            "Emotion": "EMOTION_INFO.json"
        }
        self._info_name = names_collection[MotionClass.__name__]


    def the_worst_comparison(self, fps):
        """
         Computes the worst and the best out-of-sample error.
        :param fps: fps to be set in each gesture
        """
        print("%s: the_worst_between_comparison is running" % self.MotionClass.__name__)
        start = time.time()
        trn_path = os.path.join(self.proj_path, "Training")

        patterns = {}
        supremum = {}
        infimum = {}

        for directory in os.listdir(trn_path):
            patterns[directory] = []
            infimum[directory] = 0.
            supremum[directory] = 0.
            trn_subdir = os.path.join(trn_path, directory)
            for short_name in os.listdir(trn_subdir):
                fname = os.path.join(trn_subdir, short_name)
                knownGest = self.MotionClass(fname, fps)
                patterns[directory].append(knownGest)
        
        for directory in os.listdir(trn_path):
            tst_subfolder = os.path.join(self.proj_path, "Testing", directory)
            print(" testing %s" % directory)
            how_many_samples = len(os.listdir(tst_subfolder))
            for _sampleID, test_name in enumerate(os.listdir(tst_subfolder)):
                # sys.stdout.write("\r testing %s: %.2f" %
                #                  (directory, (float(_sampleID+1)/how_many_samples)*100))
                # sys.stdout.flush()

                fpath_test = os.path.join(tst_subfolder, test_name)
                unknownGest = self.MotionClass(fpath_test, fps)
                the_same_costs = []
                other_costs = []

                for theSamePattern in patterns[directory]:
                    dist, path = compare(theSamePattern, unknownGest)
                    dist /= float(len(path))
                    the_same_costs.append(dist)

                for class_name, gestsLeft in patterns.items():
                    if class_name != directory:
                        for knownGest in gestsLeft:
                            dist, path = compare(knownGest, unknownGest)
                            dist /= float(len(path))
                            other_costs.append(dist)
                min_other_cost = min(other_costs)

                if max(the_same_costs) >= min_other_cost:
                    supremum[directory] += 1.
                if min(the_same_costs) >= min_other_cost:
                    infimum[directory] += 1

        total_samples = 0
        print("The result is shown in #misclassified: ")
        for dir in supremum.keys():
            tst_subfolder = os.path.join(self.proj_path, "Testing", dir)
            tst_samples = len(os.listdir(tst_subfolder))
            total_samples += tst_samples
            msg = "%s: \t\t min = %d, max = %d out of %d test samples" % (dir, infimum[dir], supremum[dir], tst_samples)
            print(msg)
        total_supremum = sum(supremum.values())
        total_infimum = sum(infimum.values())
        print("*** INF: %d; \tSUP: %d; \t TOTAL SAMPLES: %d" % (total_infimum, total_supremum, total_samples))
        duration = time.time() - start
        print("Duration: %d sec" % duration)

        proj_info = json.load(open(self._info_name, 'r'))
        proj_info["error"] = {
            "inf": float(total_infimum) / total_samples,
            "sup": float(total_supremum) / total_samples
        }
        json.dump(proj_info, open(self._info_name, 'w'))

        return total_infimum, total_supremum


    def collect_first_patterns(self, fps):
        """
         Collects first patterns from each subfolder of Training set.
         :param fps: fps to be set in each gesture
        """
        pattern_gestures = []
        trn_path = os.path.join(self.proj_path, "Training")
        for root, _, logs in os.walk(trn_path):
            if any(logs):
                full_filename = os.path.join(root, logs[0])
                gest = self.MotionClass(full_filename, fps)
                pattern_gestures.append(gest)
        print("Took %d patterns as the first log in each training dir." % \
              len(pattern_gestures))
        return pattern_gestures


    def compare_with_first(self, fps):
        """
         Compares each test sample with the first one in Training folder.
         :param fps: fps to be set in each gesture
        """
        print("%s: comparing them all with fps = %s" % (self.MotionClass.__name__, fps))
        start = time.time()
        patterns = self.collect_first_patterns(fps)
        misclassified = 0.
        total_samples = 0
        tst_path = os.path.join(self.proj_path, "Testing")
        for root, _, logs in os.walk(tst_path):
            for test_log in logs:
                full_filename = os.path.join(root, test_log)
                unknown_gest = self.MotionClass(full_filename, fps)
                offset = []
                for known_gest in patterns:
                    dist, path = compare(known_gest, unknown_gest)
                    dist /= float(len(path))
                    offset.append(dist)
                ind = np.argmin(offset)
                possible_gest = patterns[ind]

                if possible_gest.name != unknown_gest.name:
                    print("got %s, should be %s" % (possible_gest.name, unknown_gest.name))
                    misclassified += 1.

                total_samples += 1

        Etest = misclassified / float(total_samples)
        print("Etest: %g <----> (%d / %d)" % (Etest, misclassified, total_samples))
        duration = time.time() - start
        print("Duration: %d sec" % duration)

        return Etest


    def error_vs_fps(self):
        """
         Plots the E-test VS fps dependency.
        """
        fps_range = range(5, 13, 1)
        test_errors = []
        for fps in fps_range:
            Etest, conf = self.compare_with_first(fps)
            test_errors.append(Etest)

        plt.plot(fps_range, test_errors, marker='o', ms=8)
        plt.xlabel("data freq (fps), 1/s")
        plt.ylabel("Etest")
        plt.title("out-of-sample error VS fps")
        plt.savefig("png/error_vs_fps_%s.png" % self.MotionClass.__name__)
        plt.show()


    def show_a_comparison(self):
        """
         Shows a comparison for two randomly picked samples.
        """
        trn_path = os.path.join(self.proj_path, "Training")
        tst_path = os.path.join(self.proj_path, "Testing")

        directory = np.random.choice(os.listdir(trn_path))
        trn_subdir = os.path.join(trn_path, directory)
        tst_subdir = os.path.join(tst_path, directory)

        trn_random_file = np.random.choice(os.listdir(trn_subdir))
        tst_random_file = np.random.choice(os.listdir(tst_subdir))
        trn_random_file = os.path.join(trn_subdir, trn_random_file)
        tst_random_file = os.path.join(tst_subdir, tst_random_file)

        trn_sample = self.MotionClass(trn_random_file, fps=None)
        tst_sample = self.MotionClass(tst_random_file, fps=None)
        show_comparison(trn_sample, tst_sample)


########################################################################################################################
#                                              T R A I N I N G                                                         #
########################################################################################################################

class Training(object):
    def __init__(self, MotionClass):
        self.MotionClass = MotionClass
        _paths = {
            "HumanoidUkr": MOCAP_PATH,
            "HumanoidKinect": KINECT_PATH,
            "Emotion": EMOTION_PATH_PICKLES
        }
        names_collection = {
            "HumanoidUkr": "MOCAP_INFO.json",
            "HumanoidKinect": "KINECT_INFO.json",
            "Emotion": "EMOTION_INFO.json"
        }
        self._info_name = names_collection[MotionClass.__name__]
        self.proj_path = _paths[MotionClass.__name__]
        self.proj_info = {}


    def load_info(self):
        """
         Initializes empty PROJECT_INFO.
        """
        try:
            self.proj_info = json.load(open(self._info_name, 'r'))
        except:
            self.proj_info = {
                "weights": {},
                "beta": None,
                "within_variance": None,
                "between_variance": None,
                "within_std": None,
                "between_std": None,
                "d-ratio": None,
                "d-ratio-std": None,
                "error": {"inf": None, "sup": None}
            }


    def compute_weights(self, mode, beta, fps):
        """
         Computes aver weights from the Training dataset.
        :param mode: defines moving markers
        :param beta: to be choosing to yield the biggest ratio
        :param fps: frames per second to be set
        """
        self.load_info()
        self.proj_info["beta"] = beta
        trn_path = os.path.join(self.proj_path, "Training")
        global_weights = {}

        for directory in os.listdir(trn_path):
            global_weights[directory] = []
            current_dir_weights = []
            trn_subfolder = os.path.join(trn_path, directory)
            for trn_name in os.listdir(trn_subfolder):
                fpath_trn = os.path.join(trn_subfolder, trn_name)
                gest = self.MotionClass(fpath_trn, fps)
                gest.compute_weights(mode, beta)
                current_dir_weights.append(gest.get_weights())
            global_weights[directory] = np.average(current_dir_weights, axis=0).tolist()

        self.proj_info["weights"] = global_weights
        json.dump(self.proj_info,  open(self._info_name,  'w'))
        print("New weights are saved in %s" % self._info_name)


    def compute_within_variance(self, fps):
        """
         Computes aver within variance from the Training dataset.
         :param fps: frames per second to be set
        """
        self.load_info()
        print("%s: COMPUTING WITHIN VARIANCE" % self.MotionClass.__name__)
        trn_path = os.path.join(self.proj_path, "Training")

        one_vs_the_same_var = []
        for directory in os.listdir(trn_path):
            trn_subfolder = os.path.join(trn_path, directory)
            log_examples = os.listdir(trn_subfolder)
            current_dir_var = []
            while len(log_examples) > 1:
                fpath_trn = os.path.join(trn_subfolder, log_examples[0])
                firstGest = self.MotionClass(fpath_trn, fps)
                for another_log in log_examples[1:]:
                    full_filename = os.path.join(trn_subfolder, another_log)
                    goingGest = self.MotionClass(full_filename, fps)
                    dist, path = compare(firstGest, goingGest)
                    dist /= float(len(path))
                    current_dir_var.append(dist)
                log_examples.pop(0)
            if any(current_dir_var):
                one_vs_the_same_var.append(np.average(current_dir_var))

        if any(one_vs_the_same_var):
            within_var = np.average(one_vs_the_same_var)
            within_std = np.std(one_vs_the_same_var)
        else:
            within_var = None
            within_std = None

        self.proj_info["within_variance"] = within_var
        self.proj_info["within_std"] = within_std
        json.dump(self.proj_info, open(self._info_name, 'w'))

        info = "Done with: \n\t within-var: %s \n\t " % within_var
        info += "within-std: %s\n" % within_std
        print(info)


    def compute_between_variance(self, fps):
        """
         Computes aver between variance from the Training dataset.
         :param fps: frames per second to be set
        """
        print("%s: COMPUTING BETWEEN VARIANCE" % self.MotionClass.__name__)
        self.load_info()
        trn_path = os.path.join(self.proj_path, "Training")
        dirs = os.listdir(trn_path)
        roots = [os.path.join(self.proj_path, "Training", one) for one in dirs]

        one_vs_others_var = []
        while len(roots) > 1:
            first_dir = roots[0]
            dirs_left = roots[1:]
            for first_log in os.listdir(first_dir):
                first_log_full = os.path.join(first_dir, first_log)
                firstGest = self.MotionClass(first_log_full, fps)
                for other_dir in dirs_left:
                    for other_log in os.listdir(other_dir):
                        full_filename = os.path.join(other_dir, other_log)
                        goingGest = self.MotionClass(full_filename, fps)
                        dist, path = compare(firstGest, goingGest)
                        dist /= float(len(path))
                        one_vs_others_var.append(dist)
            roots.pop(0)

        between_var = np.average(one_vs_others_var)
        between_std = np.std(one_vs_others_var)

        self.proj_info["between_variance"] = between_var
        self.proj_info["between_std"] = between_std
        json.dump(self.proj_info, open(self._info_name, 'w'))

        info = "Done with: \n\t between-var: %g \n\t " % between_var
        info += "between-std: %g\n" % between_std
        print(info)


    def update_ratio(self, mode, beta, fps):
        """
         Updates weights, within and between variance for the given beta param.
        :param mode: defines moving markers
        :param beta: to be choosing to yield the biggest ratio
        :param fps: frames per second to be set
        """
        self.compute_weights(mode, beta, fps)
        self.compute_within_variance(fps)
        self.compute_between_variance(fps)

        within_var = self.proj_info["within_variance"]
        within_std = self.proj_info["within_std"]
        between_var = self.proj_info["between_variance"]
        between_std = self.proj_info["between_std"]

        if within_var is not None:
            sigma_between = between_std / within_var
            sigma_within = within_std * between_var / within_var ** 2
            ratio_std = norm([sigma_between, sigma_within])
            self.proj_info["d-ratio"] = between_var / within_var
            self.proj_info["d-ratio-std"] = ratio_std
        else:
            self.proj_info["d-ratio"] = between_var
            self.proj_info["d-ratio-std"] = between_var

        print("(!) New discriminant ratio: %g" % self.proj_info["d-ratio"])
        json.dump(self.proj_info, open(self._info_name, 'w'))


    def choose_beta(self, mode, fps):
        """
         Chooses the best beta to get the biggest discriminant ratio.
         :param mode: defines moving markers
         :param fps: frames per second to be set
        """
        print("%s: choosing the beta with fps = %s" % (self.MotionClass.__name__, fps))
        begin = time.time()
        beta_range = [1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        gained_ratios = []
        gained_rstds = []
        for beta in beta_range:
            print("BETA: %f" % beta)
            self.update_ratio(mode, beta, fps)
            gained_ratios.append(self.proj_info["d-ratio"])
            gained_rstds.append(self.proj_info["d-ratio-std"])

        choosing_beta = zip(beta_range, gained_ratios, gained_rstds)
        self.proj_info["choosing beta"] = choosing_beta
        json.dump(self.proj_info, open(self._info_name, 'w'))
        print(zip(beta_range, gained_ratios, gained_rstds))

        best_ratio = max(gained_ratios)
        ind = np.argmax(gained_ratios)
        best_beta = beta_range[ind]
        print("BEST RATIO: %g, w.r.t. beta = %g" % (best_ratio, best_beta))

        plt.errorbar(np.log(beta_range), gained_ratios, gained_rstds,
                     linestyle='None', marker='^', ms=8)
        plt.xlabel("beta, log")
        plt.ylabel("discriminant ratio")
        plt.title("Choosing the best beta")
        plt.savefig("png/choosing_beta_%s.png" % self.MotionClass.__name__)
        end = time.time()
        print("\t Duration: ~%d m" % ((end - begin) / 60.))
        plt.show()


    def ratio_vs_fps(self, mode, beta, start, end):
        """
         Displays discriminant ratio vs fps.
        :param mode: defines moving markers
        :param beta: to be choosing to yield the biggest ratio
        :param start: lower fps
        :param end: upper fps
        :return:
        """
        fps_range = np.arange(start, end, 1)
        ratios = []
        ratios_std = []
        for fps in fps_range:
            self.update_ratio(mode, beta, fps)
            ratios.append(self.proj_info["d-ratio"])
            ratios_std.append(self.proj_info["d-ratio-std"])
        plt.errorbar(fps_range, ratios, ratios_std, marker='^', ms=8)
        plt.xlabel("data freq (fps), 1/s")
        mean_std = 100. * np.average(ratios_std)
        plt.ylabel("between variance, std=%.1f%%" % mean_std)
        plt.title("Between variance VS fps")
        plt.savefig("png/Db_fps.png")
        plt.show()