# coding=utf-8

import os
import time
import json

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from tools.comparison import compare, show_comparison
from Kinect.kreader import KINECT_PATH
from MOCAP.mreader import MOCAP_PATH
from Emotion.emotion import EMOTION_PATH_PICKLES


class InstrumentCollector(object):
    def __init__(self, MotionClass, prefix):
        self.MotionClass = MotionClass
        self.prefix = prefix
        _paths = {
            "HumanoidUkr": MOCAP_PATH,
            "HumanoidKinect": KINECT_PATH,
            "Emotion": EMOTION_PATH_PICKLES,
            "EmotionArea": None
        }
        self.proj_path = _paths[MotionClass.__name__]
        names_collection = {
            "HumanoidUkr": "MOCAP_INFO.json",
            "HumanoidKinect": "KINECT_INFO.json",
            "Emotion": "EMOTION_INFO.json",
            "EmotionArea": "EMOTION_AREAS_INFO.json"
        }
        self.proj_info = {}
        self._info_name = names_collection[MotionClass.__name__]
        if prefix == "":
            self.trn_path = os.path.join(self.proj_path, "Training")
            self.tst_path = os.path.join(self.proj_path, "Testing")
        else:
            self.trn_path = os.path.join(prefix, "Training")
            self.tst_path = os.path.join(prefix, "Testing")
        self.train_gestures = []
        self.test_gestures = []

    def load_info(self):
        """
         Initializes empty PROJECT_INFO.
        """
        try:
            self.proj_info = json.load(open(self._info_name, 'r'))
        except FileNotFoundError:
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

    def load_train_samples(self, fps):
        """
        :param fps: frames per second to be set
                    pass as None not to change default fps
        :return: training gestures
        """
        for directory in os.listdir(self.trn_path):
            trn_subfolder = os.path.join(self.trn_path, directory)
            for trn_name in os.listdir(trn_subfolder):
                fpath_trn = os.path.join(trn_subfolder, trn_name)
                gest = self.MotionClass(fpath_trn, fps)
                self.train_gestures.append(gest)
        return self.train_gestures

    def load_test_samples(self, fps):
        """
        :param fps: frames per second to be set
                    pass as None not to change default fps
        :return: testing gestures
        """
        for directory in os.listdir(self.tst_path):
            tst_subfolder = os.path.join(self.tst_path, directory)
            for tst_name in os.listdir(tst_subfolder):
                fpath_tst = os.path.join(tst_subfolder, tst_name)
                gest = self.MotionClass(fpath_tst, fps)
                self.test_gestures.append(gest)
        return self.test_gestures

    def compute_weights(self, mode, beta, fps):
        """
         Computes aver weights from the Training dataset.
        :param mode: defines moving markers
        :param beta: to be choosing to yield the biggest ratio
        :param fps: frames per second to be set
                    pass as None not to change default fps
        """
        self.load_info()
        self.proj_info["beta"] = beta

        global_weights = {}

        for directory in os.listdir(self.trn_path):
            global_weights[directory] = []
            current_dir_weights = []
            trn_subfolder = os.path.join(self.trn_path, directory)
            for trn_name in os.listdir(trn_subfolder):
                fpath_trn = os.path.join(trn_subfolder, trn_name)
                gest = self.MotionClass(fpath_trn, fps)
                gest.compute_weights(mode, beta)
                current_dir_weights.append(gest.get_weights())
                assert not np.isnan(gest.get_weights()).any(), \
                    "got np.nan weights in %s" % gest.fname
            global_weights[directory] = np.average(current_dir_weights, axis=0).tolist()

        if self.prefix == "":
            self.proj_info["weights"] = global_weights
        else:
            sub_project = os.path.basename(self.prefix)
            self.proj_info["weights"][sub_project] = global_weights

        json.dump(self.proj_info,  open(self._info_name,  'w'))
        print("New weights are saved in %s" % self._info_name)


########################################################################################################################
#                                              T E S T I N G                                                           #
########################################################################################################################

class Testing(InstrumentCollector):
    def __init__(self, MotionClass, prefix=""):
        InstrumentCollector.__init__(self, MotionClass, prefix)

    def the_worst_comparison(self, fps):
        """
         Computes the worst and the best out-of-sample error.
         NOTE: comparison is made by choosing the ABSOLUTE lowest DTW cost
               among examples -- without its normalizing by the path length,
               since the last one rises in-sample and out-of-sample errors.
        :param fps: fps to be set in each gesture
                    pass as None not to change default fps
        """
        print("%s: the_worst_between_comparison is running" % self.MotionClass.__name__)
        start = time.time()

        patterns = {}
        supremum = {}
        infimum = {}

        for directory in os.listdir(self.trn_path):
            patterns[directory] = []
            infimum[directory] = 0.
            supremum[directory] = 0.
            trn_subdir = os.path.join(self.trn_path, directory)
            for short_name in os.listdir(trn_subdir):
                fname = os.path.join(trn_subdir, short_name)
                knownGest = self.MotionClass(fname, fps)
                patterns[directory].append(knownGest)
        
        for directory in os.listdir(self.trn_path):
            tst_subfolder = os.path.join(self.tst_path, directory)
            print(" testing %s" % directory)
            for _sampleID, test_name in enumerate(os.listdir(tst_subfolder)):
                fpath_test = os.path.join(tst_subfolder, test_name)
                unknownGest = self.MotionClass(fpath_test, fps)
                the_same_costs = []
                other_costs = []

                for theSamePattern in patterns[directory]:
                    dist = compare(theSamePattern, unknownGest)
                    the_same_costs.append(dist)

                other_patterns = []
                for class_name, gestsLeft in patterns.items():
                    if class_name != directory:
                        for knownGest in gestsLeft:
                            dist = compare(knownGest, unknownGest)
                            other_costs.append(dist)
                            other_patterns.append(knownGest)
                min_other_cost = min(other_costs)

                if max(the_same_costs) >= min_other_cost:
                    ind = np.argmin(other_costs)
                    got_pattern = other_patterns[ind]
                    if got_pattern.name != unknownGest.name:
                        supremum[directory] += 1.

                        msg = "got %s" % got_pattern.name
                        if hasattr(got_pattern, "fname"):
                            msg += " (file: %s)" % got_pattern.fname
                        msg += ", should be %s" % unknownGest.name
                        if hasattr(unknownGest, "fname"):
                            msg += " (file: %s)" % unknownGest.fname
                        print(msg)

                if min(the_same_costs) >= min_other_cost:
                    ind = np.argmin(other_costs)
                    got_pattern = other_patterns[ind]
                    if got_pattern.name != unknownGest.name:
                        infimum[directory] += 1

        total_samples = 0
        print("The result is shown in #misclassified: ")
        for dir in supremum.keys():
            tst_subfolder = os.path.join(self.tst_path, dir)
            tst_samples = len(os.listdir(tst_subfolder))
            total_samples += tst_samples
            msg = "%s: \t\t min = %d, max = %d out of %d test samples" % (dir, infimum[dir], supremum[dir], tst_samples)
            print(msg)
        total_supremum = sum(supremum.values())
        total_infimum = sum(infimum.values())
        print("*** THE BEST CASE: %d; \tTHE WORST CASE: %d; \t TOTAL SAMPLES: %d" %
              (total_infimum, total_supremum, total_samples))
        duration = time.time() - start
        print("Duration: %d sec" % duration)

        proj_info = json.load(open(self._info_name, 'r'))
        proj_info["error"] = {
            "inf": float(total_infimum) / total_samples,
            "sup": float(total_supremum) / total_samples
        }
        json.dump(proj_info, open(self._info_name, 'w'))

        return total_infimum, total_supremum, total_samples


    def collect_first_patterns(self, fps):
        """
         Collects first patterns from each subfolder of Training set.
         :param fps: fps to be set in each gesture
                     pass as None not to change default fps
        """
        pattern_gestures = []
        for root, _, logs in os.walk(self.trn_path):
            if any(logs):
                log_path = os.path.join(root, logs[0])
                gest = self.MotionClass(log_path, fps)
                pattern_gestures.append(gest)
        print("Took %d patterns as the first log in each training dir." % \
              len(pattern_gestures))
        return pattern_gestures


    def compare_with_first(self, fps):
        """
         Compares each test sample with the first one in Training folder.
         :param fps: fps to be set in each gesture
                     pass as None not to change default fps
        """
        print("%s: comparing them all with fps = %s" % (self.MotionClass.__name__, fps))
        start = time.time()
        patterns = self.collect_first_patterns(fps)
        misclassified = 0.
        total_samples = 0
        for root, _, logs in os.walk(self.tst_path):
            for test_log in logs:
                unknown_log_path = os.path.join(root, test_log)
                unknown_gest = self.MotionClass(unknown_log_path, fps)
                costs = []
                for known_gest in patterns:
                    dist = compare(known_gest, unknown_gest)
                    costs.append(dist)
                ind = np.argmin(costs)
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
        directory = np.random.choice(os.listdir(self.trn_path))
        trn_subdir = os.path.join(self.trn_path, directory)
        tst_subdir = os.path.join(self.tst_path, directory)

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

class Training(InstrumentCollector):

    # TODO how to compute a variance

    def __init__(self, MotionClass, prefix=""):
        InstrumentCollector.__init__(self, MotionClass, prefix)

    def compute_within_variance(self, fps):
        """
         Computes aver within variance from the Training dataset.
         Makes it None in case when
         :param fps: frames per second to be set
                     pass as None not to change default fps
         :return (float), within_var
        """
        self.load_info()
        print("%s: COMPUTING WITHIN VARIANCE" % self.MotionClass.__name__)

        one_vs_the_same_var = []
        for directory in os.listdir(self.trn_path):
            trn_subfolder = os.path.join(self.trn_path, directory)
            log_examples = os.listdir(trn_subfolder)
            current_dir_var = []
            while len(log_examples) > 1:
                fpath_trn = os.path.join(trn_subfolder, log_examples[0])
                firstGest = self.MotionClass(fpath_trn, fps)
                for another_log in log_examples[1:]:
                    full_filename = os.path.join(trn_subfolder, another_log)
                    goingGest = self.MotionClass(full_filename, fps)
                    dist = compare(firstGest, goingGest)
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

        return within_var


    def compute_between_variance(self, fps):
        """
         Computes aver between variance from the Training dataset.
         :param fps: frames per second to be set
                     pass as None not to change default fps
         :return: (float), between_var
        """
        print("%s: COMPUTING BETWEEN VARIANCE" % self.MotionClass.__name__)
        self.load_info()
        trndirs = os.listdir(self.trn_path)
        roots = [os.path.join(self.trn_path, one) for one in trndirs]

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
                        dist = compare(firstGest, goingGest)
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

        return between_var


    def update_ratio(self, mode, beta, fps):
        """
         Updates weights, within and between variance for the given beta param.
        :param mode: defines moving markers
        :param beta: to be choosing to yield the biggest ratio
        :param fps: frames per second to be set
                    pass as None not to change default fps
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
                     pass as None not to change default fps
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

        choosing_beta = tuple(zip(beta_range, gained_ratios, gained_rstds))
        print(choosing_beta)
        self.proj_info["choosing beta"] = choosing_beta
        json.dump(self.proj_info, open(self._info_name, 'w'))

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