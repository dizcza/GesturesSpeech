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
from Emotion.em_reader import EMOTION_PATH_PICKLES


class InstrumentCollector(object):
    def __init__(self, MotionClass, prefix=""):
        self.MotionClass = MotionClass
        self.prefix = prefix
        _paths = dict(HumanoidUkr=MOCAP_PATH,
                      HumanoidKinect=KINECT_PATH,
                      Emotion=EMOTION_PATH_PICKLES,
                      EmotionArea=None)
        self.proj_path = _paths[MotionClass.__name__]
        names_collection = dict(HumanoidUkr="MOCAP_INFO.json",
                                HumanoidKinect="KINECT_INFO.json",
                                Emotion="EMOTION_INFO.json",
                                EmotionArea="EMOTION_AREAS_INFO.json")
        self.proj_info = {}
        self._info_name = names_collection[MotionClass.__name__]
        if prefix == "":
            self.trn_path = os.path.join(self.proj_path, "Training")
            self.tst_path = os.path.join(self.proj_path, "Testing")
        else:
            self.trn_path = os.path.join(prefix, "Training")
            self.tst_path = os.path.join(prefix, "Testing")

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

    def dump_info(self):
        json.dump(self.proj_info, open(self._info_name, 'w'))

    def load_train_samples(self, fps):
        """
        :param fps: frames per second to be set;
                    pass as None to use the default fps
        :return: training gestures
        """
        train_gestures = []
        for class_name in os.listdir(self.trn_path):
            class_path = os.path.join(self.trn_path, class_name)
            for trn_name in os.listdir(class_path):
                fpath_trn = os.path.join(class_path, trn_name)
                gest = self.MotionClass(fpath_trn, fps)
                train_gestures.append(gest)
        return tuple(train_gestures)

    def load_test_samples(self, fps):
        """
        :param fps: frames per second to be set;
                    pass as None to use the default fps
        :return: testing gestures
        """
        test_gestures = []
        for directory in os.listdir(self.tst_path):
            tst_subfolder = os.path.join(self.tst_path, directory)
            for tst_name in os.listdir(tst_subfolder):
                fpath_tst = os.path.join(tst_subfolder, tst_name)
                gest = self.MotionClass(fpath_tst, fps)
                test_gestures.append(gest)
        return tuple(test_gestures)

    def compute_weights(self, mode, beta, fps):
        """
         Computes aver weights from the Training dataset.
        :param mode: defines moving markers
        :param beta: (float), defines weights activity;
                      the best beta value is around 1e2;
                      set it to None to model when beta vanishes;
        :param fps: frames per second to be set;
                    pass as None to use the default fps
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
                weights_array = gest.get_weights()
                if np.isnan(weights_array).any(): continue
                current_dir_weights.append(weights_array)
            assert len(current_dir_weights) > 0, "too many files with NaN weights"
            global_weights[directory] = np.average(current_dir_weights, axis=0).tolist()

        if self.prefix == "":
            self.proj_info["weights"] = global_weights
        else:
            sub_project = os.path.basename(self.prefix)
            self.proj_info["weights"][sub_project] = global_weights

        self.dump_info()
        print("New weights are saved in %s" % self._info_name)


########################################################################################################################
#                                              T E S T I N G                                                           #
########################################################################################################################

class Testing(InstrumentCollector):
    def __init__(self, MotionClass, prefix=""):
        InstrumentCollector.__init__(self, MotionClass, prefix)

    def the_worst_comparison(self, fps, verbose=True, weighted=True):
        """
         Computes the worst and the best out-of-sample error, using WDTW algorithm.
         The confidence measure is set to be a margin between the chosen positive
         result and the first negative result.
        :param fps: fps to be set in each gesture;
                    pass as None to use the default fps
        :param verbose: verbose display (True) or silent (False)
        :param weighted: use weighted FastDTW modification or just FastDTW
        """

        def print_err(got_pattern, unknownGest):
            if verbose:
                msg = "got %s" % got_pattern.name
                if hasattr(got_pattern, "fname"):
                    msg += " (file: %s)" % got_pattern.fname
                msg += ", should be %s" % unknownGest.name
                if hasattr(unknownGest, "fname"):
                    msg += " (file: %s)" % unknownGest.fname
                if hasattr(unknownGest, "author") and hasattr(got_pattern, "author"):
                    if unknownGest.author == got_pattern.author:
                        msg += " -- the same author"
                    else:
                        msg += " -- different authors"
                print(msg)

        print("%s: TWE WORST COMPARISON is running (FPS = %s)" % (self.MotionClass.__name__, fps))
        start = time.time()
        self.load_info()

        patterns = {}
        supremum = {}
        infimum = {}
        margin = 0

        for directory in os.listdir(self.trn_path):
            patterns[directory] = []
            infimum[directory] = 0.
            supremum[directory] = 0.
            trn_subdir = os.path.join(self.trn_path, directory)
            for short_name in os.listdir(trn_subdir):
                fname = os.path.join(trn_subdir, short_name)
                knownGest = self.MotionClass(fname, fps)
                patterns[directory].append(knownGest)
        
        for directory in os.listdir(self.tst_path):
            tst_subfolder = os.path.join(self.tst_path, directory)
            if verbose: print(" testing %s" % directory)
            for _sampleID, test_name in enumerate(os.listdir(tst_subfolder)):
                fpath_test = os.path.join(tst_subfolder, test_name)
                unknownGest = self.MotionClass(fpath_test, fps)
                the_same_costs = []
                other_costs = []

                for theSamePattern in patterns[directory]:
                    dist = compare(theSamePattern, unknownGest, weighted=weighted)
                    the_same_costs.append(dist)

                other_patterns = []
                for class_name, gestsLeft in patterns.items():
                    if class_name != directory:
                        for knownGest in gestsLeft:
                            dist = compare(knownGest, unknownGest, weighted=weighted)
                            other_costs.append(dist)
                            other_patterns.append(knownGest)
                min_other_cost = min(other_costs)
                min_the_same_cost = min(the_same_costs)
                max_the_same_cost = max(the_same_costs)

                if max_the_same_cost >= min_other_cost:
                    # the worst test scenario is FAILED
                    ind = np.argmin(other_costs)
                    got_pattern = other_patterns[ind]
                    assert got_pattern.name != unknownGest.name, "invalid data structure"
                    supremum[directory] += 1.
                    # print_err(got_pattern, unknownGest)

                if min_the_same_cost >= min_other_cost:
                    # both the worst and the best test scenarios are FAILED
                    ind = np.argmin(other_costs)
                    got_pattern = other_patterns[ind]
                    assert got_pattern.name != unknownGest.name, "invalid data structure"
                    infimum[directory] += 1
                    print_err(got_pattern, unknownGest)

                interval = max_the_same_cost - min_the_same_cost
                how_good = (min_other_cost - min_the_same_cost) / interval
                margin += min(1, max(0, how_good))

        total_samples = 0
        print("The result is shown in number of misclassified samples: ")
        for dir in supremum.keys():
            tst_subfolder = os.path.join(self.tst_path, dir)
            tst_samples = len(os.listdir(tst_subfolder))
            total_samples += tst_samples
            if verbose:
                msg = "  %s: \t\t min = %d, max = %d out of %d test samples" % (
                    dir, infimum[dir], supremum[dir], tst_samples
                )
                print(msg)
        total_supremum = sum(supremum.values())
        total_infimum = sum(infimum.values())
        margin *= 100. / total_samples
        self.proj_info["error"]["inf"] = float(total_infimum) / total_samples
        self.proj_info["error"]["sup"] = float(total_supremum) / total_samples
        print("*** THE BEST CASE: %d; \tTHE WORST CASE: %d; \t TOTAL SAMPLES: %d" %
              (total_infimum, total_supremum, total_samples))
        print("*** margin: %.3g%%" % margin)
        duration = time.time() - start
        print("Duration: %d sec" % duration)

        return total_infimum, total_supremum, total_samples


    def error_vs_fps(self, mode, beta):
        """
         Plots the out-of-sample error VS fps.
         :param mode: defines moving markers
         :param beta: (float), defines weights activity;
                      the best beta value is around 100;
                      set it to None to model when beta vanishes;
        """
        fps_range = range(2, 11, 1)
        test_errors = []
        for fps in fps_range:
            self.compute_weights(mode, beta, fps)
            inf, sup, tot = self.the_worst_comparison(fps, verbose=False)
            Etest = float(sup) / tot
            test_errors.append(Etest)
        plt.plot(fps_range, test_errors, 'o--', ms=8)
        plt.ylim(ymin=-0.01)
        plt.xlim(1, 14)
        plt.ylabel("Etest")
        plt.xlabel("FPS")
        plt.title("out-of-sample error VS fps")
        plt.grid()
        plt.savefig("png/error_vs_fps.png")
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

    def __init__(self, MotionClass, prefix=""):
        InstrumentCollector.__init__(self, MotionClass, prefix)

    def compute_within_variance(self, fps, verbose=True):
        """
         Computes averaged within-class variance from the Training dataset.
         :param fps: frames per second to be set;
                     pass as None to use the default fps
         :param verbose: verbose display (True) or silent (False)
         :return (float), averaged variance between two different samples
                          within the same class
        """
        self.load_info()
        print("%s: COMPUTING WITHIN VARIANCE" % self.MotionClass.__name__)
        start_timer = time.time()

        one_vs_the_same_var = []
        for directory in os.listdir(self.trn_path):
            trn_subfolder = os.path.join(self.trn_path, directory)
            log_examples = os.listdir(trn_subfolder)
            while len(log_examples) > 1:
                first_log_path = os.path.join(trn_subfolder, log_examples[0])
                firstGest = self.MotionClass(first_log_path, fps)
                for another_log in log_examples[1:]:
                    other_log_path = os.path.join(trn_subfolder, another_log)
                    goingGest = self.MotionClass(other_log_path, fps)

                    # since both firstGest and goingGest have the same weights
                    # (stored in PROJECTNAME_INFO.json), there is no need to
                    # alter arguments and compute it explicitly, because
                    # compare(goingGest, firstGest) == compare(firstGest, goingGest)
                    dist = compare(firstGest, goingGest)

                    one_vs_the_same_var.append(dist)

                log_examples.pop(0)

        if any(one_vs_the_same_var):
            within_var = np.average(one_vs_the_same_var)
            within_std = np.std(one_vs_the_same_var)
        else:
            within_var = None
            within_std = None

        self.proj_info["within_variance"] = within_var
        self.proj_info["within_std"] = within_std
        self.dump_info()

        if verbose:
            duration = time.time() - start_timer
            info = "Done with: \n\t within-var: %s \n\t " % within_var
            info += "within-std: %s\n\t" % within_std
            info += "duration: %d\n" % duration
            print(info)

        return within_var


    def compute_between_variance(self, fps, verbose=True):
        """
         Computes averaged between-class variance from the Training dataset.
         :param fps: frames per second to be set;
                     pass as None to use the default fps
         :param verbose: verbose display (True) or silent (False)
         :return: (float), the averaged dist between two samples from different classes
        """
        print("%s: COMPUTING BETWEEN VARIANCE" % self.MotionClass.__name__)
        start_timer = time.time()
        one_vs_others_var = []
        trn_samples = self.load_train_samples(fps)
        for firstGest in trn_samples:
            for goingGest in (trn_samples + tuple()):
                if firstGest.name != goingGest.name:
                    dist = compare(firstGest, goingGest)
                    one_vs_others_var.append(dist)
        between_var = np.average(one_vs_others_var)
        between_std = np.std(one_vs_others_var)
        self.proj_info["between_variance"] = between_var
        self.proj_info["between_std"] = between_std
        json.dump(self.proj_info, open(self._info_name, 'w'))

        if verbose:
            duration = time.time() - start_timer
            info = "Done with: \n\t between-var: %f \n\t " % between_var
            info += "between-std: %f\n\t" % between_std
            info += "duration: %d sec\n" % duration
            print(info)

        return between_var


    def update_ratio(self, mode, beta, fps, verbose=False):
        """
         Updates weights, within and between variance for the given beta param.
        :param mode: defines moving markers
        :param beta: (float), defines weights activity;
                      the best beta value is around 100;
                      set it to None to model when beta vanishes;
        :param fps: frames per second to be set;
                    pass as None to use the default fps
        :param verbose: verbose display (True) or silent (False)
        """
        self.compute_weights(mode, beta, fps)
        self.compute_within_variance(fps, verbose)
        self.compute_between_variance(fps, verbose)

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
            self.proj_info["d-ratio-std"] = between_std

        print("(!) New discriminant ratio: %f (FPS = %s)" % (self.proj_info["d-ratio"], fps))
        json.dump(self.proj_info, open(self._info_name, 'w'))


    def choose_beta_simple(self, mode, fps):
        """
         Chooses the best beta to get the biggest discriminant ratio.
         It's a simple form of plotting the results.
         Use choose_beta_pretty for nice plotting.
         :param mode: defines moving markers
         :param fps: frames per second to be set;
                     pass as None to use the default fps
        """
        print("%s: choosing the beta (simple) with FPS = %s" % (self.MotionClass.__name__, fps))
        beta_range = 1e-6, 1e-3, 1e0, 1e1, 1e2, 1e3
        gained_ratios = []
        gained_rstds = []
        for beta in beta_range:
            print("BETA: %.1e" % beta)
            self.update_ratio(mode, beta, fps)
            gained_ratios.append(self.proj_info["d-ratio"])
            gained_rstds.append(self.proj_info["d-ratio-std"])

        ind = np.argmax(gained_ratios)
        best_ratio = gained_ratios[ind]
        best_beta = beta_range[ind]
        print("BEST RATIO: %g, w.r.t. beta = %.1e" % (best_ratio, best_beta))
        plt.errorbar(np.log10(beta_range), gained_ratios, gained_rstds,
                     linestyle='None', marker='^', ms=8)
        plt.xlabel("log(beta)")
        plt.ylabel("discriminant ratio R")
        plt.title("Choosing the best beta")
        plt.grid()
        plt.show()


    def choose_beta_pretty(self, mode, fps, reset=False):
        """
         Chooses the best beta to get the biggest discriminant ratio.
         It's a pretty version of plotting the results.
         Use a simple one, if you cannot get the logic.
         :param mode: defines moving markers
         :param fps: frames per second to be set;
                     pass as None to use the default fps
         :param reset: reset (True) or continue (False) progress
        """
        begin = time.time()
        print("%s: choosing the beta with FPS = %s" % (self.MotionClass.__name__, fps))
        if not os.path.exists("progress/choosing_beta.json"): reset = True
        beta_range = 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5
        if reset:
            betas_left = beta_range
            progress = {
                "betas_used": [],
                "wthnvars": [],
                "btwvars": [],
                "ratios": [],
                "wthnvar_stds": [],
                "btwvar_stds": [],
                "ratios_stds": [],
                "duration": 0
            }
            print("Reset progress.")
        else:
            progress = json.load(open("progress/choosing_beta.json"))
            start = len(progress["betas_used"])
            betas_left = beta_range[start:]
            print("Last computed beta was %.1e" % beta_range[start-1])

        for beta in betas_left:
            start_clock = time.ctime() + ":\t"
            print(start_clock + "PROCESSING BETA = %.1e" % beta)
            self.update_ratio(mode, beta, fps, verbose=False)

            progress["wthnvars"].append(self.proj_info["within_variance"])
            progress["btwvars"].append(self.proj_info["between_variance"])
            progress["ratios"].append(self.proj_info["d-ratio"])

            progress["wthnvar_stds"].append(self.proj_info["within_std"])
            progress["btwvar_stds"].append(self.proj_info["between_std"])
            progress["ratios_stds"].append(self.proj_info["d-ratio-std"])

            progress["betas_used"].append(beta)
            progress["duration"] += int(time.time() - begin)

            json.dump(progress, open("progress/choosing_beta.json", 'w'))

        ind_highlight = np.argmax(progress["ratios"])
        best_ratio = progress["ratios"][ind_highlight]
        best_beta = beta_range[ind_highlight]
        print("BEST RATIO: %g, w.r.t. beta = %.1e" % (best_ratio, best_beta))

        if None in progress["wthnvars"]:
            # only between-class variance is available
            plt.plot(np.log10(beta_range), progress["ratios"], 'b^-', ms=8)
            plt.ylabel("Db")
            std_pct = np.divide(progress["ratios_stds"], progress["ratios"])
            std_mean = 100. * np.average(std_pct)
            plt.legend(["Db, std=%.1f%%" % std_mean], numpoints=1, loc=3)
            plt.grid()
        else:
            keys = "wthnvars", "btwvars", "ratios"
            keys_std = "wthnvar_stds", "btwvar_stds", "ratios_stds"
            std_inf = []
            for i in range(3):
                std_pct = np.divide(progress[keys_std[i]], progress[keys[i]])
                std_pct = 100. * np.average(std_pct)
                std_inf.append("std=%.1f%%" % std_pct)
            legends = ["%s, %s" % pair for pair in zip(("Dw", "Db", "R"), std_inf)]
            markers = 'ys-', 'b^-', 'go-'
            for i, (key, lgnd, mark) in enumerate(zip(keys, legends, markers), start=1):
                plt.subplot(3, 1, i)
                plt.plot(np.log10(beta_range), progress[key], mark)
                plt.legend([lgnd], numpoints=1, loc=3)
                plt.ylabel(lgnd.split(',')[0])
                plt.grid()
        plt.ylim(0.995 * min(progress["ratios"]), 1.005 * max(progress["ratios"]))
        plt.xlabel("log(beta)")
        plt.suptitle("Choosing the best beta")
        plt.savefig("png/choosing_beta.png")
        json.dump(progress, open("progress/choosing_beta.json", 'w'))
        print("\t Duration: ~%d m" % (progress["duration"] / 60.))
        plt.show()


    def ratio_vs_fps(self, mode, beta, start, end, step=1, reset=False):
        """
         Displays discriminant ratio vs fps.
        :param mode: defines moving markers
        :param beta: (float), defines weights activity;
                      the best beta value is around 100;
                      set it to None to model when beta vanishes;
        :param start: lower fps
        :param end: upper fps
        :param step: fps step
        :param reset: reset (True) or continue (False) progress
        """
        if not os.path.exists("progress/ratio_vs_fps.json"): reset = True
        if reset:
            fps_left = np.arange(start, end+step, step)
            progress = {"fps_used": [], "r_got": [], "rstd_got": []}
            print("Reset progress.")
        else:
            progress = json.load(open("progress/ratio_vs_fps.json", 'r'))
            next_fps = progress["fps_used"][-1] + step
            print("Continue progress from FPS = %d." % next_fps)
            fps_left = np.arange(next_fps, end+step, step)

        for fps in fps_left:
            try:
                self.update_ratio(mode, beta, fps)
            except AssertionError:
                continue
            progress["fps_used"].append(int(fps))
            progress["r_got"].append(self.proj_info["d-ratio"])
            progress["rstd_got"].append(self.proj_info["d-ratio-std"])
            json.dump(progress, open("progress/ratio_vs_fps.json", 'w'))

        fps_used = progress["fps_used"]
        ratios = progress["r_got"]
        ratios_std = progress["rstd_got"]
        plt.errorbar(fps_used, ratios, ratios_std, marker='^', ms=8)
        plt.xlabel("FPS")
        mean_std = 100. * np.average(ratios_std)
        print("mean std: %.1f%%" % mean_std)
        plt.ylabel("R")
        plt.title("Discriminant ratio VS fps")
        plt.grid()
        plt.xlim(0, end + 1)
        plt.savefig("png/ratio_vs_fps.png")
        plt.show()
