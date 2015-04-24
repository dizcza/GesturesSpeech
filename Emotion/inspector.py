# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Button
import matplotlib.animation as animation
from Emotion.emotion import Emotion
from Emotion.preparation import EMOTION_PATH_PICKLES, \
    define_valid_face_actions, get_face_markers, get_face_areas
from Emotion.excel_parser import parse_xls
import os
import json


def load_emotions(specify):
    """
    :return: list of Emotion instances
    """
    database = []
    if specify is None:
        print("Loading all emotion database")
        for pkl_log in os.listdir(EMOTION_PATH_PICKLES):
            if pkl_log.endswith(".pkl"):
                pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log)
                em = Emotion(pkl_path)
                database.append(em)
    else:
        emotions_basket, _, _ = parse_xls()
        assert specify in emotions_basket, "invalid emotion is specified"
        print("Loading %s" % specify)
        for pkl_log in emotions_basket[specify]:
            pkl_path = os.path.join(EMOTION_PATH_PICKLES, pkl_log + ".pkl")
            em = Emotion(pkl_path)
            database.append(em)
    return tuple(database)


class Inspector(object):
    def __init__(self, emotion, reset):
        self.emotion = emotion
        self.checked_basket = set([])
        self.face_areas = get_face_areas()
        self.labels = get_face_markers()
        cashed_emotion = r"inspector_cache/%s.json" % emotion
        if not reset and os.path.exists(cashed_emotion):
            self._result = json.load(open(cashed_emotion, 'r'))
        else:
            self._result = init_empty_result()
        self.valid_actions = define_valid_face_actions()
        self.database = load_emotions(specify=emotion)
        self.toggles = {}
        self.iterator = 0
        self.current_obj = self.database[self.iterator]
        self.hot_ids = []
        self.scat = None
        self.display_only_unchecked = False
        self.check_buttons = None
        self.active_area = self.face_areas[0]

        self.act_axes = None
        self.fig = plt.figure()
        self.set_navigate_buttons()
        self.set_radio_buttons()
        self.set_switch_button()
        self.set_action_buttons()
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        plt.subplots_adjust(left=0.3, bottom=0.2)
        self.ax = self.fig.add_subplot(111)

    def handle_close(self, event):
        """
         Saves before shutting down.
        """
        self.save_result(None)

    def show(self):
        self.animate()
        try:
            plt.show()
        except AttributeError:
            pass

    def set_navigate_buttons(self):
        """
         Sets revert, prev and next buttons to move around emotion files.
        """
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_revert = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.5, 0.05, 0.1, 0.075])

        bnext = Button(ax_next, "Next")
        bprev = Button(ax_prev, "Previous")
        brevert = Button(ax_revert, "Revert")
        bsave = Button(ax_save, "Save")

        bnext.on_clicked(self.take_next)
        bprev.on_clicked(self.take_prev)
        brevert.on_clicked(self.revert)
        bsave.on_clicked(self.save_result)

    def set_radio_buttons(self):
        """
         Sets radio buttons to track specific face place.
        """
        rax = plt.axes([0.05, 0.52, 0.2, 0.22], axisbg="lightgoldenrodyellow")
        radio = RadioButtons(rax, self.face_areas)
        radio.on_clicked(self.choose_face_part)

    def set_switch_button(self):
        """
         Switches check mode.
        """
        rax = plt.axes([0.05, 0.75, 0.2, 0.15], axisbg="lightgoldenrodyellow")
        check = CheckButtons(rax, ("only \nunchecked",), (self.display_only_unchecked,))
        check.on_clicked(self.choose_unchecked)

    def update_action_buttons(self):
        """
         Updates valid actions for the tracking face area.
        """
        if self.check_buttons is not None:
            self.check_buttons.disconnect_events()
        self.act_axes.clear()
        menu = self.valid_actions[self.active_area]
        loaded_act = self._result[self.current_obj.fname][self.active_area]
        self.toggles = {}
        bool_vals = []
        for act in menu:
            display_it = act in loaded_act
            self.toggles[act] = display_it
            bool_vals.append(display_it)
        self.check_buttons = CheckButtons(self.act_axes, menu, bool_vals)
        self.check_buttons.on_clicked(self.choose_action)

    def set_action_buttons(self):
        """
         Sets valid actions for the tracking face area.
        """
        self.act_axes = plt.axes([0.05, 0.2, 0.2, 0.3], axisbg="lightblue")
        self.update_action_buttons()

    def choose_action(self, label):
        """
        :param label: valid face area action
        """
        self.toggles[label] = not self.toggles[label]

    def choose_unchecked(self, label):
        """
         Choose manually to display only unchecked emotion objects or all of them.
        :param label == "only unchecked"
        """
        self.display_only_unchecked = not self.display_only_unchecked

    def choose_face_part(self, face_part):
        """
         Chooses what face part to be displayed.
        :param face_part: is it mouth, eyes, ...
        """
        self.active_area = face_part
        self.update_action_buttons()
        self.animate()

    def remember_last_choice(self):
        """
         Remembers the last choice in the current fname, w.r.t. active area.
        """
        outcome = [act for act in self.toggles if self.toggles[act]]
        if not any(outcome):
            outcome = ["default"]
            msg = "%s (%s)" % (self.current_obj.fname, self.active_area)
            print("WARNING: got empty action in ", msg)
        self._result[self.current_obj.fname][self.active_area] = outcome
        self.checked_basket.add(self.current_obj.fname)

    def save_result(self, event):
        """
         Saves the results.
        """
        print("Saved in %s.json" % self.emotion)
        for fname in self.checked_basket:
            self._result[fname]["is_checked"] = True
        json.dump(self._result, open(r"inspector_cache/%s.json" % self.emotion, 'w'))

    def take_next(self, event):
        """
         Takes next emotion object.
        :param event: mouse click event
        """
        self.remember_last_choice()
        seek_next = True
        if self.iterator == len(self.database) - 1:
            print("THE END.")
            seek_next = False

        while seek_next:
            self.iterator = min(len(self.database) - 1, self.iterator + 1)
            self.current_obj = self.database[self.iterator]
            already_checked = self._result[self.current_obj.fname]["is_checked"]
            seek_next = already_checked and self.iterator < len(self.database) - 1
            seek_next = seek_next and self.display_only_unchecked
        self.update_action_buttons()
        self.animate()

    def take_prev(self, event):
        """
         Takes previous emotion object.
        :param event: mouse click event
        """
        self.remember_last_choice()
        seek_prev = True
        if self.iterator == 0:
            seek_prev = False
            print("BEGINNING.")

        while seek_prev:
            self.iterator = max(0, self.iterator - 1)
            self.current_obj = self.database[self.iterator]
            already_checked = self._result[self.current_obj.fname]["is_checked"]
            seek_prev = already_checked and self.iterator > 0
            seek_prev = seek_prev and self.display_only_unchecked
        self.update_action_buttons()
        self.animate()

    def revert(self, event):
        """
         Reverts the iterator at the beginning.
        :param event: mouse click event
        """
        self.iterator = 0
        self.current_obj = self.database[self.iterator]
        print("Returned back.")
        self.animate()

    def next_frame(self, frame):
        """
        :param frame: frame ID to be displayed
        """
        frame = frame % self.current_obj.frames
        self.scat.set_offsets(self.current_obj.data[:, frame, :])
        return []

    def animate(self):
        """
         Animates current emotion file.
        """
        if self.scat is not None:
            self.scat.remove()
        self.ax.clear()
        self.ax.grid()

        self.hot_ids = self.current_obj.get_ids(*self.labels[self.active_area])
        markers = self.current_obj.data.shape[0]
        sizes = np.ones(markers) * 30
        rgba_colors = np.zeros(shape=(markers, 4))
        for rgb in range(3):
            rgba_colors[:, rgb] = 0.5
        rgba_colors[:, 3] = 0.5
        for interested_id in self.hot_ids:
            sizes[interested_id] *= 2
            rgba_colors[interested_id, 1] = 0
            rgba_colors[interested_id, 2] = 1
            rgba_colors[interested_id, 3] = 1

        self.scat = plt.scatter(self.current_obj.data[:, 0, 0],
                                self.current_obj.data[:, 0, 1], color=rgba_colors, s=sizes)
        title = "%s: %s" % (self.current_obj.fname, self.current_obj.emotion)
        self.ax.set_title(title)

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=self.current_obj.frames,
                                       interval=1,
                                       blit=True)
        try:
            plt.draw()
        except AttributeError:
            pass


class CheckInspector(Inspector):
    def __init__(self, emotion, active_area="mouth"):
        Inspector.__init__(self, emotion, reset=False)
        self.display_only_unchecked = False
        self.active_area = active_area
        self.both = self._result[self.current_obj.fname][self.active_area]

    def set_radio_buttons(self):
        ax_mouth = plt.axes([0.05, 0.8, 0.15, 0.075], axisbg="lightgoldenrodyellow")
        ax_eyes = plt.axes([0.05, 0.7, 0.15, 0.075], axisbg="lightgoldenrodyellow")
        ax_eyebrows = plt.axes([0.05, 0.6, 0.15, 0.075], axisbg="lightgoldenrodyellow")
        ax_cheeks = plt.axes([0.05, 0.5, 0.15, 0.075], axisbg="lightgoldenrodyellow")
        ax_nostrils = plt.axes([0.05, 0.4, 0.15, 0.075], axisbg="lightgoldenrodyellow")

        b_mouth = Button(ax_mouth, "track \nmouth")
        b_eyes = Button(ax_eyes, "track \neyes")
        b_eyebrows = Button(ax_eyebrows, "track \neyebrows")
        b_cheeks = Button(ax_cheeks, "track \ncheeks")
        b_nostrils = Button(ax_nostrils, "track \nnostrils")

        b_mouth.on_clicked(self.track_mouth)
        b_eyes.on_clicked(self.track_eyes)
        b_eyebrows.on_clicked(self.track_eyebrows)
        b_cheeks.on_clicked(self.track_cheeks)
        b_nostrils.on_clicked(self.track_nostrils)

    def track_mouth(self, event):
        self.active_area = "mouth"
        self.save_and_move_to_next()

    def track_eyes(self, event):
        self.active_area = "eyes"
        self.save_and_move_to_next()

    def track_eyebrows(self, event):
        self.active_area = "eyebrows"
        self.save_and_move_to_next()

    def track_cheeks(self, event):
        self.active_area = "cheeks"
        self.save_and_move_to_next()

    def track_nostrils(self, event):
        self.active_area = "nostrils"
        self.save_and_move_to_next()

    def save_and_move_to_next(self):
        self.save_result(None)
        self.iterator = -1
        self.take_next(None)

    def set_action_buttons(self):
        ax_second = plt.axes([0.15, 0.2, 0.1, 0.05], axisbg="blue")
        ax_first = plt.axes([0.02, 0.2, 0.1, 0.05], axisbg="blue")
        ax_reject = plt.axes([0.02, 0.05, 0.23, 0.14], axisbg="blue")
        ax_raw_input = plt.axes([0.02, 0.26, 0.23, 0.05], axisbg="blue")

        bsecond = Button(ax_second, "Second")
        bfirst = Button(ax_first, "First")
        breject = Button(ax_reject, "Reject")
        braw_input = Button(ax_raw_input, "Enter (raw input)")

        bsecond.on_clicked(self.choose_second)
        bfirst.on_clicked(self.choose_first)
        breject.on_clicked(self.reject_action)
        braw_input.on_clicked(self.enter_an_action)

    def choose_second(self, event):
        if len(self.both) == 2:
            second = [self.both[1]]
            self._result[self.current_obj.fname][self.active_area] = second
            print("\t --> chose second: %s" % second)

    def choose_first(self, event):
        if len(self.both) == 2:
            first = [self.both[0]]
            self._result[self.current_obj.fname][self.active_area] = first
            print("\t --> chose first: %s" % first)

    def reject_action(self, event):
        self._result[self.current_obj.fname][self.active_area] = ["(undef)"]
        print("\t --> chose rejected: (undef)")

    def enter_an_action(self, event):
        act = input("\t -- > Enter an action: ")
        self._result[self.current_obj.fname][self.active_area] = [act]
        print("\t --> chose first: %s" % act)

    def __len__(self):
        return len(self._result[self.current_obj.fname][self.active_area])

    def show(self):
        if len(self) < 2:
            self.take_next(None)
            plt.show()
        else:
            Inspector.show(self)

    def take_next(self, event):
        """
         Takes next already inspected emotion object,
         which has both possible actions, w.r.t. active face area.
        :param event: mouse click event
        """
        seek_next = True
        if self.iterator == len(self.database) - 1:
            print("THE END.")
            seek_next = False

        while seek_next:
            self.iterator = min(len(self.database) - 1, self.iterator + 1)
            self.current_obj = self.database[self.iterator]
            seek_next = self.iterator < len(self.database) - 1 and len(self) < 2
        self.both = self._result[self.current_obj.fname][self.active_area]
        self.animate()

    def take_prev(self, event):
        """
         Takes prev already inspected emotion object,
         which has both possible actions, w.r.t. active face area.
        :param event: mouse click event
        """
        seek_prev = True
        if self.iterator == 0:
            seek_prev = False
            print("BEGINNING.")

        while seek_prev:
            self.iterator = max(0, self.iterator - 1)
            self.current_obj = self.database[self.iterator]
            seek_prev = self.iterator > 0 and len(self) < 2
        self.both = self._result[self.current_obj.fname][self.active_area]
        self.animate()

    def animate(self):
        self.current_obj.data = self.current_obj.norm_data
        Inspector.animate(self)
        print("%s: %s" % (self.active_area, self.both))

    def set_switch_button(self):
        pass
    
    def update_action_buttons(self):
        pass


def init_empty_result():
    db = load_emotions(None)
    face_areas = get_face_areas()
    a_dic = {}
    labels = get_face_markers()
    for em in db:
        em_info = {
            "is_checked": False,
            "emotion": em.emotion
        }
        for fplace in face_areas:
            em_info[fplace] = ["default"]

        if em.emotion == u"улыбка":
            em_info["mouth"] = ["smile"]
            em_info["cheeks"] = ["default", "up"]

        elif em.emotion == u"закрыл глаза":
            em_info["eyes"] = ["closed"]

        elif em.emotion in (u"отвращение", u"пренебрежение", u"так себе"):
            em_info["mouth"] = ["default", "shifted"]

        elif em.emotion == u"боль":
            em_info["eyes"] = ["closed"]
            em_info["mouth"] = ["open"]
            em_info["cheeks"] = ["up"]
            em_info["nostrils"] = ["up"]
            em_info["eyebrows"] = ["default", "down"]

        elif em.emotion == u"ярость":
            em_info["mouth"] = ["open"]
            em_info["cheeks"] = ["up"]
            em_info["nostrils"] = ["up"]
            em_info["eyebrows"] = ["default", "up"]

        elif em.emotion == u"ужас":
            em_info["mouth"] = ["open"]
            em_info["nostrils"] = ["up"]
            em_info["eyebrows"] = ["up"]

        elif em.emotion == u"озадаченность":
            em_info["mouth"] = ["default", "shifted"]
            em_info["eyebrows"] = ["default", "down"]

        elif em.emotion == u"удивление":
            em_info["mouth"] = ["default", "open"]
            em_info["eyebrows"] = ["up"]

        elif em.emotion == u"плакса":
            em_info["nostrils"] = ["up"]
            em_info["eyebrows"] = ["down"]
            em_info["eyes"] = ["closed"]

        for fplace in face_areas:
            ids = em.get_ids(*labels[fplace])
            if np.isnan(em.data[ids, ::]).any():
                em_info[fplace].append("(undef)")

        a_dic[em.fname] = em_info
    return a_dic


def merge_result():
    """
     Merges all json dictionaries of emotions into one dic.
    """
    remove_undefined_and_unplaced_emotion_file_names()
    merged_dic = {}
    for json_log in os.listdir(r"inspector_cache"):
        if json_log.endswith(".json"):
            a_dic = json.load(open(r"inspector_cache/%s" % json_log, 'r'))
            merged_dic.update(a_dic)
    print("Merged %d files." % len(merged_dic))
    json.dump(merged_dic, open("face_structure_merged.json", 'w'))


def remove_undefined_and_unplaced_emotion_file_names():
    """
     For each json dict, stored in 'inspector_cache' folder,
     removes the file name if it either
        - corresponds to undefined emotion (those who not in emotion_basket)
        - corresponds to other emotion, that differs from json dict name
    """
    emotion_basket, _, _ = parse_xls()
    for json_log in os.listdir("inspector_cache"):
        if json_log.endswith(".json"):
            em_dic = json.load(open(r"inspector_cache/%s" % json_log, 'r'))
            undef_fnames = []
            other_emotion_fnames = []
            for fname in em_dic:
                if em_dic[fname]["emotion"] not in emotion_basket:
                    undef_fnames.append(fname)
                elif em_dic[fname]["emotion"] != json_log[:-5]:
                    other_emotion_fnames.append(fname)
            for bad_name in undef_fnames + other_emotion_fnames:
                del em_dic[bad_name]
            json.dump(em_dic, open(r"inspector_cache/%s" % json_log, 'w'))


def eyes_test():
    emotions_basket, _, _ = parse_xls()
    for emotion in emotions_basket:
        try:
            CheckInspector(emotion, "eyes").show()
        except AttributeError:
            pass


if __name__ == "__main__":
    # Inspector(u"ужас", reset=False).show()
    # CheckInspector(u"плакса", "eyes").show()
    merge_result()
    # remove_undefined()
    # eyes_test()
