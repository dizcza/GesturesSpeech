from Emotion.FaceLocations.inspector import Inspector
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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


if __name__ == "__main__":
    CheckInspector("улыбка").show()