# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation
from Emotion.FaceLocations.inspector import Inspector


class MultipleAnimation(Inspector):
    def __init__(self, emotion):
        Inspector.__init__(self, emotion, reset=False)
        self.scatters = []
        self.objects = []
        self.plot_sizes = None
        self.rgba_colors = None
        self.next_it = 0

    def set_navigate_buttons(self):
        """
         Sets a button to add the next object to display it.
        """
        Inspector.set_navigate_buttons(self)
        ax_add = plt.axes([0.1, 0.05, 0.1, 0.075])
        b_add = Button(ax_add, "Add")
        b_add.on_clicked(self.add_next)

    def add_next(self, event):
        self.next_it = (self.next_it + 1) % len(self.database)
        next_obj = self.database[self.next_it]
        if next_obj not in self.objects:
            self.objects.append(next_obj)
            rand_shift = np.random.random(3)
            rgba_colors = self.rgba_colors.copy()
            for interested_id in self.hot_ids:
                rgba_colors[interested_id, :-1] = rand_shift

            scat = plt.scatter(next_obj.norm_data[:, 0, 0],
                               next_obj.norm_data[:, 0, 1],
                               color=rgba_colors,
                               s=self.plot_sizes)
            self.scatters.append(scat)
            title = self.ax.get_title()
            title += ", %s" % next_obj.fname
            self.ax.set_title(title)
            print("Added %s." % next_obj.fname)

    def next_frame(self, frame):
        """
        :param frame: frame ID to be displayed
        """
        for obj_id in range(len(self.objects)):
            obj = self.objects[obj_id]
            obj_frame = frame % obj.frames
            self.scatters[obj_id].set_offsets(obj.norm_data[:, obj_frame, :])
        return []

    def define_plot_style(self):
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
        self.rgba_colors = rgba_colors
        self.plot_sizes = sizes

    def animate(self):
        """
         Animates current emotion file.
        """
        if self.scat is not None:
            self.scat.remove()
        self.ax.clear()
        self.ax.grid()

        self.define_plot_style()

        title = "%s: %s" % (self.current_obj.emotion, self.current_obj.fname)
        scat = plt.scatter(self.current_obj.norm_data[:, 0, 0],
                           self.current_obj.norm_data[:, 0, 1],
                           color=self.rgba_colors,
                           s=self.plot_sizes)
        self.objects = [self.current_obj]
        self.scatters = [scat]
        self.ax.set_title(title)
        self.next_it = self.iterator

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=self.current_obj.frames,
                                       interval=1,
                                       blit=True)
        try:
            plt.draw()
        except AttributeError:
            pass


if __name__ == "__main__":
    MultipleAnimation(u"улыбка").show()