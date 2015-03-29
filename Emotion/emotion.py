# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

EMOTION_PATH_DATA = "D:\GesturesDataset\Emotion\\txt\\data"


class Emotion(object):
    def __init__(self, obj_path, fps=None):
        self.project = "Emotion"
        self.name = obj_path.split('\\')[-1].strip(".pkl")
        # self.labels = np.genfromtxt('valid_labels.txt', dtype='str')

        # not for sure
        self.fps = 30

        # loading data from a pickle
        info = pickle.load(open(obj_path, 'rb'))
        self.data = info["data"]
        self.author = info["author"]
        self.emotion = info["emotion"]
        self.labels = info["labels"]
        self.frames = self.data.shape[1]

        self.set_fps(fps)
        self.preprocessor()

    def __str__(self):
        s = "File name:\t\t%s\n" % self.name
        s += "\temotion:\t%s\n" % self.emotion
        s += "\tauthor:\t\t%s\n" % self.author
        s += "\tdata.shape:\t%s\n" % str(self.data.shape)
        return s

    def preprocessor(self):
        # TODO implement me
        pass

    def set_fps(self, new_fps):
        """
            Modify data, w.r.t. new fps (taken from HumanoidBasic class).
        :param new_fps: fps (or points frequency) to be made in the data
        """
        if new_fps is None or new_fps >= self.fps:
            # does nothing
            return
        step_to_throw = float(self.fps) / (self.fps - new_fps)
        indices_thrown = np.arange(self.data.shape[1]) * step_to_throw
        indices_thrown = indices_thrown.astype(dtype="int")
        self.data = np.delete(self.data, indices_thrown, axis=1)
        self.frames = self.data.shape[1]
        self.fps = new_fps

    def next_frame(self, frame):
        """
        :param frame: frame id
        """
        self.scat.set_offsets(self.data[:, frame, :])
        return []

    def animate(self):
        """
         Animates 3d data.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scat = plt.scatter(self.data[:, 0, 0], self.data[:, 0, 1])
        self.ax.grid()

        anim = animation.FuncAnimation(self.fig,
                                       func=self.next_frame,
                                       frames=self.frames,
                                       interval=50.,     # in ms
                                       blit=True)
        try:
            plt.show(self.fig)
        except AttributeError:
            pass


if __name__ == "__main__":
    em = Emotion(r"D:\GesturesDataset\Emotion\txt\data\28-4-1.pkl")
    em.animate()
    print(em)