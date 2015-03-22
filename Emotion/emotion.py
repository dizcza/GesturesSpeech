# coding=utf-8

from humanoid import HumanoidBasic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cPickle as pickle

FACE_PATH_DATA = "D:\GesturesDataset\FACE\\txt\\data"


class Emotion(object):
    def __init__(self, obj_path, fps=None):
        self.project = "Emotion"
        self.name = obj_path.split('\\')[-1].strip(".pkl")
        # self.labels = np.genfromtxt('valid_labels.txt', dtype='str')

        info = pickle.load(open(obj_path, 'rb'))
        self.data = info["data"]
        self.author = info["author"]
        self.emotion = info["emotion"]
        self.labels = info["labels"]
        self.frames = self.data.shape[1]

    def __str__(self):
        s = "File name:\t\t%s\n" % self.name
        s += "\temotion:\t%s\n" % self.emotion
        s += "\tauthor:\t\t%s\n" % self.author
        s += "\tdata.shape:\t%s\n" % str(self.data.shape)
        return s

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
                                       interval=10.,     # in ms
                                       blit=True)
        try:
            plt.show(self.fig)
        except AttributeError:
            pass


if __name__ == "__main__":
    em = Emotion("D:\GesturesDataset\FACE\\txt\data\\28-4-1.pkl")
    # em.animate()
    print em