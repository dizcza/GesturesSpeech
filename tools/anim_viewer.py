# coding=utf-8

"""
A simple OpenGL MoCap Coordinate 3D viewer.                 # MocapViewer
Also provides a visualizer for arrays of 3D motion data.    # DataViewer
Original: https://github.com/EmbodiedCognition/py-c3d/blob/master/scripts/c3d-viewer
Required packages:
    - c3d: https://github.com/EmbodiedCognition/py-c3d
    - climate: http://github.com/lmjohns3/py-cli
    - curses: http://www.lfd.uci.edu/~gohlke/pythonlibs/#curses
    - pyglet: http://pyglet.readthedocs.org/en/pyglet-1.2-maintenance
    - numpy: http://sourceforge.net/projects/numpy/?source=directory
"""

import c3d
import climate
import collections
import contextlib
import numpy as np
import pyglet

from pyglet.gl import *

climate.add_arg('inputs', nargs='+', metavar='FILE', help='show these c3d files')

BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0.2, 0.2)
YELLOW = (1, 1, 0.2)
ORANGE = (1, 0.7, 0.2)
GREEN = (0.2, 0.9, 0.2)
BLUE = (0.2, 0.3, 0.9)
COLORS = (WHITE, RED, YELLOW, GREEN, BLUE, ORANGE)


@contextlib.contextmanager
def gl_context(scale=None, translate=None, rotate=None, mat=None):
    glPushMatrix()
    if mat is not None:
        glMultMatrixf(vec(*mat))
    if translate is not None:
        glTranslatef(*translate)
    if rotate is not None:
        glRotatef(*rotate)
    if scale is not None:
        glScalef(*scale)
    yield
    glPopMatrix()


def vec(*args):
    return (GLfloat * len(args))(*args)


def sphere_vertices(n=2):
    idx = [[0, 1, 2], [0, 5, 1], [0, 2, 4], [0, 4, 5],
           [3, 2, 1], [3, 4, 2], [3, 5, 4], [3, 1, 5]]
    vtx = list(np.array([
        [ 1, 0, 0], [0,  1, 0], [0, 0,  1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1]], 'f'))
    for _ in range(n):
        idx_ = []
        for ui, vi, wi in idx:
            u, v, w = vtx[ui], vtx[vi], vtx[wi]
            d, e, f = u + v, v + w, w + u
            di = len(vtx)
            vtx.append(d / np.linalg.norm(d))
            ei = len(vtx)
            vtx.append(e / np.linalg.norm(e))
            fi = len(vtx)
            vtx.append(f / np.linalg.norm(f))
            idx_.append([ui, di, fi])
            idx_.append([vi, ei, di])
            idx_.append([wi, fi, ei])
            idx_.append([di, ei, fi])
        idx = idx_
    vtx = np.array(vtx, 'f').flatten()
    return np.array(idx).flatten(), vtx, vtx


class Viewer(pyglet.window.Window):
    """
     An abstract Viewer. Should be overridden with provided data source.
    """
    def __init__(self, trace=None, paused=False):
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        try:
            config = screen.get_best_config(Config(
                alpha_size=8,
                depth_size=24,
                double_buffer=True,
                sample_buffers=1,
                samples=4))
        except pyglet.window.NoSuchConfigException:
            config = screen.get_best_config(Config())

        super(Viewer, self).__init__(
            width=800, height=650, resizable=True, vsync=False, config=config)

        # ------------------ BEGIN -------------------- #
        # -- THESE THREE PARAMS SHOULD BE OVERRIDDEN -- #
        """
        1) generator of (frame_no, points, analog) pairs
        2) frame rate -- frames per second
        3) marker trails
        """
        self._frames = iter([])
        self._frame_rate = 0
        self._trails = []
        # ------------------- END --------------------- #

        self.clock = pyglet.clock.get_default()
        self._maxlen = 16   # max trace length (in frames)
        self.interval = 0   # time interval between two frames
        self._frame_id = 0

        self.trace = trace
        self.paused = paused

        self.zoom = 2.2
        self.ty = 0
        self.tz = -1
        self.ry = 30
        self.rz = -50

        self.on_resize(self.width, self.height)

        glEnable(GL_BLEND)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glEnable(GL_POLYGON_SMOOTH)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LEQUAL)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, vec(3.0, 3.0, 10.0, 1.0))
        glEnable(GL_LIGHT0)

        BLK = [100, 100, 100] * 6
        WHT = [150, 150, 150] * 6
        N = 10
        z = 0
        vtx = []
        for i in range(N, -N, -1):
            for j in range(-N, N, 1):
                vtx.extend((j,   i, z, j, i-1, z, j+1, i,   z,
                            j+1, i, z, j, i-1, z, j+1, i-1, z))

        self.floor = pyglet.graphics.vertex_list(
            len(vtx) // 3,
            ('v3f/static', vtx),
            ('c3B/static', ((BLK + WHT) * N + (WHT + BLK) * N) * N),
            ('n3i/static', [0, 0, 1] * (len(vtx) // 3)))

        idx, vtx, nrm = sphere_vertices()
        self.sphere = pyglet.graphics.vertex_list_indexed(
            len(vtx) // 3, idx, ('v3f/static', vtx), ('n3f/static', nrm))

    def on_mouse_scroll(self, x, y, dx, dy):
        if dy == 0: return
        self.zoom *= 1.1 ** (-1 if dy < 0 else 1)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:
            # pan
            self.ty += 0.03 * dx
            self.tz += 0.03 * dy
        else:
            # roll
            self.ry += 0.2 * -dy
            self.rz += 0.2 * dx
        # print('z', self.zoom, 't', self.ty, self.tz, 'r', self.ry, self.rz)

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glu.gluPerspective(45, float(width) / height, 1, 100)

    def change_interval(self, multiplier):
        """
         Changes FPS via changing clock interval between two frames.
         :param multiplier: (float) how long/short new interval will be
        """
        self.interval *= float(multiplier)
        self.clock.unschedule(self.update)
        self.clock.schedule_interval(self.update, self.interval)
        print("FPS SET: %g; \t FPS DISPLAYED: %g" % (1. / self.interval, self.clock.get_fps()))

    def on_key_press(self, key, modifiers):
        k = pyglet.window.key
        if key == k.ESCAPE:
            pyglet.app.exit()
        elif key == k.SPACE:
            self.paused = False if self.paused else True
            print("PAUSED" if self.paused else "CONTINUE")
        elif key in (k.PLUS, k.EQUAL, k.NUM_ADD):
            self.change_interval(0.5)
        elif key in (k.MINUS, k.UNDERSCORE, k.NUM_SUBTRACT):
            self.change_interval(2)
        elif key == k.RIGHT:
            # skips 1 sec
            skip = int(self._frame_rate)
            for _ in range(skip):
                self._next_frame()

    def on_draw(self):
        self.clear()

        # http://njoubert.com/teaching/cs184_fa08/section/sec09_camera.pdf
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1, 0, 0, 0, 0, 0, 0, 0, 1)
        glTranslatef(-self.zoom, 0, 0)
        glTranslatef(0, self.ty, self.tz)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz, 0, 0, 1)

        self.floor.draw(GL_TRIANGLES)

        for t, trail in enumerate(self._trails):
            glColor4f(*(COLORS[t % len(COLORS)] + (0.7, )))
            point = None
            glBegin(GL_LINES)
            for point in trail:
                glVertex3f(*point)
            glEnd()
            with gl_context(translate=point, scale=(0.02, 0.02, 0.02)):
                self.sphere.draw(GL_TRIANGLES)

    def _reset_trails(self):
        self._trails = [collections.deque(t, self._maxlen) for t in self._trails]

    def _next_frame(self):
        # saves current frame
        # pyglet.image.get_buffer_manager().get_color_buffer().save("%d.png" % self._frame_id)
        self._frame_id += 1
        return next(self._frames)

    def update(self, dt):
        if self.paused:
            return
        for trail, point in zip(self._trails, self._next_frame()[1]):
            if point[3] > -1 or not len(trail):
                trail.append(point[:3] / 1000.)
            else:
                trail.append(trail[-1])

    def mainloop(self, slow_down=1):
        self.interval = float(slow_down) / self._frame_rate
        self.clock.schedule_interval(self.update, self.interval)
        pyglet.app.run()


class MocapViewer(Viewer):
    def __init__(self, c3d_reader):
        """
        :param c3d_reader: Reader from a c3d module with a given path to c3d file.
        """
        Viewer.__init__(self)
        # ------------------ BEGIN -------------------- #
        self._frames = c3d_reader.read_frames(copy=False)
        self._frame_rate = c3d_reader.header.frame_rate
        self._trails = [[] for _ in range(c3d_reader.point_used)]
        # ------------------- END --------------------- #
        self._reset_trails()


class DataViewer(Viewer):
    def __init__(self, data, rate):
        """
        :param data: (#frames, #markers, #dim) ndarray
                     of XYZ of body joint markers;
                     should be measured in mm
                     with feet on the floor
        :param rate: frames per sec
        """
        Viewer.__init__(self)
        self._maxlen = 5
        frame_no = np.arange(data.shape[0])
        analog = [[] for _ in frame_no]
        # ------------------ BEGIN -------------------- #
        self._frames = iter(zip(frame_no, data, analog))
        self._frame_rate = rate
        self._trails = [[] for _ in range(data.shape[1])]
        # ------------------- END --------------------- #
        self._reset_trails()


def demo():
    c3d_file_path = r"D:\GesturesDataset\MoCap\Hospital\H2_mcraw.c3d"
    try:
        MocapViewer(c3d.Reader(open(c3d_file_path, 'rb'))).mainloop()
    except StopIteration:
        pass


if __name__ == '__main__':
    demo()
