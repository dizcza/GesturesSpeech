# coding=utf-8

import os
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from MOCAP.mreader import HumanoidUkr, MOCAP_PATH

TRANSPARENCY = 0.7
BLACK = (0, 0, 0, TRANSPARENCY)
WHITE = (1, 1, 1, TRANSPARENCY)
RED = (1, 0.2, 0.2, TRANSPARENCY)
YELLOW = (1, 1, 0.2, TRANSPARENCY)
ORANGE = (1, 0.7, 0.2, TRANSPARENCY)
GREEN = (0.2, 0.9, 0.2, TRANSPARENCY)
BLUE = (0.2, 0.3, 0.9, TRANSPARENCY)
COLORS = (WHITE, RED, YELLOW, GREEN, BLUE, ORANGE)


MOUSE_LEFT_BUTTON = 1
MOUSE_SCROLL_BUTTON = 2
MOUSE_SCROLL_UP = 4
MOUSE_SCROLL_DOWN = 5

SCROLL_STEP = 1



def draw_points(data, frame):
    glBegin(GL_POINTS)
    for marker in range(data.shape[0]):
        glColor4fv(COLORS[marker % len(COLORS)])
        glVertex3fv(data[marker, frame, :])
    glEnd()




def main():
    gest_path = r"D:\GesturesDataset\MoCap\Hospital\H2_mcraw.c3d"
    assert os.path.exists(gest_path), "Unable to find the %s" % gest_path
    gest = HumanoidUkr(gest_path)
    print(gest)

    pygame.init()
    width = 800
    height = 600
    zNear = 0.0
    zFar = 40.0
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    gluPerspective(45, float(width) / height, zNear, zFar)
    gluLookAt(0, 4, 0, 0, 0, 0, 0, 0, 1)

    glAlphaFunc(GL_NOTEQUAL, 0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)

    glPointSize(8)

    scroll_amount = 0
    is_scroll_button_pressed = False
    zoomFactor = 1.01

    for frame in range(gest.data.shape[1]):
        camera_pos = glGetDoublev(GL_MODELVIEW_MATRIX)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(event.button)
                if event.button == MOUSE_SCROLL_BUTTON:
                    is_scroll_button_pressed = True
                elif event.button == MOUSE_SCROLL_UP:
                    # glTranslatef(0.0, -0.5, 0)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    # gluPerspective(45.0, float(width) / height, zNear, zFar)
                    glScalef(1, 1, 1)
                elif event.button == MOUSE_SCROLL_DOWN:
                    zoomFactor = 0.99
                    gluPerspective(45.0, float(width) / height, zNear, zFar)
                    # glTranslatef(0.0, 0.5, 0)

            elif event.type == pygame.MOUSEMOTION and is_scroll_button_pressed:
                # rotate around own axis _|_ lookAt
                print(pygame.mouse.get_rel())

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == MOUSE_SCROLL_BUTTON:
                    is_scroll_button_pressed = False

        # glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_points(gest.data, frame)
        pygame.display.flip()


main()
