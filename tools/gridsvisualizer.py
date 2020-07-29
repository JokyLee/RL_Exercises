#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.27'


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environments import gridworld


class GridsVisualizer:
    def __init__(self, name, gridHeight, gridWidth):
        self.name = name
        self.fig = None
        self.grid_h = gridHeight
        self.grid_w = gridWidth
        self.cmap = matplotlib.cm.viridis

    def onInteractive(self):
        self.fig = plt.figure()
        plt.ion()

    def offInteractive(self):
        plt.close(self.fig)
        plt.ioff()
        self.fig = None

    def _drawArrow(self, x, y, action, prob):
        if prob == 0:
            return
        action = gridworld.Actions(action)
        if action == gridworld.Actions.UP:
            dir = (0, -0.5 * prob)
        elif action == gridworld.Actions.DOWN:
            dir = (0, 0.5 * prob)
        elif action == gridworld.Actions.LEFT:
            dir = (-0.5 * prob, 0)
        elif action == gridworld.Actions.RIGHT:
            dir = (0.5 * prob, 0)
        else:
            raise ValueError("Invalid action.")
        plt.arrow(x, y, *dir, fill=False, length_includes_head=True, head_width=0.1, alpha=0.5)

    def visualize(self, values, policy, episodeNum):
        if self.fig is None:
            fig = plt.figure()
        else:
            fig = self.fig

        fig.clear()
        plt.xticks([])
        plt.yticks([])
        im = plt.imshow(values.reshape(self.grid_h, self.grid_w), cmap=self.cmap, interpolation='nearest', origin='upper')

        for state in range(policy.shape[0]):
            for action in range(policy.shape[1]):
                y, x = np.unravel_index(state, (self.grid_h, self.grid_w))
                self._drawArrow(x, y, action, policy[state][action])

        plt.title((("" or self.name) + "\n") + "Predicted Values, Episode: %d" % episodeNum)
        plt.colorbar(im, orientation='horizontal')
        fig.canvas.draw()
        plt.show()
