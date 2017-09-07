#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def pta(*args, color_rotate=True):
    '''pta = plot them all'''
    count = len(args)
    w = 20
    h = 4 * count
    fig = plt.figure(1, figsize=(w, h))
    for i, a in enumerate(args):
        ax = fig.add_subplot(count, 1, i+1)
        if color_rotate:
            for _ in range(i):
                ax.plot([], [])
        ax.plot(a)


def ptt(*args):
    '''ptt = plot them together'''
    w = 20
    h = 4
    fig = plt.figure(1, figsize=(w, h))
    ax = fig.add_subplot(1, 1, 1)
    for a in args:
        ax.plot(a)
