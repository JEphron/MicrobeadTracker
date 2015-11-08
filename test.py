import cv2
import numpy as np
import os
from os import listdir
from os.path import splitext
import sys
import itertools
import collections
import math
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':
    # todo: print a usage string (argv[1] is directory to scan)
    # assert len(sys.argv) == 3, "WRONG ARGS"

    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])

    # get all of the files from the directory
    dir_path = sys.argv[1]
    files = filter(lambda x: splitext(x)[1] == '.pgm', listdir(dir_path))
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    files = [os.path.join(dir_path, x) for x in files]

    # read files
    left, top, right, bottom = 0, 0, 1024, 1024
    imgs = [cv2.imread(x, 0)[left:right, top:bottom]
            for x in files[start_frame:end_frame]]

    divisor = 1
    imgs = [cv2.resize(x, (x.shape[1] / divisor, x.shape[0] / divisor))
            for x in imgs]

    print 'read in', end_frame - start_frame, 'frames'

    cyclableImages = itertools.cycle(imgs)

    # set up windows
    cv2.namedWindow('frame')
    cv2.namedWindow('frame2')
    cv2.namedWindow('controls')

    cv2.moveWindow('frame1', 0, 100)
    cv2.moveWindow('frame2', 550, 100)
    cv2.moveWindow('controls', 250, 750)

    # set up controls
    def nothing(x):
        print x

    cv2.createTrackbar('a', 'controls', 1, 20, nothing)
    cv2.createTrackbar('b', 'controls', 1, 20, nothing)
    cv2.createTrackbar('c', 'controls', 1, 255, nothing)
    cv2.createTrackbar('d', 'controls', 1, 255, nothing)
    cv2.createTrackbar('e', 'controls', 2, 255, nothing)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []

    def init():
        ax.set_ylim(30000, 40000) # todo: autoscale this!
        ax.set_xlim(0, 10)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line,

    def generate_frames():
        t = 0
        linecolor = (np.random.randint(100, 255), np.random.randint(
            100, 255), np.random.randint(100, 255))
        while True:
            # Capture frame-by-frame
            frame = next(cyclableImages)
            frame = cv2.GaussianBlur(frame, (7, 7), 2)
            frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                        cv2.THRESH_BINARY,19,2)
            frame = cv2.medianBlur(frame, 15)
            # cv2.imshow('frame', frame)

            contours, hierarchy = cv2.findContours(
                frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(frame, contours, -1, linecolor, 3)
            area = 0
            for i in contours:
                area = area + cv2.arcLength(i, True)
            # print area
            t = t + 0.1
            yield t, area

            # cv2.imshow('contour output', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def update(data):
        # update the data 
        t, y = data
        y = y

        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if t >= xmax:
            ax.set_xlim(xmin, 2 * xmax)
            ax.figure.canvas.draw()
        if y >= ymax:
            ax.set_ylim(ymin, 2 * ymax)
            ax.figure.canvas.draw()

        line.set_data(xdata, ydata)

        return line,

    ani = animation.FuncAnimation(fig, update, generate_frames, blit=False, interval=10,
                                  repeat=False, init_func=init)

    # ani = animation.FuncAnimation(fig, update, generate_frames, interval=100)
    plt.show()

    # while(True):
    #     process_frame(cyclableImages)
    # print area
    # if circles is not None:
    # for i in circles[0, :]:
    # draw the outer circle
    # cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    # cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    # print len(circles[0, :])
    #     cv2.imshow('frame', frame)
    cv2.destroyAllWindows()
