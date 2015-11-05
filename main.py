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
    left, top, right, bottom = 0, 0, 512, 512
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
        ax.set_ylim(0, 300000)
        ax.set_xlim(0, 10)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line,

    def generate_frames():
        t = 0
        while True:
            # Capture frame-by-frame
            frame = next(cyclableImages)
            original = np.copy(frame)
            a = cv2.getTrackbarPos('a', 'controls')
            b = cv2.getTrackbarPos('b', 'controls')
            # blar
            frame = cv2.medianBlur(frame, 15)
            # frame = cv2.GaussianBlur(frame, (5, 5), 2)
            cv2.imshow('frame2', frame)  # original

            hist, bins = np.histogram(frame.flatten(), 256, [0, 256])

            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))

            frame = clahe.apply(frame)

            _, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

            # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,4)

            contours, hierarchy = cv2.findContours(
                frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
            area = 0
            for i in contours:
                area = area + cv2.contourArea(i)
            # print area
            t = t + 0.1
            yield t, area

            cv2.imshow('contour output', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def update(data):
        # update the data

        t, y = data
        y = y

        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()

        if t >= xmax:
            ax.set_xlim(xmin, 2 * xmax)
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
