import cv2
import numpy as np
import os
from os import listdir
from os import path
import sys
import itertools
import collections
import math
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import timeit
import functools

# boring


def read_image_sequence(dir_path, start_frame, end_frame, crop_rect):
    """ Reads in a sequence of .pgm files from a directory and returns it
            dir_path -- path to the directory containing the files
            start_frame -- number at which to start
            end_frame -- number at which to end
            crop_rect -- a rectangle represented as a tuple: (left, right, top, bottom)
            returns a list of images
    """
    # we only want pgm files
    files = filter(lambda x: path.splitext(x)[1] == '.pgm', listdir(dir_path))
    # they have numerical names and we want them sorted
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    # get us the file paths
    files = [os.path.join(dir_path, x) for x in files]
    # read in the images and crop them
    left, top, right, bottom = crop_rect
    imgs = [cv2.imread(x, 0)[left:right, top:bottom]
            for x in files[start_frame:end_frame]]
    # scale here if desired
    scale = 1
    imgs = [cv2.resize(x, x.shape * scale) for x in imgs]
    return imgs

# idea:
# downsample
# compute circle centers
# apply circle algorithm to those centers


def process_and_draw(frames):
    """ Do fun stuff. 
            frames -- iteratable containing the images to process/draw
            Should yield some data for matplotlib 
    """
    for frame in frames:
        # .. process ..

        frame = cv2.GaussianBlur(frame, (7, 7), 2)
        cv2.imshow('original', frame)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 19, 2)
        frame = cv2.medianBlur(frame, 21)
        cv2.imshow('frame', frame)
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    dir_path = sys.argv[1]
    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])
    mode = sys.argv[4]

    # square for now
    width = 1024
    crop_rect = (0, 0, width, width) 
    img_sequence = read_image_sequence(
        dir_path, start_frame, end_frame, crop_rect)
    print 'read in', end_frame - start_frame, 'frames'

    cyclable_frames = itertools.cycle(img_sequence)

    if mode == 'raw' or mode == '':
        for i in process_and_draw(cyclable_frames):
            pass
    elif mode == 'histogram':
        fig, ax = plt.subplots()

        def animate(param):
            param = param.ravel()
            print param
            plt.clf()
            plt.hist(param, 256, [0, 256])

        # bind cyclable_frames to the first argument of process_and_draw
        frame_generator = functools.partial(process_and_draw, cyclable_frames)
        # animate the histogram!
        ani = animation.FuncAnimation(fig, animate, frame_generator, interval=10,
                                      repeat=False, blit=False)
        plt.show()
    elif mode == 'contour-area':
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.grid()
        xdata, ydata = [], []
        
        global_ymin = 0
        global_ymax = 1

        def init():
            ax.set_ylim(30000, 40000)  # todo: autoscale this!
            ax.set_xlim(0, 10)
            del xdata[:]
            del ydata[:]
            line.set_data(xdata, ydata)
            return line,

        def update(data):
            # update the data
            t, y = data

            xdata.append(t)
            ydata.append(y)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            print ymax, max(ydata), ymin, min(ydata)
            ax.set_ylim(min(ydata), max(ydata))
            ax.figure.canvas.draw()

            if t >= xmax:
                ax.set_xlim(xmin, 2 * xmax)
                ax.figure.canvas.draw()

            # if y >= ymax:
            #     ax.set_ylim(ymin, 2 * ymax)
            #     ax.figure.canvas.draw()


            line.set_data(xdata, ydata)

            return line,

        def gen_area():
            t = 0
            for frame in process_and_draw(cyclable_frames):
                contours, hierarchy = cv2.findContours(
                    frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                
                area = 0
                for i in contours:
                    area = area + cv2.arcLength(i, True)
                # print area
                t = t + 0.1
                yield t, area

        frame_generator = functools.partial(process_and_draw, cyclable_frames)

        ani = animation.FuncAnimation(fig, update, gen_area, blit=False, interval=10,
                                      repeat=False, init_func=init)

        # ani = animation.FuncAnimation(fig, update, generate_frames, interval=100)
        plt.show()

