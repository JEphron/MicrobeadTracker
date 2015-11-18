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
import argparse

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


def process_frames(frames):
    """ Do fun stuff. 
            frames -- iteratable containing the images to process/draw
            Should yield some data for matplotlib 
    """
    print len(frames)
    for frame in frames:
        # .. process ..
        # cv2.imshow('blah', frame)

        frame = cv2.GaussianBlur(frame, (7, 7), 2)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 19, 2)
        frame = cv2.medianBlur(frame, 21)
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def get_contours(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def show_contours(contours, frame, linecolor):
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)        
    cv2.drawContours(frame, contours, -1, linecolor, 3)
    # cv2.imshow('contours', frame)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze a sequence of frames from the microbead camera')
    parser.add_argument("indirectory", type=str, help="path to a directory containing a sequence of sequentially titled .pgm files")
    parser.add_argument("startframe", type=int, help="the index of the first frame (integer)")
    parser.add_argument("endframe", type=int, help="the index of the last frame (integer)")
    parser.add_argument("fps", type=float, help="frames per second (integer or float)")
    parser.add_argument("--show", action="store_true", help="toggles displaying the image sequence")
    parser.add_argument("--png", help="output graph to a .png image")
    parser.add_argument("--csv", help="output data to a .csv file")
    
    args = parser.parse_args()

    # args.show # boolean
    args.png # string
    args.csv # string

    # square for now
    width = 1024
    crop_rect = (0, 0, width, width) 
    img_sequence = read_image_sequence(
        args.indirectory, args.startframe, args.endframe, crop_rect)
    print 'read in', args.endframe - args.startframe, 'frames'

    # unprocessed_frames = itertools.cycle(img_sequence) if args.loop else img_sequence
    
    unprocessed_frames = img_sequence # eh

    fig, ax = plt.subplots()
    fig.suptitle('Microbead Analysis', fontsize=20)
    plt.xlabel('time (seconds)', fontsize=18)
    plt.ylabel('beads', fontsize=16)
    line, = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []
    
    global_ymin = 0
    global_ymax = 1

    def init():
        ax.set_ylim(0, 500)
        ax.set_xlim(0, len(img_sequence)/args.fps)
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

        # print ymax, max(ydata), ymin, min(ydata)
        ax.set_ylim(min(ydata)-1, max(ydata)+1)
        ax.figure.canvas.draw()

        if t >= xmax:
            ax.set_xlim(xmin, 2 * xmax)
            ax.figure.canvas.draw()

        line.set_data(xdata, ydata)

        return line,

    def gen_area():
        # time axis for the plot
        t = 0
    
        # calculate the mean area of contours in the final frame
        average_area = 0
        final_frame_contours = get_contours(img_sequence[-1])
        for i in final_frame_contours:
            average_area = average_area + cv2.arcLength(i, True)
        average_area = average_area / len(final_frame_contours)

        min_area = sys.maxint
        for frame in process_frames(unprocessed_frames):
            if args.show:
                cv2.imshow('window', frame)
            contours = get_contours(frame) # note: modifies source image
            show_contours(contours, frame, (0, 255, 0))
            
            area = 0
            average_area = 0
            
            for i in contours:
                area = area + cv2.arcLength(i, True)
            
            if average_area is 0:
                average_area = area / len(contours)
            
            if area / average_area < min_area:
                min_area = area / average_area
                print min_area

            # todo: don't do this
            if min_area <  10:
                break
            
            t = t + 1

            yield t/args.fps, min_area

        if args.png:
            print "saved graph to", args.png
            plt.savefig(args.png)


    # frame_generator = functools.partial(process_frames, unprocessed_frames)

    ani = animation.FuncAnimation(fig, update, gen_area, blit=False, interval=100,
                                  repeat=False, init_func=init)
    plt.show()
