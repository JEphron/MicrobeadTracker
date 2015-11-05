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
    while True:
        frame = next(frames)

        # .. process ..
        frame = cv2.GaussianBlur(frame, (5, 5), 2)
        cv2.imshow('frame', frame)
        yield frame.ravel()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    dir_path = sys.argv[1]
    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])
    mode = sys.argv[4]
    
    crop_rect = (512, 512, 1024, 1024)
    img_sequence = read_image_sequence(
        dir_path, start_frame, end_frame, crop_rect)
    print 'read in', end_frame - start_frame, 'frames'

    cyclable_frames = itertools.cycle(img_sequence)
    
    if mode == 'default' or mode == '':
        for i in process_and_draw(cyclable_frames):
            print i
    elif mode == 'histogram':
        fig, ax = plt.subplots()
        
        def animate(param):
            print param
            plt.clf() 
            plt.hist(param,256,[0,256])

        # bind cyclable_frames to the first argument of process_and_draw
        frame_generator = functools.partial(process_and_draw, cyclable_frames)
        # animate the histogram!
        ani = animation.FuncAnimation(fig, animate, frame_generator, interval=10,
                                      repeat=False, blit=False)
        plt.show()
