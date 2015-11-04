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

pi_4 = 4*math.pi


def find_circles(frame):
    frame_gray = cv2.GaussianBlur(frame, (5, 5), 2)

    edges = frame_gray - cv2.erode(frame_gray, None)
    _, bin_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)
    height, width = bin_edge.shape
    mask = np.zeros((height+2, width+2), dtype=np.uint8)
    cv2.floodFill(bin_edge, mask, (0, 0), 255)

    components = segment_on_dt(bin_edge)

    circles, obj_center = [], []
    contours, _ = cv2.findContours(components,
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        c = c.astype(np.int64) # XXX OpenCV bug.
        area = cv2.contourArea(c)
        if 100 < area < 3000:
            arclen = cv2.arcLength(c, True)
            circularity = (pi_4 * area) / (arclen * arclen)
            if circularity > 0.5: # XXX Yes, pretty low threshold.
                circles.append(c)
                box = cv2.boundingRect(c)
                obj_center.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))
    
    return circles, obj_center


def segment_on_dt(img):
    border = img - cv2.erode(img, None)

    dt = cv2.distanceTransform(255 - img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)

    lbl, ncc = label(dt)
    lbl[border == 255] = ncc + 1

    lbl = lbl.astype(np.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lbl)
    lbl[lbl < 1] = 0
    lbl[lbl > ncc] = 0

    lbl = lbl.astype(np.uint8)
    lbl = cv2.erode(lbl, None)
    lbl[lbl != 0] = 255
    return lbl


def track_center(objcenter, newdata):
    for i in xrange(len(objcenter)):
        ostr, oc = objcenter[i]
        best = min((abs(c[0]-oc[0])**2+abs(c[1]-oc[1])**2, j)
                for j, c in enumerate(newdata))
        j = best[1]
        if i == j:
            objcenter[i] = (ostr, new_center[j])
        else:
            print objcenter[i]
            print j
            print objcenter[j]
            print "Swapping %s <-> %s" % ((i, objcenter[i]), (j, objcenter[j]))
            objcenter[i], objcenter[j] = objcenter[j], objcenter[i]


if __name__ == '__main__':
    # todo: print a usage string (argv[1] is directory to scan)
    assert len(sys.argv) == 3, "WRONG ARGS"

    frame_count = int(sys.argv[2])

    # get all of the files from the directory
    dir_path = sys.argv[1]
    files = filter(lambda x: splitext(x)[1] == '.pgm', listdir(dir_path))
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    files = [os.path.join(dir_path, x) for x in files]
    
    # read files
    left, top, right, bottom = 0, 0, 1024, 1024
    imgs = [cv2.imread(x, 0)[left:right, top:bottom] for x in files[:frame_count]]

    divisor = 1
    imgs = [cv2.resize(x, (x.shape[1] / divisor, x.shape[0] / divisor)) for x in imgs]

    print 'read in', frame_count, 'frames'
    imgs = imgs[5:]
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

    cv2.createTrackbar('a', 'controls', 55, 50, nothing)
    cv2.createTrackbar('b', 'controls', 2, 50, nothing)
    cv2.createTrackbar('c', 'controls', 1, 255, nothing)
    cv2.createTrackbar('d', 'controls', 1, 255, nothing)
    cv2.createTrackbar('e', 'controls', 2, 255, nothing)

    obj_center = None

    while(True):

        # Capture frame-by-frame
        frame = next(cyclableImages)

        a = cv2.getTrackbarPos('a', 'controls')
        b = cv2.getTrackbarPos('b', 'controls')

        frame = cv2.medianBlur(frame, 15)
        cv2.imshow('frame2', frame)
        low = (a if a%2==1 and a > 1 else a+1)
        high = (b if b%2==1 else b+1)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, low, high)

        circles = cv2.HoughCircles(frame, cv2.cv.CV_HOUGH_GRADIENT,2,12,
                                    param1=100,param2=44,minRadius=10,maxRadius=28)


        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # circles = np.uint16(np.around(circles))
        if circles is not None:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                # cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            print len(circles[0,:])

        
        # original = np.copy(frame)
        
        
        # frame = 255 - frame

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
