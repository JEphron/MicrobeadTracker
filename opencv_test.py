import cv2
import numpy as np
import os
from os import listdir
from os.path import splitext
import sys
import itertools
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 60)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


if __name__ == '__main__':
    # todo: print a usage string (argv[1] is directory to scan)
    assert len(sys.argv) == 3, "WRONG ARGS"

    frame_count = int(sys.argv[2])

    # Get all of the files from the directory
    dir_path = sys.argv[1]
    files = filter(lambda x: splitext(x)[1] == '.pgm', listdir(dir_path))
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    files = [os.path.join(dir_path, x) for x in files]
    # print files
    imgs = [cv2.imread(x, 0) for x in files[:frame_count]]
    # src.create(rows,cols,CV_8UC1);
    # src = imread(your-file, CV_8UC1);
    divisor = 4
    imgs = [cv2.resize(x, (x.shape[1]/divisor, x.shape[0]/divisor)) for x in imgs]
    # imgs = [cv2.threshold(x, 127, 255 , cv2.THRESH_BINARY)[1] for x in imgs]
    print 'read in', frame_count, 'frames'

    cyclableImages = itertools.cycle(imgs)

    cv2.namedWindow('controls')

    def nothing(x):
        pass

    cv2.createTrackbar('a','controls',0,255,nothing)
    cv2.createTrackbar('b','controls',1,255,nothing)
    cv2.createTrackbar('c','controls',1,255,nothing)
    cv2.createTrackbar('d','controls',1,255,nothing)
    cv2.createTrackbar('e','controls',2,255,nothing)

    while(True):

        # Capture frame-by-frame
        frame = next(cyclableImages)
        a = cv2.getTrackbarPos('a','controls')
        b = cv2.getTrackbarPos('b','controls')
        c = cv2.getTrackbarPos('c','controls')
        d = cv2.getTrackbarPos('d','controls')
        e = cv2.getTrackbarPos('e','controls')
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        frame = cv2.medianBlur(frame, 5)
        # frame = cv2.Canny(frame,a,b)
        # circles = cv2.HoughCircles(frame, cv2.cv.CV_HOUGH_GRADIENT, 1, 4, param1=100, param2=100, minRadius=5, maxRadius=10)
        # # print circles
        # # circles = cv2.HoughCircles(frame, cv2.cv.CV_HOUGH_GRADIENT,1,20,
        # #                             param1=50,param2=30,minRadius=0,maxRadius=0)
        

        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # # circles = np.uint16(np.around(circles))
        # if circles is not None:
        #     for i in circles[0,:]:
        #         # draw the outer circle
        #         cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        #         # draw the center of the circle
        #         cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)


        # frame = cv2.adaptiveThreshold(frame, high, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY)[1]
        # cv2
        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()