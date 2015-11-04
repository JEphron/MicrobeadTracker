import cv2
import numpy as np
img = cv2.imread('./Vd0YK.png', 0)
img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
# im = cv2.Canny(im, 0, 50)
img = cv2.medianBlur(img, 15)

size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while(not done):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()
    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True



contours, hierarchy = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



imgcol = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

cv2.drawContours(imgcol, contours, -1, (0,255,0), 3)

cv2.imshow("skel", imgcol)

cv2.waitKey(0)
cv2.destroyAllWindows()
