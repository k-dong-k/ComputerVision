import cv2 as cv
import numpy as np
import sys

img = cv.imread('tree.png')

row, col = img.shape[:2]
cp = (col / 2, row / 2) 
rot = cv.getRotationMatrix2D(cp, 45, 1.5) 
dst = cv.warpAffine(img, rot, (int(col * 1.5), int(row * 1.5)), flags = cv.INTER_LINEAR)

if img.shape[0] != dst.shape[0]:
    dst = cv.resize(dst, (int(col * 1.5), img.shape[0]))
    
result = np.hstack([img, dst])

cv.imshow('result', result)

cv.waitKey()
cv.destroyAllWindows()