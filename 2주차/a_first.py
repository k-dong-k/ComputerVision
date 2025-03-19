import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

gh = cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(gh,color = 'b', linewidth = 1)
plt.show()

t,bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

h = cv.calcHist([bin_img],[0],None,[256],[0,256])
cv.imshow('R channel binarization', bin_img)
plt.plot(h,color = 'r', linewidth = 1)
plt.show()