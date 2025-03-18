import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

t,bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적 임곘값 = ', t)

h = cv.calcHist([bin_img],[0],None,[256],[0,256])
cv.imshow('R channel binarization', bin_img)
plt.plot(h,color = 'r', linewidth = 1)
plt.show()